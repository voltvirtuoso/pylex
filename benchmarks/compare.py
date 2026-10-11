"""Compare Pylex with OCRmyPDF (Tesseract) and OCRmyPDF + the PaddleOCR plugin.

Builds scan-like test PDFs whose word positions are known, runs each tool, and scores the
searchable text layer it produced:

  recall   share of ground-truth words whose text appears in the layer
  aligned  share found in the right place (word centre within 0.6 line-heights vertically
           and 4 line-heights horizontally of the true word; lines are placed as a whole by
           Pylex, so the horizontal tolerance is deliberately coarse)
  extra    OCR words that match no ground-truth word, as a share of all OCR words

Cases: plain text page, dense drawing-like page with many small labels, and the same plain page
stored sideways with /Rotate 90 or 270 (what scanners and rotated CAD exports produce).

    python benchmarks/compare.py                     # all tools that are installed
    python benchmarks/compare.py --only pylex --pylex-args "--model-size small"
    python benchmarks/compare.py --input-dir scans/  # your own files: timing and word counts only
    python benchmarks/compare.py --self-test         # validate the scoring with a perfect-OCR oracle

Install the tools you want to compare: pip install ocrmypdf ocrmypdf-paddleocr (plus Tesseract and
Ghostscript on the system) and a Pylex profile (see docs/installation.md).
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pypdfium2 as pdfium
import reportlab
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

FONT = str(Path(reportlab.__file__).parent / "fonts" / "Vera.ttf")
VOCAB = ["pump", "valve", "inlet", "outlet", "drain", "vent", "tank", "line", "flow", "gauge", "check", "relief",
         "P-101", "FV-2203", "PT-114", "TIC-300", "6-PG-1001", "HV-17", "LAH-42", "XV-9001", "250mm", "DN50"]
DPI = 200


# ---------------------------------------------------------------- test documents
def _page_image(kind: str, seed: int):
    """Return (PIL image, [(word, (x0, y0, x1, y1)) in pixels, upright])."""
    rng = random.Random(seed)
    if kind == "dense":
        w, h, size = 3600, 2400, 22
    else:
        w, h, size = 1654, 2339, 34
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, size)
    words = []
    serial = iter(range(1, 100000))  # unique tags: a repeated word could match by chance
    if kind == "dense":
        cols, rows = 14, 24
        cw, rh = w // cols, h // rows
        for r in range(rows):
            for c in range(cols):
                if rng.random() < 0.55:
                    text = f"{rng.choice(VOCAB)}-{next(serial):03d}"
                    x = c * cw + rng.randint(8, cw // 4)
                    y = r * rh + rng.randint(6, rh // 3)
                    box = draw.textbbox((x, y), text, font=font)
                    draw.text((x, y), text, font=font, fill=0)
                    words.append((text, box))
        for r in range(1, rows):  # line work in the gaps between label rows
            draw.line([(0, r * rh - 2), (w, r * rh - 2)], fill=0, width=2)
    else:
        x, y, line_h = 120, 140, int(size * 1.9)
        while y < h - 160:
            x = 120
            while True:
                text = f"{rng.choice(VOCAB[:12])}-{next(serial):03d}"
                tw = draw.textlength(text, font=font)
                if x + tw > w - 120:
                    break
                box = draw.textbbox((x, y), text, font=font)
                draw.text((x, y), text, font=font, fill=0)
                words.append((text, box))
                x += int(tw + size * 0.6)
            y += line_h
    return img, words


def build_case(name: str, outdir: Path, seed: int = 7):
    """Write <name>.pdf and return the ground truth in displayed points (origin top-left)."""
    kind, _, rot = name.partition("-rot")
    img, words = _page_image(kind, seed)
    scale = 72 / DPI
    truth = [(t, tuple(v * scale for v in b)) for t, b in words]
    stored = img
    if rot:  # store the scan sideways and let /Rotate display it upright
        stored = img.rotate(90 if rot == "90" else -90, expand=True)
    pdf = outdir / f"{name}.pdf"
    stored.convert("RGB").save(pdf, resolution=DPI)
    if rot:
        reader, writer = PdfReader(str(pdf)), PdfWriter()
        page = reader.pages[0]
        page.rotate(int(rot))
        writer.add_page(page)
        with open(pdf, "wb") as fh:
            writer.write(fh)
    return pdf, truth, (img.width * scale, img.height * scale)


# ---------------------------------------------------------------- scoring
def layer_words(pdf_path):
    """Words in the PDF's text layer as (text, centre_x, centre_y) in displayed points, top-left origin."""
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    left, bottom, right, top = page.get_mediabox()
    width, height = right - left, top - bottom
    rot = page.get_rotation() % 360
    tp = page.get_textpage()
    text = tp.get_text_range()
    out, cur = [], []

    def flush():
        if cur:
            xs = [c[0] for c in cur]
            ys = [c[1] for c in cur]
            out.append(("".join(c[2] for c in cur), sum(xs) / len(xs), sum(ys) / len(ys)))
            cur.clear()

    for i, ch in enumerate(text):
        if ch.isspace() or ch in "\ufffe\uffff":
            flush()
            continue
        l, b, r, t = tp.get_charbox(i, loose=True)
        ux, uy = (l + r) / 2 - left, (b + t) / 2 - bottom
        if rot == 0:
            x, y = ux, height - uy
        elif rot == 90:
            x, y = uy, ux
        elif rot == 180:
            x, y = width - ux, uy
        else:
            x, y = height - uy, width - ux
        cur.append((x, y, ch))
    flush()
    return out


def score(pdf_path, truth):
    got = layer_words(pdf_path)
    norm = lambda s: s.casefold().strip(".,;:")
    pool = {}
    for text, x, y in got:
        pool.setdefault(norm(text), []).append([x, y, False])
    # Pass 1 claims in-place matches first, so a stray word elsewhere on the page can never
    # steal a match that belongs to a different ground-truth word with the same text.
    aligned = 0
    unmatched = []
    for text, (x0, y0, x1, y1) in truth:
        free = [c for c in pool.get(norm(text), []) if not c[2]]
        lh = y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        near = [c for c in free if abs(c[1] - cy) <= 0.6 * lh and abs(c[0] - cx) <= 4 * lh + (x1 - x0) / 2]
        if near:
            min(near, key=lambda c: abs(c[0] - cx) + abs(c[1] - cy))[2] = True
            aligned += 1
        else:
            unmatched.append(text)
    # Pass 2: position-free recall for what is left (right text, wrong place).
    found = aligned
    for text in unmatched:
        free = [c for c in pool.get(norm(text), []) if not c[2]]
        if free:
            free[0][2] = True
            found += 1
    total = max(len(truth), 1)
    n_got = max(len(got), 1)
    matched = sum(1 for v in pool.values() for c in v if c[2])
    return {"truth_words": len(truth), "ocr_words": len(got), "recall": found / total,
            "aligned": aligned / total, "extra": max(0, len(got) - matched) / n_got}


# ---------------------------------------------------------------- tools
def tools(pylex_args: list[str]):
    exe = sys.executable
    specs = {"pylex": [exe, "-m", "pylex", "{in}", "-o", "{out}", "--force", "-q", *pylex_args]}
    if shutil.which("ocrmypdf"):
        specs["ocrmypdf-tesseract"] = ["ocrmypdf", "--force-ocr", "-l", "eng", "{in}", "{out}"]
        if _module_available("ocrmypdf_paddleocr"):
            specs["ocrmypdf-paddleocr"] = ["ocrmypdf", "--plugin", "ocrmypdf_paddleocr", "--force-ocr", "{in}", "{out}"]
    return specs


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def run_tool(cmd, src: Path, dst: Path, timeout: int):
    argv = [a.replace("{in}", str(src)).replace("{out}", str(dst)) for a in cmd]
    dst.unlink(missing_ok=True)
    start = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return time.perf_counter() - start, proc.returncode, proc.stderr[-300:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", help="Run only this tool (repeatable): pylex, ocrmypdf-tesseract, ocrmypdf-paddleocr")
    ap.add_argument("--pylex-args", default="", help='Extra Pylex options, e.g. "--device gpu --model-size small"')
    ap.add_argument("--cases", default="plain,dense,plain-rot90,plain-rot270")
    ap.add_argument("--repeat", type=int, default=1, help="Timed runs per case; the median is reported")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--no-warmup", action="store_true", help="Skip the untimed warm-up run (model download / compile)")
    ap.add_argument("--input-dir", type=Path, help="Run on your own PDFs/images: timing and word counts only")
    ap.add_argument("--out", type=Path, default=Path("benchmark-results"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    args.out.mkdir(parents=True, exist_ok=True)
    avail = tools(args.pylex_args.split())
    chosen = {k: v for k, v in avail.items() if not args.only or k in args.only}
    missing = [k for k in (args.only or []) if k not in avail]
    if missing:
        print("not installed / unavailable:", ", ".join(missing))
    if not chosen:
        return 1

    work = Path(tempfile.mkdtemp(prefix="pylex-bench-"))
    cases = []
    if args.input_dir:
        files = sorted(p for p in args.input_dir.rglob("*") if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        cases = [(p.name, p, None) for p in files]
    else:
        for name in args.cases.split(","):
            pdf, truth, _ = build_case(name.strip(), work)
            cases.append((name.strip(), pdf, truth))

    rows = []
    for tool, cmd in chosen.items():
        if not args.no_warmup and cases:
            run_tool(cmd, cases[0][1], work / "warm.pdf", args.timeout)
        for name, src, truth in cases:
            times, rc, err = [], 0, ""
            dst = args.out / f"{tool}__{Path(name).stem}.pdf"
            for _ in range(args.repeat):
                t, rc, err = run_tool(cmd, src, dst, args.timeout)
                times.append(t)
            row = {"tool": tool, "case": name, "seconds": round(statistics.median(times), 2), "exit": rc}
            if rc != 0 or not dst.exists():
                row["error"] = err.strip()[-200:]
            elif truth is not None:
                row.update({k: round(v, 3) if isinstance(v, float) else v for k, v in score(dst, truth).items()})
            else:
                row["ocr_words"] = len(layer_words(dst))
            rows.append(row)
            print(row, flush=True)
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    (args.out / "results.md").write_text(table(rows, args.input_dir is not None))
    print("\n" + table(rows, args.input_dir is not None))
    print(f"Wrote {args.out}/results.json and results.md. Outputs are kept for visual inspection.")
    return 0


def table(rows, own: bool) -> str:
    if own:
        head = "| Tool | File | Seconds | Words in text layer | Exit |\n|---|---|---:|---:|---:|\n"
        return head + "\n".join(f"| {r['tool']} | {r['case']} | {r['seconds']} | {r.get('ocr_words', '')} | {r['exit']} |" for r in rows)
    head = "| Tool | Case | Seconds | Recall | Aligned | Extra |\n|---|---|---:|---:|---:|---:|\n"
    pct = lambda v: "" if v is None else f"{v:.0%}"
    return head + "\n".join(
        f"| {r['tool']} | {r['case']} | {r['seconds']} | {pct(r.get('recall'))} | {pct(r.get('aligned'))} | {pct(r.get('extra'))} |"
        if "error" not in r else f"| {r['tool']} | {r['case']} | {r['seconds']} | failed: {r['error'][:60]} | | |"
        for r in rows
    )


# ---------------------------------------------------------------- scoring self-test
def oracle(truth, size_pt, rotate: int):
    """A PDF whose invisible text sits exactly on the truth, then rotated by `rotate` degrees."""
    w, h = size_pt
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    for text, (x0, y0, x1, y1) in truth:
        t = c.beginText()
        t.setTextRenderMode(3)
        t.setFont("Helvetica", max(y1 - y0, 4))
        t.setTextOrigin(x0, h - y1)
        t.textOut(text)
        c.drawText(t)
    c.save()
    buf.seek(0)
    writer = PdfWriter()
    page = PdfReader(buf).pages[0]
    if rotate:
        page.rotate(rotate)
    writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out


def rotate_truth(truth, size_pt, deg):
    w, h = size_pt
    out = []
    for text, (x0, y0, x1, y1) in truth:
        if deg == 0:
            box = (x0, y0, x1, y1)
        elif deg == 90:
            box = (h - y1, x0, h - y0, x1)
        elif deg == 180:
            box = (w - x1, h - y1, w - x0, h - y0)
        else:
            box = (y0, w - x1, y1, w - x0)
        out.append((text, box))
    return out


def self_test() -> int:
    work = Path(tempfile.mkdtemp())
    _, truth, size = build_case("plain", work)
    ok = True
    for deg in (0, 90, 180, 270):
        p = work / f"oracle{deg}.pdf"
        p.write_bytes(oracle(truth, size, deg).read())
        good = score(p, rotate_truth(truth, size, deg))
        bad = score(p, truth if deg else rotate_truth(truth, size, 90))
        verdict = good["aligned"] > 0.97 and bad["aligned"] < 0.2
        ok &= verdict
        print(f"rotate {deg:>3}: correct truth aligned={good['aligned']:.0%}, wrong truth aligned={bad['aligned']:.0%} -> {'OK' if verdict else 'FAIL'}")
    print("scoring self-test:", "passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
