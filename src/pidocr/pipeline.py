"""Turn one input file into one searchable-PDF output, plus batch discovery."""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover
    pdfium = None

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:  # pragma: no cover
    PdfReader = PdfWriter = None

try:
    from reportlab.pdfgen import canvas as rl_canvas
except ImportError:  # pragma: no cover
    rl_canvas = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from pidocr.config import ALL_EXTS, PDF_EXTS, Config
from pidocr.engine import ocr_image
from pidocr.geometry import get_page_rotation, render_pdf_page_for_ocr
from pidocr.overlay import build_invisible_text_overlay

log = logging.getLogger("pidocr")

REQUIRED_PACKAGES = {
    "pypdfium2": pdfium,
    "pypdf": PdfReader,
    "reportlab": rl_canvas,
    "Pillow": Image,
}


def check_dependencies() -> None:
    missing = [name for name, mod in REQUIRED_PACKAGES.items() if mod is None]
    if missing:
        raise SystemExit(
            "Missing required package(s): "
            + ", ".join(missing)
            + "\nInstall with: pip install pidocr[ocr]"
        )


# ======================================================================
# Page range parsing (tesseract-style "1-5,9,12-14")
# ======================================================================
def parse_page_ranges(spec: str) -> list[tuple[int, int]]:
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
        else:
            a = b = int(part)
        if a < 1 or b < a:
            raise ValueError(f"Invalid page range: {part!r}")
        ranges.append((a, b))
    return ranges


def page_in_ranges(page_num_1based: int, ranges: list[tuple[int, int]] | None) -> bool:
    if not ranges:
        return True
    return any(a <= page_num_1based <= b for a, b in ranges)


# ======================================================================
# Result tracking
# ======================================================================
@dataclass
class FileResult:
    input_path: Path
    output_path: Path | None = None
    ok: bool = False
    pages_processed: int = 0
    lines_added: int = 0
    error: str | None = None
    elapsed: float = 0.0


# ======================================================================
# PDF processing
# ======================================================================
def process_pdf(engine, input_path: Path, output_path: Path, cfg: Config) -> FileResult:
    result = FileResult(input_path=input_path, output_path=output_path)
    t0 = time.time()

    pdfium_doc = pdfium.PdfDocument(str(input_path))
    reader = PdfReader(str(input_path))
    writer = PdfWriter()

    n_pages = len(reader.pages)

    for i in range(n_pages):
        page_num = i + 1
        page = reader.pages[i]

        if not page_in_ranges(page_num, cfg.page_ranges):
            writer.add_page(page)
            log.debug("  Page %d/%d: skipped (outside --pages)", page_num, n_pages)
            continue

        original_rotation = get_page_rotation(page)
        log.debug(
            "  Page %d/%d: %.1f x %.1f pt, /Rotate=%d",
            page_num,
            n_pages,
            float(page.mediabox.width),
            float(page.mediabox.height),
            original_rotation,
        )

        pdfium_page = pdfium_doc[i]
        img_array, img_size_px = render_pdf_page_for_ocr(pdfium_page, original_rotation, cfg.dpi)

        try:
            items = ocr_image(engine, img_array, cfg)
        except Exception as e:
            log.warning("  Page %d: OCR failed (%s), keeping page unchanged", page_num, e)
            items = []

        # Transfer /Rotate into page content so PDF-content coordinates match
        # the OCR image's orientation. This is what fixes 90/180/270 pages.
        if original_rotation:
            page.transfer_rotation_to_content()

        canvas_w_pt = float(page.mediabox.width)
        canvas_h_pt = float(page.mediabox.height)
        canvas_left_pt = float(page.mediabox.left)
        canvas_bottom_pt = float(page.mediabox.bottom)

        render_w_pt = float(page.cropbox.width)
        render_h_pt = float(page.cropbox.height)
        render_left_pt = float(page.cropbox.left)
        render_bottom_pt = float(page.cropbox.bottom)

        if items:
            overlay_buf = build_invisible_text_overlay(
                items,
                img_size_px,
                canvas_w_pt,
                canvas_h_pt,
                render_w_pt,
                render_h_pt,
                render_left_pt,
                render_bottom_pt,
                canvas_left_pt,
                canvas_bottom_pt,
                cfg,
            )
            overlay_reader = PdfReader(overlay_buf)
            page.merge_page(overlay_reader.pages[0])
            result.lines_added += len(items)
            log.info("  Page %d/%d: %d OCR line(s) added", page_num, n_pages, len(items))
        else:
            log.info("  Page %d/%d: no text detected", page_num, n_pages)

        result.pages_processed += 1
        writer.add_page(page)

    if reader.metadata:
        writer.add_metadata(reader.metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        writer.write(f)
    tmp_path.replace(output_path)

    pdfium_doc.close()

    result.ok = True
    result.elapsed = time.time() - t0
    return result


# ======================================================================
# Image processing (image -> single-page searchable PDF)
# ======================================================================
def process_image(engine, input_path: Path, output_path: Path, cfg: Config) -> FileResult:
    result = FileResult(input_path=input_path, output_path=output_path)
    t0 = time.time()

    pil_img = Image.open(input_path).convert("RGB")
    img_array = np.array(pil_img)
    img_w_px, img_h_px = pil_img.size

    try:
        items = ocr_image(engine, img_array, cfg)
    except Exception as e:
        log.warning("  OCR failed (%s), producing an unsearchable page", e)
        items = []

    # The page is sized in points from the image's pixel size at --image-dpi,
    # exactly like tesseract's own image-to-pdf convention.
    page_w_pt = img_w_px * 72.0 / cfg.image_dpi
    page_h_pt = img_h_px * 72.0 / cfg.image_dpi

    # 1. Draw the visible image as the page background.
    img_buf = io.BytesIO()
    c = rl_canvas.Canvas(img_buf, pagesize=(page_w_pt, page_h_pt))
    c.drawImage(str(input_path), 0, 0, width=page_w_pt, height=page_h_pt)
    c.save()
    img_buf.seek(0)
    base_reader = PdfReader(img_buf)
    base_page = base_reader.pages[0]

    # 2. Build the invisible OCR overlay in the same page coordinate space.
    if items:
        overlay_buf = build_invisible_text_overlay(
            items,
            (img_w_px, img_h_px),
            page_w_pt,
            page_h_pt,
            page_w_pt,
            page_h_pt,
            0.0,
            0.0,
            0.0,
            0.0,
            cfg,
        )
        overlay_reader = PdfReader(overlay_buf)
        base_page.merge_page(overlay_reader.pages[0])
        result.lines_added = len(items)
        log.info("  %d OCR line(s) added", len(items))
    else:
        log.info("  No text detected")

    writer = PdfWriter()
    writer.add_page(base_page)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        writer.write(f)
    tmp_path.replace(output_path)

    result.pages_processed = 1
    result.ok = True
    result.elapsed = time.time() - t0
    return result


def process_file(engine, input_path: Path, output_path: Path, cfg: Config) -> FileResult:
    """Dispatch to process_pdf or process_image based on the input's extension."""
    if input_path.suffix.lower() in PDF_EXTS:
        return process_pdf(engine, input_path, output_path, cfg)
    return process_image(engine, input_path, output_path, cfg)


# ======================================================================
# File discovery & output-path resolution
# ======================================================================
def discover_inputs(paths: list[str], recursive: bool, exts: set[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            walker = p.rglob("*") if recursive else p.glob("*")
            for child in sorted(walker):
                if child.is_file() and child.suffix.lower() in exts:
                    found.append(child)
        else:
            log.error("Input path does not exist: %s", raw)
    return found


def resolve_output_path(
    in_file: Path,
    base_input: Path | None,
    output_arg: str | None,
    suffix: str,
    in_place: bool,
) -> Path:
    if in_place:
        return in_file

    if output_arg is None:
        return in_file.with_name(f"{in_file.stem}{suffix}.pdf")

    out = Path(output_arg)

    # Multiple/folder inputs, or an explicit trailing separator, or an
    # existing directory -> treat output_arg as a directory and mirror
    # the relative structure under it.
    treat_as_dir = (
        out.is_dir()
        or output_arg.endswith(os.sep)
        or (base_input is not None and base_input.is_dir())
    )

    if treat_as_dir:
        if base_input is not None and base_input.is_dir():
            rel = in_file.relative_to(base_input)
        else:
            rel = Path(in_file.name)
        rel = rel.with_suffix(".pdf")
        return out / rel

    return out
