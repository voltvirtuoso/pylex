"""Command-line interface: `pidocr <inputs...> [options]`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pidocr import __version__
from pidocr.config import ALL_EXTS, Config
from pidocr.engine import get_ocr_engine
from pidocr.pipeline import (
    FileResult,
    check_dependencies,
    discover_inputs,
    parse_page_ranges,
    process_file,
    resolve_output_path,
)

log = logging.getLogger("pidocr")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pidocr",
        description=(
            "Add an invisible, searchable OCR text layer to PDFs and images "
            "(PaddleOCR-based). Handles rotated PDF pages (/Rotate 90/180/270) "
            "correctly \u2014 the visible drawing is never altered, only made "
            "searchable/selectable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  pidocr drawing.pdf\n"
            "  pidocr drawing.pdf -o searchable.pdf\n"
            "  pidocr scans/ -o out/ -r\n"
            "  pidocr page.png page.tif -o out/\n"
            "  pidocr drawing.pdf --pages 1-5,9\n"
            "  pidocr big_folder/ -r --dry-run\n"
            "  pidocr file.pdf --in-place --force\n"
            "  pidocr drawing.pdf --dpi 400 --min-confidence 0.6 -v\n"
        ),
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more PDF/image files or folders to process.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "Output file (single input) or output folder (multiple/folder "
            "input, structure is mirrored). Default: next to each input as "
            "'<name><suffix>.pdf'."
        ),
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subfolders when an input is a directory.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input file with its OCR'd version (requires --force).",
    )
    parser.add_argument(
        "--suffix",
        default=".ocr",
        metavar="SUFFIX",
        help="Suffix inserted before .pdf when -o is not given (default: '.ocr').",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing output files (and required for --in-place).",
    )
    parser.add_argument(
        "--formats",
        default=None,
        metavar="ext,ext,...",
        help=(
            "Comma-separated extensions to pick up when an input is a folder "
            f"(default: all of {', '.join(sorted(e.lstrip('.') for e in ALL_EXTS))})."
        ),
    )
    parser.add_argument(
        "--pages",
        metavar="RANGE",
        default=None,
        help="Only OCR these 1-based PDF pages, e.g. '1-5,9,12-14'. Other pages are copied through untouched.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be processed and their output paths, then exit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inputs whose output file already exists (instead of erroring or overwriting).",
    )

    ocr_group = parser.add_argument_group("OCR options")
    ocr_group.add_argument("--dpi", type=int, default=300, help="PDF page render DPI for OCR (default: 300).")
    ocr_group.add_argument(
        "--image-dpi", type=int, default=300, help="Assumed DPI for standalone raster images (default: 300)."
    )
    ocr_group.add_argument("--lang", default="en", help="PaddleOCR language code (default: en).")
    ocr_group.add_argument(
        "--min-confidence", type=float, default=0.5, help="Minimum OCR confidence to keep a line (default: 0.5)."
    )
    ocr_group.add_argument(
        "--det-limit-side-len", type=int, default=8000, help="Max side length for text detection (default: 8000)."
    )
    ocr_group.add_argument("--gpu", action="store_true", help="Run PaddleOCR on GPU instead of CPU.")
    ocr_group.add_argument(
        "--no-textline-orientation", action="store_true", help="Disable per-textline orientation classification."
    )

    overlay_group = parser.add_argument_group("Overlay tuning (advanced)")
    overlay_group.add_argument("--font", default="Helvetica", help="Invisible-layer font (default: Helvetica).")
    overlay_group.add_argument(
        "--baseline-fraction",
        type=float,
        default=0.18,
        help="Perpendicular baseline offset as a fraction of box height (default: 0.18).",
    )

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity (-v info, -vv debug).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print warnings/errors.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(
        dpi=args.dpi,
        image_dpi=args.image_dpi,
        min_confidence=args.min_confidence,
        lang=args.lang,
        font=args.font,
        det_limit_side_len=args.det_limit_side_len,
        baseline_fraction=args.baseline_fraction,
        use_textline_orientation=not args.no_textline_orientation,
        use_gpu=args.gpu,
    )
    if args.pages:
        try:
            cfg.page_ranges = parse_page_ranges(args.pages)
        except ValueError as e:
            raise SystemExit(f"pidocr: error: {e}") from e
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    check_dependencies()

    if args.in_place and not args.force:
        parser.error("--in-place requires --force (it overwrites your input files).")

    exts = ALL_EXTS
    if args.formats:
        exts = {("." + e.strip().lstrip(".")).lower() for e in args.formats.split(",") if e.strip()}

    input_files = discover_inputs(args.inputs, args.recursive, exts)
    if not input_files:
        log.error("No matching input files found.")
        return 1

    # base_input: if the user passed exactly one directory, mirror structure
    # relative to it; otherwise outputs are flattened into --output.
    base_input = None
    if len(args.inputs) == 1 and Path(args.inputs[0]).is_dir():
        base_input = Path(args.inputs[0])

    jobs: list[tuple[Path, Path]] = []
    for in_file in input_files:
        if in_file.suffix.lower() not in ALL_EXTS:
            log.warning("Skipping unsupported file: %s", in_file)
            continue
        out_file = resolve_output_path(in_file, base_input, args.output, args.suffix, args.in_place)
        jobs.append((in_file, out_file))

    if args.dry_run:
        print(f"{len(jobs)} file(s) would be processed:\n")
        for in_file, out_file in jobs:
            print(f"  {in_file}  ->  {out_file}")
        return 0

    cfg = config_from_args(args)

    # Filter existing outputs up front per --skip-existing / --force.
    filtered_jobs = []
    for in_file, out_file in jobs:
        if out_file.exists() and not args.in_place:
            if args.skip_existing:
                log.info("Skipping (output exists): %s", out_file)
                continue
            if not args.force:
                log.error(
                    "Output already exists: %s (use --force to overwrite or --skip-existing to skip)",
                    out_file,
                )
                continue
        filtered_jobs.append((in_file, out_file))

    if not filtered_jobs:
        log.error("Nothing to do.")
        return 1

    engine = get_ocr_engine(cfg)

    results: list[FileResult] = []
    for idx, (in_file, out_file) in enumerate(filtered_jobs, 1):
        print(f"[{idx}/{len(filtered_jobs)}] {in_file}")
        try:
            res = process_file(engine, in_file, out_file, cfg)
            results.append(res)
            print(
                f"    -> {out_file}  ({res.pages_processed} page(s), "
                f"{res.lines_added} line(s), {res.elapsed:.1f}s)"
            )
        except Exception as e:
            log.error("  FAILED: %s -> %s", in_file, e)
            results.append(FileResult(input_path=in_file, ok=False, error=str(e)))

    ok = sum(1 for r in results if r.ok)
    failed = [r for r in results if not r.ok]

    print(f"\nDone: {ok}/{len(results)} succeeded.")
    if failed:
        print(f"{len(failed)} file(s) failed:")
        for r in failed:
            print(f"  - {r.input_path}: {r.error}")
        return 1

    return 0
