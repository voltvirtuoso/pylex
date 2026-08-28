"""Command-line interface: `pylex <inputs...> [options]`."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from pylex import __version__
from pylex.config import ALL_EXTS, Config
from pylex.engine import configure_cpu_threads, get_ocr_engine
from pylex.pipeline import (
    FileResult,
    check_dependencies,
    discover_inputs,
    parse_page_ranges,
    process_file,
    resolve_output_path,
)
from pylex.progress import ProgressBar
from pylex.runner import run_parallel

log = logging.getLogger("pylex")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pylex",
        description=(
            "Add an invisible, searchable OCR text layer to PDFs and images "
            "(PaddleOCR-based). Handles rotated PDF pages (/Rotate 90/180/270) "
            "correctly \u2014 the visible drawing is never altered, only made "
            "searchable/selectable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  pylex drawing.pdf\n"
            "  pylex drawing.pdf -o searchable.pdf\n"
            "  pylex scans/ -o out/ -r\n"
            "  pylex page.png page.tif -o out/\n"
            "  pylex drawing.pdf --pages 1-5,9\n"
            "  pylex big_folder/ -r --dry-run\n"
            "  pylex file.pdf --in-place --force\n"
            "  pylex drawing.pdf --dpi 400 --min-confidence 0.6 -v\n"
            "  pylex big_folder/ -r -o out/ -j 4       # 4 files in parallel\n"
            "  pylex dense_pid.pdf --tile-size 1200 --tile-overlap 250\n"
            "                                            # more text in crowded symbol clusters\n"
            "  pylex scans/ -r -xf '*_draft*' -xf 'backup/*'\n"
            "                                            # exclude by wildcard pattern\n"
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
        "-xf",
        "--exclude",
        action="append",
        default=None,
        metavar="PATTERN",
        help=(
            "Exclude files matching this glob/wildcard pattern (fnmatch rules: '*' matches "
            "anything, including path separators, so 'backup/*' and '*_draft*' both work). "
            "Use wildcards around a token when matching part of a filename, e.g. '*_OCR*'. "
            "Matched against the filename and the path relative to the input folder. Repeat "
            "-xf for multiple patterns, or comma-separate several within one, e.g. "
            "-xf '*.tmp,*_old*,archive/*'."
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
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Process N files in parallel using separate worker processes "
            "(each loads its own OCR engine, reused across the files it "
            "picks up). Default: 1 (sequential). Pages within one file are "
            "always OCR'd sequentially; parallelism is across files."
        ),
    )

    ocr_group = parser.add_argument_group("OCR options")
    ocr_group.add_argument("--dpi", type=int, default=300, help="PDF page render DPI for OCR (default: 300).")
    ocr_group.add_argument(
        "--image-dpi", type=int, default=300, help="Assumed DPI for standalone raster images (default: 300)."
    )
    ocr_group.add_argument("--lang", default="en", help="PaddleOCR language code (default: en).")
    ocr_group.add_argument(
        "--ocr-version",
        choices=("PP-OCRv4", "PP-OCRv5", "PP-OCRv6"),
        default="PP-OCRv6",
        help="PaddleOCR model generation (default: PP-OCRv6).",
    )
    ocr_group.add_argument(
        "--model-size",
        choices=("tiny", "small", "medium"),
        default="medium",
        help="PP-OCRv6 model tier (default: medium; small balances speed, tiny favors latency).",
    )
    ocr_group.add_argument(
        "--inference-engine",
        choices=("auto", "paddle_static", "paddle_dynamic", "onnxruntime", "openvino"),
        default="auto",
        help=(
            "Runtime engine (default: auto; uses OpenVINO when available, then "
            "ONNX Runtime, otherwise dynamic Paddle)."
        ),
    )
    ocr_group.add_argument(
        "--runtime-cache-dir",
        default=None,
        metavar="PATH",
        help="Persistent OpenVINO/ONNX Runtime compiled-engine cache directory (default: platform cache).",
    )
    ocr_group.add_argument(
        "--enable-hpi",
        action="store_true",
        help="Enable PaddleOCR high-performance inference and its cached backend selection.",
    )
    ocr_group.add_argument(
        "--hpi-backend",
        choices=("auto", "paddle", "openvino", "onnxruntime"),
        default=None,
        help="Optional HPI backend; OpenVINO/ONNX require supported PaddleOCR 3.x deployment dependencies.",
    )
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
    ocr_group.add_argument(
        "--no-vertical-text-retry",
        action="store_true",
        help="Disable targeted 90-degree retries for low-confidence vertical labels.",
    )
    ocr_group.add_argument(
        "--strict-tiling",
        action="store_true",
        help="Always use configured overlapping tiles; disables the adaptive full-page optimization.",
    )
    ocr_group.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Fast profile for simple images: one full-image pass, no text-line orientation model, "
            "and a 3200 px detector cap. Use normal mode for dense drawings."
        ),
    )
    ocr_group.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="Enable oneDNN/MKLDNN for explicit static CPU inference; dynamic mode avoids static oneDNN compatibility issues.",
    )
    ocr_group.add_argument(
        "--recognition-batch-size",
        type=int,
        default=6,
        metavar="N",
        help="Recognition batch size per image/tile (default: 6).",
    )
    ocr_group.add_argument(
        "--orientation-batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Text-line orientation batch size (default: 8).",
    )
    ocr_group.add_argument(
        "--no-runtime-fallback",
        action="store_true",
        help="Fail instead of falling back if the selected Paddle runtime is unsupported.",
    )
    ocr_group.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        metavar="N",
        help="Native CPU threads per process (default: auto-balanced across --jobs workers).",
    )
    ocr_group.add_argument(
        "--retry-upscale",
        action="store_true",
        help="If a pass finds no text, retry once at 1.5x resolution to recover tiny/low-resolution labels.",
    )

    overlay_group = parser.add_argument_group("Overlay tuning (advanced)")
    overlay_group.add_argument("--font", default="Helvetica", help="Invisible-layer font (default: Helvetica).")
    overlay_group.add_argument(
        "--baseline-fraction",
        type=float,
        default=0.18,
        help="Perpendicular baseline offset as a fraction of box height (default: 0.18).",
    )

    dense_group = parser.add_argument_group(
        "Dense-drawing tuning",
        (
            "A whole-page OCR pass resizes the entire page down before detecting text, so a tag "
            "squeezed between several valve/instrument symbols can shrink below what the detector "
            "picks up, even though the same text on open pipe run is fine. These flags recover it."
        ),
    )
    dense_group.add_argument(
        "--tile-size",
        type=int,
        default=2000,
        metavar="PX",
        help=(
            "OCR the page in overlapping PX x PX windows at full resolution instead of one whole-"
            "page pass (default: 2000). Lower it (e.g. 900-1200) for very crowded drawings. "
            "0 disables tiling (same as --no-tile)."
        ),
    )
    dense_group.add_argument(
        "--tile-overlap",
        type=int,
        default=220,
        metavar="PX",
        help="Overlap between adjacent tiles in pixels, so text near a tile edge isn't cut in half (default: 220).",
    )
    dense_group.add_argument(
        "--no-tile",
        action="store_true",
        help="Disable tiling; OCR each page in a single whole-page pass (equivalent to --tile-size 0).",
    )
    dense_group.add_argument(
        "--dedupe-tol-px",
        type=float,
        default=20.0,
        metavar="PX",
        help=(
            "When tiling, two detections with identical text whose centers land within this many "
            "page-pixels of each other are treated as the same duplicate from tile overlap and merged "
            "(default: 20)."
        ),
    )
    dense_group.add_argument(
        "--det-thresh",
        type=float,
        default=None,
        metavar="F",
        help="PaddleOCR text_det_thresh override (detector pixel-score cutoff). Leave unset to use PaddleOCR's default.",
    )
    dense_group.add_argument(
        "--det-box-thresh",
        type=float,
        default=None,
        metavar="F",
        help=(
            "PaddleOCR text_det_box_thresh override. Lowering this (e.g. 0.4-0.5) recovers weaker "
            "detections in crowded symbol clusters, at some risk of more false positives."
        ),
    )
    dense_group.add_argument(
        "--det-unclip-ratio",
        type=float,
        default=None,
        metavar="F",
        help=(
            "PaddleOCR text_det_unclip_ratio override. Raising this (e.g. 1.8-2.2) expands detected "
            "boxes, helping recover text that's touching or squeezed against line-art."
        ),
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
        ocr_version=args.ocr_version,
        model_size=args.model_size,
        inference_engine=args.inference_engine,
        enable_hpi=args.enable_hpi,
        hpi_backend=args.hpi_backend,
        runtime_cache_dir=args.runtime_cache_dir,
        font=args.font,
        det_limit_side_len=min(args.det_limit_side_len, 3200) if args.fast else args.det_limit_side_len,
        baseline_fraction=args.baseline_fraction,
        use_textline_orientation=not args.no_textline_orientation and not args.fast,
        vertical_text_retry=not args.no_vertical_text_retry and not args.fast,
        use_gpu=args.gpu,
        disable_mkldnn=not args.enable_mkldnn,
        cpu_threads=args.cpu_threads,
        orientation_batch_size=args.orientation_batch_size,
        recognition_batch_size=args.recognition_batch_size,
        allow_runtime_fallback=not args.no_runtime_fallback,
        fast_mode=args.fast,
        retry_upscale=args.retry_upscale,
        tile_size=0 if args.no_tile or args.fast else args.tile_size,
        tile_overlap=args.tile_overlap,
        adaptive_tiling=not args.strict_tiling,
        dedupe_tol_px=args.dedupe_tol_px,
        det_thresh=args.det_thresh,
        det_box_thresh=args.det_box_thresh,
        det_unclip_ratio=args.det_unclip_ratio,
    )
    if args.pages:
        try:
            cfg.page_ranges = parse_page_ranges(args.pages)
        except ValueError as e:
            raise SystemExit(f"pylex: error: {e}") from e
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

    if args.jobs < 1:
        parser.error("--jobs must be >= 1.")
    if args.enable_hpi and args.inference_engine != "auto":
        parser.error("--enable-hpi and --inference-engine are mutually exclusive; use --inference-engine auto.")
    if args.hpi_backend and not args.enable_hpi:
        parser.error("--hpi-backend requires --enable-hpi.")
    if args.jobs > 1 and args.gpu:
        log.warning(
            "--jobs %d with --gpu: multiple worker processes will share the same GPU, "
            "which can contend for memory rather than speed things up. "
            "Consider --jobs 1 with --gpu, or --jobs N without --gpu.",
            args.jobs,
        )
    if args.tile_size < 0:
        parser.error("--tile-size must be >= 0 (0 disables tiling).")
    if args.tile_overlap < 0:
        parser.error("--tile-overlap must be >= 0.")
    if args.tile_size > 0 and args.tile_overlap >= args.tile_size:
        parser.error("--tile-overlap must be smaller than --tile-size.")

    jobs_requested = args.jobs

    exts = ALL_EXTS
    if args.formats:
        exts = {("." + e.strip().lstrip(".")).lower() for e in args.formats.split(",") if e.strip()}

    exclude_patterns: list[str] = []
    for raw in args.exclude or []:
        exclude_patterns.extend(p.strip() for p in raw.split(",") if p.strip())

    input_files = discover_inputs(args.inputs, args.recursive, exts, exclude_patterns)
    if not input_files:
        if exclude_patterns:
            log.error("No matching input files found (check your -xf patterns aren't excluding everything).")
        else:
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
        if exclude_patterns:
            print(f"\n(excluding files matching: {', '.join(exclude_patterns)})")
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

    effective_jobs = min(jobs_requested, len(filtered_jobs))
    cpu_count = os.cpu_count() or 1
    if cfg.cpu_threads is None:
        cfg.cpu_threads = max(1, cpu_count // effective_jobs)
    elif cfg.cpu_threads < 1:
        parser.error("--cpu-threads must be >= 1.")
    if effective_jobs > cpu_count:
        log.info(
            "Requesting %d worker(s) on a machine with %d detected CPU core(s); "
            "proceeding, but returns may diminish past core count.",
            effective_jobs,
            cpu_count,
        )

    if effective_jobs > 1:
        bar = ProgressBar(len(filtered_jobs), label="OCR", enabled=not args.quiet)
        bar.println(
            f"Running with {effective_jobs} parallel worker(s) "
            "(each worker loads its own OCR engine; startup may take a while)..."
        )
        bar.start_spinner(f"Loading OCR engine in {effective_jobs} worker(s)")

        def _progress(done: int, total: int, in_file: Path, res: FileResult) -> None:
            if res.ok:
                bar.println(
                    f"[{done}/{total}] {in_file} -> {res.output_path}  "
                    f"({res.pages_processed} page(s), {res.lines_added} line(s), {res.elapsed:.1f}s)"
                )
            else:
                log.error("[%d/%d] FAILED: %s -> %s", done, total, in_file, res.error)
            bar.update(1)

        try:
            results: list[FileResult] = run_parallel(
                filtered_jobs, cfg, effective_jobs, log_level=level, progress_cb=_progress
            )
        finally:
            bar.stop_spinner()
            bar.close()
    else:
        configure_cpu_threads(cfg.cpu_threads)
        bar = ProgressBar(len(filtered_jobs), label="OCR", enabled=not args.quiet)
        bar.start_spinner(
            f"Loading {cfg.ocr_version} {cfg.model_size} model ({cfg.inference_engine})"
        )
        engine_started = time.perf_counter()
        try:
            engine = get_ocr_engine(cfg)
        finally:
            bar.stop_spinner()
        bar.println(f"OCR engine ready in {time.perf_counter() - engine_started:.1f}s")

        results = []
        for idx, (in_file, out_file) in enumerate(filtered_jobs, 1):
            bar.println(f"[{idx}/{len(filtered_jobs)}] {in_file}")

            file_base = idx - 1

            def _on_page(page_num: int, n_pages: int, _in_file=in_file, _base=file_base) -> None:
                bar.stop_spinner()
                page_fraction = page_num / max(n_pages, 1)
                bar.set_progress(
                    _base + page_fraction,
                    suffix=f"{_in_file.name} (page {page_num}/{n_pages})",
                )

            def _on_stage(stage: str, _in_file=in_file, _base=file_base) -> None:
                # Numeric progress is only advanced at truthful checkpoints.
                # Paddle's predict() call is monolithic, so animate activity
                # during that blocking interval instead of inventing progress.
                bar.stop_spinner()
                if not stage.startswith("Building"):
                    bar.set_progress(_base + 0.05)
                bar.start_spinner(f"{_in_file.name}: {stage}")

            def _on_tile(tile_number: int, n_tiles: int, _in_file=in_file, _base=file_base) -> None:
                tile_fraction = 0.1 + 0.85 * ((tile_number - 0.5) / max(n_tiles, 1))
                bar.stop_spinner()
                bar.set_progress(_base + tile_fraction)
                bar.start_spinner(f"{_in_file.name}: OCR tile {tile_number}/{n_tiles}")

            def _on_tile_done(tile_number: int, n_tiles: int, _base=file_base) -> None:
                completed_fraction = 0.1 + 0.85 * (tile_number / max(n_tiles, 1))
                bar.stop_spinner()
                bar.set_progress(_base + completed_fraction)

            try:
                res = process_file(
                    engine,
                    in_file,
                    out_file,
                    cfg,
                    on_page=_on_page,
                    on_stage=_on_stage,
                    on_tile=_on_tile,
                    on_tile_done=_on_tile_done,
                )
                results.append(res)
                bar.println(
                    f"    -> {out_file}  ({res.pages_processed} page(s), "
                    f"{res.lines_added} line(s), {res.elapsed:.1f}s)"
                )
            except Exception as e:  # noqa: BLE001 - keep processing the remaining batch files
                log.error("  FAILED: %s -> %s", in_file, e)
                results.append(FileResult(input_path=in_file, ok=False, error=str(e)))

            bar.update(1)

        bar.close()

    ok = sum(1 for r in results if r.ok)
    failed = [r for r in results if not r.ok]

    if not args.quiet:
        print(f"\nDone: {ok}/{len(results)} succeeded.")
    if failed:
        if not args.quiet:
            print(f"{len(failed)} file(s) failed:")
        for r in failed:
            log.error("  - %s: %s", r.input_path, r.error)
        return 1

    return 0
