"""Parallel batch execution.

File-level parallelism via a process pool, not threads: PaddleOCR's engine
isn't safe to hammer with concurrent predict() calls from multiple threads,
and OCR is CPU-bound anyway (the GIL would erase most of the benefit from
threads on the inference itself). A ProcessPoolExecutor gives each worker
its own engine, built once and reused for every file that worker picks up.

Within a single PDF, pages still OCR sequentially inside whichever worker
owns that file — the parallelism is across files, not across pages of one
file. That keeps one big multi-hundred-page drawing set from starving
everything else onto a single worker while still using every core across
a batch of files.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pidocr.config import Config
from pidocr.engine import configure_cpu_threads, get_ocr_engine
from pidocr.pipeline import FileResult, process_file

log = logging.getLogger("pidocr")

# Set once per worker process by _init_worker; each worker process has its
# own copy of module globals, so this is never shared across workers.
_WORKER_CFG: Config | None = None


def _init_worker(cfg: Config, log_level: int) -> None:
    """Runs once when a worker process starts: warm its own OCR engine."""
    global _WORKER_CFG
    _WORKER_CFG = cfg
    logging.basicConfig(level=log_level, format="%(message)s")
    configure_cpu_threads(cfg.cpu_threads)
    get_ocr_engine(cfg)  # load + cache in this process now, not on first file


def _run_one(in_file: Path, out_file: Path) -> FileResult:
    assert _WORKER_CFG is not None, "worker process was not initialized"
    t0 = time.time()
    try:
        engine = get_ocr_engine(_WORKER_CFG)
        return process_file(engine, in_file, out_file, _WORKER_CFG)
    except Exception as e:  # noqa: BLE001 - keep batch processing alive
        return FileResult(
            input_path=in_file,
            output_path=out_file,
            ok=False,
            error=str(e),
            elapsed=time.time() - t0,
        )


def run_parallel(
    jobs_list: list[tuple[Path, Path]],
    cfg: Config,
    workers: int,
    log_level: int = logging.WARNING,
    progress_cb: Callable[[int, int, Path, FileResult], None] | None = None,
) -> list[FileResult]:
    """Run (in_file, out_file) jobs across `workers` worker processes.

    Returns results in the SAME order as jobs_list (completion order may
    differ across workers; progress_cb, if given, fires in completion order
    with (done_count, total, in_file, result)).
    """
    results: list[FileResult | None] = [None] * len(jobs_list)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(cfg, log_level),
    ) as pool:
        future_to_idx = {
            pool.submit(_run_one, in_file, out_file): idx
            for idx, (in_file, out_file) in enumerate(jobs_list)
        }

        for done_count, future in enumerate(as_completed(future_to_idx), start=1):
            idx = future_to_idx[future]
            in_file, out_file = jobs_list[idx]
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001 - worker crashed outright
                result = FileResult(input_path=in_file, output_path=out_file, ok=False, error=str(e))

            results[idx] = result
            if progress_cb:
                progress_cb(done_count, len(jobs_list), in_file, result)

    return results  # type: ignore[return-value]
