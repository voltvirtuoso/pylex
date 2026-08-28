"""Measure profile-level OCR pass counts without loading PaddleOCR.

This is intentionally a pipeline benchmark, not an accuracy benchmark. Run it
from the project root with `PYTHONPATH=src python benchmarks/benchmark_profiles.py`.
"""

from __future__ import annotations

import time

import numpy as np

from pidocr.config import Config
from pidocr.tiling import compute_tiles, ocr_image_tiled


class StubEngine:
    pass


def run(label: str, cfg: Config, image: np.ndarray) -> None:
    calls = 0

    def fake_ocr(_engine, crop, _cfg):
        nonlocal calls
        calls += 1
        return []

    import pidocr.tiling as tiling

    original = tiling.ocr_image
    tiling.ocr_image = fake_ocr
    try:
        started = time.perf_counter()
        ocr_image_tiled(StubEngine(), image, cfg)
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        tiling.ocr_image = original

    planned = 1 if cfg.fast_mode else len(compute_tiles(image.shape[1], image.shape[0], cfg.tile_size, cfg.tile_overlap))
    print(f"{label:12} passes={calls:2d} planned={planned:2d} orchestration_ms={elapsed_ms:8.2f}")


def main() -> None:
    image = np.zeros((3200, 4200, 3), dtype=np.uint8)
    run("accuracy", Config(tile_size=1600, tile_overlap=220), image)
    run("fast", Config(fast_mode=True, tile_size=1600, tile_overlap=220), image)


if __name__ == "__main__":
    main()
