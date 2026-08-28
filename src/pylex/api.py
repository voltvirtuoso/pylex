"""Reusable Python API for generic OCR and searchable-document extraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pylex.config import Config
from pylex.engine import get_ocr_engine
from pylex.pipeline import FileResult, process_file
from pylex.tiling import ocr_image_tiled


@dataclass(frozen=True)
class TextRegion:
    """One recognized text region in source-image pixel coordinates."""

    text: str
    polygon: np.ndarray
    confidence: float


def create_engine(config: Config | None = None):
    """Create or reuse the configured OCR engine."""
    return get_ocr_engine(config or Config())


def _as_rgb_array(source: Any) -> np.ndarray:
    if isinstance(source, np.ndarray):
        array = source
    elif hasattr(source, "convert"):
        array = np.asarray(source.convert("RGB"))
    else:
        from PIL import Image

        with Image.open(source) as image:
            array = np.asarray(image.convert("RGB"))
    if array.ndim not in (2, 3):
        raise ValueError("source must be a 2-D grayscale or 3-D color image")
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return np.ascontiguousarray(array)


def extract_text(
    source: Any,
    *,
    config: Config | None = None,
    engine: Any = None,
    on_tile: Callable[[int, int], None] | None = None,
    on_tile_done: Callable[[int, int], None] | None = None,
) -> list[TextRegion]:
    """Extract text regions from a raster image, PIL image, or image path.

    Returned polygons use the original image's pixel coordinate system. Pass a
    pre-created ``engine`` to reuse it across many images in one process.
    """
    cfg = config or Config()
    image = _as_rgb_array(source)
    ocr_engine = engine if engine is not None else create_engine(cfg)
    items = ocr_image_tiled(
        ocr_engine,
        image,
        cfg,
        on_tile=on_tile,
        on_tile_done=on_tile_done,
    )
    return [TextRegion(text=text, polygon=polygon, confidence=score) for text, polygon, score in items]


def create_searchable_document(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: Config | None = None,
    engine: Any = None,
    on_page: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_tile: Callable[[int, int], None] | None = None,
    on_tile_done: Callable[[int, int], None] | None = None,
) -> FileResult:
    """Create a searchable PDF from an image or PDF input."""
    cfg = config or Config()
    ocr_engine = engine if engine is not None else create_engine(cfg)
    return process_file(
        ocr_engine,
        Path(input_path),
        Path(output_path),
        cfg,
        on_page=on_page,
        on_stage=on_stage,
        on_tile=on_tile,
        on_tile_done=on_tile_done,
    )


__all__ = ["TextRegion", "create_engine", "create_searchable_document", "extract_text"]

