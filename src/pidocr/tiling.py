"""Tile a page image into overlapping windows so OCR keeps full local
resolution on dense drawings, then merge and de-duplicate detections back
into whole-page pixel coordinates.

Why this exists: a single whole-page OCR pass resizes the entire page
down to the detector's max side length before it looks for text. On an
open, sparse area of a drawing that's fine — the text is still big
relative to everything around it after the resize. But in a dense
cluster (several valves, an instrument bubble, and a line tag all
crammed together), the tag text is small to begin with, and shrinking
the whole page down shrinks it further, often below what the detector
can reliably pick up — even though the exact same size of text sitting
on an open pipe run gets OCR'd fine.

Splitting the page into overlapping tiles and running detection on each
tile at full resolution fixes this generically, for any drawing, without
hand-tuning per page: within a tile, the crowded cluster is a much
larger fraction of the (still-limited) detector input, so the same
resize no longer erases the text.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from pidocr.config import Config
from pidocr.engine import ocr_image, ocr_images

log = logging.getLogger("pidocr")


def compute_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) windows covering the image, with overlap.

    tile_size <= 0, or an image already smaller than tile_size on both
    axes, returns a single tile spanning the whole image (no tiling).
    """
    if tile_size <= 0 or (width <= tile_size and height <= tile_size):
        return [(0, 0, width, height)]

    overlap = max(0, min(overlap, tile_size - 1))
    step = max(1, tile_size - overlap)

    def axis_starts(total: int) -> list[int]:
        starts = list(range(0, max(total - tile_size, 0) + 1, step))
        if not starts or starts[-1] + tile_size < total:
            starts.append(max(0, total - tile_size))
        return starts

    x_starts = axis_starts(width)
    y_starts = axis_starts(height)

    tiles = []
    seen = set()
    for y0 in y_starts:
        y1 = min(y0 + tile_size, height)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, width)
            key = (x0, y0, x1, y1)
            if key not in seen:
                seen.add(key)
                tiles.append(key)

    return tiles


def _quad_center(poly: np.ndarray) -> np.ndarray:
    return poly.mean(axis=0)


def _vertical_candidates(
    items: list[tuple[str, np.ndarray, float]],
    aspect_ratio: float,
    max_confidence: float,
) -> bool:
    """Return whether any detected polygon is tall enough to indicate vertical text."""
    for _text, poly, _score in items:
        xs = poly[:, 0]
        ys = poly[:, 1]
        width = float(max(xs) - min(xs))
        height = float(max(ys) - min(ys))
        if height > max(width, 1.0) * aspect_ratio and _score < max_confidence:
            return True
    return False


def _rotate_items_back(items: list[tuple[str, np.ndarray, float]], width: int):
    """Map polygons from a counter-clockwise 90-degree image back to original coordinates."""
    mapped = []
    for text, poly, score in items:
        restored = np.column_stack((width - 1 - poly[:, 1], poly[:, 0]))
        mapped.append((text, restored, score))
    return mapped


def _augment_vertical_text(engine, img_array: np.ndarray, cfg: Config, items):
    """Add a rotated retry only when normal OCR found a tall candidate polygon."""
    if cfg.fast_mode or not cfg.vertical_text_retry or not items:
        return items
    if not _vertical_candidates(
        items,
        cfg.vertical_text_aspect_ratio,
        cfg.vertical_text_retry_max_confidence,
    ):
        return items

    rotated = np.rot90(img_array, k=1)
    rotated_items = ocr_image(engine, rotated, cfg)
    restored = _rotate_items_back(rotated_items, img_array.shape[1])
    return dedupe_items(items + restored, cfg.dedupe_tol_px)


def _ocr_with_vertical_retry(engine, img_array: np.ndarray, cfg: Config):
    """Run normal OCR and retry only when its output suggests vertical text."""
    return _augment_vertical_text(engine, img_array, cfg, ocr_image(engine, img_array, cfg))


def dedupe_items(items: list[tuple[str, np.ndarray, float]], tol_px: float) -> list[tuple[str, np.ndarray, float]]:
    """Drop near-duplicate detections produced by tile overlap regions.

    Two detections are treated as the same physical text if they have
    identical recognized text AND their polygon centers land within
    tol_px of each other (in whole-page pixel space). Among duplicates,
    the highest-confidence detection is kept.
    """
    kept: list[tuple[str, np.ndarray, float]] = []

    for text, poly, score in sorted(items, key=lambda it: -it[2]):
        center = _quad_center(poly)
        is_dup = False
        for kt, kpoly, _kscore in kept:
            if kt != text:
                continue
            if float(np.linalg.norm(center - _quad_center(kpoly))) < tol_px:
                is_dup = True
                break
        if not is_dup:
            kept.append((text, poly, score))

    return kept


def ocr_image_tiled(
    engine,
    img_array: np.ndarray,
    cfg: Config,
    on_tile: Callable[[int, int], None] | None = None,
    on_tile_done: Callable[[int, int], None] | None = None,
):
    """Tiled OCR pass: split img_array, OCR each tile, merge into page
    coordinates, and de-duplicate detections that both tiles caught in
    their shared overlap strip. Falls back to a single ocr_image() call
    when tiling is disabled or the image is already small.
    """
    height, width = img_array.shape[:2]
    if cfg.fast_mode:
        if on_tile:
            on_tile(1, 1)
        items = _ocr_with_vertical_retry(engine, img_array, cfg)
        if on_tile_done:
            on_tile_done(1, 1)
        return items

    # A small-model full-page pass keeps the same detector/recognizer and
    # orientation settings as tiled mode. It only avoids tiling when the
    # image can fit within the detector's configured resolution, so this is
    # a throughput optimization rather than a lower-accuracy profile.
    fits_detector = max(width, height) <= cfg.det_limit_side_len
    fits_pixel_budget = width * height <= cfg.full_page_max_pixels
    if cfg.adaptive_tiling and cfg.model_size == "small" and fits_detector and fits_pixel_budget:
        if on_tile:
            on_tile(1, 1)
        items = _ocr_with_vertical_retry(engine, img_array, cfg)
        if on_tile_done:
            on_tile_done(1, 1)
        return items

    tiles = compute_tiles(width, height, cfg.tile_size, cfg.tile_overlap)

    if len(tiles) == 1:
        if on_tile:
            on_tile(1, 1)
        items = _ocr_with_vertical_retry(engine, img_array, cfg)
        if on_tile_done:
            on_tile_done(1, 1)
        return items

    log.debug("Tiling %dx%d image into %d tile(s) (tile_size=%d, overlap=%d)",
              width, height, len(tiles), cfg.tile_size, cfg.tile_overlap)

    all_items: list[tuple[str, np.ndarray, float]] = []
    batch_size = 1 if cfg.retry_upscale else max(1, cfg.tile_batch_size)
    for batch_start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[batch_start : batch_start + batch_size]
        for tile_number in range(batch_start + 1, batch_start + len(batch_tiles) + 1):
            if on_tile:
                on_tile(tile_number, len(tiles))

        crops = [img_array[y0:y1, x0:x1] for x0, y0, x1, y1 in batch_tiles]
        try:
            if len(crops) == 1:
                batch_items = [_ocr_with_vertical_retry(engine, crops[0], cfg)]
            else:
                batch_items = ocr_images(engine, crops, cfg)
        except Exception as e:  # noqa: BLE001 - continue with other tiles
            log.warning("  Tile batch %d-%d: OCR failed (%s), skipping batch", batch_start + 1, batch_start + len(batch_tiles), e)
            batch_items = [[] for _ in batch_tiles]

        for tile_offset, ((x0, y0, x1, y1), tile_items) in enumerate(zip(batch_tiles, batch_items)):
            tile_number = batch_start + tile_offset + 1
            tile_items = _augment_vertical_text(engine, crops[tile_offset], cfg, tile_items)
            if on_tile_done:
                on_tile_done(tile_number, len(tiles))
            offset = np.array([x0, y0], dtype=np.float64)
            for text, poly, score in tile_items:
                all_items.append((text, poly + offset, score))

    merged = dedupe_items(all_items, cfg.dedupe_tol_px)

    log.debug("  %d raw detection(s) across tiles -> %d after de-duplication", len(all_items), len(merged))

    return merged
