from itertools import pairwise

import numpy as np
import pytest

from pylex.tiling import compute_tiles, dedupe_items, ocr_image_tiled


def _quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


# ----------------------------------------------------------------------
# compute_tiles
# ----------------------------------------------------------------------
def test_compute_tiles_disabled_returns_single_tile():
    assert compute_tiles(3000, 2000, tile_size=0, overlap=200) == [(0, 0, 3000, 2000)]


def test_compute_tiles_small_image_returns_single_tile():
    assert compute_tiles(800, 600, tile_size=1600, overlap=200) == [(0, 0, 800, 600)]


def test_compute_tiles_covers_whole_image_with_overlap():
    width, height, tile_size, overlap = 3000, 2200, 1600, 220
    tiles = compute_tiles(width, height, tile_size, overlap)

    assert len(tiles) > 1

    # Every tile must stay within bounds.
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height

    # The full image must be covered: every pixel column/row is inside at
    # least one tile (checked via edges, cheap proxy for full coverage).
    xs = sorted({x0 for x0, _, _, _ in tiles} | {x1 for _, _, x1, _ in tiles})
    assert xs[0] == 0
    assert xs[-1] == width

    ys = sorted({y0 for _, y0, _, _ in tiles} | {y1 for _, _, _, y1 in tiles})
    assert ys[0] == 0
    assert ys[-1] == height

    # Adjacent tiles along x actually overlap by roughly `overlap` px.
    x_starts = sorted({x0 for x0, _, _, _ in tiles})
    for a, b in pairwise(x_starts):
        assert b - a <= tile_size  # otherwise there'd be a gap


def test_compute_tiles_no_duplicate_tiles():
    tiles = compute_tiles(3000, 2200, 1600, 220)
    assert len(tiles) == len(set(tiles))


# ----------------------------------------------------------------------
# dedupe_items
# ----------------------------------------------------------------------
def test_dedupe_removes_close_duplicate_same_text():
    items = [
        ("TAG-101", _quad(100, 100, 150, 120), 0.80),
        ("TAG-101", _quad(103, 101, 153, 121), 0.95),  # near-identical, higher score
    ]
    merged = dedupe_items(items, tol_px=20.0)
    assert len(merged) == 1
    assert merged[0][2] == pytest.approx(0.95)


def test_dedupe_keeps_distinct_text_even_if_close():
    items = [
        ("TAG-101", _quad(100, 100, 150, 120), 0.9),
        ("TAG-102", _quad(103, 101, 153, 121), 0.9),
    ]
    merged = dedupe_items(items, tol_px=20.0)
    assert len(merged) == 2


def test_dedupe_keeps_same_text_when_far_apart():
    items = [
        ("VALVE", _quad(0, 0, 50, 20), 0.9),
        ("VALVE", _quad(2000, 2000, 2050, 2020), 0.9),
    ]
    merged = dedupe_items(items, tol_px=20.0)
    assert len(merged) == 2


# ----------------------------------------------------------------------
# ocr_image_tiled: end-to-end merge behaviour with a stub engine
# ----------------------------------------------------------------------
def test_ocr_image_tiled_merges_across_tiles_without_duplicating(monkeypatch):
    import pylex.tiling as tiling_mod
    from pylex.config import Config

    # Build a "page" big enough to force tiling into (at least) 2x2 tiles.
    width, height = 3000, 2200
    img = np.zeros((height, width, 3), dtype=np.uint8)

    calls = []

    def fake_ocr_image(engine, crop, cfg):
        h, w = crop.shape[:2]
        calls.append((w, h))
        # Every tile "sees" one detection near its own top-left corner —
        # simulates a real tag that only one tile's overlap region contains.
        poly = np.array([[5, 5], [80, 5], [80, 25], [5, 25]], dtype=float)
        return [(f"TILE-TAG-{w}x{h}", poly, 0.9)]

    monkeypatch.setattr(tiling_mod, "ocr_image", fake_ocr_image)

    cfg = Config(tile_size=1600, tile_overlap=220, tile_batch_size=1, adaptive_tiling=False)
    items = ocr_image_tiled(object(), img, cfg)

    assert len(calls) > 1  # tiling actually happened
    # Each tile's synthetic top-left detection lands in a different place in
    # page coordinates, so none of them should collide/dedupe with another.
    assert len(items) == len(calls)


def test_ocr_image_tiled_falls_back_to_single_pass_when_small(monkeypatch):
    import pylex.tiling as tiling_mod
    from pylex.config import Config

    img = np.zeros((400, 600, 3), dtype=np.uint8)
    calls = []

    def fake_ocr_image(engine, crop, cfg):
        calls.append(crop.shape)
        return [("SMALL-PAGE-TAG", _quad(5, 5, 80, 25), 0.9)]

    monkeypatch.setattr(tiling_mod, "ocr_image", fake_ocr_image)

    cfg = Config(tile_size=1600, tile_overlap=220, tile_batch_size=1, adaptive_tiling=False)
    items = ocr_image_tiled(object(), img, cfg)

    assert len(calls) == 1  # no tiling for a page smaller than tile_size
    assert len(items) == 1


def test_tiling_recovers_a_tag_a_whole_page_pass_misses(monkeypatch):
    """Reproduces the reported failure mode: a whole-page OCR pass resizes
    the entire page before detecting text, so a tag squeezed into a dense
    symbol cluster can fall below the detector's threshold even though the
    exact same text on an open pipe run gets picked up fine. Tiling OCRs
    each region at full resolution and should recover it.
    """
    import pylex.tiling as tiling_mod
    from pylex.config import Config

    width, height = 3400, 2500
    OPEN_TEXT_XY = (300, 300)  # sparse area: any pass should catch this
    DENSE_TAG_XY = (2600, 1900)  # crowded symbol cluster: only a small-enough crop catches this
    WHOLE_PAGE_DETECTION_LIMIT = 1600 * 1600  # simulated resize/detectability ceiling

    img = np.zeros((height, width, 3), dtype=np.uint8)
    call_count = {"n": 0}

    def fake_ocr_image(engine, crop, cfg):
        h, w = crop.shape[:2]
        tiles_now = tiling_mod.compute_tiles(width, height, cfg.tile_size, cfg.tile_overlap)
        idx = call_count["n"]
        call_count["n"] += 1
        x0, y0, x1, y1 = tiles_now[idx]

        items = []
        if x0 <= OPEN_TEXT_XY[0] < x1 and y0 <= OPEN_TEXT_XY[1] < y1:
            lx, ly = OPEN_TEXT_XY[0] - x0, OPEN_TEXT_XY[1] - y0
            items.append(("OPEN-TEXT", _quad(lx, ly, lx + 80, ly + 20), 0.9))

        # The dense tag only survives detection when the crop is small
        # enough — i.e. tiling actually reduced the effective downscale.
        if (
            w * h <= WHOLE_PAGE_DETECTION_LIMIT
            and x0 <= DENSE_TAG_XY[0] < x1
            and y0 <= DENSE_TAG_XY[1] < y1
        ):
            lx, ly = DENSE_TAG_XY[0] - x0, DENSE_TAG_XY[1] - y0
            items.append(("DENSE-TAG", _quad(lx, ly, lx + 60, ly + 16), 0.85))

        return items

    monkeypatch.setattr(tiling_mod, "ocr_image", fake_ocr_image)

    # Whole-page pass (tiling disabled): misses the dense tag.
    call_count["n"] = 0
    whole_cfg = Config(tile_size=0)
    whole_items = ocr_image_tiled(object(), img, whole_cfg)
    whole_texts = {t for t, _, _ in whole_items}
    assert whole_texts == {"OPEN-TEXT"}
    assert "DENSE-TAG" not in whole_texts

    # Tiled pass: recovers the dense tag, without duplicating either tag.
    call_count["n"] = 0
    tiled_cfg = Config(tile_size=1600, tile_overlap=220, tile_batch_size=1, adaptive_tiling=False)
    tiled_items = ocr_image_tiled(object(), img, tiled_cfg)
    tiled_texts = [t for t, _, _ in tiled_items]
    assert sorted(tiled_texts) == ["DENSE-TAG", "OPEN-TEXT"]


def test_ocr_image_tiled_fast_mode_uses_one_pass(monkeypatch):
    import pylex.tiling as tiling_mod
    from pylex.config import Config

    calls = []

    def fake_ocr_image(engine, crop, cfg):
        calls.append(crop.shape[:2])
        return []

    monkeypatch.setattr(tiling_mod, "ocr_image", fake_ocr_image)
    img = np.zeros((3200, 4200, 3), dtype=np.uint8)
    cfg = Config(fast_mode=True, tile_size=1000, tile_overlap=200)

    assert tiling_mod.ocr_image_tiled(object(), img, cfg) == []
    assert calls == [(3200, 4200)]


def test_ocr_image_tiled_reports_tile_progress(monkeypatch):
    import pylex.tiling as tiling_mod
    from pylex.config import Config

    progress = []
    monkeypatch.setattr(tiling_mod, "ocr_image", lambda engine, crop, cfg: [])
    image = np.zeros((2200, 3000, 3), dtype=np.uint8)

    tiling_mod.ocr_image_tiled(
        object(),
        image,
        Config(tile_size=1600, tile_overlap=220, tile_batch_size=1, adaptive_tiling=False),
        on_tile=lambda tile_number, total: progress.append((tile_number, total)),
    )

    assert progress == [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]


def test_ocr_image_tiled_adapts_to_one_pass_within_detector_budget(monkeypatch):
    import pylex.tiling as tiling_mod
    from pylex.config import Config

    calls = []
    monkeypatch.setattr(
        tiling_mod,
        "ocr_image",
        lambda engine, crop, cfg: calls.append(crop.shape[:2]) or [],
    )
    image = np.zeros((2200, 3000, 3), dtype=np.uint8)

    tiling_mod.ocr_image_tiled(object(), image, Config(tile_size=1600, tile_overlap=220))

    assert calls == [(2200, 3000)]


def test_vertical_text_retry_recovers_label_from_rotated_pass(monkeypatch):
    import pylex.tiling as tiling_mod
    from pylex.config import Config

    calls = []

    def fake_ocr_image(engine, image, cfg):
        calls.append(image.shape)
        if len(calls) == 1:
            return [("MISREAD", _quad(10, 10, 20, 100), 0.5)]
        return [("VERTICAL-LABEL", _quad(5, 5, 75, 20), 0.95)]

    monkeypatch.setattr(tiling_mod, "ocr_image", fake_ocr_image)
    image = np.zeros((200, 300, 3), dtype=np.uint8)

    items = tiling_mod.ocr_image_tiled(object(), image, Config(tile_size=2000))

    assert len(calls) == 2
    assert "VERTICAL-LABEL" in {text for text, _poly, _score in items}
