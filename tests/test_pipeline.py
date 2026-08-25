from pathlib import Path

import numpy as np
import pytest

from pidocr.geometry import order_quad, polygon_geometry
from pidocr.pipeline import (
    page_in_ranges,
    parse_page_ranges,
    resolve_output_path,
)


def test_parse_page_ranges_basic():
    assert parse_page_ranges("1-5,9,12-14") == [(1, 5), (9, 9), (12, 14)]


def test_parse_page_ranges_invalid():
    with pytest.raises(ValueError):
        parse_page_ranges("0-2")
    with pytest.raises(ValueError):
        parse_page_ranges("5-2")


def test_page_in_ranges():
    ranges = [(1, 5), (9, 9)]
    assert page_in_ranges(3, ranges)
    assert page_in_ranges(9, ranges)
    assert not page_in_ranges(6, ranges)
    assert page_in_ranges(6, None)


def test_order_quad_axis_aligned():
    poly = np.array([[10, 10], [100, 10], [100, 40], [10, 40]], dtype=float)
    ordered = order_quad(poly)
    assert ordered.shape == (4, 2)


def test_polygon_geometry_axis_aligned():
    poly = np.array([[10, 10], [110, 10], [110, 40], [10, 40]], dtype=float)
    center, direction, w, h, angle = polygon_geometry(poly)
    assert w == pytest.approx(100.0, abs=1.0)
    assert h == pytest.approx(30.0, abs=1.0)
    assert angle == pytest.approx(0.0, abs=1.0)


def test_resolve_output_path_default_suffix(tmp_path):
    f = tmp_path / "drawing.pdf"
    f.touch()
    out = resolve_output_path(f, None, None, ".ocr", in_place=False)
    assert out == tmp_path / "drawing.ocr.pdf"


def test_resolve_output_path_in_place(tmp_path):
    f = tmp_path / "drawing.pdf"
    f.touch()
    out = resolve_output_path(f, None, None, ".ocr", in_place=True)
    assert out == f


def test_resolve_output_path_mirrors_folder_structure(tmp_path):
    base = tmp_path / "scans"
    sub = base / "sub"
    sub.mkdir(parents=True)
    f = sub / "a.pdf"
    f.touch()
    out_dir = tmp_path / "out"
    out = resolve_output_path(f, base, str(out_dir), ".ocr", in_place=False)
    assert out == out_dir / "sub" / "a.pdf"


def test_process_file_dispatches_by_extension(tmp_path, monkeypatch):
    import pidocr.pipeline as pipeline

    calls = []
    monkeypatch.setattr(pipeline, "process_pdf", lambda *a, **k: calls.append("pdf"))
    monkeypatch.setattr(pipeline, "process_image", lambda *a, **k: calls.append("image"))

    pdf_path = tmp_path / "a.pdf"
    img_path = tmp_path / "a.png"
    pipeline.process_file(object(), pdf_path, tmp_path / "out.pdf", None)
    pipeline.process_file(object(), img_path, tmp_path / "out2.pdf", None)

    assert calls == ["pdf", "image"]
