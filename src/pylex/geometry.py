"""Rotation and polygon geometry helpers.

The rotation-transfer + start-of-polygon anchoring here is the actual fix
for OCR text landing in the wrong place on rotated PDF pages. Don't
"simplify" it — it's the whole reason this tool works on rotated
engineering drawings where a naive overlay doesn't.
"""

from __future__ import annotations

import math

import numpy as np


def get_page_rotation(page) -> int:
    try:
        return int(page.get("/Rotate", 0) or 0) % 360
    except Exception:  # noqa: BLE001 - malformed PDF metadata should not stop processing
        return 0


def render_pdf_page_for_ocr(pdfium_page, rotation: int, dpi: int):
    """Render a PDF page exactly as the viewer displays it (rotation applied)."""
    try:
        pdfium_page.set_rotation(rotation)
    except Exception:  # noqa: BLE001 - renderer versions may not expose rotation
        rotation = 0

    scale = dpi / 72.0
    bitmap = pdfium_page.render(scale=scale)
    pil_img = bitmap.to_pil().convert("RGB")
    return np.array(pil_img), pil_img.size


def order_quad(poly):
    """Normalize an OCR quadrilateral to approximately TL, TR, BR, BL."""
    pts = np.asarray(poly, dtype=np.float64)
    if pts.shape[0] != 4:
        return pts[:4]

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]

    ordered = np.array(
        [pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]],
        dtype=np.float64,
    )

    if len({tuple(np.round(p, 6)) for p in ordered}) < 4:
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        p = pts[np.argsort(angles)]
        start = np.argmin(p[:, 0] + p[:, 1])
        ordered = np.roll(p, -start, axis=0)

    return ordered


def display_poly_to_pdf(
    poly,
    img_size_px,
    render_w_pt,
    render_h_pt,
    render_left_pt,
    render_bottom_pt,
    canvas_left_pt,
    canvas_bottom_pt,
):
    """Map displayed-crop-box pixel coords into overlay-canvas (media-box) coords."""
    img_w_px, img_h_px = img_size_px
    sx = render_w_pt / float(img_w_px)
    sy = render_h_pt / float(img_h_px)

    dx = render_left_pt - canvas_left_pt
    dy = render_bottom_pt - canvas_bottom_pt

    result = []
    for x_px, y_px in np.asarray(poly, dtype=np.float64)[:4]:
        x_pt = dx + float(x_px) * sx
        y_pt = dy + render_h_pt - float(y_px) * sy
        result.append((x_pt, y_pt))

    return np.asarray(result, dtype=np.float64)


def polygon_geometry(poly_pdf):
    """Center, unit direction, width, height, angle(deg) of a PDF-space quad."""
    pts = order_quad(poly_pdf)

    edges = [pts[1] - pts[0], pts[2] - pts[1], pts[3] - pts[2], pts[0] - pts[3]]
    lengths = [float(np.linalg.norm(e)) for e in edges]

    long_idx = sorted(range(4), key=lambda i: lengths[i], reverse=True)[:2]
    v1, v2 = edges[long_idx[0]], edges[long_idx[1]]
    if np.dot(v1, v2) < 0:
        v2 = -v2

    direction = v1 + v2
    if np.linalg.norm(direction) < 1e-9:
        direction = v1
    direction = direction / max(np.linalg.norm(direction), 1e-9)

    width_pt = max(0.25, (lengths[long_idx[0]] + lengths[long_idx[1]]) / 2.0)
    short_idx = [i for i in range(4) if i not in long_idx]
    height_pt = max(0.25, (lengths[short_idx[0]] + lengths[short_idx[1]]) / 2.0)

    center = pts.mean(axis=0)
    angle_deg = math.degrees(math.atan2(direction[1], direction[0]))

    return center, direction, width_pt, height_pt, angle_deg
