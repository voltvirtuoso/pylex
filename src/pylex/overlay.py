"""Build the invisible, searchable text layer as a one-page PDF overlay."""

from __future__ import annotations

import io
import logging

import numpy as np
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as rl_canvas

from pylex.config import Config
from pylex.geometry import display_poly_to_pdf, order_quad, polygon_geometry

log = logging.getLogger("pylex")


def build_invisible_text_overlay(
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
    cfg: Config,
):
    """
    Build the invisible OCR layer in the SAME coordinate system as the
    rotation-transferred PDF page. No extra 90/180/270 rotation here —
    that was already baked into the page content upstream.
    """
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(canvas_w_pt, canvas_h_pt))

    skipped = 0
    for text, poly_display, _score in items:
        try:
            poly_display = order_quad(poly_display)
            poly_pdf = display_poly_to_pdf(
                poly_display,
                img_size_px,
                render_w_pt,
                render_h_pt,
                render_left_pt,
                render_bottom_pt,
                canvas_left_pt,
                canvas_bottom_pt,
            )

            center, direction, box_w_pt, box_h_pt, angle_deg = polygon_geometry(poly_pdf)

            if box_w_pt < cfg.min_box_width_pt or box_h_pt < cfg.min_box_height_pt:
                continue

            font_size = max(cfg.min_font_size, min(cfg.max_font_size, box_h_pt * 0.88))
            natural_w = stringWidth(text, cfg.font, font_size)
            if natural_w <= 0:
                continue

            hscale = max(10.0, min(500.0, 100.0 * box_w_pt / natural_w))

            # Anchor at the START of the polygon (not its center) so long OCR
            # text can't drift outside the page — this is the key placement fix.
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            baseline = (
                center
                - direction * (box_w_pt * 0.5)
                - normal * (box_h_pt * cfg.baseline_fraction)
            )

            c.saveState()
            c.translate(float(baseline[0]), float(baseline[1]))
            c.rotate(angle_deg)

            t = c.beginText(0, 0)
            t.setFont(cfg.font, font_size)
            t.setTextRenderMode(3)  # invisible
            t.setHorizScale(hscale)
            t.textOut(text)

            c.drawText(t)
            c.restoreState()
        except Exception as e:  # noqa: BLE001 - skip only malformed overlay regions
            skipped += 1
            log.debug("Overlay item skipped: %s", e)

    if skipped:
        log.debug("%d overlay item(s) skipped", skipped)

    c.save()
    buf.seek(0)
    return buf
