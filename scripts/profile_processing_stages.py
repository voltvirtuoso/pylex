from __future__ import annotations

import io
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas as rl_canvas

from pidocr.config import Config
from pidocr.engine import get_ocr_engine
from pidocr.overlay import build_invisible_text_overlay
from pidocr.tiling import ocr_image_tiled


def main() -> None:
    width, height = 1845, 832
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row in range(102):
        draw.text((40 + (row % 4) * 430, 20 + (row // 4) * 31), f"TAG-{row:03d} PUMP PRESSURE", fill="black", font=font)
    array = np.asarray(image)
    cfg = Config(
        model_size="small",
        inference_engine=os.environ.get("PIDOCR_ENGINE", "paddle_dynamic"),
        disable_mkldnn=True,
        adaptive_tiling=True,
        runtime_cache_dir=os.environ.get("PIDOCR_RUNTIME_CACHE_DIR"),
        recognition_batch_size=int(os.environ.get("PIDOCR_REC_BATCH", "6")),
        orientation_batch_size=int(os.environ.get("PIDOCR_ORI_BATCH", "8")),
    )

    started = time.perf_counter()
    engine = get_ocr_engine(cfg)
    print(
        f"batches=rec:{cfg.recognition_batch_size},ori:{cfg.orientation_batch_size} "
        f"engine_init={time.perf_counter() - started:.3f}"
    )

    started = time.perf_counter()
    items = ocr_image_tiled(engine, array, cfg)
    print(f"ocr_tiled={time.perf_counter() - started:.3f} items={len(items)}")

    started = time.perf_counter()
    overlay = build_invisible_text_overlay(
        items, (width, height), width * 72 / cfg.image_dpi, height * 72 / cfg.image_dpi,
        width * 72 / cfg.image_dpi, height * 72 / cfg.image_dpi, 0.0, 0.0, 0.0, 0.0, cfg,
    )
    print(f"overlay_build={time.perf_counter() - started:.3f} bytes={overlay.getbuffer().nbytes}")

    started = time.perf_counter()
    base_buf = io.BytesIO()
    c = rl_canvas.Canvas(base_buf, pagesize=(width * 72 / cfg.image_dpi, height * 72 / cfg.image_dpi))
    c.drawImage("/tmp/pidocr_profile_stage.png", 0, 0, width=width * 72 / cfg.image_dpi, height=height * 72 / cfg.image_dpi)
    c.save()
    base_reader = PdfReader(base_buf)
    base_page = base_reader.pages[0]
    overlay_reader = PdfReader(overlay)
    base_page.merge_page(overlay_reader.pages[0])
    writer = PdfWriter()
    writer.add_page(base_page)
    out = io.BytesIO()
    writer.write(out)
    print(f"pdf_build={time.perf_counter() - started:.3f} bytes={out.getbuffer().nbytes}")


if __name__ == "__main__":
    image = Image.new("RGB", (1845, 832), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row in range(102):
        draw.text((40 + (row % 4) * 430, 20 + (row // 4) * 31), f"TAG-{row:03d} PUMP PRESSURE", fill="black", font=font)
    image.save("/tmp/pidocr_profile_stage.png")
    main()
