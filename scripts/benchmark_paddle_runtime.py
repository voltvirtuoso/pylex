from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pidocr.config import Config
from pidocr.engine import get_ocr_engine
from pidocr.tiling import ocr_image_tiled


def run(model_size: str) -> None:
    image = Image.new("RGB", (640, 280), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((80, 100), "PUMP P-101  PRESSURE 12.5 bar", fill="black", font=font)
    draw.text((80, 220), "FLOW FT-204  125 m3/h", fill="black", font=font)
    array = np.asarray(image)
    config = Config(
        fast_mode=False,
        tile_size=1600,
        use_textline_orientation=True,
        retry_upscale=False,
        disable_mkldnn=True,
        inference_engine="paddle_dynamic",
        model_size=model_size,
    )

    started = time.perf_counter()
    engine = get_ocr_engine(config)
    init_seconds = time.perf_counter() - started
    started = time.perf_counter()
    items = ocr_image_tiled(engine, array, config)
    inference_seconds = time.perf_counter() - started
    print(f"model={model_size} engine_init_seconds={init_seconds:.3f}")
    print(f"model={model_size} inference_seconds={inference_seconds:.3f} detections={len(items)}")
    print(f"model={model_size} texts={repr([text for text, _, _ in items])}")


if __name__ == "__main__":
    run("small")
    run("medium")
