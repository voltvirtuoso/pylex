from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pidocr.config import Config
from pidocr.engine import get_ocr_engine
from pidocr.tiling import ocr_image_tiled


def main() -> None:
    image = Image.new("RGB", (3000, 2200), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for y, text in ((500, "PUMP P-101 PRESSURE 12.5 bar"), (1100, "FLOW FT-204 125 m3/h"), (1700, "VALVE XV-301 OPEN")):
        draw.text((300, y), text, fill="black", font=font)
    array = np.asarray(image)

    for adaptive in (True,):
        config = Config(
            model_size="small",
            inference_engine="paddle_dynamic",
            disable_mkldnn=True,
            use_textline_orientation=True,
            tile_size=1600,
            tile_overlap=220,
            adaptive_tiling=adaptive,
        )
        started = time.perf_counter()
        engine = get_ocr_engine(config)
        init_seconds = time.perf_counter() - started
        passes: list[tuple[int, int]] = []
        started = time.perf_counter()
        items = ocr_image_tiled(
            engine,
            array,
            config,
            on_tile=lambda n, total, bucket=passes: bucket.append((n, total)),
        )
        inference_seconds = time.perf_counter() - started
        print(
            f"adaptive={adaptive} init_seconds={init_seconds:.3f} "
            f"ocr_seconds={inference_seconds:.3f} passes={passes} detections={len(items)}"
        )


if __name__ == "__main__":
    main()
