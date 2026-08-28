from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pylex.config import Config
from pylex.engine import get_ocr_engine
from pylex.tiling import ocr_image_tiled

SOURCE = Path('/home/ubuntu/upload/pasted_file_PgMc78_image.png')


def main() -> None:
    engine = get_ocr_engine(Config(model_size='small', inference_engine='auto', use_textline_orientation=True))
    original = np.asarray(Image.open(SOURCE).convert('RGB'))
    for angle in (0, 90, 270):
        image = np.rot90(original, k=angle // 90)
        cfg = Config(model_size='small', inference_engine='auto', tile_size=2000, use_textline_orientation=True)
        items = ocr_image_tiled(engine, image, cfg)
        print(f'angle={angle} detections={len(items)}')
        for text, poly, score in items:
            if angle == 0:
                xs = poly[:, 0]
                ys = poly[:, 1]
                box_width = max(xs) - min(xs)
                box_height = max(ys) - min(ys)
                if box_height > box_width * 1.3:
                    print(f'  vertical_candidate aspect={box_height / max(box_width, 1):.2f} {score:.3f} {text!r}')
            if any(token in text.upper() for token in ('ESDV', 'BDV', '0101', '0107', '15038')):
                print(f'  {score:.3f} {text!r}')


if __name__ == '__main__':
    main()
