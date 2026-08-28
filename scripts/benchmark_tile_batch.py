from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from pypdf import PdfReader

from pidocr.config import Config
from pidocr.engine import get_ocr_engine, _parse_results
from pidocr.pipeline import get_page_rotation, render_pdf_page_for_ocr
from pidocr.tiling import compute_tiles


PDF = Path('/home/ubuntu/upload/10020-01-PID-005.pdf')


def main() -> None:
    cfg = Config(model_size='small', inference_engine='auto', dpi=300, tile_size=1600, tile_overlap=220)
    reader = PdfReader(str(PDF))
    doc = pdfium.PdfDocument(str(PDF))
    page = reader.pages[0]
    image, _ = render_pdf_page_for_ocr(doc[0], get_page_rotation(page), cfg.dpi)
    height, width = image.shape[:2]
    tiles = compute_tiles(width, height, cfg.tile_size, cfg.tile_overlap)
    engine = get_ocr_engine(cfg)
    crops = [np.ascontiguousarray(image[y0:y1, x0:x1, ::-1]) for x0, y0, x1, y1 in tiles[:4]]

    started = time.perf_counter()
    individual = []
    for crop in crops:
        individual.extend(engine.predict(crop))
    individual_seconds = time.perf_counter() - started

    started = time.perf_counter()
    batched = list(engine.predict(crops))
    batch_seconds = time.perf_counter() - started
    print(f'individual_seconds={individual_seconds:.3f} result_objects={len(individual)}')
    print(f'batch_seconds={batch_seconds:.3f} result_objects={len(batched)}')
    print(f'parsed_batch_items={sum(len(_parse_results([result], cfg)) for result in batched)}')
    doc.close()


if __name__ == '__main__':
    main()
