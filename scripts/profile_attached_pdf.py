from __future__ import annotations

import os
import time
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

from pylex.config import Config
from pylex.engine import get_ocr_engine
from pylex.pipeline import get_page_rotation, render_pdf_page_for_ocr
from pylex.tiling import compute_tiles, ocr_image_tiled

PDF = Path('/home/ubuntu/upload/10020-01-PID-005.pdf')


def main() -> None:
    tile_size = int(os.environ.get('PYLEX_TILE_SIZE', '1600'))
    tile_batch_size = int(os.environ.get('PYLEX_TILE_BATCH_SIZE', '4'))
    tile_overlap = int(os.environ.get('PYLEX_TILE_OVERLAP', '220'))
    dpi = int(os.environ.get('PYLEX_DPI', '300'))
    use_orientation = os.environ.get('PYLEX_ORIENTATION', '1') != '0'
    cfg = Config(
        model_size='small',
        inference_engine='auto',
        dpi=dpi,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        use_textline_orientation=use_orientation,
    )
    reader = PdfReader(str(PDF))
    doc = pdfium.PdfDocument(str(PDF))
    page = reader.pages[0]
    rotation = get_page_rotation(page)
    pdf_page = doc[0]
    started = time.perf_counter()
    image, _image_size = render_pdf_page_for_ocr(pdf_page, rotation, cfg.dpi)
    render_seconds = time.perf_counter() - started
    height, width = image.shape[:2]
    tiles = compute_tiles(width, height, cfg.tile_size, cfg.tile_overlap)
    print(f'rotation={rotation} rendered={width}x{height} pixels={width*height:,} render_seconds={render_seconds:.3f}')
    print(f'tile_count={len(tiles)} adaptive={cfg.adaptive_tiling} model={cfg.model_size}')

    started = time.perf_counter()
    engine = get_ocr_engine(cfg)
    init_seconds = time.perf_counter() - started
    print(f'engine_init_seconds={init_seconds:.3f}')

    started = time.perf_counter()
    progress = []
    items = ocr_image_tiled(engine, image, cfg, on_tile=lambda n, total: progress.append((n, total)))
    print(f'ocr_seconds={time.perf_counter() - started:.3f} detections={len(items)} callbacks={progress}')
    doc.close()


if __name__ == '__main__':
    main()
