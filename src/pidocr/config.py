"""Runtime configuration for the OCR engine and overlay renderer."""

from __future__ import annotations

from dataclasses import dataclass

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ALL_EXTS = PDF_EXTS | IMAGE_EXTS


@dataclass
class Config:
    dpi: int = 300
    image_dpi: int = 300
    min_confidence: float = 0.5
    lang: str = "en"
    ocr_version: str = "PP-OCRv4"
    font: str = "Helvetica"
    min_font_size: float = 1.0
    max_font_size: float = 200.0
    det_limit_side_len: int = 8000
    baseline_fraction: float = 0.18
    min_box_width_pt: float = 0.25
    min_box_height_pt: float = 0.25
    use_textline_orientation: bool = True
    use_gpu: bool = False
    disable_mkldnn: bool = True
    page_ranges: list[tuple[int, int]] | None = None  # 1-based, inclusive; None = all
