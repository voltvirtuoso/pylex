"""pidocr — add an invisible, searchable OCR text layer to PDFs and images.

Built around a PaddleOCR-based text-layer engine that anchors every OCR
line at the START of its polygon (not its center) and transfers PDF page
rotation into the page CONTENT before the OCR overlay is merged. That
combination is what makes OCR placement correct on rotated engineering
drawings (/Rotate = 90 / 180 / 270) as well as normal upright pages.
"""

from pidocr.config import Config
from pidocr.pipeline import FileResult, process_image, process_pdf

__version__ = "1.0.0"

__all__ = ["Config", "FileResult", "process_pdf", "process_image", "__version__"]
