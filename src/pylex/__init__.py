"""Pylex: PYthon LEXicon and language extraction.

Pylex provides generic OCR text-region extraction for raster images and
searchable-PDF generation for images and PDFs. It can be used from Python or
through the ``pylex`` command-line program.
"""

from pylex.api import TextRegion, create_engine, create_searchable_document, extract_text
from pylex.config import Config
from pylex.engine import PylexSetupError
from pylex.pipeline import FileResult, process_image, process_pdf
from pylex.runtime import detect_hardware, doctor, recommend_install, verify_gpu

__version__ = "2.0.0"

__all__ = [
    "Config",
    "FileResult",
    "PylexSetupError",
    "TextRegion",
    "__version__",
    "create_engine",
    "create_searchable_document",
    "detect_hardware",
    "doctor",
    "extract_text",
    "process_image",
    "process_pdf",
    "recommend_install",
    "verify_gpu",
]
