"""Pylex: PYthon LEXicon and language extraction.

Pylex provides generic OCR text-region extraction for raster images and
searchable-PDF generation for images and PDFs. It can be used from Python or
through the ``pylex`` command-line program.
"""

from pylex.api import TextRegion, create_engine, create_searchable_document, extract_text
from pylex.config import Config
from pylex.pipeline import FileResult, process_image, process_pdf

__version__ = "2.0.0"

__all__ = [
    "Config",
    "FileResult",
    "TextRegion",
    "__version__",
    "create_engine",
    "create_searchable_document",
    "extract_text",
    "process_image",
    "process_pdf",
]
