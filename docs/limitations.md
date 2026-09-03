# Pylex Limitations and Supported Use Cases

Pylex performs OCR on raster images and rendered PDF pages, then adds recognized text to searchable-PDF output or returns OCR regions through the Python API. It is designed for scans, screenshots, forms, receipts, diagrams, technical drawings, and photographs.

OCR is probabilistic. Results may be wrong when text is handwritten, very small, blurred, skewed, noisy, low contrast, occluded, multilingual without the appropriate model support, or mixed with complex artwork. Confidence scores help identify uncertain regions but do not prove correctness.

Pylex preserves the visible source page and focuses on text extraction and text-layer placement. It does not promise perfect reading order, table reconstruction, semantic headings, handwriting recognition, document understanding, or translation. It is also not a hosted OCR API.

For important workflows, validate output against a representative document set. Record the input resolution, language, model, runtime, and configuration. If a document contains sensitive information, process it in an environment appropriate for that data and avoid uploading it to public issues or external services.
