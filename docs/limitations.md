# Limitations

Pylex runs OCR on raster images and rendered PDF pages, then adds the recognized text to a searchable PDF or returns regions through the Python API. It suits scans, screenshots, forms, receipts, diagrams, technical drawings and photographs.

## OCR is probabilistic

Results can be wrong for handwriting, very small, blurred, skewed, noisy, low-contrast or occluded text, and for languages without the right model. Confidence scores help find uncertain regions but do not prove correctness. Validate output on representative documents when errors have operational, legal, safety or financial consequences.

## What Pylex does not do

- No handwriting recognition, translation, table reconstruction, semantic headings or guaranteed reading order.
- No PDF/A output, deskewing or image clean-up. If you need those, OCRmyPDF is a better fit.
- It is a library and CLI, not a hosted OCR service.

## Scripts and fonts

The invisible text layer defaults to Helvetica, which covers Latin-1. Other scripts need `--font` with a Unicode `.ttf` or a built-in CJK font. Right-to-left scripts such as Arabic and Hebrew are not verified: shaping and ordering in the text layer may not match what a reader expects.

## Hardware

CPUs and NVIDIA GPUs (CUDA, through the `[gpu]` profiles) are what Pylex's install profiles support. Google TPUs are not supported. Apple Silicon, AMD and DirectML GPUs run on the CPU in Pylex. Other Paddle accelerator types are passed through to PaddleOCR but untested.

## Failure handling

If OCR fails on a PDF page, Pylex keeps the page unchanged, logs a warning and records it in `FileResult.failed_pages`; the file is still written. The CLI then exits with status 1 and names the pages. `create_engine` raises `PylexSetupError` when no backend is installed.

## Sensitive documents

Process sensitive files in an appropriate environment and do not attach them to public issues. Models are downloaded on first use; after that no network is needed.
