# Pylex documentation

Pylex is an open-source Python OCR package and command-line tool. It reads text from images and scanned PDFs, and adds an invisible, searchable text layer to a PDF while keeping the original page untouched. The same engine is available as a Python library that returns text, confidence and position for every line.

It is built for pages where plain OCR struggles: **large, dense or rotated documents** such as engineering drawings, where small labels sit between symbols and PDF pages carry a rotation flag.

## Two ways to use it

| | Command line | Python SDK |
|---|---|---|
| Best for | Folders of scans, one-off jobs, scripts | Applications, services, notebooks |
| Start with | `pylex scan.pdf` | `extract_text("page.png")` |
| Reference | [CLI reference](cli.md) | [Python SDK](python-api.md) |

Both share one install and one set of models.

## Sixty seconds

```bash
python -m pip install pylex-ocr
pylex doctor --install        # detects your CPU or NVIDIA GPU and installs the right runtime
pylex scan.pdf                # writes scan.ocr.pdf
```

Hardware profiles (CPU, Intel, NVIDIA CUDA 12 and 13) and what is and is not supported are covered in [Installation](installation.md).

## When to choose something else

Pylex is not a general document-conversion suite. If you need PDF/A output, deskewing and clean-up, OCRmyPDF is the established choice and has a PaddleOCR plugin. Pylex is not a handwriting recognizer, a layout or table reconstruction engine, or a hosted service. See [Limitations](limitations.md).

## Where to go

- New: [Installation](installation.md), then [Quickstart](quickstart.md)
- Understanding results and speed: [How it works](how-it-works.md), [Accuracy and benchmarks](accuracy-and-benchmarks.md)
- Something wrong: [Troubleshooting](troubleshooting.md)
