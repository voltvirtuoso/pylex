# Pylex: Python OCR for searchable PDFs and document text extraction

[![CI](https://github.com/voltvirtuoso/pylex/actions/workflows/ci.yml/badge.svg)](https://github.com/voltvirtuoso/pylex/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Latest release](https://img.shields.io/github/v/release/voltvirtuoso/pylex)](https://github.com/voltvirtuoso/pylex/releases)

**Pylex** is an open-source Python OCR package and command-line tool for extracting text from images and scanned PDFs. It creates searchable PDFs while preserving the original visible page, and it provides a reusable Python API for document-processing pipelines.

Use Pylex when you need to add an invisible searchable text layer to a scanned PDF, OCR PNG/JPEG/TIFF/BMP/WebP/PDF files, process many files or folders from a CLI, recover small text in dense documents with overlapping tiles, or integrate OCR regions and confidence scores into a Python application.

> Pylex is an OCR and searchable-PDF tool. It is not a handwriting-recognition system, a hosted OCR service, or a full document-layout reconstruction engine. Accuracy depends on image quality, language, model, and document layout.

## Quick start

### Install

The distribution name is `pylex-ocr` to avoid confusion with an unrelated legacy PyPI project that already uses the name `pylex`. The installed command and Python import remain `pylex`.

```bash
python -m pip install "pylex-ocr[ocr]"
```

For development or tests without the OCR backend:

```bash
python -m pip install -e ".[dev]"
```

### Create a searchable PDF

```bash
pylex scan.pdf                         # writes scan.ocr.pdf
pylex scan.pdf -o searchable.pdf
pylex page.png -o searchable.pdf       # image input becomes a searchable PDF
```

### Process a folder

```bash
pylex scans/ -r -o out/ -j 4
```

Pylex processes independent files in parallel. Pages inside one PDF remain sequential. Use `-j 1` with `--gpu`; multiple GPU workers can compete for the same device memory.

## Why Pylex

Many OCR-to-PDF workflows render a page, run OCR, and place text over the original image without carefully reconciling display orientation and PDF coordinates. On rotated pages, this can make the invisible text layer appear in the wrong location.

Pylex handles this by rendering the page in viewer orientation, transferring the PDF page rotation into the content stream before merging the OCR overlay, and anchoring each OCR line at the start of its detected polygon. The original visible page is preserved while the output becomes searchable and selectable.

For large or crowded documents, Pylex uses overlapping tiles at local resolution by default. This helps recover small labels that can disappear when an entire page is resized before detection.

## Supported inputs

Pylex accepts `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, and `.webp` files. Inputs may be a single file, multiple files, or a folder scanned recursively with `-r`.

## Common workflows

### OCR a dense drawing

```bash
pylex dense_document.pdf --tile-size 1000 --tile-overlap 250
```

If text touches line art or is tightly crowded, adjust the detector settings:

```bash
pylex dense_document.pdf \
  --det-box-thresh 0.45 \
  --det-unclip-ratio 2.0
```

### Run a quick pass on a simple image

```bash
pylex simple_scan.png --fast
pylex simple_scan.png --fast --retry-upscale
```

`--fast` disables some accuracy-oriented processing. Use the default profile for dense layouts, rotated pages, or documents where recall matters more than latency.

### Select pages and exclude prior outputs

```bash
pylex drawing.pdf --pages 1-5,9
pylex scans/ -r -xf "*_OCR*" -xf "backup/*"
```

### Choose a runtime

```bash
pylex input.pdf --inference-engine onnxruntime -v
pylex input.pdf --inference-engine openvino -v
pylex input.pdf --model-size medium
```

The automatic runtime prefers OpenVINO when the OpenVINO-enabled ONNX Runtime package is available, then standard ONNX Runtime, and finally dynamic Paddle inference. The first optimized-runtime invocation may be slower while models and compiled artifacts are prepared.

## Installation options

The `[ocr]` extra installs PaddleOCR and PaddlePaddle. The optional runtime extras are useful for explicit benchmarking or deployment choices:

```bash
python -m pip install "pylex-ocr[ocr]"
python -m pip install "pylex-ocr[ocr,onnx]"
python -m pip install "pylex-ocr[ocr,openvino]"
```

On Intel CPUs, compare OpenVINO and standard ONNX Runtime on the target machine rather than assuming one is always faster. On Windows, use the native ONNX/OpenVINO packages instead of PaddleOCR’s Linux-oriented HPI dependency installer.

From source:

```bash
git clone https://github.com/voltvirtuoso/pylex.git
cd pylex
python -m pip install -e ".[ocr,dev]"
```

## Python API

Pylex returns recognized regions with text, confidence, and polygons in the original image pixel coordinates.

```python
from pylex import Config, create_engine, extract_text

config = Config(inference_engine="auto", model_size="small")
engine = create_engine(config)  # Reuse this engine for multiple images.
regions = extract_text("page.png", config=config, engine=engine)

for region in regions:
    print(region.text, region.confidence, region.polygon.tolist())
```

Create a searchable PDF from an image or existing PDF:

```python
from pylex import Config, create_engine, create_searchable_document

config = Config(inference_engine="auto")
engine = create_engine(config)
result = create_searchable_document(
    "input.pdf",
    "output_searchable.pdf",
    config=config,
    engine=engine,
)

print(result.ok, result.pages_processed, result.lines_added)
```

Pass one engine to multiple calls to avoid rebuilding the OCR backend for every file. The API also supports `on_page`, `on_stage`, `on_tile`, and `on_tile_done` callbacks for applications, services, notebooks, and custom progress interfaces. Lower-level `process_image` and `process_pdf` functions are available when an application needs direct output-path and callback control.

## CLI reference

Run the following command for the complete option list and defaults:

```bash
pylex --help
```

The main options include input/output paths, recursive folder processing, in-place output, page ranges, dry runs, skip-existing behavior, exclusions, worker processes, DPI, language, model size, inference engine, runtime cache, confidence thresholds, tiling, detector controls, GPU mode, progress output, and version reporting.

## Performance and accuracy guidance

| Workload | Recommended starting point | Trade-off |
|---|---|---|
| General CPU OCR | Default PP-OCRv6 small with automatic runtime | Chooses an available runtime automatically |
| Small or medium raster image | Default adaptive full-page path | Avoids unnecessary tiled passes |
| Dense or large document | Default tiled mode | More passes, but better local resolution |
| Higher recognition accuracy | `--model-size medium` | More CPU time and memory |
| Simple low-latency pass | `--fast` | Less tolerant of dense or rotated layouts |
| Many independent files | `-j N` | Each worker owns an OCR engine |
| NVIDIA GPU | `--gpu -j 1` | Requires a compatible Paddle GPU install |

Benchmarks are machine- and document-dependent. When reporting performance, include the CPU/GPU, operating system, Python version, model, runtime, language, input resolution, and document type.

## Limitations

OCR can misrecognize text, especially when scans are low-resolution, skewed, noisy, handwritten, multilingual, or heavily occluded. Pylex preserves the visible document and adds recognized text; it does not guarantee semantic structure, table reconstruction, reading-order perfection, or perfect recognition. Always validate OCR output in workflows where errors have operational, legal, safety, or financial consequences.

Pylex is intentionally domain-neutral. It can process forms, receipts, screenshots, scans, diagrams, technical drawings, and photographs, but domain-specific results should be evaluated against a representative validation set.

## Development

```bash
python -m pip install -e ".[ocr,dev]"
pytest
ruff check .
```

Please open an issue with the Pylex version, Python version, operating system, input type, language, command, model/runtime, and a redacted or synthetic reproduction when possible.

## Project links

- [Quickstart](docs/quickstart.md)
- [CLI reference](docs/cli.md)
- [Python API guide](docs/python-api.md)
- [Accuracy and benchmarks](docs/accuracy-and-benchmarks.md)
- [Limitations](docs/limitations.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Releases](https://github.com/voltvirtuoso/pylex/releases)
- [Issue tracker](https://github.com/voltvirtuoso/pylex/issues)

## License

Pylex is released under the [MIT License](LICENSE).
