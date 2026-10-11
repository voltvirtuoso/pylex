# Pylex: Python OCR for searchable PDFs and document text extraction

[![CI](https://github.com/voltvirtuoso/pylex/actions/workflows/ci.yml/badge.svg)](https://github.com/voltvirtuoso/pylex/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Latest release](https://img.shields.io/github/v/release/voltvirtuoso/pylex)](https://github.com/voltvirtuoso/pylex/releases)

**Documentation: https://voltvirtuoso.github.io/pylex/**

**Pylex** is an open-source Python OCR package and command-line tool for extracting text from images and scanned PDFs. It creates searchable PDFs while preserving the original visible page, and it provides a reusable Python API for document-processing pipelines.

Use Pylex when you need to add an invisible searchable text layer to a scanned PDF, OCR PNG/JPEG/TIFF/BMP/WebP/PDF files, process many files or folders from a CLI, recover small text in dense documents with overlapping tiles, or integrate OCR regions and confidence scores into a Python application.

> Pylex is an OCR and searchable-PDF tool. It is not a handwriting-recognition system, a hosted OCR service, or a full document-layout reconstruction engine. Accuracy depends on image quality, language, model, and document layout.

## What Pylex does

- **Searchable PDFs.** Adds an invisible text layer to scanned PDFs and images; the visible page is not changed.
- **Text extraction in Python.** `extract_text()` returns each line's text, confidence (0 to 1) and four-corner position.
- **Dense and large pages.** Overlapping tiles keep small labels at full resolution, with a retry for vertical text.
- **CPU and NVIDIA GPU.** One install profile per hardware (`[cpu]`, `[intel]`, `[gpu]`, `[gpu-cu12]`), and `pylex doctor` recommends one for your machine.
- **Batch processing.** Folders, recursion, parallel workers, page ranges, exclude patterns, dry runs, and skip-existing.
- **Languages and scripts.** PaddleOCR language codes through `--lang`; non-Latin searchable text through `--font`.
- **Open source.** MIT licensed, with a Python SDK and a command line tool from one package.

## Quick start

### Install

Pylex is not on PyPI yet, so install it from this repository. You need Python 3.10 to 3.12 and `git`. The package name is `pylex-ocr` (the name `pylex` on PyPI is an unrelated project); the command and import are `pylex`. Pick one profile for your hardware:

```bash
# Any CPU (recommended default)
python -m pip install "pylex-ocr[cpu] @ git+https://github.com/voltvirtuoso/pylex.git"

# NVIDIA GPU, nvidia-smi reports CUDA 13 or newer
python -m pip install "pylex-ocr[gpu] @ git+https://github.com/voltvirtuoso/pylex.git"

# NVIDIA GPU, nvidia-smi reports CUDA 12.x
python -m pip install "pylex-ocr[gpu-cu12] @ git+https://github.com/voltvirtuoso/pylex.git"

# Intel CPU with OpenVINO
python -m pip install "pylex-ocr[intel] @ git+https://github.com/voltvirtuoso/pylex.git"
```

Not sure which one? Install the core and let Pylex choose:

```bash
python -m pip install "pylex-ocr @ git+https://github.com/voltvirtuoso/pylex.git"
pylex doctor --install
```

Profiles are mutually exclusive. Google TPUs are not supported. No `git`? Use `https://github.com/voltvirtuoso/pylex/archive/refs/heads/main.zip` instead of `git+https://github.com/voltvirtuoso/pylex.git`. Full guide: [docs/installation.md](docs/installation.md). For development: `python -m pip install -e ".[cpu,dev]"`.

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

Pylex processes independent files in parallel. Pages inside one PDF remain sequential. Use `-j 1` with a GPU; multiple GPU workers can compete for the same device memory.

## Why Pylex

A searchable-PDF tool has to reconcile the page's display orientation (`/Rotate`) with PDF coordinates, or the invisible text layer lands in the wrong place. Pylex does this by rendering the page in viewer orientation, transferring the PDF page rotation into the content stream before merging the OCR overlay, and anchoring each OCR line at the start of its detected polygon. The original visible page is preserved while the output becomes searchable and selectable.

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

See the full guide in [docs/installation.md](docs/installation.md). In short: one of `[cpu]`, `[intel]`, `[gpu]`, `[gpu-cu12]`; `pylex doctor` recommends one and `pylex doctor --verify-gpu` confirms the GPU really loads. The older `[ocr]`, `[onnx]` and `[openvino]` extras still work as building blocks.

On Intel CPUs, compare `[intel]` (OpenVINO) with `[cpu]` on your own documents rather than assuming one is faster.

From source:

```bash
git clone https://github.com/voltvirtuoso/pylex.git
cd pylex
python -m pip install -e ".[cpu,dev]"
```

## Python API

Pylex returns recognized regions with text, confidence, and polygons in the original image pixel coordinates.

```python
from pylex import Config, create_engine, extract_text

config = Config(inference_engine="auto", model_size="small")  # Config() defaults to small; the CLI defaults to medium
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
| General CPU OCR | CLI default PP-OCRv6 medium (`--model-size small` is faster) with automatic runtime | Chooses an available runtime automatically |
| Small or medium raster image | Default adaptive full-page path | Avoids unnecessary tiled passes |
| Dense or large document | Default tiled mode | More passes, but better local resolution |
| Higher recognition accuracy | `--model-size medium` | More CPU time and memory |
| Simple low-latency pass | `--fast` | Less tolerant of dense or rotated layouts |
| Many independent files | `-j N` | Each worker owns an OCR engine |
| NVIDIA GPU | `--device gpu -j 1` (or `--device auto`) | Needs a `[gpu]` or `[gpu-cu12]` install |

Benchmarks are machine- and document-dependent. When reporting performance, include the CPU/GPU, operating system, Python version, model, runtime, language, input resolution, and document type.

## FAQ

**What is Pylex?**
A Python OCR library and command-line tool built on PaddleOCR models. It makes scanned PDFs searchable and extracts text, confidence and position from images.

**How do I make a scanned PDF searchable from Python?**

```python
from pylex import Config, create_engine, create_searchable_document

config = Config()
engine = create_engine(config)
result = create_searchable_document("scan.pdf", "scan-searchable.pdf", config=config, engine=engine)
print(result.ok, result.lines_added)
```

**How do I install it?**
Pylex is not on PyPI yet, so install it from GitHub with the profile for your hardware. See [Install](#install) above and [docs/installation.md](docs/installation.md).

**Does Pylex support GPUs?**
Yes, NVIDIA GPUs through the `[gpu]` (CUDA 13) and `[gpu-cu12]` (CUDA 12) profiles. Run `pylex doctor --verify-gpu` to confirm CUDA really loads, then use `--device gpu`.

**Does Pylex support Google TPUs?**
No. PaddleOCR's documented hardware list does not include TPUs and Pylex has no TPU code path. It runs on the host CPU of a TPU machine.

**How is Pylex different from OCRmyPDF?**
OCRmyPDF uses Tesseract by default, produces PDF/A, and can deskew and clean pages; it also has a plugin interface with PaddleOCR plugins. Pylex uses PaddleOCR models directly, with tiling for dense pages and a Python API that returns text regions. In this project's synthetic tests OCRmyPDF with Tesseract placed text correctly on rotated pages; Pylex has not been measured in the same harness yet. See [docs/accuracy-and-benchmarks.md](docs/accuracy-and-benchmarks.md).

**Which languages does it support?**
Those PaddleOCR supports, selected with `--lang`. The default invisible-text font covers Latin-1; use `--font` with a Unicode `.ttf` for other scripts.

**Does it send my documents anywhere?**
OCR runs on your machine. Models are downloaded on first use, and PaddleOCR checks model-host connectivity at startup unless `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` is set.

**Which file types can it read?**
PDF, PNG, JPEG, TIFF, BMP and WebP. The output is always a PDF.

## Limitations

OCR can misrecognize text, especially when scans are low-resolution, skewed, noisy, handwritten, multilingual, or heavily occluded. Pylex preserves the visible document and adds recognized text; it does not guarantee semantic structure, table reconstruction, reading-order perfection, or perfect recognition. Always validate OCR output in workflows where errors have operational, legal, safety, or financial consequences.

Pylex is intentionally domain-neutral. It can process forms, receipts, screenshots, scans, diagrams, technical drawings, and photographs, but domain-specific results should be evaluated against a representative validation set.

## Development

```bash
python -m pip install -e ".[cpu,dev]"
pytest
ruff check .
```

Please open an issue with the Pylex version, Python version, operating system, input type, language, command, model/runtime, and a redacted or synthetic reproduction when possible.

## Project links

- [Documentation site](https://voltvirtuoso.github.io/pylex/) (the same pages as `docs/`, in one scrollable page; the source is `docs/index.html`)
- [Installation (CPU, GPU, TPU)](docs/installation.md)
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
