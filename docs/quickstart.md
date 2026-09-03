# Pylex Quickstart: Create a Searchable PDF

Pylex is a Python OCR tool for turning scanned PDFs and raster images into searchable, selectable PDF documents. The original visible page is preserved while an invisible OCR text layer is added.

## Install

```bash
python -m pip install "pylex-ocr[ocr]"
```

The package installs the `pylex` command and the `pylex` Python module. The `[ocr]` extra installs the PaddleOCR backend required for real OCR processing.

## OCR one scanned PDF

```bash
pylex scan.pdf -o scan-searchable.pdf
```

If `-o` is omitted, Pylex writes `scan.ocr.pdf` beside the input file:

```bash
pylex scan.pdf
```

## OCR an image

```bash
pylex page.png -o page-searchable.pdf
```

Supported raster formats include PNG, JPEG, TIFF, BMP, and WebP.

## OCR a folder

```bash
pylex scans/ -r -o out/ -j 4
```

The `-r` flag recursively scans the input folder. The `-j 4` flag processes independent files with four worker processes. It does not parallelize pages within one PDF.

## Improve dense-document recall

For crowded technical drawings, diagrams, or forms where small labels are missed, use smaller overlapping tiles:

```bash
pylex drawing.pdf --tile-size 1000 --tile-overlap 250
```

For text touching line art:

```bash
pylex drawing.pdf --det-box-thresh 0.45 --det-unclip-ratio 2.0
```

## Check the result

Open the output PDF in a viewer and try to select or search for text. OCR quality depends on resolution, language, noise, rotation, document layout, model, and runtime. Validate the output before relying on it for high-stakes decisions.

## Next steps

Read the [CLI reference](cli.md), [Python API guide](python-api.md), [accuracy guidance](accuracy-and-benchmarks.md), and [limitations](limitations.md).
