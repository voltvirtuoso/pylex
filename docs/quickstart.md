# Quickstart

Five minutes from nothing to a searchable PDF, once from the command line and once from Python. Install first: [Installation](installation.md).

```bash
python -m pip install "pylex-ocr[cpu] @ git+https://github.com/voltvirtuoso/pylex.git"   # or [gpu], [gpu-cu12], [intel]
```

## Command line

### One scanned PDF

```bash
pylex scan.pdf -o scan-searchable.pdf
```

Without `-o`, Pylex writes `scan.ocr.pdf` next to the input. The first run downloads models, so it is slower than later runs. Open the output in any viewer and search or select text: the page looks the same, with an invisible text layer on top.

### An image

```bash
pylex page.png -o page-searchable.pdf
```

PNG, JPEG, TIFF, BMP and WebP become single-page searchable PDFs.

### A folder

```bash
pylex scans/ -r -o out/ -j 4
```

`-r` recurses, `out/` mirrors the folder structure, `-j 4` processes four files at once. Add `--dry-run` first to preview.

### Dense or crowded pages

```bash
pylex drawing.pdf --tile-size 1000 --tile-overlap 250
pylex drawing.pdf --det-box-thresh 0.45 --det-unclip-ratio 2.0   # text touching line art
```

### Use a GPU

```bash
pylex doctor --verify-gpu      # confirm CUDA really loads
pylex scan.pdf --device gpu    # or --device auto
```

Keep `-j 1` with a GPU: worker processes compete for its memory.

## Python

```python
from pylex import Config, create_engine, extract_text

config = Config(model_size="small")
engine = create_engine(config)          # load once, reuse

for region in extract_text("page.png", config=config, engine=engine):
    print(f"{region.confidence:.2f}  {region.text}")
```

Each region has `text`, a `confidence` from 0 to 1, and a four-corner `polygon` in the image's pixel coordinates.

```python
from pylex import create_searchable_document

result = create_searchable_document("input.pdf", "output.pdf", config=config, engine=engine)
if not result.ok:
    raise RuntimeError(result.error)
print(result.pages_processed, "pages,", result.lines_added, "lines")
```

For a GPU use `Config(device="gpu")`.

## Check the result

OCR is probabilistic. Search for words you know are on the page and look at confidence values for anything important. See [Limitations](limitations.md).

## Next

[CLI reference](cli.md) · [Python SDK](python-api.md) · [Accuracy and benchmarks](accuracy-and-benchmarks.md) · [Troubleshooting](troubleshooting.md)
