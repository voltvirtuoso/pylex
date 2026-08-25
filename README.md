# pidocr

Add an invisible, searchable OCR text layer to PDFs and images — the
visible drawing is never touched, it just becomes selectable/searchable.
Built for messy engineering drawings (P&IDs, scanned sheets) rather than
clean document scans.

Works like `tesseract`, but as a batch-capable CLI:

```bash
pidocr drawing.pdf                        # -> drawing.ocr.pdf
pidocr drawing.pdf -o searchable.pdf
pidocr scans/ -o out/ -r                  # recurse a folder, mirror structure
pidocr page.png page.tif -o out/          # images become searchable PDFs
pidocr drawing.pdf --pages 1-5,9          # only OCR those pages
pidocr big_folder/ -r --dry-run           # preview what would run
pidocr file.pdf --in-place --force        # overwrite the input file
pidocr big_folder/ -r -o out/ -j 4         # 4 files in parallel
```

## Why this exists

Naive "render page -> OCR -> paste text back on top" pipelines misplace
text on rotated PDF pages (`/Rotate` 90/180/270), because the OCR image
is in display/viewer orientation while the PDF content stream is in a
different coordinate system. pidocr fixes this by:

1. Rendering each page exactly as a viewer would show it (rotation applied).
2. Running OCR on that rendered image.
3. **Transferring the page's `/Rotate` into its content stream** before
   merging the overlay, so the PDF-content coordinate system matches the
   OCR image's orientation.
4. Anchoring each OCR line at the **start** of its detected polygon
   (not its center), so long lines can't drift outside the page.

That combination is what makes OCR placement correct on rotated
engineering drawings as well as normal upright pages.

## Parallelism

`-j/--jobs N` runs N files at a time using separate **worker processes**,
each with its own PaddleOCR engine instance that it reuses across every
file it picks up (so the (slow) engine load only happens once per worker,
not once per file). This is process-based, not thread-based, because
PaddleOCR inference isn't safe to call concurrently from multiple threads
on one engine, and OCR is CPU-bound anyway — threads wouldn't help.

Pages within a single PDF are still OCR'd sequentially by whichever
worker owns that file; the parallelism is across files in a batch, not
across pages of one file. Progress prints in completion order (whichever
worker finishes first), not input order.

`--gpu` combined with `-j > 1` will have every worker process share the
same GPU, which usually contends for memory rather than actually speeding
things up — prefer `--gpu -j 1`, or drop `--gpu` and use `-j N` on CPU.

## Install

```bash
pip install pidocr[ocr]
```

`[ocr]` pulls in `paddleocr` + `paddlepaddle`, which are large — the base
install (`pip install pidocr`) is enough for development/testing with a
stubbed engine, but you need the `[ocr]` extra to actually run OCR.

From source:

```bash
git clone <repo>
cd pidocr
pip install -e ".[ocr]"
```

## Usage

```
pidocr [-h] [-o PATH] [-r] [--in-place] [--suffix SUFFIX] [-f]
       [--formats ext,ext,...] [--pages RANGE] [--dry-run]
       [--skip-existing] [--dpi DPI] [--image-dpi IMAGE_DPI] [--lang LANG]
       [--min-confidence MIN_CONFIDENCE]
       [--det-limit-side-len DET_LIMIT_SIDE_LEN] [--gpu]
       [--no-textline-orientation] [--font FONT]
       [--baseline-fraction BASELINE_FRACTION] [-v] [-q] [--version]
       inputs [inputs ...]
```

Run `pidocr --help` for the full flag reference with defaults.

Supported input: `.pdf`, `.png`, `.jpg`/`.jpeg`, `.tif`/`.tiff`, `.bmp`,
`.webp`, as single files, multiple files, or folders (`-r` to recurse).

## Development

```bash
pip install -e ".[ocr,dev]"
pytest
ruff check .
```

## License

MIT
