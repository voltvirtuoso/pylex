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
Because each worker owns an independent PaddleOCR engine, a multi-worker run
can spend a noticeable amount of time loading and compiling the backend before
the first file completes. pidocr shows a startup spinner/status message during
that phase, so this delay is visible rather than looking like a stalled run.

## Dense / crowded drawings (missed text near symbol clusters)

A single whole-page OCR pass resizes the entire page down before
detecting text. On an open area of a drawing that's harmless — the text
stays big relative to everything around it. But in a crowded cluster
(several valves, an instrument bubble, and a line tag squeezed together),
the tag text is small to begin with, and the whole-page resize can shrink
it below what the detector picks up — even though the exact same size of
text on an open pipe run gets OCR'd fine.

pidocr OCRs each page in overlapping tiles at full local resolution by
default (`--tile-size 2000 --tile-overlap 220`), then merges and
de-duplicates detections back into page coordinates, which recovers most
of this. If a particular drawing is still missing tags in its busiest
areas:

```bash
# Smaller tiles = more effective resolution per region.
pidocr dense_pid.pdf --tile-size 1000 --tile-overlap 250

# Recover text that's touching or squeezed against line-art.
pidocr dense_pid.pdf --det-box-thresh 0.45 --det-unclip-ratio 2.0

# Combine both for very busy drawings.
pidocr dense_pid.pdf --tile-size 1000 --tile-overlap 250 \
                      --det-box-thresh 0.45 --det-unclip-ratio 2.0
```

`--tile-size 0` (or `--no-tile`) disables tiling entirely and falls back
to one whole-page pass — useful for quick tests or very simple drawings
where tiling only adds runtime. The equivalent convenience profile is:

```bash
pidocr simple_scan.png --fast
# If tiny text is missed on a low-resolution image:
pidocr simple_scan.png --fast --retry-upscale
```

`--det-thresh`/`--det-box-thresh`/
`--det-unclip-ratio` map directly to PaddleOCR's detector knobs and are
passed through only when set, so they're safe to leave alone on drawings
that already OCR well.

## Install

```bash
pip install pidocr[ocr]
```

`[ocr]` pulls in `paddleocr`, `paddlepaddle`, and `onnxruntime`, which are
large — the base install (`pip install pidocr`) is enough for development/testing
with a stubbed engine, but you need the `[ocr]` extra to actually run OCR.
The default `--inference-engine auto` selects OpenVINO when the OpenVINO-enabled
ONNX Runtime package is installed, then standard ONNX Runtime, and finally
dynamic Paddle. The first OpenVINO process may be slower while detector,
recognizer, and orientation engines are compiled; pidocr stores compiled
artifacts in a persistent platform cache. Override the location with
`--runtime-cache-dir` if needed. On an Intel CPU, install the optional OpenVINO build with

`pip install pidocr[openvino]` after installing the OCR dependencies. You can
force `--inference-engine openvino` or `--inference-engine onnxruntime` for
benchmarking. PaddleOCR's HPI dependency installer currently targets Linux
x86-64; on Windows, use these native ONNX/OpenVINO packages rather than HPI.

From source:

```bash
git clone <repo>
cd pidocr
pip install -e ".[ocr]"
```

## Usage

```
pidocr [-h] [-o PATH] [-r] [--in-place] [--suffix SUFFIX] [-f]
       [--formats ext,ext,...] [-xf PATTERN] [--pages RANGE] [--dry-run]
       [--skip-existing] [-j N] [--dpi DPI] [--image-dpi IMAGE_DPI]
       [--lang LANG] [--ocr-version {PP-OCRv4,PP-OCRv5,PP-OCRv6}]
       [--model-size {tiny,small,medium}]
       [--inference-engine {auto,paddle_static,paddle_dynamic,onnxruntime,openvino}]
       [--runtime-cache-dir PATH] [--enable-hpi]
       [--hpi-backend {auto,paddle,openvino,onnxruntime}]
       [--min-confidence MIN_CONFIDENCE]
       [--det-limit-side-len DET_LIMIT_SIDE_LEN] [--gpu]
       [--no-textline-orientation] [--no-vertical-text-retry]
       [--fast] [--disable-mkldnn]
       [--recognition-batch-size N] [--orientation-batch-size N]
       [--no-runtime-fallback] [--cpu-threads N] [--retry-upscale] [--font FONT]
       [--baseline-fraction BASELINE_FRACTION] [--tile-size PX]
       [--tile-overlap PX] [--no-tile] [--dedupe-tol-px PX]
       [--det-thresh F] [--det-box-thresh F] [--det-unclip-ratio F]
       [-v] [-q] [--version]
       inputs [inputs ...]
```

Run `pidocr --help` for the full flag reference with defaults.

Supported input: `.pdf`, `.png`, `.jpg`/`.jpeg`, `.tif`/`.tiff`, `.bmp`,
`.webp`, as single files, multiple files, or folders (`-r` to recurse).

### Excluding files (`-xf`)

`-xf`/`--exclude` skips files matching a glob/wildcard pattern when
scanning folders (fnmatch rules: `*` matches anything, including path
separators). To match a token anywhere inside a filename, include wildcards
around it; for example, use `*_OCR*` to skip outputs from an earlier run.
Patterns are matched against both the filename and the path relative to the
input folder. Repeat the flag for multiple patterns, or comma-separate several
within one:

```bash
pidocr scans/ -r -xf "*_draft*" -xf "backup/*"
pidocr scans/ -r -xf "*.tmp,*_old*,archive/*"
pidocr scans/ -r -xf "*_OCR*"             # skip prior OCR outputs
```

### Speed and accuracy profiles

The default profile uses PP-OCRv6 small with an automatic CPU runtime. When
ONNX Runtime is installed, pidocr uses it for the same PaddleOCR detector,
recognizer, and orientation models; otherwise it falls back to dynamic Paddle
inference. For images that fit within the detector resolution and 8-megapixel
safety budget, the normal path automatically uses one full-page pass; larger
images retain overlapping tiles. For higher recognition and detection accuracy,
use `--model-size medium`; use `--strict-tiling` when dense drawing recall is
more important than avoiding extra passes. The normal path also retries only
low-confidence tall detections at 90 degrees, which recovers vertical labels
without rotating every tile. Use `--no-vertical-text-retry` only as a diagnostic
or when throughput is more important than vertical-label recall.
If a low-resolution image produces no text, add `--retry-upscale` to make one
1.5x retry; it is deliberately not automatic because the retry adds work.

| Workload | Recommended options | Trade-off |
|---|---|---|
| General CPU OCR | Default `PP-OCRv6 small` + automatic ONNX/dynamic runtime | Uses ONNX Runtime when installed; dynamic Paddle fallback |
| Small/medium raster image | Adaptive full-page pass | Avoids unnecessary overlapping passes without changing OCR settings |
| Higher-accuracy OCR | `--model-size medium` with automatic or explicit runtime | More CPU time and memory |
| Linux/WSL optimized deployment | `--enable-hpi` or tested static runtime | First engine build may be slower; cache is reused |
| Simple image with tiny text | Default mode + `--retry-upscale` | One extra pass only when the first pass finds nothing |
| Lowest-latency trade-off | `--fast` | Less tolerant of rotated text and dense layouts |
| Large or dense P&ID drawing | Default tiled mode (`--tile-size 2000`) | Fewer passes than the previous default while retaining local-resolution recall |
| Many CPU files/pages | `--enable-hpi` or `--inference-engine paddle_static` with a measured `-j` value | First optimized-engine setup may be slower; later repeated inference may improve |
| NVIDIA GPU installation | `--gpu -j 1` | Requires a compatible Paddle GPU installation; avoid duplicate GPU workers |

For a single image, use `-j 1`; multiple workers initialize separate OCR
engines and are intended for batches of independent files, not for speeding up
one image. When `-j N` is used, pidocr automatically divides the available CPU
thread budget across workers. Override that with `--cpu-threads N` if a machine
has a known optimal setting. For supported Linux/WSL deployments, `--enable-hpi`
can build and cache PaddleOCR's optimized backend; the first run may be slower.
Use `-v` to see the selected runtime. On Intel CPUs, compare `auto` and
`--inference-engine openvino`; OpenVINO is not universally faster than standard
ONNX Runtime, so keep whichever is faster and accurate on the target machine.
On Paddle builds where static oneDNN/PIR inference is compatible,
`--inference-engine paddle_static --enable-mkldnn` remains an explicit advanced
option; automatic mode does not force it.

### Progress

The runtime cache can be made explicit on Windows:

```powershell
pidocr "C:\path\test.png" --model-size small --inference-engine openvino `
  --runtime-cache-dir "$env:LOCALAPPDATA\pidocr\openvino" -v --force
```

The first invocation can still be slower because the models and compiled
kernels must be prepared. Later processes can reuse the cache, although model
loading and filesystem/antivirus overhead remain machine-dependent.

A live progress bar tracks files processed when stdout is a real
terminal, with the current file (and page, for a large multi-page PDF)
shown as a trailing suffix. While PaddleOCR is importing, initializing, or
compiling its backend, a startup spinner keeps the terminal visibly active.
Piped or redirected output (CI logs, `> file.log`) falls back to plain status
lines instead of animating — no carriage-return noise in a log file. PaddleOCR
and PaddleX backend status chatter is suppressed so it does not drown out
pidocr's progress display. `-q/--quiet` suppresses both, leaving only warnings
and errors.

## Development

```bash
pip install -e ".[ocr,dev]"
pytest
ruff check .
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT
