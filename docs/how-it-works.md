# How it works

Pylex turns a page into a searchable page without changing how it looks. Each file goes through the same stages.

## 1. Render

A PDF page is rendered with pypdfium2 at `--dpi` (default 300), **exactly as a viewer displays it**, with the page's `/Rotate` value applied. A raster image is loaded as is.

## 2. Detect and recognize

The image goes to PaddleOCR models (PP-OCRv6 by default): a detector finds text lines, an optional classifier flips upside-down lines, and a recognizer reads each line. Pylex runs these through one of the runtimes below.

| Runtime | When it is used |
|---|---|
| ONNX Runtime (CPU or CUDA) | Auto choice when installed. Fastest in the project's measurements on CPU. |
| ONNX Runtime with OpenVINO | Auto choice when the OpenVINO build is installed (`[intel]`). |
| Paddle dynamic | Fallback when ONNX Runtime is absent, or when only `paddlepaddle-gpu` provides the GPU. |
| Paddle static | Only if requested explicitly. Can hit oneDNN operator errors on some builds. |

With a GPU requested, auto selects ONNX Runtime if it has the CUDA provider, otherwise Paddle's GPU build; if neither exists it explains the fix and continues on the CPU.

## 3. Tiling for dense pages

A whole-page pass shrinks the page to the detector's size limit, which can make small labels squeezed between symbols too small to detect. Pylex instead cuts large pages into overlapping tiles (`--tile-size`, default 2000 px; `--tile-overlap`, default 220 px), runs OCR on each at full resolution, shifts the results back into page coordinates, and merges duplicates: the same text whose centers lie within `--dedupe-tol-px` is kept once, with the highest confidence.

Images no larger than one tile are not tiled. For `--model-size small`, an adaptive shortcut also skips tiling when the image fits the detector limit and 8 megapixels; `--strict-tiling` turns that off.

## 4. Vertical text retry

The orientation classifier only distinguishes 0 and 180 degrees, so text running vertically is often misread. When a tile contains tall, low-confidence boxes, Pylex re-runs that tile rotated 90 degrees and merges the better results. Disable with `--no-vertical-text-retry`.

## 5. Build the invisible layer

For each recognized line, Pylex draws the text in PDF text render mode 3 (invisible). The font size comes from the box height, the horizontal scale stretches the text to the box width, and the angle follows the polygon. Each line is anchored at the **start** of its polygon, so long text cannot drift off the page.

## 6. Merge without disturbing the page

For rotated PDF pages, Pylex first transfers `/Rotate` into the page content, so the overlay and the OCR image share one orientation, then merges the overlay onto the original page. The visible content is never re-rendered or re-compressed. Output is written to a temporary file and renamed.

## Parallelism

`-j N` runs N worker processes, each loading its own engine. Pages inside one PDF are sequential. CPU threads are divided between workers automatically unless you set `--cpu-threads`.

## Fonts

The text layer uses Helvetica by default, which covers Latin-1. Other scripts need `--font` with a Unicode TrueType file or a built-in CJK font. See [CLI reference](cli.md).
