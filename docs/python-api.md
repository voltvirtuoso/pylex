# Python SDK

Pylex is a library as well as a command. Everything the CLI does is available from Python, with the same install ([Installation](installation.md)).

```python
from pylex import Config, create_engine, extract_text, create_searchable_document
```

## The three objects you need

- **`Config`**: every setting, as a dataclass with sensible defaults.
- **engine**: the loaded OCR backend from `create_engine(config)`. Loading is slow (seconds); create it once and reuse it.
- **results**: `TextRegion` objects from `extract_text`, or a `FileResult` from `create_searchable_document`.

## extract_text

```python
extract_text(source, *, config=None, engine=None, on_tile=None, on_tile_done=None) -> list[TextRegion]
```

`source` can be a file path, a PIL image, or a NumPy array (2-D grayscale or 3-D RGB; an alpha channel is dropped). NumPy arrays are assumed to be RGB. Large images are processed in overlapping tiles according to the config, and results are merged back into whole-image coordinates.

```python
config = Config(model_size="small")
engine = create_engine(config)
regions = extract_text("page.png", config=config, engine=engine)
for r in regions:
    print(r.text, r.confidence, r.polygon.tolist())
```

### TextRegion

| Attribute | Type | Meaning |
|---|---|---|
| `text` | `str` | Recognized text for one line. |
| `confidence` | `float` | OCR confidence, 0 to 1. A signal, not proof of correctness. |
| `polygon` | `ndarray` (4, 2) | Four corner points `(x, y)` in original image pixels. |

Regions are lines of text, not a document model: reading order, tables and headings are not reconstructed.

## create_searchable_document

```python
create_searchable_document(input_path, output_path, *, config=None, engine=None,
                           on_page=None, on_stage=None, on_tile=None, on_tile_done=None) -> FileResult
```

Takes a PDF or image and writes a searchable PDF.

| Attribute | Meaning |
|---|---|
| `ok` | `True` when the output was written. |
| `pages_processed` | Pages that were OCR'd (pages excluded by `page_ranges` are copied, not counted). |
| `lines_added` | Text lines placed in the invisible layer. |
| `error` | Error text when `ok` is false. |
| `elapsed` | Seconds. |
| `input_path`, `output_path` | The paths used. |
| `failed_pages` | 1-based pages whose OCR raised an error. Those pages are kept unchanged, without a text layer. |

A file can report `ok=True` and still have entries in `failed_pages`, so check it when completeness matters. The CLI exits with status 1 in that case.

## Callbacks

| Callback | Called with | When |
|---|---|---|
| `on_page(page, n_pages)` | ints | After each PDF page, including skipped ones. |
| `on_stage(text)` | str | Stage changes such as "Rendering page 2/10" or "Running OCR". |
| `on_tile(n, total)` | ints | Before a tile starts. |
| `on_tile_done(n, total)` | ints | After a tile finishes. |

```python
from pylex import create_searchable_document
create_searchable_document("in.pdf", "out.pdf", config=config, engine=engine,
                           on_page=lambda p, n: print(f"page {p}/{n}"))
```

`on_tile` and `on_tile_done` apply to images; PDFs report through `on_page` and `on_stage`.

## Lower-level functions

`process_pdf(engine, input_path, output_path, cfg, on_page=None, on_stage=None)` and `process_image(engine, input_path, output_path, cfg, on_stage=None, on_tile=None, on_tile_done=None)` do the work directly. They take `pathlib.Path` arguments and a `Config`, and return a `FileResult`.

## GPU in the SDK

```python
config = Config(device="gpu")          # "gpu:1" for a specific card, "auto" to use one if present
engine = create_engine(config)
regions = extract_text("page.png", config=config, engine=engine)
```

If the GPU is not usable, a warning explains why and the engine runs on the CPU (set `allow_runtime_fallback=False` to raise instead). Check beforehand:

```python
import pylex
print(pylex.doctor())                          # readable report and install command
hw, rec = pylex.doctor(as_text=False)          # HardwareReport, Recommendation
ok, detail = pylex.verify_gpu()                # does CUDA really load?
```

`rec.commands` is the list of pip commands for this machine and `rec.ready` says whether it is already set up. Google TPUs are not supported; see [Installation](installation.md).

## Recipes

### Draw boxes on an image

```python
from PIL import Image, ImageDraw
img = Image.open("page.png").convert("RGB")
draw = ImageDraw.Draw(img)
for r in extract_text(img, config=config, engine=engine):
    draw.polygon([tuple(p) for p in r.polygon.tolist()], outline="red")
img.save("boxes.png")
```

### Export JSON

```python
import json
data = [{"text": r.text, "confidence": r.confidence, "polygon": r.polygon.tolist()} for r in regions]
json.dump(data, open("page.json", "w"), ensure_ascii=False, indent=2)
```

### Text from a PDF page without making a PDF

`extract_text` reads images, so render the page first:

```python
import pypdfium2 as pdfium
page = pdfium.PdfDocument("doc.pdf")[0]
image = page.render(scale=300 / 72).to_pil().convert("RGB")
regions = extract_text(image, config=config, engine=engine)
```

### Many files

Reuse one engine in a loop. For parallelism use separate processes, each with its own engine (the engine is not safe for concurrent calls from threads); the CLI's `-j` does exactly this.

### Non-Latin searchable PDFs

```python
config = Config(lang="ru", font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
```

For Chinese, Japanese or Korean use a built-in font such as `font="STSong-Light"`.

## Errors

- A missing OCR backend, an unusable device with `allow_runtime_fallback=False`, or unreachable model hosts make `create_engine` raise `pylex.PylexSetupError` (a `RuntimeError`). The message says how to fix it, including the install command for your machine.
- An unknown `device` or `font` raises `ValueError`.

## Config reference

Pass any of these as keyword arguments to `Config(...)`.

| Field | Default | Meaning |
|---|---|---|
| `dpi` | `300` | Render resolution for PDF pages. |
| `image_dpi` | `300` | Assumed DPI of raster images; sets output page size. |
| `min_confidence` | `0.5` | Drop lines below this confidence. |
| `lang` | `'en'` | PaddleOCR language code. |
| `ocr_version` | `'PP-OCRv6'` | `PP-OCRv4`, `PP-OCRv5` or `PP-OCRv6`. |
| `model_size` | `'small'` | `tiny`, `small` or `medium` (PP-OCRv6). Note the CLI default is `medium`. |
| `inference_engine` | `'auto'` | `auto`, `paddle_static`, `paddle_dynamic`, `onnxruntime`, `openvino`. |
| `enable_hpi` | `False` | PaddleOCR high-performance inference. |
| `hpi_backend` | `None` | HPI backend: `auto`, `paddle`, `openvino`, `onnxruntime`. |
| `runtime_cache_dir` | `None` | Cache for compiled OpenVINO/ONNX engines; `None` = platform default. |
| `allow_runtime_fallback` | `True` | Fall back to another runtime, or to the CPU when a GPU is unusable, instead of failing. |
| `font` | `'Helvetica'` | Invisible-text font: standard PDF font, built-in CJK font name, or path to a Unicode `.ttf`. |
| `min_font_size` | `1.0` | Smallest font size in the text layer (pt). |
| `max_font_size` | `200.0` | Largest font size in the text layer (pt). |
| `det_limit_side_len` | `8000` | Largest side the detector sees. |
| `baseline_fraction` | `0.18` | Baseline offset as a fraction of box height. |
| `min_box_width_pt` | `0.25` | Skip overlay boxes narrower than this. |
| `min_box_height_pt` | `0.25` | Skip overlay boxes shorter than this. |
| `use_textline_orientation` | `True` | Run the upside-down line classifier. |
| `vertical_text_retry` | `True` | Retry tall low-confidence boxes rotated 90 degrees. |
| `vertical_text_aspect_ratio` | `1.3` | Height/width ratio that marks a box as tall. |
| `vertical_text_retry_max_confidence` | `0.8` | Only retry boxes below this confidence. |
| `device` | `None` | `None` (use `use_gpu`), `auto`, `cpu`, `gpu`, `gpu:N`, `cuda`, or a Paddle accelerator type. |
| `use_gpu` | `False` | Legacy; same as `device="gpu"`. |
| `disable_mkldnn` | `True` | Keep oneDNN off for static Paddle inference. |
| `cpu_threads` | `None` | Native threads per process; `None` = backend default. |
| `orientation_batch_size` | `8` | Orientation classifier batch size. |
| `recognition_batch_size` | `6` | Recognition batch size. |
| `fast_mode` | `False` | Fast profile: one pass, no orientation model. |
| `retry_upscale` | `False` | Retry once at 1.5x when a pass finds nothing. |
| `page_ranges` | `None` | List of inclusive 1-based `(start, end)` tuples; `None` = all pages. |
| `tile_size` | `2000` | Tile edge in pixels; `<= 0` disables tiling. |
| `tile_overlap` | `220` | Tile overlap in pixels. |
| `tile_batch_size` | `4` | Tiles per backend call. |
| `adaptive_tiling` | `True` | Skip tiling for `small` models when the image fits the detector and pixel budget. |
| `full_page_max_pixels` | `8000000` | Pixel budget for the adaptive shortcut. |
| `dedupe_tol_px` | `20.0` | Duplicate-merge distance for tile overlaps. |
| `det_thresh` | `None` | Detector pixel-score cutoff. |
| `det_box_thresh` | `None` | Detector box-score cutoff. |
| `det_unclip_ratio` | `None` | Detector box expansion ratio. |

`Config` is a plain dataclass, so `dataclasses.replace(config, dpi=200)` makes a modified copy.
