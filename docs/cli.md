# CLI reference

```text
pylex [options] INPUT [INPUT ...]
pylex doctor [--json] [--verify-gpu] [--install [-y]]
```

Run `pylex --help` for the authoritative list for your installed version. Inputs are PDF, PNG, JPEG, TIFF, BMP or WebP files, or folders. Output is always a searchable PDF.

## Input and output

- A single file with no `-o` gives `<name>.ocr.pdf` beside the input. Change the suffix with `--suffix`.
- `-o file.pdf` names the output for a single input.
- `-o folder/` writes into a folder. With several inputs, or a folder input, `-o` is always treated as a folder, and a folder input has its structure mirrored.
- Existing outputs are an error unless you pass `--force` (overwrite) or `--skip-existing` (skip).
- Files are written to a temporary name first and then renamed, so a crash never leaves a half-written PDF.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Every file succeeded |
| 1 | No matching input, nothing to do, a file failed, or OCR failed on some pages (the file was written but those pages have no text layer) |
| 2 | Invalid arguments |

## Selecting files

| Option | Meaning |
|---|---|
| `inputs` | Files or folders (one or more). |
| `-r`, `--recursive` | Recurse into subfolders of folder inputs. |
| `--formats ext,ext` | Only pick up these extensions in folders. Default: all supported. |
| `-xf`, `--exclude PATTERN` | Skip files matching a wildcard, matched against the file name and relative path (`*` also matches `/`). Repeat it or comma-separate: `-xf '*_draft*,backup/*'`. |
| `--pages RANGE` | OCR only these 1-based PDF pages, e.g. `1-5,9`. Other pages are copied through unchanged. |
| `--dry-run` | List inputs and output paths, then exit. |
| `--skip-existing` | Skip inputs whose output already exists. |
| `-o`, `--output PATH` | Output file or folder. |
| `--in-place` | Overwrite each input with its OCR version. Requires `--force`. |
| `--suffix SUFFIX` | Suffix before `.pdf` when `-o` is absent. Default `.ocr`. |
| `-f`, `--force` | Overwrite existing outputs. |

## Speed and parallelism

| Option | Default | Meaning |
|---|---|---|
| `-j`, `--jobs N` | 1 | Process N files in parallel worker processes, each with its own engine. Pages inside one PDF are always sequential. |
| `--cpu-threads N` | CPU count ÷ jobs | Native threads per process. |
| `--fast` | off | One full-page pass, no text-line orientation model, detector capped at 3200 px. For simple images only. |
| `--recognition-batch-size N` | 6 | Recognition batch size. |
| `--orientation-batch-size N` | 8 | Text-line orientation batch size. |

## Using a GPU

```bash
pylex scan.pdf --device gpu        # first GPU
pylex scan.pdf --device gpu:1      # second GPU
pylex scan.pdf --device auto       # GPU if a CUDA runtime is installed, else CPU
pylex doctor --verify-gpu          # confirm CUDA actually loads
```

`--gpu` is a shortcut for `--device gpu`. With `auto`, Pylex picks ONNX Runtime when it has the CUDA provider, otherwise Paddle's GPU build. If you ask for a GPU that isn't usable, Pylex prints why and the fix, then continues on the CPU. Add `--no-runtime-fallback` to stop instead. Use `-j 1` with a GPU; Pylex warns if you don't. Install steps are in [Installation](installation.md).

## OCR options

| Option | Default | Meaning |
|---|---|---|
| `--device DEV` | cpu | `auto`, `cpu`, `gpu`, `gpu:N`, or a Paddle accelerator type (`xpu`, `npu`, `mlu`, `gcu`, `dcu`, ...). |
| `--gpu` | off | Shortcut for `--device gpu`. |
| `--dpi N` | 300 | Resolution used to render PDF pages for OCR. |
| `--image-dpi N` | 300 | Assumed DPI of raster inputs; sets the output page size. |
| `--lang CODE` | en | PaddleOCR language code. |
| `--ocr-version` | PP-OCRv6 | `PP-OCRv4`, `PP-OCRv5` or `PP-OCRv6`. |
| `--model-size` | **medium** | `tiny`, `small` or `medium` (PP-OCRv6 tiers). The Python `Config()` default is `small`. |
| `--inference-engine` | auto | `auto`, `paddle_static`, `paddle_dynamic`, `onnxruntime`, `openvino`. Auto prefers OpenVINO, then ONNX Runtime, then dynamic Paddle on CPU. |
| `--min-confidence F` | 0.5 | Drop lines below this confidence. |
| `--det-limit-side-len N` | 8000 | Largest side the detector sees. |
| `--no-textline-orientation` | off | Disable the upside-down line classifier. |
| `--no-vertical-text-retry` | off | Disable the 90-degree retry for tall low-confidence boxes. |
| `--retry-upscale` | off | If a pass finds nothing, retry once at 1.5x. |
| `--runtime-cache-dir PATH` | platform cache | Where OpenVINO/ONNX store compiled engines. |
| `--enable-mkldnn` | off | oneDNN for static Paddle CPU inference. |
| `--enable-hpi`, `--hpi-backend` | off | PaddleOCR high-performance inference. Linux-oriented; not combinable with a non-auto `--inference-engine`. |
| `--no-runtime-fallback` | off | Fail instead of falling back to another runtime or the CPU. |

## Dense-document tuning

| Option | Default | Meaning |
|---|---|---|
| `--tile-size PX` | 2000 | OCR in overlapping PX-sized windows at full resolution. Try 900-1200 for crowded drawings. `0` disables tiling. |
| `--tile-overlap PX` | 220 | Overlap between tiles. Must be smaller than the tile size. |
| `--no-tile` | off | Same as `--tile-size 0`. |
| `--strict-tiling` | off | Always tile; turns off the adaptive full-page shortcut (which applies only to `--model-size small`). |
| `--dedupe-tol-px F` | 20 | Same text with centers closer than this is treated as an overlap duplicate. |
| `--det-thresh F` | unset | Detector pixel-score cutoff. |
| `--det-box-thresh F` | unset | Lower (0.4-0.5) recovers weak detections in crowded areas. |
| `--det-unclip-ratio F` | unset | Higher (1.8-2.2) expands boxes for text touching line art. |

## Invisible text layer

| Option | Default | Meaning |
|---|---|---|
| `--font` | Helvetica | Helvetica covers Latin-1 only. For other scripts pass a Unicode `.ttf` path (DejaVuSans.ttf for Cyrillic/Greek) or a built-in CJK font: `STSong-Light`, `MSung-Light`, `HeiseiMin-W3`, `HeiseiKakuGo-W5`, `HYSMyeongJo-Medium`, `HYGothic-Medium`. |
| `--baseline-fraction F` | 0.18 | Baseline offset as a fraction of box height. |

Example for Russian: `pylex scan.pdf --lang ru --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`. Right-to-left scripts (Arabic, Hebrew) have not been verified.

## Output and logging

| Option | Meaning |
|---|---|
| `-v`, `-vv` | Info or debug logging. `-v` also shows the selected runtime. |
| `-q`, `--quiet` | Warnings and errors only. |
| `--version` | Print the version. |

## pylex doctor

```bash
pylex doctor                  # report and recommended install command
pylex doctor --json           # machine-readable
pylex doctor --verify-gpu     # confirm CUDA loads; exit code 1 if not
pylex doctor --install -y     # run the recommended pip commands
```

It reports CPU, NVIDIA GPU and driver CUDA version, TPU environments (not usable), installed runtime packages, and ONNX Runtime providers. A file literally named `doctor` can be passed as `./doctor`.

## Environment variables

| Variable | Effect |
|---|---|
| `PADDLE_PDX_MODEL_SOURCE` | Model download source: `huggingface` (default), `modelscope`, `aistudio`, `bos`. |
| `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` | Set `True` to skip the connectivity check. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | Set from `--cpu-threads` unless already set. |

## Recipes

```bash
pylex scans/ -r -o out/ -j 4 --skip-existing          # resumable batch
pylex big.pdf --pages 1-10 --dpi 200                  # quick look at part of a file
pylex simple.png --fast --retry-upscale               # fast pass on a clean image
pylex scans/ -r -xf '*_OCR*' -xf 'backup/*'           # skip earlier outputs
pylex file.pdf --in-place --force                     # replace the original
```
