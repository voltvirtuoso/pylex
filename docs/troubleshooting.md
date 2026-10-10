# Troubleshooting

Start with `pylex doctor`. It lists the hardware found, the packages installed, and the command to fix the setup.

## "paddleocr is not installed"

No OCR runtime is installed. The message includes the command for your machine; or run `pylex doctor --install`. Do not use `pip install pylex` (an unrelated project).

## `pip` cannot find `pylex-ocr`

If no release is on PyPI yet, install from GitHub: `pip install "pylex-ocr[cpu] @ git+https://github.com/voltvirtuoso/pylex.git"`.

## I asked for the GPU but it runs on the CPU

Pylex prints a warning saying why. Common causes:

- **Plain `onnxruntime` is installed.** It has no CUDA. Run `pip uninstall -y onnxruntime`, then `pip install "pylex-ocr[gpu]"`.
- **Two ONNX Runtime packages are installed.** `doctor` flags this. Uninstall all of `onnxruntime`, `onnxruntime-gpu`, `onnxruntime-openvino`, then install one profile.
- **CUDA libraries did not load.** Run `pylex doctor --verify-gpu`. If it says the provider is listed but did not load, your driver is probably too old for the build. Check `nvidia-smi`: CUDA 13 needs `[gpu]`, CUDA 12 needs `[gpu-cu12]`.
- **No NVIDIA GPU.** Intel, AMD and Apple hardware run on the CPU in Pylex, and TPUs are not supported. See [Installation](installation.md).

To confirm the GPU is busy while a job runs, watch `nvidia-smi`.

## `--gpu` is slower with `-j 4`

Workers share one GPU and compete for memory. Use `-j 1` with a GPU.

## The search finds nothing, or text is garbled

- **Non-Latin scripts** (Russian, Greek, Chinese, Arabic...) need `--font`. The default font only covers Latin-1, and Pylex logs a warning when it meets other characters.
- **No text detected:** try `--retry-upscale`, a higher `--dpi`, or lower `--det-box-thresh`.
- **A page was kept unchanged:** if OCR fails on a page, Pylex logs a warning, keeps the page without text, and exits with status 1 naming the pages (`FileResult.failed_pages` in Python). Run with `-v` to see the cause.

## Small labels are missed on a dense drawing

Use smaller tiles and a more tolerant detector: `--tile-size 1000 --tile-overlap 250 --det-box-thresh 0.45 --det-unclip-ratio 2.0`.

## The first run is very slow

Models are being downloaded (cached in `~/.paddlex/official_models`), and OpenVINO compiles models on first use. Later runs are faster. Choose a different download host with `PADDLE_PDX_MODEL_SOURCE=modelscope` (or `aistudio`, `bos`), or copy `~/.paddlex` from a connected machine for offline use.

## "No available model hosting platforms detected"

Pylex prints "could not download its OCR models". The machine cannot reach any model host. Allow access to one of them, set `PADDLE_PDX_MODEL_SOURCE`, or copy a populated `~/.paddlex`.

## Output already exists

Pass `--force` to overwrite or `--skip-existing` to skip.

## oneDNN or PIR errors with Paddle static

Use the default runtime, or `--inference-engine paddle_dynamic`. Pylex retries dynamic Paddle automatically unless `--no-runtime-fallback` is set.

## Reporting a problem

Include `pylex --version`, the output of `pylex doctor`, the exact command, the input type and resolution, and a redacted or synthetic example. See [CONTRIBUTING](https://github.com/voltvirtuoso/pylex/blob/main/CONTRIBUTING.md).
