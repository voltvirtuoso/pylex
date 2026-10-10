# Installation

Pylex needs an OCR runtime that matches your hardware, and the right pip package is different for a laptop CPU, an Intel server, and an NVIDIA GPU. This page shows the one command for each, and how to let Pylex pick for you.

The distribution on PyPI is **`pylex-ocr`**. The Python module and the command are both **`pylex`**. (The name `pylex` on PyPI belongs to an unrelated project, so never run `pip install pylex`.)

Requirements: Python 3.10, 3.11 or 3.12 on Linux, macOS or Windows.

## The fastest path: let Pylex choose

Install the light core package, then ask Pylex what your machine needs:

```bash
python -m pip install pylex-ocr
pylex doctor            # inspects CPU, NVIDIA GPU, TPU and installed packages
pylex doctor --install  # runs the recommended pip command (asks first)
```

`pylex doctor` prints the hardware it found, the runtime packages already installed, and the exact command to run. From Python the same thing is available as `pylex.doctor()`, so a notebook or deployment script can check itself.

If you run `pylex` before any OCR runtime is installed, the error message already contains the right command for your machine.

## Pick one profile

Choose exactly one of these. They are alternatives, not add-ons.

| Your hardware | Install command | What it installs |
|---|---|---|
| Any CPU (recommended default) | `pip install "pylex-ocr[cpu]"` | PaddleOCR, PaddlePaddle, ONNX Runtime |
| Intel CPU, to try OpenVINO | `pip install "pylex-ocr[intel]"` | PaddleOCR, PaddlePaddle, ONNX Runtime with the OpenVINO provider |
| NVIDIA GPU, `nvidia-smi` reports CUDA 13 or newer | `pip install "pylex-ocr[gpu]"` | PaddleOCR, PaddlePaddle, `onnxruntime-gpu` for CUDA 13, plus CUDA and cuDNN libraries from pip |
| NVIDIA GPU, `nvidia-smi` reports CUDA 12.x | `pip install "pylex-ocr[gpu-cu12]"` | The same, built for CUDA 12 |

These work the same for the command line tool and the Python SDK, because they are the same package.

> **Why one profile only?** `onnxruntime`, `onnxruntime-gpu` and `onnxruntime-openvino` all provide the same Python module name. Installing two of them corrupts each other's files. If you switch hardware profiles, remove the old one first (see [Switching profiles](#switching-profiles)).

### Which CUDA version do I have?

Run `nvidia-smi` and read **CUDA Version** in the top-right corner. That is the newest CUDA your **driver** supports, which is what matters here. You do not need the CUDA toolkit installed: the `gpu` extras pull the CUDA and cuDNN runtime libraries from pip.

- **13.x or newer** → `[gpu]`. ONNX Runtime's documentation says its CUDA 13.0 builds require CUDA 13.0 or newer, and that PyPI's `onnxruntime-gpu` has been built with CUDA 13.0 by default since version 1.27.
- **12.x** → `[gpu-cu12]`, which pins ONNX Runtime below 1.27. The same documentation says builds made with CUDA 12.8 require CUDA 12.8 or newer, so if the GPU check below fails on an older 12.x driver, update the driver.
- **11.x or older** → the ONNX Runtime GPU builds Pylex installs target CUDA 12 and 13. Update the driver, or use the CPU profile.

### Check that the GPU is really used

Having `onnxruntime-gpu` installed does not prove CUDA loads. ONNX Runtime silently falls back to the CPU if a CUDA or cuDNN library is missing, and Pylex hides backend output, so verify explicitly:

```bash
pylex doctor --verify-gpu
```

It builds a tiny model and asks ONNX Runtime which provider it actually registered. `OK - CUDAExecutionProvider is active` means you are on the GPU. Then run:

```bash
pylex scan.pdf --device gpu -v
```

or, in Python, `Config(device="gpu")`. See [Using a GPU](cli.md#using-a-gpu) and [GPU in the SDK](python-api.md#gpu-in-the-sdk).

## Installing from GitHub

Before a release is on PyPI, or to try the latest code, install straight from the repository. Quote the whole thing:

```bash
python -m pip install "pylex-ocr[cpu] @ git+https://github.com/voltvirtuoso/pylex.git"
```

Replace `[cpu]` with `[gpu]`, `[gpu-cu12]` or `[intel]` as needed. For development:

```bash
git clone https://github.com/voltvirtuoso/pylex.git
cd pylex
python -m pip install -e ".[cpu,dev]"
pytest
```

## Use a virtual environment

OCR runtimes are large and version-sensitive. Keep Pylex in its own environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install "pylex-ocr[cpu]"
```

To use only the command line tool, `pipx` isolates it for you:

```bash
pipx install "pylex-ocr[cpu]"
```

## Paddle GPU build (alternative to ONNX Runtime)

Pylex can also run on the GPU through PaddlePaddle's own CUDA build, instead of ONNX Runtime. PyPI only carries `paddlepaddle-gpu` up to version 2.6.2 (checked October 2026); the 3.x GPU builds come from Paddle's own package index, so a pip extra cannot select them and you install in two steps. Copy the exact command for your CUDA version and platform from [Paddle's installation guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html); it has the form:

```bash
python -m pip install paddlepaddle-gpu==<version> -i <package-index-url-from-the-guide>
python -m pip install pylex-ocr "paddleocr>=3.0,<4"
```

Do **not** use the `[ocr]` or `[cpu]` extras here: they would also install the CPU `paddlepaddle`, which conflicts with `paddlepaddle-gpu`. Leave `onnxruntime` out as well, or add `--inference-engine paddle_dynamic` so Pylex does not pick it.

This route needs a system CUDA setup that matches the wheel. For most people the `[gpu]` profile above is simpler.

## Other accelerators

PaddleOCR can target several accelerator types through its device setting, and Pylex passes the value through on the Paddle runtimes:

```bash
pylex scan.pdf --device npu:0 --inference-engine paddle_dynamic
```

Accepted device types are `xpu`, `npu`, `mlu`, `gcu`, `dcu`, `iluvatar_gpu` and `metax_gpu`, the device names PaddleOCR and PaddleX document. Each needs the matching hardware build of PaddlePaddle, installed from Paddle's instructions for that hardware. **This path is passed through but has not been tested by the Pylex project**, so treat it as experimental.

### What Pylex does not use

| Hardware | Status |
|---|---|
| **Google TPU** | **Not supported.** PaddleOCR's documented hardware list does not include TPUs, and Pylex has no TPU code path. On a TPU machine Pylex can only use the host CPU: install `[cpu]`. `pylex doctor` detects TPU environments from environment variables and says so. |
| **Apple Silicon** (Metal, Neural Engine) | Pylex's extras run on the CPU; no Metal or Core ML provider is wired in. PaddleOCR documents Apple Silicon support through Paddle's own builds, which Pylex has not tested. |
| **AMD GPUs** (ROCm) | Pylex's extras run on the CPU; no ROCm runtime is included. PaddleOCR documents AMD GPU support through Paddle's own builds, which Pylex has not tested. |
| **Windows DirectML** | Not wired in. Runs on the CPU. |

## First run: models are downloaded

The first OCR run downloads the detection and recognition models and caches them under `~/.paddlex/official_models`. After that, no network is needed.

- The default download source is Hugging Face. To change it, set `PADDLE_PDX_MODEL_SOURCE` to `huggingface`, `modelscope`, `aistudio` or `bos` before running.
- To skip the connectivity check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`.
- For an offline machine, run Pylex once on a connected machine and copy the `~/.paddlex` folder across.

The first run with OpenVINO is also slower while the runtime compiles the models. It caches the result (see `--runtime-cache-dir`).

## Switching profiles

To move from one profile to another, remove every ONNX Runtime package and then install the new profile:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-openvino
python -m pip install "pylex-ocr[gpu]"
```

`pylex doctor` detects when more than one is installed and prints this for you.

## Check the installation

```bash
pylex --version
pylex doctor
```

```python
import pylex
print(pylex.__version__)
print(pylex.doctor())
```

The status line reads **ready** when the packages and providers match the recommended profile. If something is off, the report says what to change. See also [Troubleshooting](troubleshooting.md).
