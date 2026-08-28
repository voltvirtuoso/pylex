# OCR Performance and Accuracy Guide

## Executive summary

The project was slow on simple images for two separate reasons. First, the default accuracy-oriented workflow is designed for large, dense P&ID drawings: it can run multiple overlapping tile inferences and it enables PaddleOCR's optional text-line orientation classifier. Second, PaddleOCR model import and backend initialization are expensive on the first run, particularly when multiple worker processes each load their own model stack. The current normal path uses an automatic CPU runtime: ONNX Runtime when installed, with dynamic Paddle as the fallback. Static oneDNN/MKLDNN remains an explicit option because a real Paddle 3.3.1 test hit an unsupported PIR/oneDNN operator during inference.

For a simple upright image, use the default automatic runtime with `-j 1`. Install `pip install pylex[ocr]` so ONNX Runtime is available; use `-v` to confirm the selected engine. The verified medium-model command is `pylex image.png --model-size medium --inference-engine auto -j 1`. For a low-resolution image that produces no usable text on the first pass, add `--retry-upscale`. PP-OCRv6 small now uses one full-page pass automatically when the image fits the detector side limit and the 8-megapixel safety budget; larger or dense P&ID pages retain overlapping tiles. Use `--strict-tiling` to force tiles for a small-model image. The CLI progress display advances through page rendering and tile inference rather than updating only when the entire file completes. This release uses PaddleOCR only; no Windows OCR fallback is included.

## What the research found

PaddleOCR's official OCR documentation describes text-line orientation as an optional module and separates detection from recognition. It also publishes materially different speed/size trade-offs between mobile and server model tiers. The current docs identify PP-OCRv6_medium as the current general-pipeline default and report higher internal-set detection and recognition scores for the medium tier than for the small tier. The published CPU figures are model-inference-only and exclude preprocessing and postprocessing, so they should be used to understand direction rather than as a guarantee for a complete CLI run [1].

PaddleOCR's high-performance inference documentation describes automatic or explicit use of Paddle Inference, OpenVINO, ONNX Runtime, and TensorRT, together with thread and precision controls. The same documentation currently emphasizes Linux x86-64 support for its high-performance dependency installation and recommends Docker or WSL for Windows, so the project does not force HPI on Windows [2].

Windows-native OCR was evaluated but deliberately excluded from this release. The project’s performance and accuracy path is now Paddle-only so that model selection, preprocessing, runtime acceleration, and output behavior remain consistent across supported environments.

## Changes in pylex 1.5.7

The 1.5.7 patch adds an explicit OpenVINO execution-provider mode through
PaddleX's supported ONNX Runtime runner. Automatic mode selects OpenVINO when
the installed ONNX Runtime build exposes `OpenVINOExecutionProvider`, then
standard ONNX Runtime, then dynamic Paddle. On the equivalent 1845×832,
102-line fixture, OpenVINO prediction took about 1.06 seconds, compared with
0.81–0.95 seconds for standard ONNX Runtime and 10.29 seconds for dynamic
Paddle. OpenVINO is optimized for Intel hardware and is not guaranteed to beat
standard ONNX Runtime on every CPU, so target-machine validation remains
important.

## Changes in pylex 1.5.6

The 1.5.6 patch makes automatic runtime selection use ONNX Runtime when its
portable CPU package is installed, with dynamic Paddle as the fallback. In a
sandbox benchmark on an equivalent 1845×832, 102-line image, dynamic Paddle
prediction took 10.292 seconds while ONNX Runtime took 0.814 seconds. Engine
initialization was 4.835 seconds versus 3.035 seconds on the repeat ONNX run.
These are directional measurements, not guarantees for every Windows CPU or
image; validate the selected runtime on representative P&ID samples.

## Changes in pylex 1.5.5

The 1.5.4 patch makes progress meaningful within a file. During standalone-image
OCR, rendering and each tile pass update the displayed percentage; during PDF
OCR, rendering and completed pages do the same. A one-file run can therefore
show intermediate progress before the final `1/1` completion state.

## Changes in pylex 1.5.3

The 1.5.3 patch adds adaptive full-page inference for PP-OCRv6 small images
that fit within the configured detector resolution and 8-megapixel budget. It
avoids redundant overlapping tile passes without disabling text-line orientation
or changing detection/recognition thresholds. Use `--strict-tiling` when dense
P&ID recall is more important than the adaptive latency optimization.

## Changes in pylex 1.5.2

The 1.5.2 patch makes the previously opaque single-image phase visible. The
CLI now reports the model tier and inference engine during initialization,
prints the measured engine-ready time, reports `Running OCR`, and shows
`OCR tile N/M` before each blocking tile pass. These messages do not speed the
model itself, but they distinguish model download/initialization from inference
and prevent a long 0% display from looking like a deadlock.

## Changes in pylex 1.5.1

The 1.5.1 patch corrects a PaddleOCR 3.7 configuration mismatch: `cpu_threads`
is not accepted by the dynamic runner, so it is now sent only to static and
ONNX Runtime configurations. If an explicitly selected optimized runtime fails,
pylex rebuilds the same Paddle model with dynamic Paddle inference instead of
retrying static oneDNN. The user-reported PP-OCRv6 medium CLI path was then
smoke-tested end to end and produced a searchable PDF successfully.

## Changes in pylex 1.5.0

| Area | Change | Expected effect |
|---|---|---|
| Default model quality | PP-OCRv6 small replaces PP-OCRv4 | Current practical detection and recognition baseline; medium remains available for higher accuracy |
| CPU inference | Automatic ONNX Runtime when installed; dynamic Paddle fallback | Same OCR model family; avoids the tested Paddle 3.3.1 PIR/oneDNN failure |
| Simple-image latency | Retained `--fast` as an explicit trade-off | One full-image inference, no text-line orientation model, detector cap of 3200 px |
| Tiny/weak text | Added `--retry-upscale` | One 1.5x Lanczos retry only after a no-result first pass; coordinates are mapped back to the original image |
| CPU contention | Added automatic per-worker thread budgeting and `--cpu-threads` | Reduces OpenMP/MKL/OpenBLAS oversubscription when `-j N` is used |
| Memory copies | Explicit contiguous BGR conversion | Avoids an unpredictable internal copy from a negative-stride channel-reversed view |
| Dense drawings | Retained tiled accuracy-first default | Protects small labels in crowded drawing regions |

## Recommended commands

For a simple image or upright scan, run the normal optimized path:

```powershell
pylex "C:\path\simple.png" --inference-engine paddle_static -j 1
```

For a low-resolution image where the first pass misses all text:

```powershell
pylex "C:\path\simple.png" --inference-engine paddle_static --retry-upscale -j 1
```

For a large, crowded P&ID, keep the accuracy-first mode:

```powershell
pylex "C:\path\drawing.pdf" --tile-size 1000 --tile-overlap 250 `
  --det-box-thresh 0.45 --det-unclip-ratio 2.0 -j 1
```

Multiple workers are useful for independent files in a batch; they do not make one image faster and each worker owns a separate Paddle engine.

## Benchmark interpretation

The included benchmarks measure both pipeline pass counts and a real Paddle smoke run. On a 4200×3200 drawing-sized array, the accuracy profile scheduled 9 OCR passes while the fast profile scheduled 1. On a 640×280 synthetic two-line image using the normal dynamic path with orientation enabled, PP-OCRv6 small took 0.553 seconds for inference after 11.246 seconds of engine initialization in a clean process; PP-OCRv6 medium took 4.178 seconds after 1.028 seconds of initialization. Both recognized the two test lines. These are directional measurements rather than guarantees: actual wall-clock performance depends on Paddle/PaddleX versions, CPU/GPU hardware, model-cache state, image dimensions, and text density.

For production use, benchmark the selected Paddle model and runtime against representative P&ID samples and check the generated searchable PDFs, especially for rotated labels and dense symbol clusters.

## References

[1]: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html "PaddleOCR General OCR Pipeline Usage Tutorial"
[2]: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/inference_deployment/local_inference/high_performance_inference.html "PaddleOCR High-Performance Inference"
