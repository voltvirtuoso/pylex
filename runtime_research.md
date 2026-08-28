# Runtime support findings

## ONNX Runtime

The official ONNX Runtime documentation lists the Python CPU package `onnxruntime` and identifies Windows x64 as a supported platform. It also notes that Windows builds require the Microsoft Visual C++ 2019 runtime. Native CPU installation is `pip install onnxruntime`.

Sources:
- https://onnxruntime.ai/docs/install/
- https://onnxruntime.ai/docs/get-started/with-python.html

## OpenVINO

OpenVINO provides native Windows installation and CPU inference support. However, PaddleOCR/PaddleX high-performance inference dependency documentation currently recommends Linux x86-64 for the HPI plugin and recommends Docker or WSL on Windows. That is distinct from direct ONNX Runtime on Windows.

OpenVINO can be preferable on Intel CPUs, but it should be benchmarked on the target machine. It is not universally faster than ONNX Runtime, and automatic model conversion or first-time engine construction may add startup time.

Sources:
- https://docs.openvino.ai/install
- https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/inference_deployment/local_inference/high_performance_inference.html
- https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/high_performance_inference.html

## Local benchmark

On the sandbox’s equivalent 1845x832, 102-line image using PP-OCRv6 small with orientation enabled:
- Dynamic Paddle OCR prediction: 10.292 seconds.
- Standard ONNX Runtime OCR prediction: 0.814 seconds on the first measured run and 0.949 seconds on a repeat run.
- OpenVINO Execution Provider through the ONNX Runtime runner: 1.060 seconds on the measured run.
- Overlay construction: approximately 0.010 seconds.
- PDF build/write: approximately 0.055–0.063 seconds.

PaddleX does not accept `engine='openvino'` directly; the supported configuration is `engine='onnxruntime'` with `OpenVINOExecutionProvider` and CPU provider fallback. Automatic mode implements that mapping and falls back safely if provider construction fails.

These measurements are directional and are not a Windows-user guarantee. OpenVINO is a valid native Windows option, especially for Intel CPUs, but it was not faster than standard ONNX Runtime in this sandbox run. The target Windows CPU should decide between the two.
