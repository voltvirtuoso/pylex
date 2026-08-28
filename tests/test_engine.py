import os
from types import SimpleNamespace

import numpy as np


def test_paddle_backend_output_is_suppressed(monkeypatch, capfd):
    import sys

    import pylex.engine as engine_mod
    from pylex.config import Config

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            print("backend loading status")
            sys.stderr.write("backend warning status\n")

        def predict(self, image, **kwargs):
            print("backend predict status")
            sys.stderr.write("backend predict warning\n")
            return [
                {
                    "rec_texts": ["TAG-101"],
                    "rec_polys": [
                        [[0, 0], [60, 0], [60, 20], [0, 20]],
                    ],
                    "rec_scores": [0.95],
                }
            ]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))
    monkeypatch.setattr(engine_mod, "_ENGINE", None)

    cfg = Config()
    engine = engine_mod.get_ocr_engine(cfg)
    items = engine_mod.ocr_image(engine, np.zeros((30, 80, 3), dtype=np.uint8), cfg)

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert [text for text, _, _ in items] == ["TAG-101"]


def test_backend_file_descriptor_output_is_suppressed(capfd):
    import pylex.engine as engine_mod

    with engine_mod._quiet_backend_output():
        os.write(1, b"backend child status\n")
        os.write(2, b"backend child warning\n")

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_retry_upscale_runs_only_after_no_result_and_maps_coordinates(monkeypatch):
    import pylex.engine as engine_mod
    from pylex.config import Config

    calls = []

    class RetryEngine:
        def predict(self, image, **kwargs):
            calls.append(image.shape[:2])
            if len(calls) == 1:
                return []
            return [
                {
                    "rec_texts": ["TINY"],
                    "rec_polys": [[[15, 15], [45, 15], [45, 30], [15, 30]]],
                    "rec_scores": [0.9],
                }
            ]

    cfg = Config(retry_upscale=True)
    items = engine_mod.ocr_image(RetryEngine(), np.zeros((20, 30, 3), dtype=np.uint8), cfg)

    assert calls == [(20, 30), (30, 45)]
    assert items[0][0] == "TINY"
    np.testing.assert_allclose(items[0][1], np.array([[10, 10], [30, 10], [30, 20], [10, 20]]))


def test_configure_cpu_threads_sets_common_pool_limits(monkeypatch):
    import pylex.engine as engine_mod

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    engine_mod.configure_cpu_threads(3)

    assert all(
        os.environ[name] == "3"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )


def test_paddle_runtime_defaults_use_current_model_and_automatic_runtime():
    import pylex.engine as engine_mod
    from pylex.config import Config

    kwargs = engine_mod._paddle_runtime_kwargs(Config())

    assert kwargs["text_detection_model_name"] == "PP-OCRv6_small_det"
    assert kwargs["text_recognition_model_name"] == "PP-OCRv6_small_rec"
    assert kwargs["lang"] is None
    assert kwargs["ocr_version"] is None
    assert kwargs["engine"] in {"paddle_dynamic", "onnxruntime"}
    assert kwargs["engine_config"]["device_type"] == "cpu"
    assert "run_mode" not in kwargs["engine_config"]


def test_hpi_backend_is_configured_without_explicit_engine():
    import pylex.engine as engine_mod
    from pylex.config import Config

    kwargs = engine_mod._paddle_runtime_kwargs(Config(enable_hpi=True, hpi_backend="openvino"))

    assert kwargs["enable_hpi"] is True
    assert kwargs["hpi_config"] == {"backend": "openvino", "auto_config": False}
    assert "engine" not in kwargs
    assert "engine_config" not in kwargs


def test_static_runtime_failure_retries_with_dynamic(monkeypatch):
    import pylex.engine as engine_mod
    from pylex.config import Config

    class StaticEngine:
        def predict(self, image, **kwargs):
            raise NotImplementedError("ConvertPirAttribute2RuntimeAttribute oneDNN")

    class DynamicEngine:
        def predict(self, image, **kwargs):
            return [
                {
                    "rec_texts": ["RECOVERED"],
                    "rec_polys": [[[0, 0], [40, 0], [40, 12], [0, 12]]],
                    "rec_scores": [0.9],
                }
            ]

    fallback_configs = []
    monkeypatch.setattr(
        engine_mod,
        "get_ocr_engine",
        lambda cfg: fallback_configs.append(cfg) or DynamicEngine(),
    )
    cfg = Config(inference_engine="paddle_static", disable_mkldnn=False)

    result = engine_mod._predict(StaticEngine(), np.zeros((20, 40, 3), dtype=np.uint8), cfg)

    assert result[0]["rec_texts"] == ["RECOVERED"]
    assert fallback_configs[0].inference_engine == "paddle_dynamic"
    assert fallback_configs[0].disable_mkldnn is True


def test_cpu_threads_are_mapped_per_inference_engine():
    import pylex.engine as engine_mod
    from pylex.config import Config

    dynamic = engine_mod._paddle_runtime_kwargs(Config(cpu_threads=12))
    assert "cpu_threads" not in dynamic["engine_config"]

    static = engine_mod._paddle_runtime_kwargs(Config(inference_engine="paddle_static", cpu_threads=12))
    assert static["engine_config"]["cpu_threads"] == 12

    onnx = engine_mod._paddle_runtime_kwargs(Config(inference_engine="onnxruntime", cpu_threads=12))
    assert onnx["engine_config"]["intra_op_num_threads"] == 12
    assert onnx["engine_config"]["inter_op_num_threads"] == 1


def test_auto_runtime_prefers_onnxruntime_when_openvino_is_unavailable(monkeypatch):
    import pylex.engine as engine_mod
    from pylex.config import Config

    monkeypatch.setattr(engine_mod.importlib.util, "find_spec", lambda name: object() if name == "onnxruntime" else None)
    monkeypatch.setattr(engine_mod, "_onnxruntime_has_openvino", lambda: False)

    assert engine_mod._effective_inference_engine(Config()) == "onnxruntime"
    assert engine_mod._paddle_runtime_kwargs(Config())["engine"] == "onnxruntime"


def test_auto_runtime_falls_back_to_dynamic_paddle_when_onnxruntime_is_unavailable(monkeypatch):
    import pylex.engine as engine_mod
    from pylex.config import Config

    monkeypatch.setattr(engine_mod.importlib.util, "find_spec", lambda _name: None)

    assert engine_mod._effective_inference_engine(Config()) == "paddle_dynamic"
    assert engine_mod._paddle_runtime_kwargs(Config())["engine"] == "paddle_dynamic"


def test_openvino_runtime_maps_to_onnxruntime_with_openvino_provider():
    import pylex.engine as engine_mod
    from pylex.config import Config

    kwargs = engine_mod._paddle_runtime_kwargs(Config(inference_engine="openvino", cpu_threads=4))

    assert kwargs["engine"] == "onnxruntime"
    assert kwargs["engine_config"]["providers"] == [
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert kwargs["engine_config"]["provider_options"][0]["device_type"] == "CPU"
    assert kwargs["engine_config"]["provider_options"][0]["num_of_threads"] == "4"


def test_openvino_runtime_uses_configured_cache_directory():
    import pylex.engine as engine_mod
    from pylex.config import Config

    kwargs = engine_mod._paddle_runtime_kwargs(
        Config(inference_engine="openvino", runtime_cache_dir="C:/pylex-cache")
    )

    provider_options = kwargs["engine_config"]["provider_options"][0]
    assert provider_options["cache_dir"] == "C:/pylex-cache"
