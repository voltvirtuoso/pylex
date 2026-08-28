"""OCR engine: lazy PaddleOCR loading and raw prediction -> (text, quad, score)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import sys
import time

import numpy as np
from PIL import Image

from pylex.config import Config

log = logging.getLogger("pylex")

_ENGINE = None
_ENGINE_KEY: tuple[object, ...] | None = None
_UNSUPPORTED_DET_KWARGS_WARNED = False

# PaddleOCR/PaddleX use a mixture of Python loggers and direct stdout/stderr
# writes. Keep their normal status chatter out of the application's progress
# display while preserving pylex's own warnings and errors.
class _BackendNullStream(io.TextIOBase):
    """Non-closing text sink for third-party handlers created during suppression."""

    encoding = "utf-8"

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


_PADDLE_LOGGERS = (
    "paddle",
    "paddleocr",
    "paddlex",
    "paddlex.utils",
    "paddlex.inference",
)


def configure_cpu_threads(threads: int | None) -> None:
    """Bound common native thread pools before Paddle is imported."""
    if threads is None or threads < 1:
        return
    value = str(int(threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, value)


def _quiet_backend_logging() -> None:
    """Mute known backend loggers at the point where their objects are used."""
    for name in _PADDLE_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


def _set_windows_std_handle(which: int, handle: int) -> None:
    """Point a native Windows standard handle at ``handle``."""
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    kernel32.SetStdHandle.restype = ctypes.c_bool
    kernel32.SetStdHandle(ctypes.c_uint32(which), ctypes.c_void_p(handle))


def _get_windows_std_handle(which: int) -> int:
    """Return a native Windows standard handle as an integer."""
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    return int(kernel32.GetStdHandle(ctypes.c_uint32(which)))


@contextlib.contextmanager
def _quiet_backend_output():
    """Suppress backend output, including writes from child processes.

    Paddle/PaddleX sometimes invokes utilities such as ``where.exe`` during
    model setup. Their output can bypass Python's ``redirect_stdout`` and go
    directly to inherited OS handles, so both Python streams, CRT descriptors,
    and Windows standard handles are redirected to the null device.
    """
    saved_fds: list[tuple[int, int]] = []
    saved_windows_handles: list[tuple[int, int, int]] = []
    null_fd: int | None = None
    try:
        null_stream = _BackendNullStream()
        with contextlib.redirect_stdout(null_stream), contextlib.redirect_stderr(null_stream):
            sys.stdout.flush()
            sys.stderr.flush()
            for fd in (1, 2):
                saved_fds.append((fd, os.dup(fd)))
            null_fd = os.open(os.devnull, os.O_WRONLY)

            if os.name == "nt":
                import msvcrt

                for which in (0xFFFFFFF5, 0xFFFFFFF4):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
                    saved_handle = _get_windows_std_handle(which)
                    null_handle_fd = os.dup(null_fd)
                    null_handle = msvcrt.get_osfhandle(null_handle_fd)
                    saved_windows_handles.append((which, saved_handle, null_handle_fd))
                    _set_windows_std_handle(which, null_handle)

            for fd, _ in saved_fds:
                os.dup2(null_fd, fd)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                for fd, saved_fd in saved_fds:
                    os.dup2(saved_fd, fd)
                if os.name == "nt":
                    for which, saved_handle, null_handle_fd in saved_windows_handles:
                        _set_windows_std_handle(which, saved_handle)
                        os.close(null_handle_fd)
    finally:
        for _, saved_fd in saved_fds:
            os.close(saved_fd)
        if null_fd is not None:
            os.close(null_fd)


def _paddle_model_names(cfg: Config) -> tuple[str, str] | None:
    """Return explicit current PaddleOCR model names when a tier is selected."""
    if cfg.ocr_version == "PP-OCRv6":
        if cfg.model_size not in {"tiny", "small", "medium"}:
            raise ValueError("PP-OCRv6 model_size must be tiny, small, or medium.")
        tier = cfg.model_size
        return f"PP-OCRv6_{tier}_det", f"PP-OCRv6_{tier}_rec"
    return None


def _default_runtime_cache_dir() -> str:
    """Return a writable persistent cache path for compiled runtime artifacts."""
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    else:
        root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "pylex", "openvino")


def _onnxruntime_has_openvino() -> bool:
    """Return whether the installed ONNX Runtime build exposes OpenVINO EP."""
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        import onnxruntime as ort

        return "OpenVINOExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001 - automatic mode must retain its fallback
        return False


def _effective_inference_engine(cfg: Config) -> str:
    """Resolve the portable automatic runtime choice once per engine build."""
    if cfg.inference_engine != "auto":
        return cfg.inference_engine
    if _onnxruntime_has_openvino():
        return "openvino"
    return "onnxruntime" if importlib.util.find_spec("onnxruntime") else "paddle_dynamic"


def _paddle_runtime_kwargs(cfg: Config) -> dict:
    """Build version-appropriate runtime options for PaddleOCR 3.x."""
    runtime_engine = _effective_inference_engine(cfg)
    kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": cfg.use_textline_orientation and not cfg.fast_mode,
        "lang": cfg.lang,
        "ocr_version": cfg.ocr_version,
        "device": "gpu" if cfg.use_gpu else "cpu",
        "textline_orientation_batch_size": cfg.orientation_batch_size,
        "text_recognition_batch_size": cfg.recognition_batch_size,
        "text_rec_score_thresh": cfg.min_confidence,
    }
    model_names = _paddle_model_names(cfg)
    if model_names is not None:
        kwargs["text_detection_model_name"], kwargs["text_recognition_model_name"] = model_names
        # PaddleOCR warns when model names are explicit alongside lang/version;
        # the names already encode the selected v6 tier and language family.
        kwargs["lang"] = None
        kwargs["ocr_version"] = None

    if cfg.enable_hpi:
        kwargs["enable_hpi"] = True
        if cfg.hpi_backend == "auto":
            kwargs["hpi_config"] = {"auto_config": True}
        elif cfg.hpi_backend:
            kwargs["hpi_config"] = {"backend": cfg.hpi_backend, "auto_config": False}
    else:
        # PaddleX exposes OpenVINO through its ONNX Runtime runner rather
        # than as a standalone `engine='openvino'` value.
        kwargs["engine"] = "onnxruntime" if runtime_engine == "openvino" else runtime_engine
        engine_config = {
            "device_type": "gpu" if cfg.use_gpu else "cpu",
        }
        if runtime_engine == "paddle_static":
            if cfg.cpu_threads:
                engine_config["cpu_threads"] = cfg.cpu_threads
            engine_config["run_mode"] = "paddle" if cfg.disable_mkldnn else "mkldnn"
        elif runtime_engine in {"onnxruntime", "openvino"}:
            if cfg.cpu_threads:
                engine_config["intra_op_num_threads"] = cfg.cpu_threads
            engine_config["inter_op_num_threads"] = 1
            if runtime_engine == "openvino":
                engine_config["providers"] = [
                    "OpenVINOExecutionProvider",
                    "CPUExecutionProvider",
                ]
                cache_dir = cfg.runtime_cache_dir or _default_runtime_cache_dir()
                os.makedirs(cache_dir, exist_ok=True)
                provider_options = {"device_type": "CPU", "cache_dir": cache_dir}
                if cfg.cpu_threads:
                    provider_options["num_of_threads"] = str(cfg.cpu_threads)
                engine_config["provider_options"] = [provider_options, {}]
        kwargs["engine_config"] = engine_config
    return kwargs


def get_ocr_engine(cfg: Config):
    """Lazily construct (and cache) the PaddleOCR engine.

    Importing paddleocr is slow, so this is only called once real work is
    about to happen — not for --help, --dry-run, or argument errors.
    """
    global _ENGINE, _ENGINE_KEY
    runtime_engine = _effective_inference_engine(cfg)
    engine_key = (
        cfg.lang.lower(),
        cfg.ocr_version,
        cfg.model_size,
        runtime_engine,
        cfg.enable_hpi,
        cfg.hpi_backend or "",
        cfg.runtime_cache_dir or "",
        cfg.use_gpu,
        cfg.use_textline_orientation and not cfg.fast_mode,
        cfg.orientation_batch_size,
        cfg.recognition_batch_size,
    )
    if _ENGINE is not None and _ENGINE_KEY == engine_key:
        return _ENGINE

    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    _quiet_backend_logging()

    log.info(
        "Loading PaddleOCR engine (version=%s, lang=%s, model=%s, runtime=%s)...",
        cfg.ocr_version,
        cfg.lang,
        cfg.model_size,
        runtime_engine,
    )
    t0 = time.time()

    try:
        with _quiet_backend_output():
            from paddleocr import PaddleOCR
    except ImportError as e:
        raise SystemExit(
            "paddleocr is not installed. Install it with:\n"
            "  pip install pylex[ocr]\n"
            "or:\n"
            "  pip install paddleocr paddlepaddle"
        ) from e

    _quiet_backend_logging()
    try:
        with _quiet_backend_output():
            _ENGINE = PaddleOCR(**_paddle_runtime_kwargs(cfg))
    except (RuntimeError, TypeError, ValueError) as exc:
        if not cfg.allow_runtime_fallback:
            raise SystemExit(
                "The installed PaddleOCR/PaddleX build rejected the requested "
                f"runtime configuration ({exc}). Use --allow-runtime-fallback "
                "to retry with the portable Paddle runtime."
            ) from exc
        log.warning("Optimized Paddle runtime unavailable (%s); retrying with paddle_dynamic.", exc)
        fallback = Config(
            **{
                **cfg.__dict__,
                "inference_engine": "paddle_dynamic",
                "enable_hpi": False,
                "hpi_backend": None,
                "disable_mkldnn": True,
            }
        )
        fallback_kwargs = _paddle_runtime_kwargs(fallback)
        with _quiet_backend_output():
            _ENGINE = PaddleOCR(**fallback_kwargs)

    _ENGINE_KEY = engine_key
    log.info("Engine ready in %.1fs", time.time() - t0)
    return _ENGINE


def _is_runtime_backend_failure(exc: BaseException) -> bool:
    """Identify known static-runtime failures that are safe to retry dynamically."""
    message = str(exc).lower()
    return isinstance(exc, NotImplementedError) or any(
        token in message for token in ("onednn", "one-dnn", "pir", "paddle_static")
    )


def _predict_dynamic_fallback(img_bgr, cfg: Config):
    """Rebuild once with dynamic Paddle inference after a static backend failure."""
    global _ENGINE, _ENGINE_KEY
    fallback = Config(
        **{
            **cfg.__dict__,
            "inference_engine": "paddle_dynamic",
            "enable_hpi": False,
            "hpi_backend": None,
            "disable_mkldnn": True,
        }
    )
    log.warning("Static Paddle inference failed; retrying this run with paddle_dynamic (%s).", cfg.inference_engine)
    _ENGINE = None
    _ENGINE_KEY = None
    return _predict(get_ocr_engine(fallback), img_bgr, fallback, allow_runtime_fallback=False)


def _predict(engine, img_bgr, cfg: Config, allow_runtime_fallback: bool = True):
    """Call PaddleOCR with detector tuning and recover from known runtime faults."""
    global _UNSUPPORTED_DET_KWARGS_WARNED

    base_kwargs = {
        "text_det_limit_side_len": cfg.det_limit_side_len,
        "text_det_limit_type": "max",
    }

    tuning_kwargs = {}
    if cfg.det_thresh is not None:
        tuning_kwargs["text_det_thresh"] = cfg.det_thresh
    if cfg.det_box_thresh is not None:
        tuning_kwargs["text_det_box_thresh"] = cfg.det_box_thresh
    if cfg.det_unclip_ratio is not None:
        tuning_kwargs["text_det_unclip_ratio"] = cfg.det_unclip_ratio

    try:
        with _quiet_backend_output():
            return engine.predict(img_bgr, **base_kwargs, **tuning_kwargs)
    except TypeError as exc:
        if tuning_kwargs and not _UNSUPPORTED_DET_KWARGS_WARNED:
            log.warning(
                "Installed PaddleOCR doesn't accept detector-tuning kwargs "
                "(%s) — ignoring --det-thresh/--det-box-thresh/--det-unclip-ratio.",
                exc,
            )
            _UNSUPPORTED_DET_KWARGS_WARNED = True
        try:
            with _quiet_backend_output():
                return engine.predict(img_bgr, **base_kwargs)
        except (NotImplementedError, RuntimeError) as runtime_exc:
            if allow_runtime_fallback and cfg.allow_runtime_fallback and _is_runtime_backend_failure(runtime_exc):
                return _predict_dynamic_fallback(img_bgr, cfg)
            raise
    except (NotImplementedError, RuntimeError) as exc:
        if allow_runtime_fallback and cfg.allow_runtime_fallback and _is_runtime_backend_failure(exc):
            return _predict_dynamic_fallback(img_bgr, cfg)
        raise


def _parse_results(results, cfg: Config, coordinate_scale: float = 1.0):
    """Convert PaddleOCR result objects into the project's normalized tuples."""
    items = []
    for res in results or []:
        if isinstance(res, dict):
            texts = res.get("rec_texts") or []
            polys = res.get("rec_polys") or res.get("dt_polys")
            scores = res.get("rec_scores") or []
        else:
            texts = getattr(res, "rec_texts", None) or []
            polys = getattr(res, "rec_polys", None) or getattr(res, "dt_polys", None)
            scores = getattr(res, "rec_scores", None) or []

        if polys is None:
            continue

        for i, text in enumerate(texts):
            if not text:
                continue
            score = float(scores[i]) if i < len(scores) else 1.0
            if score < cfg.min_confidence:
                continue

            poly = np.asarray(polys[i], dtype=np.float64)
            if poly.ndim != 2 or poly.shape[0] < 4 or poly.shape[1] != 2:
                continue

            items.append((str(text), poly[:4] * coordinate_scale, score))

    return items


def _upscale_for_retry(img_array: np.ndarray, scale: float = 1.5) -> np.ndarray | None:
    """Upscale only modest images so a no-result retry cannot exhaust memory."""
    height, width = img_array.shape[:2]
    if height * width > 12_000_000:
        return None
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(img_array).resize(new_size, Image.Resampling.LANCZOS))


def ocr_images(engine, img_arrays: list[np.ndarray], cfg: Config) -> list[list[tuple[str, np.ndarray, float]]]:
    """OCR several images in one backend call, preserving one result per image.

    Batch mode is used for tiled pages when retry-upscale is disabled. The
    PaddleOCR pipeline accepts a list of images and returns one result object
    per image; parsing each object independently keeps tile coordinates and
    confidence filtering identical to the single-image path.
    """
    if not img_arrays:
        return []
    img_bgrs = [np.ascontiguousarray(img[:, :, ::-1]) for img in img_arrays]
    results = _predict(engine, img_bgrs, cfg)
    if len(results) != len(img_arrays):
        raise RuntimeError(
            f"OCR backend returned {len(results)} result(s) for {len(img_arrays)} input image(s)."
        )
    return [_parse_results([result], cfg) for result in results]


def ocr_image(engine, img_array: np.ndarray, cfg: Config):
    """Run PaddleOCR on an RGB numpy array; return [(text, quad[4,2], score)].

    This is a single whole-image pass — for large/dense drawings, callers
    should generally go through pylex.tiling.ocr_image_tiled() instead,
    which splits into overlapping windows and merges the result. An optional
    upscale retry runs only when the first pass returns no usable text.
    """
    # Paddle may copy negative-stride channel-reversed views internally;
    # make one explicit contiguous array so that copy is predictable.
    img_bgr = np.ascontiguousarray(img_array[:, :, ::-1])
    items = _parse_results(_predict(engine, img_bgr, cfg), cfg)
    if items or not cfg.retry_upscale:
        return items

    scale = 1.5
    retry_img = _upscale_for_retry(img_array, scale)
    if retry_img is None:
        return items
    retry_bgr = np.ascontiguousarray(retry_img[:, :, ::-1])
    retry_results = _predict(engine, retry_bgr, cfg)
    return _parse_results(retry_results, cfg, coordinate_scale=1.0 / scale)
