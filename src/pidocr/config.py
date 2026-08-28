"""Runtime configuration for the OCR engine and overlay renderer."""

from __future__ import annotations

from dataclasses import dataclass

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ALL_EXTS = PDF_EXTS | IMAGE_EXTS


@dataclass
class Config:
    dpi: int = 300
    image_dpi: int = 300
    min_confidence: float = 0.5
    lang: str = "en"
    # PaddleOCR is the sole backend. PP-OCRv6 is the current accuracy baseline;
    # the small tier balances accuracy and CPU latency for engineering text.
    ocr_version: str = "PP-OCRv6"
    model_size: str = "small"
    # Paddle dynamic inference is the verified cross-platform baseline. Static
    # oneDNN is available as an explicit option but can hit PIR/oneDNN operator
    # incompatibilities in some Paddle 3.x builds.
    # Auto selects ONNX Runtime when installed; otherwise it uses dynamic
    # Paddle. Both paths use the same OCR models and pipeline settings.
    inference_engine: str = "auto"
    enable_hpi: bool = False
    hpi_backend: str | None = None
    # Persistent compiled-engine cache for OpenVINO/ONNX Runtime backends.
    # None selects the platform cache location automatically.
    runtime_cache_dir: str | None = None
    allow_runtime_fallback: bool = True
    font: str = "Helvetica"
    min_font_size: float = 1.0
    max_font_size: float = 200.0
    det_limit_side_len: int = 8000
    baseline_fraction: float = 0.18
    min_box_width_pt: float = 0.25
    min_box_height_pt: float = 0.25
    # Text-line orientation improves recall for upside-down/rotated text but
    # adds a separate classifier model. Keep it on for the accuracy-first
    # default; --no-textline-orientation can disable it.
    use_textline_orientation: bool = True
    # Retry only tiles containing tall detected polygons after a 90-degree
    # rotation to recover vertical labels that ordinary line OCR misreads.
    vertical_text_retry: bool = True
    vertical_text_aspect_ratio: float = 1.3
    vertical_text_retry_max_confidence: float = 0.8
    use_gpu: bool = False
    # Dynamic inference does not use the static oneDNN path by default.
    disable_mkldnn: bool = True
    # Optional per-process native thread budget. None preserves backend defaults.
    cpu_threads: int | None = None
    orientation_batch_size: int = 8
    recognition_batch_size: int = 6
    # Retained as a compatibility profile; the default path is now tuned for
    # better throughput and accuracy rather than requiring this flag.
    fast_mode: bool = False
    # Retry a no-result pass once with a 1.5x Lanczos upscale. This is useful
    # for low-resolution labels, but remains opt-in because it adds work when
    # an image contains no detectable text.
    retry_upscale: bool = False
    page_ranges: list[tuple[int, int]] | None = None  # 1-based, inclusive; None = all

    # Tiling: OCR dense drawings in overlapping windows at full local
    # resolution instead of one whole-page pass. A whole-page pass resizes
    # everything down to det_limit_side_len before detection, so small tags
    # squeezed between valve/instrument symbols can shrink below what the
    # detector can pick up even though the same text on open pipe run is
    # fine. tile_size <= 0 disables tiling (single full-page pass).
    tile_size: int = 2000
    tile_overlap: int = 220
    # Batch several tiles in one backend call to reduce per-call overhead while
    # preserving the same tile resolution and overlap. Retry-upscale remains
    # sequential because its second pass depends on each tile's result.
    tile_batch_size: int = 4
    # Automatically avoid tiling when a small-model image fits within the
    # detector's native resolution; strict tiling remains available for dense
    # drawings where local recall is more important than extra passes.
    adaptive_tiling: bool = True
    full_page_max_pixels: int = 8_000_000
    dedupe_tol_px: float = 20.0

    # Text-detector tuning (passed straight to PaddleOCR's predict() when
    # not None). Lower det_box_thresh and a larger det_unclip_ratio both
    # help recover text that's touching or squeezed against line-art —
    # exactly the failure mode tiling alone doesn't always fix.
    det_thresh: float | None = None
    det_box_thresh: float | None = None
    det_unclip_ratio: float | None = None
