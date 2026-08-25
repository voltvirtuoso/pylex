"""OCR engine: lazy PaddleOCR loading and raw prediction -> (text, quad, score)."""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from pidocr.config import Config

log = logging.getLogger("pidocr")

_ENGINE = None


def get_ocr_engine(cfg: Config):
    """Lazily construct (and cache) the PaddleOCR engine.

    Importing paddleocr is slow, so this is only called once real work is
    about to happen — not for --help, --dry-run, or argument errors.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise SystemExit(
            "paddleocr is not installed. Install it with:\n"
            "  pip install pidocr[ocr]\n"
            "or:\n"
            "  pip install paddleocr paddlepaddle"
        ) from e

    log.info("Loading PaddleOCR engine (%s, lang=%s)...", cfg.ocr_version, cfg.lang)
    t0 = time.time()

    _ENGINE = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=cfg.use_textline_orientation,
        lang=cfg.lang,
        ocr_version=cfg.ocr_version,
        enable_mkldnn=not cfg.disable_mkldnn,
        device="gpu" if cfg.use_gpu else "cpu",
    )

    log.info("Engine ready in %.1fs", time.time() - t0)
    return _ENGINE


def ocr_image(engine, img_array: np.ndarray, cfg: Config):
    """Run PaddleOCR on an RGB numpy array; return [(text, quad[4,2], score)]."""
    img_bgr = img_array[:, :, ::-1]

    results = engine.predict(
        img_bgr,
        text_det_limit_side_len=cfg.det_limit_side_len,
        text_det_limit_type="max",
    )

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

            items.append((str(text), poly[:4], score))

    return items
