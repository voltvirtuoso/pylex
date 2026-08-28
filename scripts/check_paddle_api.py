from inspect import signature

from paddleocr import PaddleOCR


params = signature(PaddleOCR).parameters
required = {
    "text_detection_model_name",
    "text_recognition_model_name",
    "textline_orientation_batch_size",
    "text_recognition_batch_size",
    "engine",
    "engine_config",
    "enable_hpi",
}
has_kwargs = any(p.kind.name == "VAR_KEYWORD" for p in params.values())
missing = sorted(required - params.keys())
print(f"PaddleOCR constructor supports {len(params)} declared parameters and **kwargs={has_kwargs}")
print(f"required_direct_parameters_missing={missing}")
if missing and not has_kwargs:
    raise SystemExit(1)
