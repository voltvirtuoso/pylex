from pylex.cli import build_arg_parser, config_from_args


def test_fast_profile_disables_tiling_and_orientation():
    args = build_arg_parser().parse_args(["image.png", "--fast", "--retry-upscale"])
    cfg = config_from_args(args)

    assert cfg.fast_mode is True
    assert cfg.tile_size == 0
    assert cfg.use_textline_orientation is False
    assert cfg.det_limit_side_len == 3200
    assert cfg.retry_upscale is True


def test_runtime_options_select_current_paddle_stack():
    args = build_arg_parser().parse_args(
        [
            "image.png",
            "--ocr-version",
            "PP-OCRv6",
            "--model-size",
            "medium",
            "--enable-hpi",
            "--hpi-backend",
            "openvino",
            "--recognition-batch-size",
            "12",
        ]
    )
    cfg = config_from_args(args)

    assert cfg.ocr_version == "PP-OCRv6"
    assert cfg.model_size == "medium"
    assert cfg.enable_hpi is True
    assert cfg.hpi_backend == "openvino"
    assert cfg.recognition_batch_size == 12


def test_default_profile_keeps_accuracy_options_enabled():
    args = build_arg_parser().parse_args(["image.png"])
    cfg = config_from_args(args)

    assert cfg.fast_mode is False
    assert cfg.tile_size == 2000
    assert cfg.use_textline_orientation is True
    assert cfg.det_limit_side_len == 8000
    assert cfg.retry_upscale is False
    assert cfg.disable_mkldnn is True
    assert cfg.ocr_version == "PP-OCRv6"
    assert cfg.model_size == "medium"
    assert cfg.inference_engine == "auto"
    assert cfg.enable_hpi is False


def test_vertical_text_retry_is_enabled_by_default_and_can_be_disabled():
    default_cfg = config_from_args(build_arg_parser().parse_args(["image.png"]))
    disabled_cfg = config_from_args(
        build_arg_parser().parse_args(["image.png", "--no-vertical-text-retry"])
    )

    assert default_cfg.vertical_text_retry is True
    assert disabled_cfg.vertical_text_retry is False
