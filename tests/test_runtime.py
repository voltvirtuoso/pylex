import json

import pytest

import pylex.engine as engine_mod
from pylex import runtime
from pylex.cli import build_arg_parser, config_from_args, main
from pylex.config import Config

SMI_NAMES = "NVIDIA GeForce RTX 4090, 575.57.08\n"
SMI_FULL_CUDA13 = "| NVIDIA-SMI 580.65   Driver Version: 580.65   CUDA Version: 13.0 |"
SMI_FULL_CUDA12 = "| NVIDIA-SMI 550.54   Driver Version: 550.54   CUDA Version: 12.4 |"
SMI_FULL_CUDA11 = "| NVIDIA-SMI 470.57   Driver Version: 470.57   CUDA Version: 11.4 |"


def fake_run(names="", full=""):
    def run(cmd):
        return names if "--query-gpu=name,driver_version" in " ".join(cmd) else full

    return run


def make_report(**overrides):
    base = {"system": "Linux", "machine": "x86_64", "python": "3.12.0", "cpu_name": "AMD EPYC"}
    base.update(overrides)
    return runtime.HardwareReport(**base)


@pytest.fixture(autouse=True)
def _no_real_hardware(monkeypatch):
    """Keep every test independent of the machine running it."""
    monkeypatch.setattr(runtime, "onnxruntime_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "paddle_gpu_installed", lambda: False)
    monkeypatch.setattr(runtime, "_detect_tpu", lambda: None)


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def test_cuda_version_is_parsed_from_nvidia_smi_output():
    assert runtime._parse_cuda_version(SMI_FULL_CUDA12) == (12, 4)
    assert runtime._parse_cuda_version("no gpu here") is None


def test_detect_hardware_reads_nvidia_gpu_and_driver():
    report = runtime.detect_hardware(run=fake_run(SMI_NAMES, SMI_FULL_CUDA13))

    assert report.nvidia_gpus == ["NVIDIA GeForce RTX 4090"]
    assert report.nvidia_driver == "575.57.08"
    assert report.cuda_driver_version == (13, 0)


def test_detect_hardware_without_nvidia_smi_reports_no_gpu():
    report = runtime.detect_hardware(run=fake_run("", ""))

    assert report.nvidia_gpus == []
    assert report.cuda_driver_version is None


# ----------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cuda", "profile"),
    [((13, 0), "gpu"), ((13, 2), "gpu"), ((12, 4), "gpu-cu12"), (None, "gpu-cu12")],
)
def test_nvidia_gpu_gets_matching_cuda_profile(cuda, profile):
    report = make_report(nvidia_gpus=["RTX"], cuda_driver_version=cuda)

    rec = runtime.recommend_install(report)

    assert rec.profile == profile
    assert rec.commands == [f'python -m pip install "{runtime.install_spec(profile)}"']


def test_driver_too_old_for_current_onnxruntime_falls_back_to_cpu_with_warning():
    rec = runtime.recommend_install(make_report(nvidia_gpus=["Old"], cuda_driver_version=(11, 4)))

    assert rec.profile == "cpu"
    assert any("CUDA 11.4" in w for w in rec.warnings)


def test_cuda_12_below_12_8_warns_about_onnxruntime_requirement():
    rec = runtime.recommend_install(make_report(nvidia_gpus=["Old"], cuda_driver_version=(12, 4)))

    assert rec.profile == "gpu-cu12"
    assert any("12.8" in w for w in rec.warnings)


def test_cuda_12_8_has_no_such_warning():
    rec = runtime.recommend_install(make_report(nvidia_gpus=["New"], cuda_driver_version=(12, 8)))

    assert rec.warnings == []


def test_machine_without_gpu_gets_cpu_profile():
    rec = runtime.recommend_install(make_report())

    assert rec.profile == "cpu"
    assert rec.alternatives == []


def test_intel_cpu_is_offered_openvino_as_a_benchmark_option_not_a_default():
    rec = runtime.recommend_install(make_report(cpu_name="Intel(R) Core(TM) i7"))

    assert rec.profile == "cpu"
    assert any("[intel]" in a for a in rec.alternatives)


def test_tpu_is_reported_as_unsupported_and_cpu_is_used():
    rec = runtime.recommend_install(make_report(tpu="TPU_NAME=node-1"))

    assert rec.profile == "cpu"
    assert any("TPU" in w and "CPU" in w and "no TPU support" in w for w in rec.warnings)


def test_apple_silicon_note_says_cpu_only():
    rec = runtime.recommend_install(make_report(system="Darwin", machine="arm64", apple_silicon=True))

    assert rec.profile == "cpu"
    assert any("Apple Silicon" in n for n in rec.notes)


def test_conflicting_onnxruntime_packages_are_flagged_and_uninstalled_first():
    report = make_report(
        nvidia_gpus=["RTX"],
        cuda_driver_version=(13, 0),
        installed={"onnxruntime": "1.30.0", "onnxruntime-gpu": "1.30.0", "paddleocr": "3.7.0"},
    )

    rec = runtime.recommend_install(report)

    assert any("Conflicting" in w for w in rec.warnings)
    assert rec.commands[0] == "python -m pip uninstall -y onnxruntime onnxruntime-gpu"
    assert rec.commands[-1] == f'python -m pip install "{runtime.install_spec("gpu")}"'
    assert rec.ready is False


def test_cpu_onnxruntime_on_a_gpu_machine_is_a_conflict():
    report = make_report(
        nvidia_gpus=["RTX"],
        cuda_driver_version=(13, 0),
        installed={"onnxruntime": "1.30.0", "paddleocr": "3.7.0"},
    )

    rec = runtime.recommend_install(report)

    assert rec.ready is False
    assert rec.commands[0].startswith("python -m pip uninstall -y onnxruntime")


def test_ready_requires_the_cuda_provider_on_gpu_profiles():
    installed = {"paddleocr": "3.7.0", "onnxruntime-gpu": "1.30.0"}
    base = {"nvidia_gpus": ["RTX"], "cuda_driver_version": (13, 0), "installed": installed}

    without = runtime.recommend_install(make_report(**base, onnxruntime_providers=["CPUExecutionProvider"]))
    with_cuda = runtime.recommend_install(
        make_report(**base, onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    )

    assert without.ready is False
    assert with_cuda.ready is True


def test_cpu_profile_is_ready_when_paddleocr_and_onnxruntime_are_installed():
    report = make_report(installed={"paddleocr": "3.7.0", "onnxruntime": "1.30.0"})

    assert runtime.recommend_install(report).ready is True


def test_install_hint_uses_the_real_distribution_name():
    hint = runtime.install_hint()

    assert "pylex-ocr[" in hint
    assert "pylex[ocr]" not in hint
    assert "pylex doctor" in hint


def test_doctor_text_and_tuple_modes():
    report = make_report()

    text = runtime.doctor(report=report)
    hw, rec = runtime.doctor(as_text=False, report=report)

    assert "Recommended profile: [cpu]" in text
    assert hw is report and rec.profile == "cpu"


def test_run_install_executes_each_command_after_confirmation(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "call", lambda parts: calls.append(parts) or 0)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    rec = runtime.Recommendation(
        profile="gpu",
        extra="gpu",
        commands=["python -m pip uninstall -y onnxruntime", 'python -m pip install "pylex-ocr[gpu]"'],
    )

    assert runtime.run_install(rec) == 0
    assert calls[0][1:] == ["-m", "pip", "uninstall", "-y", "onnxruntime"]
    assert calls[1][1:] == ["-m", "pip", "install", "pylex-ocr[gpu]"]


def test_run_install_does_nothing_when_declined(monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "call", lambda parts: pytest.fail("must not run"))
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    rec = runtime.Recommendation(profile="cpu", extra="cpu", commands=['python -m pip install "pylex-ocr[cpu]"'])

    assert runtime.run_install(rec) == 1


def test_verify_gpu_reports_missing_cuda_provider(monkeypatch):
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _n: object())
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)

    ok, detail = runtime.verify_gpu()

    assert ok is False and "CUDAExecutionProvider is not available" in detail


def test_verify_gpu_detects_a_provider_that_listed_but_did_not_load(monkeypatch):
    class FakeSession:
        def __init__(self, *_a, **_k):
            pass

        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider"]

    class FakeOrt:
        InferenceSession = FakeSession

        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _n: object())
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)

    ok, detail = runtime.verify_gpu()

    assert ok is False and "did not load" in detail


# ----------------------------------------------------------------------
# Device resolution and engine configuration
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (Config(), ("cpu", None)),
        (Config(use_gpu=True), ("gpu", None)),
        (Config(device="gpu"), ("gpu", None)),
        (Config(device="cuda"), ("gpu", None)),
        (Config(device="CUDA:1"), ("gpu", 1)),
        (Config(device="gpu:2"), ("gpu", 2)),
        (Config(device="npu:0"), ("npu", 0)),
        (Config(device="cpu", use_gpu=True), ("cpu", None)),
    ],
)
def test_resolve_device(cfg, expected):
    assert engine_mod.resolve_device(cfg) == expected


@pytest.mark.parametrize("bad", ["tpu", "gpu:one", "gpu:0,1", "cpu:0"])
def test_resolve_device_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        engine_mod.resolve_device(Config(device=bad))


def test_auto_device_uses_gpu_only_when_a_cuda_runtime_exists(monkeypatch):
    assert engine_mod.resolve_device(Config(device="auto")) == ("cpu", None)

    monkeypatch.setattr(runtime, "onnxruntime_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert engine_mod.resolve_device(Config(device="auto")) == ("gpu", None)


def test_auto_engine_on_gpu_prefers_onnxruntime_with_cuda_provider(monkeypatch):
    monkeypatch.setattr(runtime, "onnxruntime_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])

    cfg = Config(device="gpu:1")
    kwargs = engine_mod._paddle_runtime_kwargs(cfg)

    assert engine_mod._effective_inference_engine(cfg) == "onnxruntime"
    assert kwargs["device"] == "gpu:1"
    assert kwargs["engine"] == "onnxruntime"
    assert kwargs["engine_config"]["device_type"] == "gpu"
    assert kwargs["engine_config"]["device_id"] == 1


def test_auto_engine_on_gpu_uses_paddle_when_only_paddle_gpu_is_installed(monkeypatch):
    monkeypatch.setattr(runtime, "paddle_gpu_installed", lambda: True)

    cfg = Config(device="gpu")

    assert engine_mod._effective_inference_engine(cfg) == "paddle_dynamic"
    assert engine_mod._paddle_runtime_kwargs(cfg)["device"] == "gpu"


def test_other_accelerators_are_routed_to_paddle():
    cfg = Config(device="npu:0")

    assert engine_mod._effective_inference_engine(cfg) == "paddle_dynamic"
    assert engine_mod._paddle_runtime_kwargs(cfg)["device"] == "npu:0"


def test_gpu_without_a_cuda_runtime_falls_back_to_cpu_with_a_clear_warning(monkeypatch, caplog):
    cfg = Config(device="gpu")

    with caplog.at_level("WARNING", logger="pylex"):
        fixed = engine_mod._preflight_device(cfg)

    assert engine_mod.resolve_device(fixed) == ("cpu", None)
    assert fixed.use_gpu is False
    assert "no CUDA provider" in caplog.text or "paddlepaddle-gpu" in caplog.text
    assert "pylex-ocr[gpu]" in caplog.text


def test_gpu_without_a_cuda_runtime_exits_when_fallback_is_disabled():
    cfg = Config(device="gpu", allow_runtime_fallback=False)

    with pytest.raises(engine_mod.PylexSetupError) as info:
        engine_mod._preflight_device(cfg)

    assert "pylex doctor" in str(info.value)


def test_gpu_with_openvino_is_rejected():
    cfg = Config(device="gpu", inference_engine="openvino", allow_runtime_fallback=False)

    with pytest.raises(engine_mod.PylexSetupError) as info:
        engine_mod._preflight_device(cfg)

    assert "CPU only" in str(info.value)


def test_cpu_requests_skip_preflight_untouched():
    cfg = Config()

    assert engine_mod._preflight_device(cfg) is cfg


def test_missing_paddleocr_error_names_the_real_package(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "paddleocr", None)  # makes the import raise ImportError
    monkeypatch.setattr(engine_mod, "_ENGINE", None)
    monkeypatch.setattr(engine_mod, "_ENGINE_KEY", None)

    with pytest.raises(engine_mod.PylexSetupError) as info:
        engine_mod.get_ocr_engine(Config())

    message = str(info.value)
    assert "pylex-ocr[" in message
    assert "pip install pylex[ocr]" not in message


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_device_flag_reaches_the_config():
    cfg = config_from_args(build_arg_parser().parse_args(["a.png", "--device", "gpu:1"]))

    assert cfg.device == "gpu:1"
    assert cfg.use_gpu is False


def test_gpu_flag_is_still_supported():
    cfg = config_from_args(build_arg_parser().parse_args(["a.png", "--gpu"]))

    assert engine_mod.resolve_device(cfg) == ("gpu", None)


def test_doctor_subcommand_prints_a_report(capsys):
    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "Pylex doctor" in out and "Recommended profile" in out


def test_doctor_json_is_machine_readable(capsys):
    assert main(["doctor", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["recommendation"]["profile"] in runtime.PROFILES
    assert "hardware" in payload


def test_doctor_verify_gpu_failure_sets_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "verify_gpu", lambda: (False, "nope"))

    assert main(["doctor", "--verify-gpu"]) == 1
    assert "FAILED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["a.png", "--gpu", "--device", "cpu"],
        ["a.png", "--device", "tpu"],
        ["a.png", "--device", "gpu", "--inference-engine", "openvino"],
    ],
)
def test_invalid_device_combinations_are_rejected(argv, capsys):
    with pytest.raises(SystemExit) as info:
        main(argv)

    assert info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_unreachable_model_hosts_give_a_clear_setup_error(monkeypatch):
    import sys
    import types

    class FakePaddleOCR:
        def __init__(self, **_kwargs):
            raise Exception("No available model hosting platforms detected.")  # noqa: TRY002 - mirrors PaddleX

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakePaddleOCR))
    monkeypatch.setattr(engine_mod, "_ENGINE", None)
    monkeypatch.setattr(engine_mod, "_ENGINE_KEY", None)

    with pytest.raises(engine_mod.PylexSetupError) as info:
        engine_mod.get_ocr_engine(Config())

    assert "PADDLE_PDX_MODEL_SOURCE" in str(info.value)


def test_install_commands_use_the_github_source_until_published(monkeypatch):
    monkeypatch.setattr(runtime, "PUBLISHED_ON_PYPI", False)

    assert runtime.install_spec("cpu") == "pylex-ocr[cpu] @ git+https://github.com/voltvirtuoso/pylex.git"
    assert runtime.install_spec() == "pylex-ocr @ git+https://github.com/voltvirtuoso/pylex.git"
    assert "git+https://github.com/voltvirtuoso/pylex.git" in runtime.install_hint()


def test_install_commands_switch_to_the_short_form_once_published(monkeypatch):
    monkeypatch.setattr(runtime, "PUBLISHED_ON_PYPI", True)

    assert runtime.install_spec("gpu") == "pylex-ocr[gpu]"
    assert "git+" not in runtime.install_hint()


def test_a_deliberate_openvino_install_is_ready_not_a_conflict():
    report = make_report(installed={"paddleocr": "3.7.0", "onnxruntime-openvino": "1.24.1"})

    rec = runtime.recommend_install(report)

    assert rec.ready is True
    assert not any("Conflicting" in w for w in rec.warnings)


def test_two_onnxruntime_packages_are_never_ready():
    report = make_report(installed={"paddleocr": "3.7.0", "onnxruntime": "1.30", "onnxruntime-openvino": "1.24"})

    rec = runtime.recommend_install(report)

    assert rec.ready is False
    assert any("Conflicting" in w for w in rec.warnings)


def test_no_user_facing_text_tells_people_to_install_an_unpublished_package():
    """While the package is not on PyPI, `pip install "pylex-ocr[...]"` fails. Docs must not show it."""
    import re
    from pathlib import Path

    if runtime.PUBLISHED_ON_PYPI:
        pytest.skip("package is published; the short form is valid")
    root = Path(__file__).resolve().parent.parent
    short_form = re.compile(r"""(pip|pipx)\s+install\s+(--upgrade\s+)?["']?pylex-ocr(\[[^\]]*\])?["']?(?![\w\[ @.-])""")
    offenders = []
    files = [root / "README.md", root / "PERFORMANCE.md", *sorted((root / "docs").glob("*.md")),
             *sorted((root / "src").rglob("*.py")), root / "scripts" / "docs_template.html"]
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if short_form.search(line):
                offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()[:90]}")
    assert not offenders, "PyPI-only install commands found:\n" + "\n".join(offenders)
