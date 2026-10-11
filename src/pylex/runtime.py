"""Hardware detection and install guidance for Pylex's OCR runtimes.

Pylex runs PaddleOCR models through one of several runtimes (ONNX Runtime,
OpenVINO, or Paddle itself). Which pip packages you need depends on your
hardware, and the packages conflict with each other (``onnxruntime`` and
``onnxruntime-gpu`` cannot be installed side by side). This module looks at
the machine and tells you what to install, from both Python and the CLI
(``pylex doctor``).

Nothing here imports Paddle or ONNX Runtime unless it has to, so it is cheap
to call before an OCR backend is installed.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

DISTRIBUTION = "pylex-ocr"
GITHUB_URL = "https://github.com/voltvirtuoso/pylex.git"
# Flip to True once the package is on PyPI: every install command Pylex prints then switches
# from the GitHub form to the short requirement `pylex-ocr[cpu]`.
PUBLISHED_ON_PYPI = False


def install_spec(extra: str | None = None) -> str:
    """The requirement string to pass to pip, e.g. ``pylex-ocr[cpu] @ git+https://...``."""
    name = f"{DISTRIBUTION}[{extra}]" if extra else DISTRIBUTION
    return name if PUBLISHED_ON_PYPI else f"{name} @ git+{GITHUB_URL}"

# Every package that provides the ``onnxruntime`` import name. Installing more
# than one of them corrupts the shared files, so ``doctor`` flags it.
ONNXRUNTIME_DISTS = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-openvino",
    "onnxruntime-directml",
    "onnxruntime-rocm",
    "onnxruntime-migraphx",
)
PADDLE_DISTS = ("paddlepaddle", "paddlepaddle-gpu")
TRACKED_DISTS = ("paddleocr", *PADDLE_DISTS, *ONNXRUNTIME_DISTS, "openvino")

# Extras defined in pyproject.toml, with what each is for.
PROFILES: dict[str, str] = {
    "cpu": "Any CPU. ONNX Runtime plus PaddleOCR. The portable default.",
    "intel": "Intel CPUs. Same as cpu, but ONNX Runtime with the OpenVINO provider.",
    "gpu": "NVIDIA GPU, CUDA 13 build of ONNX Runtime (driver reporting CUDA 13 or newer).",
    "gpu-cu12": "NVIDIA GPU, CUDA 12 build of ONNX Runtime (driver reporting CUDA 12).",
}

_TPU_ENV_VARS = ("TPU_NAME", "TPU_ACCELERATOR_TYPE", "COLAB_TPU_ADDR", "CLOUD_TPU_TASK_ID")


@dataclass
class HardwareReport:
    """What Pylex could learn about this machine without loading an OCR model."""

    system: str
    machine: str
    python: str
    cpu_name: str = ""
    nvidia_gpus: list[str] = field(default_factory=list)
    nvidia_driver: str | None = None
    cuda_driver_version: tuple[int, int] | None = None  # newest CUDA the driver supports
    apple_silicon: bool = False
    amd_gpu_stack: bool = False
    tpu: str | None = None
    installed: dict[str, str] = field(default_factory=dict)  # pip distribution -> version
    onnxruntime_providers: list[str] = field(default_factory=list)

    @property
    def is_intel_cpu(self) -> bool:
        return "intel" in self.cpu_name.lower()

    @property
    def has_nvidia(self) -> bool:
        return bool(self.nvidia_gpus)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Recommendation:
    """An install plan for this machine."""

    profile: str
    extra: str
    commands: list[str]
    ready: bool = False
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Detection primitives (small, individually mockable)
# ----------------------------------------------------------------------
def _run(cmd: list[str], timeout: float = 8.0) -> str:
    """Run a command and return stdout, or '' if it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def installed_distributions(names: tuple[str, ...] = TRACKED_DISTS) -> dict[str, str]:
    """Return ``{distribution: version}`` for the tracked packages that are installed."""
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return found


def onnxruntime_providers() -> list[str]:
    """Execution providers of the installed ONNX Runtime ([] when not installed)."""
    if importlib.util.find_spec("onnxruntime") is None:
        return []
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - a broken install must not crash detection
        return []


def paddle_gpu_installed() -> bool:
    """True when the CUDA build of Paddle (``paddlepaddle-gpu``) is installed."""
    try:
        importlib_metadata.version("paddlepaddle-gpu")
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def cuda_available() -> bool:
    """True when some installed runtime can run Pylex on an NVIDIA GPU."""
    return "CUDAExecutionProvider" in onnxruntime_providers() or paddle_gpu_installed()


def _parse_cuda_version(smi_text: str) -> tuple[int, int] | None:
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _detect_cpu_name(system: str) -> str:
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if name:
            return name
    return platform.processor() or ""


def _detect_tpu() -> str | None:
    """Heuristic Cloud TPU / Colab TPU detection from the environment."""
    for var in _TPU_ENV_VARS:
        if os.environ.get(var):
            return f"{var}={os.environ[var]}"
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return "PJRT_DEVICE=TPU"
    if importlib.util.find_spec("libtpu") is not None:
        return "libtpu package installed"
    return None


def detect_hardware(run: Callable[[list[str]], str] = _run) -> HardwareReport:
    """Inspect the machine. Safe to call without any OCR backend installed."""
    system = platform.system()
    machine = platform.machine()
    report = HardwareReport(
        system=system,
        machine=machine,
        python=platform.python_version(),
        cpu_name=_detect_cpu_name(system),
        apple_silicon=(system == "Darwin" and machine == "arm64"),
        amd_gpu_stack=bool(shutil.which("rocminfo") or os.path.isdir("/opt/rocm")),
        tpu=_detect_tpu(),
        installed=installed_distributions(),
        onnxruntime_providers=onnxruntime_providers(),
    )

    names = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    for line in names.splitlines():
        if not line.strip():
            continue
        name, _, driver = line.partition(",")
        report.nvidia_gpus.append(name.strip())
        report.nvidia_driver = report.nvidia_driver or (driver.strip() or None)
    if report.nvidia_gpus:
        report.cuda_driver_version = _parse_cuda_version(run(["nvidia-smi"]))
    return report


# ----------------------------------------------------------------------
# Recommendation
# ----------------------------------------------------------------------
def _pip(*args: str) -> str:
    return "python -m pip " + " ".join(args)


def _conflicting_onnx(installed: dict[str, str]) -> list[str]:
    return [name for name in ONNXRUNTIME_DISTS if name in installed]


def recommend_install(report: HardwareReport | None = None) -> Recommendation:
    """Choose the extra to install for this machine and say how to install it."""
    hw = report or detect_hardware()
    notes: list[str] = []
    warnings: list[str] = []
    alternatives: list[str] = []

    if hw.has_nvidia:
        cuda = hw.cuda_driver_version
        if cuda is not None and cuda[0] >= 13:
            profile = "gpu"
        elif cuda is not None and cuda[0] == 12:
            profile = "gpu-cu12"
        elif cuda is None:
            profile = "gpu-cu12"
            warnings.append(
                "An NVIDIA GPU was found but its CUDA version could not be read from nvidia-smi. "
                "Recommending the CUDA 12 build; update the NVIDIA driver if it fails to load."
            )
        else:
            profile = "cpu"
            warnings.append(
                f"The NVIDIA driver reports CUDA {cuda[0]}.{cuda[1]}, but the ONNX Runtime GPU builds "
                "Pylex installs target CUDA 12 and 13. Update the NVIDIA driver to use the GPU; "
                "falling back to the CPU profile."
            )
        if profile == "gpu-cu12" and cuda is not None and cuda[1] < 8:
            warnings.append(
                f"The driver reports CUDA {cuda[0]}.{cuda[1]}. ONNX Runtime's documentation says builds made "
                "with CUDA 12.8 require CUDA 12.8 or newer. If `pylex doctor --verify-gpu` fails, update "
                "the NVIDIA driver."
            )
        if profile != "cpu":
            notes.append(
                "The GPU extras install CUDA and cuDNN libraries through pip, so no system CUDA toolkit "
                "is required, only an NVIDIA driver new enough for the build."
            )
            alternatives.append(
                "Paddle's own GPU build also works (see docs/installation.md, 'Paddle GPU build'): "
                "its wheels come from Paddle's package index, so pip extras cannot select them."
            )
    else:
        profile = "cpu"
        if hw.is_intel_cpu:
            alternatives.append(
                f'Intel CPU: also try pip install "{install_spec("intel")}" (OpenVINO) and benchmark it '
                "against [cpu] on your documents. It is not always faster."
            )

    if hw.tpu:
        warnings.append(
            f"Google TPU detected ({hw.tpu}). Pylex has no TPU support (PaddleOCR's documented hardware list "
            "does not include TPUs), so it will use the host CPU. Use the cpu profile there."
        )
    if hw.apple_silicon:
        notes.append(
            "Apple Silicon: Pylex's extras run on the CPU and do not use Metal or the Neural Engine. "
            "PaddleOCR documents Apple Silicon support through Paddle's own builds, which Pylex has not tested."
        )
    if hw.amd_gpu_stack and not hw.has_nvidia:
        notes.append(
            "An AMD ROCm stack was found. Pylex's extras do not include a ROCm runtime, so inference "
            "runs on the CPU. PaddleOCR documents AMD GPU support through Paddle's own builds, which "
            "Pylex has not tested."
        )

    extra = profile
    commands = [_pip("install", f'"{install_spec(extra)}"')]

    conflicts = _conflicting_onnx(hw.installed)
    # On CPU hardware either the plain or the OpenVINO build is a valid choice; on an NVIDIA GPU
    # only onnxruntime-gpu is. More than one ONNX Runtime package is always a conflict.
    acceptable = {"onnxruntime", "onnxruntime-openvino"} if profile in {"cpu", "intel"} else {"onnxruntime-gpu"}
    wrong = [name for name in conflicts if name not in acceptable]
    if wrong or len(conflicts) > 1:
        warnings.append(
            "Conflicting ONNX Runtime packages are installed: "
            + ", ".join(sorted(conflicts))
            + ". They share one import name; keep only the one you need."
        )
        commands.insert(0, _pip("uninstall", "-y", *sorted(conflicts)))
    if "paddlepaddle" in hw.installed and "paddlepaddle-gpu" in hw.installed:
        warnings.append(
            "Both paddlepaddle and paddlepaddle-gpu are installed; they also share one import name. "
            "Uninstall both, then reinstall only the one you want."
        )

    ready = (
        "paddleocr" in hw.installed
        and bool(conflicts)
        and not wrong
        and len(conflicts) == 1
        and (profile not in {"gpu", "gpu-cu12"} or "CUDAExecutionProvider" in hw.onnxruntime_providers)
    )
    return Recommendation(
        profile=profile,
        extra=extra,
        commands=commands,
        ready=ready,
        notes=notes,
        warnings=warnings,
        alternatives=alternatives,
    )


def install_hint() -> str:
    """Multi-line message shown when no OCR backend is installed."""
    rec = recommend_install()
    lines = ["Pylex needs an OCR runtime that matches your hardware. For this machine, run:", ""]
    lines += [f"  {cmd}" for cmd in rec.commands]
    lines += [
        "",
        ("Other choices: [cpu] any CPU, [intel] Intel CPU with OpenVINO, [gpu] NVIDIA CUDA 13, "
         "[gpu-cu12] NVIDIA CUDA 12."),
        "Run `pylex doctor` for a full report.",
    ]
    lines += [f"WARNING: {w}" for w in rec.warnings]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Human-readable report
# ----------------------------------------------------------------------
def format_report(report: HardwareReport, rec: Recommendation) -> str:
    """Render the output of ``pylex doctor``."""
    gpu_text = ", ".join(report.nvidia_gpus) if report.nvidia_gpus else "none detected"
    cuda = report.cuda_driver_version
    lines = [
        "Pylex doctor",
        "============",
        f"System    : {report.system} {report.machine}, Python {report.python}",
        f"CPU       : {report.cpu_name or 'unknown'}",
        f"NVIDIA GPU: {gpu_text}",
    ]
    if report.has_nvidia:
        cuda_text = f"{cuda[0]}.{cuda[1]}" if cuda else "unknown"
        lines.append(f"            driver {report.nvidia_driver or 'unknown'}, supports CUDA {cuda_text}")
    if report.tpu:
        lines.append(f"TPU       : {report.tpu} (not usable by Pylex)")
    lines.append("")
    lines.append("Installed runtime packages:")
    if report.installed:
        lines += [f"  {name} {version}" for name, version in sorted(report.installed.items())]
    else:
        lines.append("  none")
    providers = ", ".join(report.onnxruntime_providers) or "n/a"
    lines.append(f"ONNX Runtime providers: {providers}")
    lines.append("")
    status = "ready" if rec.ready else "needs changes"
    lines.append(f"Recommended profile: [{rec.profile}] - {PROFILES[rec.profile]}")
    lines.append(f"Status: {status}")
    if not rec.ready:
        lines.append("To set it up:")
        lines += [f"  {cmd}" for cmd in rec.commands]
    for text in rec.warnings:
        lines.append(f"Warning: {text}")
    for text in rec.notes:
        lines.append(f"Note: {text}")
    for text in rec.alternatives:
        lines.append(f"Option: {text}")
    return "\n".join(lines)


def doctor(*, as_text: bool = True, report: HardwareReport | None = None):
    """Detect hardware and recommend an install.

    Returns the report text when ``as_text`` is true, otherwise a
    ``(HardwareReport, Recommendation)`` tuple.
    """
    hw = report or detect_hardware()
    rec = recommend_install(hw)
    return format_report(hw, rec) if as_text else (hw, rec)


def run_install(rec: Recommendation, *, assume_yes: bool = False) -> int:
    """Run the recommended pip commands with the current interpreter (asks first)."""
    if not assume_yes:
        print("About to run:")
        for cmd in rec.commands:
            print(f"  {cmd}")
        if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return 1
    for cmd in rec.commands:
        parts = shlex.split(cmd)
        parts[0] = sys.executable  # "python -m pip ..." -> run with this interpreter
        status = subprocess.call(parts)
        if status != 0:
            return status
    return 0


# ----------------------------------------------------------------------
# GPU verification
# ----------------------------------------------------------------------
def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number: int, payload: bytes | int | str) -> bytes:
    """Encode one protobuf field (varint for int, length-delimited otherwise)."""
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    data = payload.encode() if isinstance(payload, str) else payload
    return _varint(number << 3 | 2) + _varint(len(data)) + data


def _identity_model() -> bytes:
    """A minimal ONNX model (one Identity node), built without the ``onnx`` package."""
    dim = _field(1, 1)  # Dimension { dim_value: 1 }
    shape = _field(1, dim)  # TensorShapeProto { dim: [Dimension] }
    tensor_type = _field(1, 1) + _field(2, shape)  # elem_type FLOAT, shape [1]
    value_type = _field(1, tensor_type)

    def value_info(name: str) -> bytes:
        return _field(1, name) + _field(2, value_type)

    node = _field(1, "x") + _field(2, "y") + _field(3, "id") + _field(4, "Identity")
    graph = _field(1, node) + _field(2, "g") + _field(11, value_info("x")) + _field(12, value_info("y"))
    opset = _field(2, 13)
    return _field(1, 8) + _field(8, opset) + _field(7, graph)


def verify_gpu(provider: str = "CUDAExecutionProvider") -> tuple[bool, str]:
    """Check that ONNX Runtime can really run on the GPU, not just list the provider.

    ONNX Runtime quietly falls back to the CPU when CUDA or cuDNN libraries
    fail to load, and Pylex hides backend output, so this builds a tiny model
    and asks the session which providers it actually registered.
    """
    if importlib.util.find_spec("onnxruntime") is None:
        return False, "onnxruntime is not installed."
    try:
        import onnxruntime as ort

        if provider not in ort.get_available_providers():
            return False, (
                f"{provider} is not available (found: {', '.join(ort.get_available_providers())}). "
                f'Install the GPU build: pip install "{install_spec("gpu")}" and uninstall plain onnxruntime.'
            )
        preload = getattr(ort, "preload_dlls", None)
        if preload is not None:
            preload()
        session = ort.InferenceSession(_identity_model(), providers=[provider, "CPUExecutionProvider"])
        active = session.get_providers()
    except Exception as exc:  # noqa: BLE001 - report any failure to the user
        return False, f"Could not create a session: {exc}"
    if active and active[0] == provider:
        return True, f"{provider} is active."
    return False, (
        f"{provider} is listed but did not load (active: {', '.join(active)}). "
        "CUDA or cuDNN libraries are probably missing; install the pip-bundled ones with the "
        'gpu extras, or check that the NVIDIA driver is new enough.'
    )
