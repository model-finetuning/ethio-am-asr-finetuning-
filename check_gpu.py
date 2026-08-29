#!/usr/bin/env python3
"""Check whether the system is ready for GPU-based ASR fine-tuning."""

from __future__ import annotations

import importlib
import platform
import re
import subprocess
import sys
from typing import Any


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "installed (version unavailable)"))


def version_pair(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def run_nvidia_smi() -> bool:
    heading("NVIDIA driver")
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        print("FAIL: nvidia-smi was not found. Install or expose the NVIDIA driver.")
        return False
    except subprocess.TimeoutExpired:
        print("FAIL: nvidia-smi timed out.")
        return False

    if result.returncode != 0:
        print(f"FAIL: nvidia-smi returned exit code {result.returncode}.")
        if result.stderr.strip():
            print(result.stderr.strip())
        return False

    for line in result.stdout.strip().splitlines():
        print(line)
    return True


def gib(byte_count: int) -> float:
    return byte_count / (1024**3)


def check_pytorch() -> tuple[bool, list[dict[str, Any]]]:
    heading("PyTorch CUDA")
    try:
        import torch
    except ImportError:
        print("FAIL: PyTorch is not installed.")
        return False, []

    print(f"PyTorch:             {torch.__version__}")
    print(f"Built with CUDA:     {torch.version.cuda or 'No (CPU-only build)'}")
    print(f"CUDA available:      {torch.cuda.is_available()}")
    print(f"cuDNN:               {torch.backends.cudnn.version() or 'Unavailable'}")

    if not torch.cuda.is_available():
        print(
            "FAIL: PyTorch cannot access CUDA. This usually means a CPU-only "
            "PyTorch build, an incompatible driver, or an unavailable GPU."
        )
        return False, []

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        device = {
            "index": index,
            "name": properties.name,
            "vram_gib": gib(properties.total_memory),
            "compute_capability": f"{major}.{minor}",
            "bf16": bool(torch.cuda.is_bf16_supported()),
            "tf32": major >= 8,
        }
        devices.append(device)

        print(f"\nGPU {index}:              {device['name']}")
        print(f"VRAM:               {device['vram_gib']:.2f} GiB")
        print(f"Compute capability:  {device['compute_capability']}")
        print(f"BF16 supported:       {device['bf16']}")
        print(f"TF32 supported:       {device['tf32']}")

    heading("CUDA allocation test")
    try:
        device = torch.device("cuda:0")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        left = torch.randn((512, 512), device=device, dtype=dtype)
        right = torch.randn((512, 512), device=device, dtype=dtype)
        output = left @ right
        torch.cuda.synchronize(device)
        del left, right, output
        torch.cuda.empty_cache()
        print(f"PASS: CUDA matrix multiplication succeeded using {dtype}.")
    except Exception as exc:
        print(f"FAIL: CUDA allocation or matrix multiplication failed: {exc}")
        return False, devices

    return True, devices


def check_packages() -> bool:
    heading("Training packages")
    packages = [
        "torch",
        "torchaudio",
        "soundfile",
        "transformers",
        "accelerate",
        "jiwer",
        "numpy",
        "safetensors",
    ]
    versions = {name: package_version(name) for name in packages}
    for name, version in versions.items():
        print(f"{name:14} {version or 'MISSING'}")

    okay = all(version is not None for version in versions.values())
    if versions["torch"] and versions["torchaudio"]:
        torch_series = version_pair(versions["torch"])
        audio_series = version_pair(versions["torchaudio"])
        if torch_series and audio_series:
            stable_abi_compatible = audio_series >= (2, 11) and torch_series >= (2, 11)
            legacy_compatible = audio_series < (2, 11) and audio_series == torch_series
            if not (stable_abi_compatible or legacy_compatible):
                okay = False
                print(
                    "FAIL: TorchAudio 2.11+ requires PyTorch 2.11+. Older "
                    "TorchAudio releases must match PyTorch's major/minor version."
                )
    return okay


def main() -> int:
    print("GPU training environment check")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"System:   {platform.platform()}")
    print(f"Machine:  {platform.machine()}")

    driver_ok = run_nvidia_smi()
    torch_ok, devices = check_pytorch()
    packages_ok = check_packages()

    heading("Recommendation")
    if torch_ok and devices:
        if devices[0]["bf16"]:
            print("Use the training script's default BF16 mode.")
        else:
            print("Use --fp16 because this GPU does not report BF16 support.")
    else:
        print(
            "Install a CUDA-enabled PyTorch build compatible with your NVIDIA "
            "driver, then run this check again."
        )

    if driver_ok and torch_ok and packages_ok:
        print("PASS: The environment is ready for GPU ASR training.")
        return 0

    print("FAIL: Fix the items reported above before starting training.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
