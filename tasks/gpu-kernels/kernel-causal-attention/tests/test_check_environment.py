"""Trusted GPU preflight; never import a submission or read worker artifacts."""

from __future__ import annotations

import json
import os

import torch

from test_sweep_protocol import hardware
from test_sweep_reward import validate_hardware


def main() -> None:
    """Check the required device, software, telemetry, and a fixed GPU operation.

    Raises:
        RuntimeError: The process is untrusted or the required environment is
            unavailable. CUDA and telemetry failures also terminate this probe.
    """
    if os.geteuid() != 0:
        raise RuntimeError("Environment probe must run as root")
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one CUDA device")
    properties = torch.cuda.get_device_properties(0)
    if (properties.name != "NVIDIA H100 80GB HBM3"
            or properties.multi_processor_count != 132
            or torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8"):
        raise RuntimeError("Unsupported GPU or software environment")
    uuid = validate_hardware(hardware())
    values = torch.ones(256, device="cuda", dtype=torch.float32)
    total = values.sum().item()
    torch.cuda.synchronize()
    if total != 256.:
        raise RuntimeError("GPU failed the fixed arithmetic check")
    print(json.dumps({"healthy": True, "gpu_uuid": uuid}))


if __name__ == "__main__":
    main()
