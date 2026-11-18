"""Read numerical settings frozen into the image from task configuration."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat


def read_tolerance() -> float:
    """Read the trusted NRMSE limit without accepting process-env overrides.

    The packager extracts this value from task.toml. The container image owns
    both this module and its adjacent configuration file as root. Shell exports
    and settings in a writable per-run contract cannot change the limit.

    Returns:
        The positive, finite NRMSE limit authored for this task.

    Raises:
        ValueError: The policy is writable, not root-owned, or malformed.
        OSError: The fixed policy file is missing or cannot be opened safely.
    """
    path = Path(__file__).with_name("numerical_policy.json")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r") as stream:
        metadata = os.fstat(stream.fileno())
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0
                or metadata.st_mode & 0o022 or metadata.st_size > 4096):
            raise ValueError("Numerical policy must be a protected root-owned file")
        settings = json.load(stream)
    if not isinstance(settings, dict) or set(settings) != {"KERNEL_NRMSE_TOLERANCE"}:
        raise ValueError("Malformed numerical policy")
    raw = settings["KERNEL_NRMSE_TOLERANCE"]
    if not isinstance(raw, str):
        raise ValueError("Numerical tolerance must be a configured string")
    value = float(raw)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("Numerical tolerance must be finite and between zero and one")
    return value
