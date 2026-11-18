"""Shared records and validation for one directly timed GPU sweep per seed."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

from numerical_policy import read_tolerance

PROTOCOL = "resident-whole-sweep-cuda-graph-v3"
WARMUP = 5


class Contract(TypedDict):
    """Fields consumed by the numerical runtime and sweep scheduler."""

    operation: str
    workloads: int
    memory_gib: float
    gradients: bool
    tolerance: float


class Hardware(TypedDict):
    """Observed GPU settings; observation does not assert clock locking."""

    name: str
    uuid: str
    driver: str
    compute_mode: str
    mig_mode: str
    sm_clock_mhz: float
    memory_clock_mhz: float
    power_limit_w: float
    temperature_c: float
    process_ids: list[int]
    throttle_counters_us: dict[str, int]


class Variation(TypedDict):
    """Descriptive statistics of the observed sweep durations."""

    minimum_ms: float
    maximum_ms: float
    mean_ms: float
    population_stddev_ms: float
    range_ms: float
    range_over_min_percent: float
    coefficient_of_variation_percent: float


def positive(value: object) -> bool:
    """Return whether a value is a finite, strictly positive real number.

    Args:
        value: Untrusted value from a measurement file.

    Returns:
        False for booleans, nonnumeric values, NaN, infinity, and nonpositive data.
    """
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def read_contract(path: Path) -> Contract:
    """Load and validate the subset of a task contract required by the worker.

    Args:
        path: Operation settings in JSON format.

    Returns:
        A typed numerical contract.

    Raises:
        ValueError: A required field has an invalid type or value.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Contract must be an object")
    if (not isinstance(raw.get("operation"), str)
            or type(raw.get("workloads")) is not int or raw["workloads"] < 1
            or not positive(raw.get("memory_gib"))
            or type(raw.get("gradients")) is not bool):
        raise ValueError("Invalid numerical contract")
    return Contract(operation=raw["operation"], workloads=raw["workloads"],
                    memory_gib=float(raw["memory_gib"]), gradients=raw["gradients"],
                    tolerance=read_tolerance())


def number(node: ET.Element, path: str) -> float:
    """Read a finite numeric XML field whose suffix contains its units.

    Args:
        node: GPU XML element returned by nvidia-smi.
        path: Relative path to a numeric telemetry field.

    Returns:
        The field's numeric component.

    Raises:
        RuntimeError: The field is missing, unsupported, or nonfinite.
    """
    text = node.findtext(path, "")
    try:
        value = float(text.split()[0])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"measurement infrastructure: unavailable GPU field {path}: {text}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"measurement infrastructure: nonfinite GPU field {path}")
    return value


def hardware() -> Hardware:
    """Read GPU identity and clock observations without creating a CUDA context.

    Returns:
        Settings and cumulative throttling counters for the sole visible GPU.

    Raises:
        RuntimeError: GPU identity or essential telemetry cannot be established.
    """
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True,
                            text=True, check=True, timeout=30)
    root = ET.fromstring(result.stdout)
    devices = root.findall("gpu")
    if len(devices) != 1:
        raise RuntimeError("measurement infrastructure: exactly one GPU required")
    gpu = devices[0]
    counters = gpu.find("clocks_event_reasons_counters")
    throttle = {} if counters is None else {
        item.tag: int(item.text.split()[0]) for item in counters
        if item.text and item.text.split()[0].isdigit()
    }
    power_path = next((path for path in ("gpu_power_readings/current_power_limit",
                      "gpu_power_readings/power_limit", "power_readings/power_limit")
                       if gpu.find(path) is not None), "power_readings/power_limit")
    report = Hardware(
        name=gpu.findtext("product_name", ""), uuid=gpu.findtext("uuid", ""),
        driver=root.findtext("driver_version", ""),
        compute_mode=gpu.findtext("compute_mode", ""),
        mig_mode=gpu.findtext("mig_mode/current_mig", ""),
        sm_clock_mhz=number(gpu, "clocks/sm_clock"),
        memory_clock_mhz=number(gpu, "clocks/mem_clock"),
        power_limit_w=number(gpu, power_path), temperature_c=number(gpu, "temperature/gpu_temp"),
        process_ids=[int(item.text) for item in gpu.findall("processes/process_info/pid")
                     if item.text and item.text.isdigit()],
        throttle_counters_us=throttle,
    )
    if (report["name"] != "NVIDIA H100 80GB HBM3" or report["mig_mode"] != "Disabled"
            or not report["uuid"].startswith("GPU-") or report["power_limit_w"] < 699):
        raise RuntimeError("measurement infrastructure: GPU differs from the H100 contract")
    return report


def variation(times: list[float]) -> Variation:
    """Describe observed seed-to-seed variation without estimating its cause.

    Args:
        times: Complete, positive sweep durations in milliseconds.

    Returns:
        Range, population standard deviation, and coefficient of variation.

    Raises:
        ValueError: No timings were supplied or a timing is invalid.
    """
    if not times or not all(positive(value) for value in times):
        raise ValueError("Positive finite sweep durations required")
    low, high, mean = min(times), max(times), statistics.mean(times)
    deviation = statistics.pstdev(times)
    return Variation(minimum_ms=low, maximum_ms=high, mean_ms=mean,
                     population_stddev_ms=deviation, range_ms=high-low,
                     range_over_min_percent=100*(high-low)/low,
                     coefficient_of_variation_percent=100*deviation/mean)
