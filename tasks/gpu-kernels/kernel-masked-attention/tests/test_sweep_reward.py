"""Continuous reward from the median of the configured full-sweep measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median

from test_sweep_protocol import PROTOCOL, SEED_COUNT, positive, variation

PARTIAL_CREDIT_BAND = 0.10


class MeasurementError(ValueError):
    """The available evidence cannot support a valid grading decision."""


def validate_target(target: Mapping[str, object], workloads: int, *, seed_count: int = SEED_COUNT) -> None:
    """Require a frozen target authored with the same whole-sweep protocol.

    Args:
        target: Private performance target supplied by the task author.
        workloads: Required number of cases in every sweep.
        seed_count: Required number of distinct seeds from the trusted contract.

    Raises:
        MeasurementError: The target is incomplete or uses another protocol.
    """
    if type(seed_count) is not int or seed_count < 1:
        raise MeasurementError("Invalid configured seed count")
    if target.get("status") != "frozen":
        raise MeasurementError("Whole-sweep reference baseline has not been frozen")
    if (target.get("schema_version") != 2 or target.get("timing_protocol") != PROTOCOL
            or target.get("seed_count") != seed_count or target.get("workloads") != workloads
            or target.get("scored_replays_per_seed") != 1
            or target.get("clock_policy") != "provider_default_unlocked"
            or target.get("aggregation") != "median_full_sweep_ms"):
        raise MeasurementError("Unexpected whole-sweep measurement contract")
    if not positive(target.get("suite_limit_ms")):
        raise MeasurementError("Invalid reference baseline")
    if "workload_limits_ms" in target:
        raise MeasurementError("Per-case latency limits belong to the superseded protocol")


def validate_hardware(snapshot: object) -> str:
    """Check comparable hardware while retaining observed clock variation.

    Args:
        snapshot: Settings recorded immediately before or after a sweep.

    Returns:
        The physical GPU UUID.

    Raises:
        MeasurementError: Identity or basic hardware evidence is invalid.
    """
    if not isinstance(snapshot, dict):
        raise MeasurementError("Missing hardware observation")
    if (snapshot.get("name") != "NVIDIA H100 80GB HBM3"
            or snapshot.get("mig_mode") != "Disabled"
            or not positive(snapshot.get("power_limit_w")) or snapshot["power_limit_w"] < 699
            or not positive(snapshot.get("sm_clock_mhz"))
            or not positive(snapshot.get("memory_clock_mhz"))
            or not isinstance(snapshot.get("compute_mode"), str)
            or not isinstance(snapshot.get("uuid"), str) or not snapshot["uuid"].startswith("GPU-")):
        raise MeasurementError("Unsupported GPU configuration or missing clock observations")
    return snapshot["uuid"]


def decide(records: Sequence[Mapping[str, object]], target: Mapping[str, object],
           seeds: list[int], workloads: int, memory_gib: float, *,
           seed_count: int = SEED_COUNT) -> tuple[dict[str, int | float | bool], dict[str, object]]:
    """Award full or partial credit after every configured sweep passes checks.

    Args:
        records: Independently checked, unprofiled sweep measurements.
        target: Frozen absolute full-sweep limit in milliseconds.
        seeds: Distinct private grading seeds in execution order.
        workloads: Number of cases required in each measured graph.
        memory_gib: Maximum allocated GPU memory under the task contract.
        seed_count: Required count from the trusted task contract.

    Returns:
        A reward in [0, 1] and descriptive seed timing statistics. The passed
        flag means full credit: the median sweep time meets the fixed limit.
        Correct timings up to 10% above the limit receive linearly decreasing
        credit; times at or above that cutoff receive zero.

    Raises:
        MeasurementError: Measurements are missing, duplicated, profiled,
            numerically unverified, or incompatible with the target.
    """
    validate_target(target, workloads, seed_count=seed_count)
    if len(seeds) != seed_count or len(set(seeds)) != seed_count or any(type(seed) is not int for seed in seeds):
        raise MeasurementError(f"Exactly {seed_count} distinct integer seeds are required")
    by_seed: dict[int, float] = {}
    for record in records:
        seed = record.get("seed")
        if type(seed) is not int or seed not in seeds or seed in by_seed:
            raise MeasurementError("Unexpected or duplicated sweep seed")
        if (record.get("timing_protocol") != PROTOCOL or record.get("status") != "completed"
                or record.get("correctness_passed") is not True
                or record.get("case_indices") != list(range(workloads))
                or record.get("scored_replays") != 1 or record.get("event_placement") != "inside_graph"
                or record.get("profiled") is not False
                or record.get("clock_policy") != "provider_default_unlocked"
                or record.get("torch") != "2.8.0+cu128" or record.get("cuda") != "12.8"):
            raise MeasurementError("Incomplete coverage, unchecked output, or incorrect timing protocol")
        elapsed, peak = record.get("sweep_ms"), record.get("peak_allocated_bytes")
        if not positive(elapsed) or not positive(peak) or peak > memory_gib * 1024**3:
            raise MeasurementError("Invalid duration or GPU memory evidence")
        if validate_hardware(record.get("hardware_before")) != validate_hardware(record.get("hardware_after")):
            raise MeasurementError("GPU identity changed during the sweep")
        by_seed[seed] = float(elapsed)
    if set(by_seed) != set(seeds):
        raise MeasurementError(f"All {seed_count} complete sweeps are required")
    times = [by_seed[seed] for seed in seeds]
    baseline = target["suite_limit_ms"]
    elapsed = median(times)
    passed = elapsed <= baseline
    relative_excess = max(0., (elapsed - baseline) / baseline)
    if passed:
        score = 1.
    elif elapsed >= (1. + PARTIAL_CREDIT_BAND) * baseline:
        score = 0.
    else:
        score = max(0., min(1., 1. - relative_excess / PARTIAL_CREDIT_BAND))
    return {"valid": 1, "passed": passed, "reward": score}, {
        "passed": passed, "timing_protocol": PROTOCOL, "aggregation": "median_full_sweep_ms",
        "suite_metric_ms": elapsed, "suite_limit_ms": baseline,
        "relative_latency_excess": relative_excess, "partial_credit_band": PARTIAL_CREDIT_BAND,
        "reward": score,
        "seed_times_ms": times, "variation": variation(times),
        "clock_policy": "provider_default_unlocked",
    }
