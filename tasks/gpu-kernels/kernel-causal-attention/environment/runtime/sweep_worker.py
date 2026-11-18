"""Capture all selected cases and directly time one complete GPU-resident sweep."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import save_file

from graph_checks import compare
from kernel_ops import inputs, output_and_gradients, reference, tensors
from restricted import load
from sweep_protocol import PROTOCOL, WARMUP, Contract, hardware, positive, read_contract

Values = dict[str, torch.Tensor | int | float | bool]
Kernel = Callable[[Values], torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]]


def refresh(static: Values, fresh: Values) -> None:
    """Replace tensor values at stable GPU addresses outside the timed graph.

    Args:
        static: Persistent capture inputs, mutated in place.
        fresh: Newly generated inputs with matching tensor and scalar signatures.

    Raises:
        ValueError: A tensor signature or scalar argument changed.
    """
    if static.keys() != fresh.keys():
        raise ValueError("Input signature changed")
    for key, value in static.items():
        replacement = fresh[key]
        if isinstance(value, torch.Tensor):
            if (not isinstance(replacement, torch.Tensor) or value.shape != replacement.shape
                    or value.dtype != replacement.dtype or replacement.device.type != "cuda"):
                raise ValueError("Input tensor signature changed")
            value.copy_(replacement)
        elif replacement != value:
            raise ValueError("Input scalar signature changed")


def clone(values: Values) -> Values:
    """Copy tensor state while preserving scalar arguments.

    Args:
        values: Kernel inputs to copy.

    Returns:
        Independent tensor buffers on the original device.
    """
    return {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in values.items()}


def optimizer_chain(fn: Kernel, values: Values, contract: Contract) -> None:
    """Preserve the existing 100-update optimizer correctness requirement.

    This untimed check runs before capture. Local update error and accumulated
    state error retain their distinct tolerances.

    Args:
        fn: Restricted candidate entrypoint.
        values: Initial optimizer state for one workload.
        contract: Numerical tolerances and operation settings.

    Raises:
        ValueError: Any local or accumulated optimizer state is incorrect.
    """
    count = values["count"]
    if type(count) is not int:
        raise ValueError("Optimizer count must be an integer")
    actual_state, expected_state = clone(values), clone(values)
    generator = torch.Generator(device="cuda").manual_seed(654321)
    for step in range(1, 101):
        actual_state["step"] = expected_state["step"] = step
        for i in range(count):
            original_gradient = values[f"g{i}"]
            if not isinstance(original_gradient, torch.Tensor):
                raise ValueError("Optimizer gradient must be a tensor")
            gradient = torch.randn(original_gradient.shape, dtype=original_gradient.dtype,
                                   device="cuda", generator=generator) * .3
            actual_state[f"g{i}"] = gradient
            expected_state[f"g{i}"] = gradient.clone()
        original, local_input = clone(expected_state), clone(actual_state)
        expected = tensors(reference("op10", expected_state))
        local_expected = tensors(reference("op10", local_input))
        actual = tensors(fn(actual_state))
        compare(actual, local_expected, contract, local_input)
        compare(actual, expected, contract, original, update_relative=False)
        for i in range(count):
            for j, key in enumerate(("p", "m", "v")):
                actual_state[f"{key}{i}"] = actual[3*i+j]
                expected_state[f"{key}{i}"] = expected[3*i+j]


def refresh_suite(states: list[Values], indices: list[int], kind: str, seed: int) -> None:
    """Populate persistent inputs one case at a time without CPU tensor copies.

    Args:
        states: GPU input buffers retained by the graph.
        indices: Workload indices in execution order.
        kind: Operation identifier understood by the task generator.
        seed: Seed used by the public generator for every selected case.
    """
    for state, index in zip(states, indices, strict=True):
        fresh = inputs(kind, seed, index, "cuda", False)
        refresh(state, fresh)
        del fresh


def measure(fn: Kernel, contract: Contract, seed: int, indices: list[int],
            output: Path, profile: bool) -> dict[str, object]:
    """Measure one graph replay containing every selected case exactly once.

    Timing events are external CUDA event nodes inside the captured graph.
    Capture and warmup use different values from the measured replay. Outputs
    are saved after the end event for numerical comparison.

    Args:
        fn: Restricted candidate entrypoint.
        contract: Numerical and memory requirements.
        seed: Seed for the fresh inputs of the measured sweep.
        indices: Case order, normally the entire ascending index range.
        output: Directory for outputs, telemetry, and the graph description.
        profile: Whether to delimit an external Nsight diagnostic capture.

    Returns:
        Direct sweep duration and coverage evidence, pending numerical checking.

    Raises:
        ValueError: Candidate results or memory usage violate the contract.
        RuntimeError: The GPU or timing infrastructure cannot produce a record.
    """
    kind = contract["operation"]
    capture_seed = seed ^ 0x5A17D3
    states: list[Values] = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for index in indices:
        state = inputs(kind, capture_seed, index, "cuda", False)
        if kind == "op10":
            optimizer_chain(fn, state, contract)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(WARMUP):
                fresh = inputs(kind, capture_seed, index, "cuda", False)
                refresh(state, fresh)
                del fresh
                actual = output_and_gradients(fn, state, contract["gradients"])
                if any(value.device.type != "cuda" for value in actual):
                    raise ValueError("All outputs must remain on CUDA")
                del actual
        torch.cuda.current_stream().wait_stream(stream)
        states.append(state)
        print(json.dumps({"event": "case_prepared", "index": index,
                          "resident_bytes": torch.cuda.memory_allocated()}), flush=True)
    refresh_suite(states, indices, kind, capture_seed)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    captured: list[list[torch.Tensor]] = []
    with torch.cuda.graph(graph, stream=stream):
        start.record()
        for state in states:
            captured.append(output_and_gradients(fn, state, contract["gradients"]))
        end.record()
    # Read telemetry before warmup so its subprocess cannot idle the GPU
    # immediately before the timed replay.
    before = hardware()
    for _ in range(WARMUP):
        refresh_suite(states, indices, kind, capture_seed)
        graph.replay()
    torch.cuda.synchronize()
    refresh_suite(states, indices, kind, seed)
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    graph.replay()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    after = hardware()
    if not positive(elapsed):
        raise RuntimeError("measurement infrastructure: invalid sweep interval")
    peak = torch.cuda.max_memory_allocated()
    if peak > contract["memory_gib"] * 1024**3:
        raise ValueError(f"GPU memory cap exceeded: {peak / 1024**3:.3f} GiB")
    for index, values in zip(indices, captured, strict=True):
        if any(type(value) is not torch.Tensor or value.device.type != "cuda" for value in values):
            raise ValueError("All measured outputs must be ordinary CUDA tensors")
        save_file({f"output_{i}": value.detach().cpu().contiguous().clone()
                   for i, value in enumerate(values)}, str(output / f"case-{index}.safetensors"))
    return {"timing_protocol": PROTOCOL, "seed": seed, "case_indices": indices,
            "sweep_ms": elapsed, "timed_replays": 1, "warmup_replays": WARMUP,
            "event_placement": "inside_graph", "capture_seed": capture_seed,
            "profiled": profile, "peak_allocated_bytes": peak,
            "hardware_before": before, "hardware_after": after,
            "clock_policy": "provider_default_unlocked",
            "exclusive_process_mode_required": False,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "status": "measured_pending_correctness"}


def main() -> None:
    """Run one sweep in a fresh process and persist its measurement evidence.

    Raises:
        RuntimeError: The GPU or software environment differs from the contract.
        ValueError: The candidate, requested case set, or output is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--indices")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    contract = read_contract(args.config)
    indices = list(map(int, args.indices.split(","))) if args.indices else list(range(contract["workloads"]))
    if (not indices or len(set(indices)) != len(indices)
            or any(index < 0 or index >= contract["workloads"] for index in indices)):
        raise ValueError("Invalid workload selection")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    if (torch.cuda.device_count() != 1 or properties.name != "NVIDIA H100 80GB HBM3"
            or properties.multi_processor_count != 132
            or torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8"):
        raise RuntimeError("measurement infrastructure: unsupported GPU or software")
    torch.cuda.set_per_process_memory_fraction(min(1., contract["memory_gib"] * 1024**3 / properties.total_memory))
    args.out.mkdir(parents=True, exist_ok=True)
    fn: Kernel = load(str(args.candidate), ["run"]).run
    report = measure(fn, contract, args.seed, indices, args.out, args.profile)
    report["candidate_sha256"] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    (args.out / "measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"event": "sweep_measured", "seed": args.seed,
                      "cases": len(indices), "sweep_ms": report["sweep_ms"]}), flush=True)


if __name__ == "__main__":
    main()
