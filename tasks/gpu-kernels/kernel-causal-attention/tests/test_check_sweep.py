"""Check every saved sweep output in a process that never imports the candidate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from test_graph_checks import compare
from test_kernel_ops import inputs, output_and_gradients, reference
from test_sweep_protocol import PROTOCOL, read_contract


def check(directory: Path, contract_path: Path) -> dict[str, object]:
    """Independently recompute all numerical outputs for a completed sweep.

    Args:
        directory: Immutable-after-worker-exit output directory.
        contract_path: Trusted task contract.

    Returns:
        Verified case coverage and seed, suitable for attachment by the grader.

    Raises:
        ValueError: Measurement metadata or any numerical output is invalid.
    """
    contract = read_contract(contract_path)
    report = json.loads((directory / "measurements.json").read_text())
    if report.get("timing_protocol") != PROTOCOL or report.get("status") != "measured_pending_correctness":
        raise ValueError("Missing completed sweep")
    seed, indices = report["seed"], report["case_indices"]
    if type(seed) is not int or not isinstance(indices, list) or len(indices) != len(set(indices)):
        raise ValueError("Invalid sweep identity")
    kind = contract["operation"]

    def numerical_reference(values: dict[str, torch.Tensor | int | float | bool]) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...]:
        """Evaluate the trusted mathematical operation for the current task.

        Args:
            values: Fresh GPU inputs generated independently of the candidate.

        Returns:
            The mathematical reference outputs.
        """
        return reference(kind, values)

    for index in indices:
        if type(index) is not int or not 0 <= index < contract["workloads"]:
            raise ValueError("Invalid workload index")
        values = inputs(kind, seed, index, "cuda", False)
        expected = output_and_gradients(numerical_reference, values, contract["gradients"])
        saved = load_file(str(directory / f"case-{index}.safetensors"), device="cuda")
        if set(saved) != {f"output_{i}" for i in range(len(expected))}:
            raise ValueError("Measured output keys differ from the contract")
        actual = [saved[f"output_{i}"] for i in range(len(expected))]
        compare(actual, expected, contract, values)
        del values, expected, actual, saved
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"event": "case_verified", "index": index}), flush=True)
    return {"seed": seed, "case_indices": indices, "correctness_passed": True,
            "timing_protocol": PROTOCOL}


def main() -> None:
    """Validate outputs after the candidate worker has released its GPU context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = check(args.directory, args.config)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
