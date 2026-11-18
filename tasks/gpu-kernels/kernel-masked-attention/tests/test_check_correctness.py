"""Independently check untimed masked-attention outputs and Q/K/V gradients."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from test_correctness_cases import CASE_COUNT, CASE_SPECS, inputs
from test_graph_checks import compare
from test_kernel_ops import output_and_gradients, reference
from test_sweep_protocol import read_contract


def numerical_reference(values: dict[str, torch.Tensor]) -> torch.Tensor:
    """Evaluate the mathematical masked-attention operation.

    Args:
        values: Independently reconstructed Q/K/V tensors and mask.

    Returns:
        Attention output whose derivatives are computed by trusted autograd.
    """
    return reference("op06", values)


def main() -> None:
    """Require all four outputs for every boundary case at the existing tolerance.

    Raises:
        ValueError: Configuration, output signature, or numerical checks fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract = read_contract(args.config)
    if contract["operation"] != "op06" or not contract["gradients"]:
        raise ValueError("Masked-attention checks require forward and backward")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases: list[dict[str, object]] = []
    for index in range(CASE_COUNT):
        values = inputs(contract["operation"], index)
        expected = output_and_gradients(numerical_reference, values, True)
        saved = load_file(str(args.directory / f"case-{index}.safetensors"), device="cuda")
        if len(expected) != 4 or set(saved) != {f"output_{i}" for i in range(4)}:
            raise ValueError("Expected attention output and Q/K/V gradients")
        actual = [saved[f"output_{i}"] for i in range(4)]
        row: dict[str, object] = {"index": index, "sequence_length": CASE_SPECS[index][0],
                                  "mask_block_width": CASE_SPECS[index][1],
                                  "head_dimension": CASE_SPECS[index][2],
                                  "correctness_passed": False}
        row["nrmse"] = {name: float((value.double() - target.double()).norm()
                                    / target.double().norm().clamp_min(1e-8))
                        for name, value, target in zip(("output", "dq", "dk", "dv"), actual, expected)}
        cases.append(row)
        try:
            compare(actual, expected, contract, values)
        except ValueError as error:
            row["error"] = str(error)
            args.out.write_text(json.dumps({"case_count": len(cases), "correctness_passed": False,
                                            "untimed": True, "cases": cases}, indent=2))
            raise
        row["correctness_passed"] = True
        print(json.dumps(row), flush=True)
        del values, expected, actual, saved
        gc.collect()
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps({"case_count": CASE_COUNT, "correctness_passed": True,
                                    "untimed": True, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
