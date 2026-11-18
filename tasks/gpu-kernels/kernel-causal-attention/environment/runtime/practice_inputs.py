"""Causal-attention examples spanning the public size range."""
from __future__ import annotations

import torch
from practice_cases import Case, default_cases

Values = dict[str, torch.Tensor | int | float | bool]


def make_inputs(case: Case, seed: int, device: str = "cuda") -> Values:
    """Generate fresh tensors and a causal mask for a public specification.

    Args:
        case: Validated dimensions and a per-case seed offset.
        seed: Sweep seed combined with the case offset to populate values.
        device: Torch device; CPU debugging uses FP32, CUDA uses BF16.

    Returns:
        Q, K and V in [batch, heads, length, head_dim] order and a Boolean
        lower-triangular mask. No values are retained between calls.
    """
    shape = (case["batch"], case["heads"], case["length"], case["head_dim"])
    generator = torch.Generator(device=device).manual_seed((seed * 65537 + case["seed"]) % 2**63)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    values: Values = {key: torch.randn(shape, generator=generator, device=device, dtype=dtype) * .3
                      for key in ("q", "k", "v")}
    values["mask"] = torch.ones(case["length"], case["length"], device=device, dtype=torch.bool).tril()
    return values


def inputs(kind: str, seed: int, index: int, device: str = "cpu", smoke: bool = False) -> Values:
    """Construct an example using the operation runtime's existing interface.

    Args:
        kind: Must be op01, the causal-attention operation.
        seed: Controls tensor contents without changing dimensions.
        index: Index of a public example, from zero through 99.
        device: Torch allocation device.
        smoke: Retains the small debugging shape for existing smoke callers.

    Returns:
        Input dictionary for the attention reference or submitted run function.

    Raises:
        ValueError: The operation or example index is unsupported.
    """
    if kind != "op01" or not 0 <= index < 100:
        raise ValueError("Expected op01 and an example index from 0 to 99")
    case = default_cases()[index]
    if smoke:
        case = Case(batch=1, heads=2, length=24, head_dim=16, seed=index)
    return make_inputs(case, seed, device)
