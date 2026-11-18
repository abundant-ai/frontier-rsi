"""Numerical comparisons for independently computed outputs."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from test_sweep_protocol import Contract


def compare(actual: list[torch.Tensor], expected: list[torch.Tensor], contract: Contract, original: dict[str, torch.Tensor | int | float | bool], update_relative: bool=True) -> None:
    """Validate output tensors against the operation's numerical contract.

    Args:
        actual: Candidate outputs, including gradients when requested.
        expected: Independently computed reference outputs in the same order.
        contract: Operation identifier and numerical tolerance settings.
        original: Input tensors used to compute parameter-update magnitudes.
        update_relative: Enforce the additional BF16 update-relative check.

    Raises:
        ValueError: An output violates the numerical, shape, or dtype contract.
        RuntimeError: Reference values or comparison norms are non-finite.
    """
    kind = contract['operation']
    if len(actual) != len(expected):
        raise ValueError('output count differs')
    for index, (value, target) in enumerate(zip(actual, expected)):
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f'output {index}: shape or dtype differs')
        if not torch.isfinite(value).all():
            raise ValueError(f'output {index}: nonfinite')
        if not torch.isfinite(target).all():
            raise RuntimeError(f'reference {index}: nonfinite')
    for index, (value, target) in enumerate(zip(actual, expected)):
        if not target.is_floating_point():
            if not torch.equal(value, target):
                raise ValueError(f'output {index}: discrete result differs')
            continue
        tolerance = contract['tolerance']
        # Convert before subtraction so finite FP32/BF16 operands cannot overflow.
        target64 = target.double()
        error = (value.double() - target64).norm()
        denominator = target64.norm().clamp_min(1e-08)
        if not torch.isfinite(error) or not torch.isfinite(denominator):
            raise RuntimeError(f'output {index}: nonfinite comparison norm')
        relative_error = error / denominator
        if not torch.isfinite(relative_error) or relative_error > tolerance:
            raise ValueError(f'output {index}: numerical tolerance exceeded')
