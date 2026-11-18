"""Untimed masked-attention cases with partial final mask and compute blocks."""
from __future__ import annotations

import torch

CASE_SPECS = ((257, 64, 64), (383, 128, 128),
              (1025, 128, 64), (8191, 64, 128))
CASE_COUNT = len(CASE_SPECS)


def inputs(kind: str, index: int) -> dict[str, torch.Tensor]:
    """Build one reproducible input within the declared masked-attention range.

    Args:
        kind: Operation identifier, which must be masked attention (op06).
        index: Case selecting sequence length, mask width, and head dimension.

    Returns:
        Fresh BF16 CUDA Q/K/V tensors and a Boolean block mask. Diagonal
        blocks remain active, including the cropped final block, so every
        query has at least one allowed key. The checker regenerates these
        values independently of the candidate process.

    Raises:
        ValueError: The operation or index is unsupported.
    """
    if kind != "op06" or index not in range(CASE_COUNT):
        raise ValueError("Unknown masked-attention correctness case")
    length, block, dimension = CASE_SPECS[index]
    generator = torch.Generator(device="cuda").manual_seed(741923 + index)
    values = {key: torch.randn((1, 8, length, dimension), generator=generator,
                               device="cuda", dtype=torch.bfloat16) * .3
              for key in ("q", "k", "v")}
    count = (length + block - 1) // block
    diagonal = torch.eye(count, device="cuda", dtype=torch.bool)
    mask = diagonal.clone().reshape(-1)
    off_diagonal = torch.arange(count * count, device="cuda").reshape(count, count)[~diagonal]
    active = max(count, round(.2 * count * count))
    chosen = off_diagonal[torch.randperm(off_diagonal.numel(), generator=generator,
                                        device="cuda")[:active - count]]
    mask[chosen] = True
    values["mask"] = mask.reshape(count, count).repeat_interleave(block, 0).repeat_interleave(block, 1)[:length, :length].contiguous()
    return values
