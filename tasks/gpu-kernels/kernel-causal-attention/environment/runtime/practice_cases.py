"""Public causal-attention case specifications and reproducible size coverage."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict


class Case(TypedDict):
    """One BF16 causal-attention input shape and a tensor-value seed offset."""

    batch: int
    heads: int
    length: int
    head_dim: int
    seed: int


def validate_cases(raw: object) -> list[Case]:
    """Validate a JSON case list against the public attention domain.

    Args:
        raw: Decoded list of dimension dictionaries, optionally containing seed.

    Returns:
        Between one and 100 complete specifications with integer seed offsets.

    Raises:
        ValueError: A field, type, count or dimension is outside the contract.
    """
    if not isinstance(raw, list) or not 1 <= len(raw) <= 100:
        raise ValueError("Provide a JSON list of 1 to 100 cases")
    result: list[Case] = []
    required = {"batch", "heads", "length", "head_dim"}
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not required <= item.keys() or item.keys() - required - {"seed"}:
            raise ValueError(f"Case {index}: expected batch, heads, length, head_dim and optional seed")
        values = dict(item, seed=item.get("seed", index))
        if any(type(value) is not int for value in values.values()):
            raise ValueError(f"Case {index}: fields must be integers")
        if (not 1 <= values["batch"] <= 4 or values["heads"] not in (8, 16)
                or not 256 <= values["length"] <= 8192 or values["head_dim"] not in (64, 128)
                or not 0 <= values["seed"] < 2**63):
            raise ValueError(f"Case {index}: dimensions or seed outside the supported range")
        result.append(Case(**values))
    return result


def read_cases(path: Path) -> list[Case]:
    """Read a bounded custom-case file without importing executable user code.

    Args:
        path: JSON file containing shape dictionaries.

    Returns:
        Validated specifications.

    Raises:
        ValueError: The file is too large or its specifications are invalid.
    """
    if path.stat().st_size > 65536:
        raise ValueError("Case file exceeds 64 KiB")
    return validate_cases(json.loads(path.read_text()))


def default_cases() -> list[Case]:
    """Return 100 public examples covering boundaries and partial sequence tiles.

    Returns:
        A deterministic shape schedule covering all stated batch, head and width
        values. Most examples are smaller so a complete sweep stays affordable;
        the largest shape and nonaligned long sequences are explicit examples.
    """
    boundary = [(1, 8, 256, 64), (4, 16, 8192, 128), (1, 8, 8191, 64),
                (3, 16, 4095, 128), (2, 8, 6145, 64), (4, 16, 257, 64)]
    cases: list[Case] = []
    for index in range(100):
        rng = random.Random(73129 + index)
        b, h, t, d = (boundary[index] if index < len(boundary) else
                      (1 + index % 4, (8, 16)[index // 4 % 2],
                       rng.choice((384, 513, 768, 1023, 1537, 2049, 3072)),
                       (64, 128)[index // 8 % 2]))
        cases.append(Case(batch=b, heads=h, length=t, head_dim=d, seed=index))
    return validate_cases(cases)
