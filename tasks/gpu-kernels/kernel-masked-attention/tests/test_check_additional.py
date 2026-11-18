"""Check new untimed cases without loading any submitted executable code."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import torch
from safetensors.torch import load_file

from test_additional_cases import LABELS, Values, advance, evaluate, inputs
from test_graph_checks import compare
from test_kernel_ops import reference
from test_sweep_protocol import Contract, read_contract


def check_case(saved: dict[str, torch.Tensor], contract: Contract, index: int) -> None:
    """Require all saved outputs to satisfy the unchanged numerical contract.

    Args:
        saved: Root-owned tensors written by the exited numerical worker.
        contract: Trusted operation and tolerance configuration.
        index: Additional-case identity.

    Raises:
        ValueError: An output is missing, extra or numerically incorrect.
    """
    kind = contract['operation']
    label = LABELS[kind][index]
    state = inputs(kind, index)
    local_state = inputs(kind, index) if kind == 'op10' else state

    def numerical_reference(values: Values) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]:
        """Compute independent reference outputs for the current operation.

        Args:
            values: Original inputs or trusted reconstructed optimizer state.

        Returns:
            Mathematical reference tensors, without importing the oracle.
        """
        return reference(kind, values)

    consumed: set[str] = set()
    for step in range(100 if kind == 'op10' else 1):
        expected = evaluate(numerical_reference, state, label)
        keys = [f'step_{step}_output_{j}' for j in range(len(expected))]
        if any(key not in saved for key in keys): raise ValueError('Missing additional output')
        actual = [saved[key] for key in keys]
        consumed.update(keys)
        compare(actual, expected, contract, state, update_relative=kind != 'op10')
        if kind == 'op10':
            local_expected = evaluate(numerical_reference, local_state, label)
            compare(actual, local_expected, contract, local_state)
            local_state = advance(local_state, actual)
            state = advance(state, expected)
    if set(saved) != consumed: raise ValueError('Unexpected additional outputs')


def main() -> None:
    """Rebuild every added input and record complete untimed coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    contract = read_contract(args.config)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    labels = LABELS[contract['operation']]
    for index, label in enumerate(labels):
        saved = load_file(str(args.directory/f'case-{index}.safetensors'), device='cuda')
        check_case(saved, contract, index)
        del saved
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({'additional_case_verified': label}), flush=True)
    args.out.write_text(json.dumps({'labels': labels, 'case_count': len(labels),
                                    'correctness_passed': True, 'untimed': True}))


if __name__ == '__main__':
    main()
