"""Save untimed submission outputs for an independent trusted checker."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

from test_additional_cases import LABELS, advance, evaluate, inputs
from test_sweep_protocol import read_contract
from test_additional_entrypoint import load_kernel


def main() -> None:
    """Execute every additional case under the numerical runner's permissions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    contract = read_contract(args.config)
    kind = contract['operation']
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., contract['memory_gib']*1024**3/total))
    fn = load_kernel(args.candidate, args.plan)
    for index, label in enumerate(LABELS[kind]):
        state = inputs(kind, index)
        saved: dict[str, torch.Tensor] = {}
        for step in range(100 if kind == 'op10' else 1):
            actual = evaluate(fn, state, label)
            if any(value.device.type != 'cuda' for value in actual):
                raise ValueError('Outputs must remain CUDA tensors')
            for j, value in enumerate(actual):
                saved[f'step_{step}_output_{j}'] = value.detach().cpu().contiguous().clone()
            if kind == 'op10': state = advance(state, actual)
        save_file(saved, str(args.out/f'case-{index}.safetensors'))
        del state, saved, actual
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({'additional_case_saved': label}), flush=True)
    (args.out/'measurements.json').write_text(json.dumps({'labels': LABELS[kind], 'untimed': True}))


if __name__ == '__main__':
    main()
