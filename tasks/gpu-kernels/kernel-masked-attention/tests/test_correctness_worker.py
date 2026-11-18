"""Save untimed candidate outputs for a separate trusted numerical checker."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from test_correctness_cases import CASE_COUNT, inputs
from test_kernel_ops import output_and_gradients
from test_restricted import load
from test_sweep_protocol import read_contract


def main() -> None:
    """Run the frozen submission on fresh structured inputs under its normal UID."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract = read_contract(args.config)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(
        min(1., contract["memory_gib"] * 1024**3 / properties.total_memory))
    fn = load(args.candidate, ["run"]).run
    for index in range(CASE_COUNT):
        values = inputs(contract["operation"], index)
        actual = output_and_gradients(fn, values, contract["gradients"])
        if any(type(value) is not torch.Tensor or value.device.type != "cuda" for value in actual):
            raise ValueError("Outputs must be ordinary CUDA tensors")
        save_file({f"output_{i}": value.detach().cpu().contiguous().clone()
                   for i, value in enumerate(actual)}, str(args.out / f"case-{index}.safetensors"))
        del values, actual
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"event": "correctness_case_saved", "index": index}), flush=True)
    (args.out / "measurements.json").write_text(json.dumps({"case_count": CASE_COUNT, "untimed": True}))


if __name__ == "__main__":
    main()
