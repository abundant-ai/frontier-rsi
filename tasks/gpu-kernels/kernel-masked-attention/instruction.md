# Block-masked attention kernel for both the forward and backward passes

Your goal is to implement a fast block-masked attention kernel for both the forward and backward passes on an NVIDIA H100. Make the implementation run as fast as possible. Preserve numerical correctness throughout the supported input range.

Implement `run(inputs)` in `/app/submission/candidate.py` and keep all helpers in that file. Read Q, K and V from `q`, `k` and `v`, with shape `[batch, heads, sequence_length, head_dimension]`, and the Boolean block mask from `mask`. Return the attention output with the same shape and dtype as Q. The output must support backpropagation through Q, K and V; do not return gradients separately. The reference operation is in `/opt/runtime/kernel_ops.py`.

Support sequence lengths from 256 to 8,192, mask block widths of 64 or 128, and target mask densities from 10% to 40%. Diagonal blocks are always present, so the smallest masks can have 50% density. Block width varies separately from head dimension. External tensors use BF16. Outputs and gradients must have NRMSE at most 0.01.

Return finite CUDA tensors with the specified shapes and dtypes. Keep peak GPU allocation within 32 GiB for the entire sweep. NRMSE is the norm of the output error divided by the reference norm, with a denominator floor of 1e-8.

Use the development tool below to check correctness and measure latency. It warms up the kernel and times a CUDA Graph replay of the full sweep. The reported latency includes GPU input copies, forward computation, the scalar loss used to drive backpropagation, and backward computation. Compilation, warmup and CPU–GPU transfers are outside the timed replay. Keep inputs and outputs on the GPU throughout the replay and compute each result from the current inputs. Use the tool's sweep timing when comparing implementations.

You have two hours on one H100 80 GB GPU. Your implementation uses the restricted Python subset defined in `/opt/runtime/restricted.py`. Check its allowed syntax and APIs before implementing helpers. The runtime files are read-only. The kernel runs without file, process, reflection or timer access. The workspace has no internet or external model API access.

To measure a sweep, run:

```bash
python3 /opt/runtime/develop.py --seconds 600 --seed 101 --count 100 --out /app/runs/experiment-01
```

The built-in examples cover the stated size boundaries. To check your own inputs, save a JSON list to `/app/submission/cases.json`:

```json
[
  {
    "batch": 1,
    "heads": 8,
    "length": 256,
    "head_dim": 64,
    "block_size": 64,
    "density": 0.1,
    "seed": 0
  }
]
```

Use `batch`, `heads`, `length`, `head_dim`, `block_size` and `density`. The tool creates a block mask with diagonal blocks present and crops its last blocks to the requested sequence length. Each case accepts an optional `seed` offset, combined with the command's `--seed` to generate values.

Run an untimed correctness check first:

```bash
python3 /opt/runtime/develop.py --cases /app/submission/cases.json --correctness-only --seconds 120 --out /app/runs/custom-check
```

This command reports numerical results in `correctness.json`. It processes cases individually and does not measure whole-sweep memory use or latency. Omit `--correctness-only` to time the supplied cases together using the same warmup and CUDA Graph replay protocol. Custom lists may contain 1 to 100 cases. Use 100 inputs for a full sweep. Compare implementations on the same input shapes and sizes, and keep the complete timed sweep within the stated memory budget.

Use a fresh output directory for each run. The `--seconds` budget covers compilation, preparation, execution and checking together. Inspect `summary.json`, `worker.log` and `check.log` for results or errors. If the budget expires, retry with fewer cases or a larger budget. Select a smaller built-in subset with `--count` or specific examples with `--indices`.
