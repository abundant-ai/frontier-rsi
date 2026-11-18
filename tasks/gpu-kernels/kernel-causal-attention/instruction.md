# Causal attention kernel for both the forward and backward passes

Your goal is to implement a fast causal attention kernel for both the forward and backward passes on an NVIDIA H100. Make the implementation run as fast as possible. Preserve numerical correctness throughout the supported input range.

Implement `run(inputs)` in `/app/submission/candidate.py` and keep all helpers in that file. The input dictionary contains `q`, `k` and `v` tensors with shape `[batch, heads, sequence_length, head_dimension]` and a Boolean causal `mask`. Return the attention output with the same shape and dtype as Q. The output must support backpropagation through Q, K and V; do not return gradients separately. The reference operation is in `/opt/runtime/kernel_ops.py`.

Support batch sizes from 1 to 4, heads 8 or 16, sequence length 256 to 8,192, and head dimension 64 or 128. External tensors use BF16. Outputs and gradients must have NRMSE at most 0.01.

Return finite CUDA tensors with the specified shapes and dtypes. Keep peak GPU allocation within 32 GiB for the entire sweep. NRMSE is the norm of the output error divided by the reference norm, with a denominator floor of 1e-8.

Use the development tool below to check correctness and measure latency. It warms up the kernel and times a CUDA Graph replay of the full sweep. The reported latency includes GPU input copies, the attention forward pass, the scalar-loss calculation used to drive backpropagation, and the backward pass. Compilation, warmup and CPU–GPU transfers are outside the timed replay. Keep inputs and outputs on the GPU throughout the replay and compute each result from the current inputs. Use the tool's sweep timing when comparing implementations; timing your kernel alone measures less work.

You have two hours on one H100 80 GB GPU. Your implementation uses the restricted Python subset defined in `/opt/runtime/restricted.py`. Check its allowed syntax and APIs before implementing helpers. The runtime files are read-only. The kernel runs without file, process, reflection or timer access. The workspace has no internet or external model API access.

To measure a sweep, run:

```bash
python3 /opt/runtime/develop.py --seconds 600 --seed 101 --count 100 --out /app/runs/experiment-01
```

The 100 built-in examples cover the stated size boundaries, including sequence lengths that leave a partial kernel tile. To check your own sizes, save a JSON list such as this to `/app/submission/cases.json`:

```json
[
  {"batch": 1, "heads": 8, "length": 257, "head_dim": 64, "seed": 17},
  {"batch": 4, "heads": 16, "length": 8191, "head_dim": 128, "seed": 29}
]
```

Each case's optional `seed` is combined with the command's `--seed` to generate tensor values. The tool creates Q, K, V and the causal mask. Run an untimed correctness check first:

```bash
python3 /opt/runtime/develop.py --cases /app/submission/cases.json --correctness-only --seconds 120 --out /app/runs/custom-check
```

This command reports NRMSE for the output and each input gradient in `correctness.json`. It does not measure whole-sweep memory use or latency. Omit `--correctness-only` to time the supplied cases together using the same warmup and CUDA Graph replay protocol. Custom lists may contain 1 to 100 cases. Use 100 inputs for a full sweep. Compare implementations on the same input shapes and sizes. Keep the complete timed sweep within the stated memory budget.

Use a fresh output directory for each run. The `--seconds` budget covers compilation, preparation, execution and checking together. If it expires, inspect `summary.json`, `worker.log` and `check.log`, then retry with fewer cases or a larger budget. You can select a smaller built-in subset with `--count` or specific cases with `--indices`.
