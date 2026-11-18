# posttrain-kit (RL)

GRPO with multi-turn rollouts on the `multifile_repair` environment, on one GPU. `/app/train_rl.py` (the run that
grades your hand-in) is a fixed wrapper around `grpo.py`; this directory is what it runs. Stack: vLLM 0.29,
transformers 5.19, TRL 1.15. Run the scripts from this directory (or put it on `PYTHONPATH`).

| File | What it is |
|---|---|
| `grpo.py` | GRPO trainer: every optimizer step plays `--episodes-per-step` episodes (`num_generations` per project) with vLLM on the same GPU, rewards them with the environment, and takes one policy-gradient step. Hooks: a schedule (what to train on next) and an algo (GRPOConfig settings, optionally the advantages or the loss). Read its docstring. |
| `rollouts/multiturn.py` | The episode loop for training and evaluation (`run_episodes`): the model replies, the environment answers as the next user turn, until it ends the episode or a limit is reached (64 replies, 4096 tokens per reply, 32768 per episode). The environment's tokens are masked out of the loss. A row's `train_prompt`, when given, replaces the environment's first message. |
| `envs/multifile_repair.py` | The environment (`Env(params)`: `reset()`, `step(text)`), with its engine in `rl_evalgen/` (`project_env.py`: tools, prompt, grading; `sandbox.py`: where the project's code runs). |
| `envs/evaluate_env.py` | Greedy evaluation of a model on episode rows: `python3 -m envs.evaluate_env --model <dir> --rows <dir of env_*.jsonl> --out r.json`. |
| `chat.py`, `accel.py` | Chat-template details (rendering, stop tokens, turn suffixes) and speed-ups (FlashAttention 3 on the H100, Liger kernels). |

## How train_rl.py uses grpo.py

`train_rl.py` checks your pool and config, writes the pool in grpo.py's format (`<dir>/env_multifile_repair.jsonl`,
each row's `difficulty` field as its arm), an algo with your settings and DAPO-style fixed ones (loss `dapo`, clip
0.2 / 0.28, no KL penalty, truncated episodes masked out, no std scaling, completions up to 24576 tokens), and a
schedule that draws arms by your stage weights (uniform over rows without a schedule). It then runs:

```bash
python3 grpo.py --model /opt/base --pool <dir> --out <model> --minutes <cap> --algo algo.py --schedule schedule.py \
    --rollout rollouts/multiturn.py --lora 64 --max-steps <max_steps> --episodes-per-step <n> --micro-batch 1 --seed 0
```

and leaves the checked pool, `algo.py`, `schedule.py` and the logs in `<model>_work/`. `<model>_work/log/steps.jsonl`
has one line per step: elapsed time, mean reward, and the shares of groups that were all-correct, all-wrong or mixed.
Only mixed groups (some episodes solved, some not) give a gradient.

You may run `grpo.py` yourself with other settings, schedules or algos to explore, and change any file here; only
`/app/train_rl.py`'s fixed run on your hand-in counts, with its own unchanged copy of this kit.

## Memory (one 80 GB GPU)

grpo.py keeps vLLM at 0.35 of the GPU (28 GB: bf16 weights plus KV cache) next to the policy. With `--lora 64` and
an fp32 base, a 4B model takes about 16 GB of weights plus 1 GB of LoRA weights, gradients and AdamW state; the
activations of a micro-batch (gradient checkpointing) come on top, which is why long multi-turn episodes use
`--micro-batch 1` (two episodes near the 32768-token cap do not fit in one pass); the fixed run also sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, so that memory PyTorch has reserved can be reused.

## Conventions

- Prompts are rendered with the model's own chat template (`chat.render`). Generation stops at the template's
  end-of-turn token or an eos (`chat.stop_ids`). The text that counts is the reply after its last `</think>`, if any
  (`chat.final_text`).
- The environment's reply is a new user turn (`chat.turn_suffix_ids`: the tokens the template puts between two
  replies, never re-encoding earlier turns).
- vLLM starts workers with `spawn`: create a `vllm.LLM` under `if __name__ == "__main__":`. One GPU job at a time.
