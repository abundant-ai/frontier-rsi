# Design the RL training data for a small coding agent

`Qwen/Qwen3-4B-Instruct-2507` (Qwen, 4.0B parameters; in `/opt/base`, read-only) will be trained with reinforcement
learning to fix the bugs in small multi-file Python projects, working with tools. You do not train the final model
yourself: you hand in **training projects and a few settings** in `/app/rl_pool/`. After your session, the original
model is trained on them with a fixed GRPO run of at most 2 hours on one H100. Your goal is a hand-in that teaches
the original model to solve as many new projects as possible.

## The projects

A project is a folder of 4 to 7 Python files, about 100 to 280 lines in all, with no subfolders:

- modules of functions, each with a docstring that says what it must return, the type of the result, and which
  inputs are valid;
- `api.py`, which imports the functions from their modules;
- `run_tests.py`, a few tests of the functions through `api.py`: it prints a `FAIL` line for each failing test,
  `k/n tests passed`, and the traceback of the first error.

Some functions have a bug: a small change (a comparison, an index, an operator, a variable, a condition, a start
value) makes them differ from their docstring. Each buggy function has one bug, and buggy functions are in different
modules where possible. Projects come at four difficulty levels, in equal shares:

| Level | Functions / modules | Buggy functions | `run_tests.py` shows | Tool calls |
|---|---|---|---|---|
| `easy` | 3 / 2 | 1 | the bug | 20 |
| `medium` | 4 / 3 | 1 or 2 | every bug | 25 |
| `hard` | 6 / 4 | 2 | the first bug only | 30 |
| `expert` | 8 / 5 | 2 or 3, most inputs do not show them | no bug | 40 |

`/app/data/samples.jsonl` holds 40 sample projects (10 per level), one JSON object per line: `id`, `difficulty`,
`prompt` (exactly the model's first message), and `params`: `files` (the project as the model first sees it), `hidden`
(for each function: its module, the hidden test inputs and the expected values, encoded as
`posttrain-kit/rl_evalgen/sandbox.py`'s `encode` writes them), `max_calls` and `timeout_s`. The new projects come from
the same distribution: the same kinds of functions, bugs and files, at the same levels, in equal shares. None of them
is among the samples.

## How the model works on a project, and how a project is graded

- Each project is one episode of the multi-turn loop in `/app/posttrain-kit/rollouts/multiturn.py`
  (`run_episodes`) with the environment `envs/multifile_repair.py`, whose engine is `rl_evalgen/project_env.py` and
  `rl_evalgen/sandbox.py`. The first message is the project's prompt, one user message in the chat template
  `/opt/base/chat_template.jinja`. The environment runs one tool call per reply and sends its output back as the
  next user message.
- When measured, the model replies greedily (temperature 0), with at most 4096 tokens per reply, at most 64 replies,
  and at most 32768 tokens for the prompt and the whole episode together.
- The tools, as the prompt describes them: `ls()`, `view("name.py")`, `write("name.py")` followed by the whole new
  file in a ```python block, `run_python("name.py")` and `submit()`. The first tool call in a reply on a line of its
  own runs; every reply uses one tool call of the project's budget, a reply without a valid call included.
  `run_python` shows what the program printed (cut to 6000 characters), the traceback with the project's files only,
  and the exit code; a run is stopped after `timeout_s` (10) seconds.
- The project is graded when the model calls `submit()` or its tool calls run out: every function of the project is
  called, from the module that defines it, on its hidden test inputs, and must return the expected value on all of
  them (numbers by value, floats within a relative 1e-9; everything else exactly: a list is not a tuple, and `True` is
  not `1`). A crash, a timeout or a value that is not plain data is wrong. Editing `run_tests.py` or `api.py` changes
  nothing. A project is solved when every function passes; an episode that ends any other way is not solved.
- The project's code runs in a sandbox: an unprivileged user, no network, limits on CPU time, memory (1 GB) and
  processes. It may import only the project's own files and `bisect`, `collections`, `copy`, `dataclasses`,
  `datetime`, `decimal`, `fractions`, `functools`, `heapq`, `itertools`, `math`, `operator`, `re`, `statistics`,
  `string` and `typing`; file, network and process access, `open`, `eval`, `exec`, `getattr`, and dunder and frame
  attributes are refused.

## What you hand in: `/app/rl_pool/`

**`pool.jsonl`**: your training projects, one JSON object per line, in the sample format:

- `params`: `files`, `hidden`, `max_calls` (1 to 64), `timeout_s` (1 to 10), and optionally `graded: "api"` (the
  prompt then says that only the functions `api.py` imports are graded; it changes only the prompt, so list only
  those functions in `hidden`; the graded projects never use it). No fix is needed:
  the reward only checks the hidden tests.
- `prompt` (optional): the first message of the episode, instead of the environment's instructions. It may say
  anything; the graded projects always get the standard prompt.
- `difficulty` (optional): a label of at most 40 characters, used by the schedule (default `"all"`).

Each row is checked and dropped if bad: at most 10000 rows; 1 to 40 files with flat `.py` names, each at most 30000
characters, that the sandbox accepts; 1 to 60 hidden functions, each of a module in `files`, with 1 to 100 inputs
and one expected value each; and the hidden tests must run on the files as given within the grading time limit
(whether they pass does not matter). A row whose first message is longer than 8192 tokens is not used.
`pool.jsonl` and `config.json` may each be at most 4 GB; a larger file scores 0.

**`config.json`** (optional, plain JSON): the training settings. A value outside its range is replaced by its default.

| Setting | Range | Default |
|---|---|---|
| `max_steps` | an int >= 1, or `null` | `null`: no step limit, the 2-hour cap stops training |
| `episodes_per_step` | 32 to 256, a multiple of 16 | 64 |
| `num_generations` | 4, 8 or 16 (attempts per project in a step) | 8 |
| `temperature` | 0.6 to 1.2 | 1.0 |
| `learning_rate` | 1e-6 to 1e-4 | 2e-5 |
| `schedule` | `{"stages": [{"from_step": k, "weights": {label: weight, ...}}, ...]}` | none: every row equally likely |

With a schedule, from optimizer step `k` on, the projects of each step are drawn from the labels with the stage's
weights (before the first stage, every row is equally likely). Each step trains on `episodes_per_step /
num_generations` projects, each played `num_generations` times.

Fixed by the task: GRPO with the DAPO loss (token-level, clipping 0.2 low and 0.28 high, no KL penalty, truncated
episodes masked out, no reward scaling by standard deviation); a LoRA of rank 64 (alpha 128) on every projection,
base weights in fp32, merged into the model at the end; the episode loop and limits above, sampled at your
temperature; reward 1 for a solved project, else 0; training from `/opt/base`, stopped at `max_steps` or after 120
minutes of wall-clock time, whichever comes first. If `/app/rl_pool/pool.jsonl` is missing or has no usable row, the
original model is measured without training. `/app/rl_pool/` must not change after your session ends.

## Tools

- `python3 /app/train_rl.py --pool /app/rl_pool --out <dir>` is **exactly the training run your pool gets** (`--minutes N`
  sets a shorter cap for a trial run; `--check` only checks the pool and the config and reports every dropped row and
  replaced setting). It saves the trained model in `<dir>` and its logs in `<dir>_work/log/` (`steps.jsonl`: one line
  per step with the reward and the shares of groups that were all solved, all unsolved or mixed; only mixed groups
  teach the model anything).
- `python3 /app/score.py --model <dir>` measures a model directory on `/app/data/samples.jsonl` as the final
  measurement does (`--items` another file of projects in the same format, `--limit N`, `--out rows.jsonl` every
  episode; `--model /opt/base` measures the original model).
- `/app/posttrain-kit` is the RL kit `train_rl.py` runs (see its `README.md`): `grpo.py`, the multi-turn rollout, the
  environment and its engine. You may run and change your copy; the training run uses its own unchanged copy.
- Python 3 with `torch`, `transformers`, `vllm`, `trl`, `peft`, `accelerate`, `datasets` and `numpy` is installed.
- You have one GPU, an NVIDIA H100 with 80 GB of memory, 16 CPU cores and 128 GB of memory. While you work, the only
  network hosts you can reach are the model APIs that agent tools use; there is no other internet access, and there
  is no other model in the container.

## Rules

- Only `/app/rl_pool/pool.jsonl` and `/app/rl_pool/config.json` are kept when your session ends; anything you create
  or change elsewhere is removed.

You have 8 hours. Run `bash /app/timer.sh` to see how much time is left.
