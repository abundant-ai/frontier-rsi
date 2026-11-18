#!/usr/bin/env python3
"""Fixed GRPO trainer over evalgen environments, with two hooks: what to train on next (the schedule), and how
rewards become an update (the algorithm).

  python3 grpo.py --model <dir> --pool <dir> --out <dir> --minutes 120 --algo algo.py
      (--schedule schedule.py | --schedule-cmd "<command that runs sched_worker.py>")
      [--guard-algo] [--token-cap N] [--log <dir>] [--seed 0] [--bf16] [--lora R]
      [--max-steps N] [--save-every N] [--stop-if-no-signal] [--attn auto|sdpa|fa2|fa3] [--no-liger]

<pool>/<env>.jsonl holds training rows (evalgen rows: prompt, answer, eval, subtask, difficulty, meta). Every
optimizer step generates --episodes-per-step completions (default 64: num_generations per prompt, 64 /
num_generations prompts) with vLLM on the same GPU, grades each with its env's evalgen checker (reward 1 or 0), and takes one policy-gradient step.
Training stops when the wall-clock budget runs out (5 minutes are kept to save) or, with --token-cap, once that many
completion tokens have been generated, or after --max-steps optimizer steps. The final policy is saved to --out;
<log>/summary.json records why it stopped. --save-every N also saves the policy (same format) to <out>_step<k> every N
steps.

Watching and stopping a run: every step prints one line (elapsed / remaining minutes of --minutes, reward, and the
share of groups that were all-correct / all-wrong / mixed) and appends it to <log>/steps.jsonl; TRL's progress bar is
off (its total is --max-steps, not the clock). The process id is in <log>/pid: `kill $(cat <out>_log/pid)` stops after
the current step and saves the policy as usual (a second kill stops at once, without saving). Do not use
`pkill -f grpo.py`: the pattern can match your own shell's command line and kill it.

Learning signal: only groups that are mixed (some completions correct, reward > 0.5, and some not) give a gradient.
From step 3 on, if more than 80% of a step's groups were all-correct (or all-wrong), a NO LEARNING SIGNAL warning is
printed once; --stop-if-no-signal stops training (and saves) once that has held for 3 consecutive steps.

Memory: by default the policy trains in fp32 with AdamW (16 bytes per parameter, about 64 GB for a 4B model) next to
vLLM's 0.35 of the GPU, which does not fit a 4B model on one 80 GB GPU. For larger models:
  --bf16    load the policy in bf16 (GRPOConfig model_init_kwargs dtype): half the weight, gradient and AdamW memory.
            Pure-bf16 AdamW can round away updates much smaller than a weight (bf16 keeps about 3 significant digits),
            so at small learning rates part of the update is lost.
  --lora R  train a rank-R LoRA (alpha 2R) on every projection of the language model (not a multimodal checkpoint's
            vision tower) instead of the weights: only the adapter has
            gradients and AdamW state. The base weights stay fp32 unless --bf16 is also given (TRL merges the adapter
            into them and unmerges it again at every vLLM weight sync, and in bf16 that round trip drifts the weights).
            vLLM gets the merged weights; at the end the adapter is merged and the full model saved, as without --lora.
            A LoRA update moves the model less per step than a full one: consider a higher learning_rate in the algo.
  A 7-8B model needs --lora 64 (fp32 base, about 63 GB with vLLM) or --lora 64 --bf16; see README.md, Memory.

Speed (accel.py; logged at start): the policy's attention runs on FlashAttention 3 on a Hopper GPU, FlashAttention 2 on
other sm80+ GPUs, else SDPA (--attn overrides auto; vLLM picks its own kernels). TRL's Liger support (use_liger_kernel)
replaces RMSNorm / SwiGLU / GeGLU / RoPE with liger's fused kernels where liger supports the family (--no-liger: off).
TRL 1.15 has no Liger GRPO loss: the log-probs come from TRL's own chunked LM head with or without Liger, so an algo's
policy_loss(), --lora, --bf16 and the vLLM weight sync work the same either way.

Schedule:  next_batch(state, n) -> list of n (env, difficulty) pairs, called before every generation.
    state.step, state.elapsed_s, state.budget_s, state.envs ({env: [difficulties]}), state.rng (random.Random),
    state.history: every checked group so far, {"step", "env", "difficulty", "rewards": [0/1 per completion],
    "lengths": [completion tokens], "truncated": count}.
  --schedule loads it in this process; --schedule-cmd runs it in its own process (sched_worker.py), which only ever
  sees `state` and returns picks. A call that fails, times out (60 s) or returns an invalid list is replaced by
  uniformly random picks for that batch and counted in summary.json.
Algorithm:  algo.py
    CONFIG = {...}   GRPOConfig overrides from ALGO_LIMITS below (plus num_generations and max_completion_length)
    advantages(rewards, info) -> tensor        optional; rewards [n_groups, G] (0/1), returns the same shape
    policy_loss(logp, old_logp, adv, mask, info) -> scalar tensor   optional; logp, old_logp, mask are [B, T] over
                                               the completion tokens, adv is [B]; replaces TRL's loss
  --guard-algo refuses an algo.py that does anything but tensor math (see guard_algo).
"""
import argparse
import ast
import importlib.util
import json
import os
import random
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # the kit dir: chat.py
import accel  # noqa: E402
import chat  # noqa: E402

EPISODES_PER_STEP, MAX_PROMPT = 64, 8192   # completions (episodes) per optimizer step: --episodes-per-step
NO_SIGNAL_FRAC, NO_SIGNAL_AFTER, NO_SIGNAL_STEPS = 0.8, 3, 3   # > 80% of groups, warn from step 3, stop after 3 steps
LOSS_TYPES = {"grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo"}
ALGO_LIMITS = {   # key: (low, high) for numbers, or the set of allowed values
    "loss_type": LOSS_TYPES, "scale_rewards": {"group", "batch", "none"},
    "importance_sampling_level": {"token", "sequence"}, "mask_truncated_completions": {True, False},
    "vllm_importance_sampling_correction": {True, False},
    "lr_scheduler_type": {"constant", "constant_with_warmup", "linear", "cosine"},
    "num_generations": {4, 8, 16}, "max_completion_length": (256, 32768),
    "epsilon": (0.0, 1.0), "epsilon_high": (0.0, 10.0), "delta": (1.0, 10.0), "beta": (0.0, 1.0),
    "num_iterations": (1, 4), "top_entropy_quantile": (0.0, 1.0), "temperature": (0.3, 1.5), "top_p": (0.5, 1.0),
    "learning_rate": (1e-8, 1e-4), "max_grad_norm": (0.01, 10.0), "warmup_steps": (0, 100), "weight_decay": (0.0, 1.0),
    "sapo_temperature_neg": (0.01, 10.0), "sapo_temperature_pos": (0.01, 10.0),
}   # epsilon_high up to 10: with loss_type "cispo" it caps the importance weight itself (5 in algos/cispo.py)
GUARD_IMPORTS = {"torch", "torch.nn", "torch.nn.functional", "math", "numpy", "collections", "functools", "itertools",
                 "typing", "dataclasses"}
GUARD_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals",
               "vars", "input", "breakpoint", "help", "memoryview", "__builtins__", "importlib", "sys", "os"}
GUARD_ATTRS = {"load", "save", "from_file", "fromfile", "tofile", "memmap", "ctypeslib", "hub", "utils", "ops",
               "library", "classes", "cpp_extension", "jit", "compile", "package", "multiprocessing", "distributed",
               "backends", "storage", "untyped_storage", "frombuffer", "fx", "export", "lib", "testing", "environ",
               "system", "popen", "modules", "builtins", "set_default_device", "set_default_dtype", "manual_seed",
               "seed", "random_seed", "f_globals", "f_locals", "gi_frame", "cr_frame", "tb_frame"}


def guard_algo(path):
    """algo.py may do tensor math on what it is given and nothing else: imports from GUARD_IMPORTS only, no file,
    process or interpreter access, no private or dunder attributes."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [f"import {a.name}" for a in node.names if a.name not in GUARD_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in GUARD_IMPORTS:
                bad.append(f"from {node.module} import ...")
            bad += [f"from {node.module} import {x.name}" for x in node.names
                    if x.name == "*" or x.name.startswith("_") or x.name in GUARD_ATTRS]
        elif isinstance(node, ast.Name) and node.id in GUARD_NAMES:
            bad.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in GUARD_ATTRS):
            bad.append(f"attribute .{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("__"):
            bad.append(f"string {node.value[:20]!r}")
    if bad:
        sys.exit(f"algo.py refused by the guard (line-independent list): {sorted(set(bad))[:12]}")


def check_config(cfg):
    for k, v in cfg.items():
        lim = ALGO_LIMITS.get(k)
        if lim is None:
            sys.exit(f"algo.CONFIG key {k!r} is not tunable (allowed: {sorted(ALGO_LIMITS)})")
        if isinstance(lim, set):
            ok = v in lim and type(v) in {type(x) for x in lim}   # 1 is not True, 8.0 is not 8
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and lim[0] <= v <= lim[1]
        if not ok:
            sys.exit(f"algo.CONFIG[{k!r}] = {v!r} is outside {lim}")


_CHECKERS = {}


def _check_one(item):
    """One check, run in a worker process (a checker can hang inside C code, where no signal reaches it): the row's
    evalgen checker, or check(row, response) from a checker file you pass with --checker."""
    row, text, path = item
    try:
        if path:
            if path not in _CHECKERS:
                _CHECKERS[path] = load_module(path, "user_checker")
            return float(_CHECKERS[path].check(row, text))   # True/False -> 1/0; a float is partial credit
        from evalgen import evals as EV
        return 1.0 if EV.load(row["eval"]).check(row, text) else 0.0
    except Exception:
        return 0.0


def _warm(path):
    """Pool initializer: import the checker (and whatever it starts) before any check is timed. A failure here is left
    to _check_one, which scores 0; an initializer that raises would make the pool respawn its workers forever."""
    try:
        if path:
            _CHECKERS[path] = load_module(path, "user_checker")
        else:
            from evalgen import evals  # noqa: F401
    except Exception:
        pass


def _ready(_):
    return True


class Checker:
    """Grades completions in a pool of spawned workers. A check that runs past `timeout` seconds scores 0 and its
    worker pool is killed and rebuilt, so no single answer can stall training. Workers load the checker when they
    start, and a new pool is waited for before anything is timed: a checker that is slow to import (a sandbox, a big
    library) would otherwise time out the first batch of every pool and score nearly everything 0."""

    def __init__(self, workers=8, timeout=10.0, path=None):
        import multiprocessing as mp
        self.mp, self.workers, self.timeout, self.timeouts = mp, workers, timeout, 0
        self.path = str(Path(path).resolve()) if path else None
        self.ctx = mp.get_context("spawn")
        self.pool = self._new_pool()

    def _new_pool(self):
        pool = self.ctx.Pool(self.workers, initializer=_warm, initargs=(self.path,))
        pool.map(_ready, range(4 * self.workers), chunksize=1)   # untimed: every worker has started and loaded
        return pool

    def _reset(self):
        self.pool.terminate(); self.pool.join()
        self.pool = self._new_pool()

    def __call__(self, items):
        items = [(r, t, self.path) for r, t in items]
        pending = [self.pool.apply_async(_check_one, (it,)) for it in items]
        deadline = time.time() + self.timeout + 0.05 * len(items)
        out = []
        for r in pending:
            try:
                out.append(r.get(timeout=max(0.1, deadline - time.time())))
            except self.mp.TimeoutError:
                out.append(None)
        if any(o is None for o in out):   # something hung: rebuild, then re-check the unresolved ones one at a time
            self._reset()
            for i, o in enumerate(out):
                if o is None:
                    try:
                        out[i] = self.pool.apply_async(_check_one, (items[i],)).get(timeout=self.timeout)
                    except self.mp.TimeoutError:
                        out[i] = 0.0; self.timeouts += 1; self._reset()
        return out

    def close(self):
        self.pool.terminate()


def group_inputs(rows, group):
    """The trainer's inputs for one generation batch: each row's prompt, `group` times in a row. Every copy is its own
    dict: TRL writes a rollout's extra fields (env_reward, ...) into inputs[i] one completion at a time, so copies that
    share one dict would all end up with the group's last reward (zero variance, no gradient)."""
    return [{"prompt": [{"role": "user", "content": r["prompt"]}], "row": json.dumps(r)}
            for r in rows for _ in range(group)]



# Image-text checkpoints (Gemma 3, Gemma 4, Ministral 3) train as a whole but only their language model learns (the LoRA
# targets skip the vision and audio towers). transformers 5 names the vision weights differently from vLLM's loader (no
# `vision_model.` level), so pushing them to vLLM fails; they never change, so the weight sync leaves them out.
FROZEN_VISION = re.compile(r"(^|\.)(vision_tower|vision_model|vision_encoder|multi_modal_projector|audio_tower|"
                           r"embed_vision|embed_audio)\.")


def skip_frozen_vision_sync(trainer):
    vg = getattr(trainer, "vllm_generation", None)
    if vg is None or not hasattr(vg, "_iter_named_params") or not hasattr(trainer.model.config, "vision_config"):
        return
    every = vg._iter_named_params

    def language_only():
        for name, param in every():   # consumed to the end: TRL merges / unmerges a LoRA adapter around this loop
            if not FROZEN_VISION.search(name):
                yield name, param
    vg._iter_named_params = language_only

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_fractions(groups):
    """Shares of groups whose completions were all correct, all wrong, or mixed; a reward > 0.5 counts as correct."""
    n = {"all_correct": 0, "all_wrong": 0, "mixed": 0}
    for g in groups:
        ok = [r > 0.5 for r in g["rewards"]]
        n["all_correct" if all(ok) else "all_wrong" if not any(ok) else "mixed"] += 1
    return {k: round(v / max(1, len(groups)), 4) for k, v in n.items()}


NO_SIGNAL_ADVICE = {
    "all_correct": "this checkpoint already solves almost every problem it is given. Train on problems it solves only "
                   "some of the time, or write harder ones.",
    "all_wrong": "this checkpoint solves almost none of the problems it is given. Use easier problems.",
}


class SignalGuard:
    """Turns each step's group fractions into at most one warning and, with stop=True, a stop decision."""

    def __init__(self, stop=False):
        self.stop, self.streak, self.warned = stop, 0, False

    def update(self, step, fr):
        """-> (warning text or None, whether to stop training) after optimizer step `step` (1-based)."""
        kind = next((k for k in ("all_correct", "all_wrong") if fr[k] > NO_SIGNAL_FRAC), None)
        self.streak = self.streak + 1 if kind else 0
        msg = None
        if kind and step >= NO_SIGNAL_AFTER and not self.warned:
            self.warned = True
            msg = (f"WARNING: NO LEARNING SIGNAL at step {step}: {fr[kind]:.0%} of groups were {kind.replace('_', '-')}, "
                   f"so they give no gradient: {NO_SIGNAL_ADVICE[kind]}")
        return msg, self.stop and step >= NO_SIGNAL_AFTER and self.streak >= NO_SIGNAL_STEPS


class State:
    def __init__(self, envs, budget_s, seed):
        self.step, self.elapsed_s, self.budget_s = 0, 0.0, budget_s
        self.envs, self.history, self.rng = envs, [], random.Random(seed)


class RemoteSchedule:
    """The schedule in its own process: JSON lines over stdin/stdout, only `state` crosses."""

    def __init__(self, cmd, log):
        self.p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(log, "a"), text=True, bufsize=1)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.p.stdout, selectors.EVENT_READ)
        self.sent, self.first = 0, True

    def next_batch(self, state, n, timeout=60):
        if self.p.poll() is not None:
            raise RuntimeError(f"schedule process exited ({self.p.returncode})")
        msg = {"n": n, "step": state.step, "elapsed_s": state.elapsed_s, "budget_s": state.budget_s,
               "new_groups": state.history[self.sent:]}
        if self.first:
            msg["envs"], self.first = state.envs, False
        self.sent = len(state.history)
        self.p.stdin.write(json.dumps(msg) + "\n"); self.p.stdin.flush()
        if not self.sel.select(timeout):
            raise TimeoutError(f"schedule did not answer within {timeout} s")
        line = self.p.stdout.readline()
        return [tuple(x) for x in json.loads(line)["picks"]]

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def policy_options(bf16, lora, model=None):
    """GRPOConfig overrides and the peft_config (or None) for --bf16 / --lora R (on the language model only, for a
    multimodal `model` checkpoint)."""
    cfg = {"model_init_kwargs": {"dtype": "bfloat16" if bf16 else "float32"}}
    peft_config = None
    if lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=lora, lora_alpha=2 * lora, lora_dropout=0.0, task_type="CAUSAL_LM",
                                 target_modules=LORA_TARGETS,
                                 exclude_modules=chat.lora_exclude(model) if model else None)
    return cfg, peft_config


def save_policy(trainer, out, lora, final=True):
    """The full model as save_pretrained safetensors (a LoRA run merges its adapter first), .
    final=False (a checkpoint mid-run) merges the adapter in place, saves, and unmerges it, so training goes on."""
    if not lora:
        trainer.save_model(out)
        return
    model = trainer.accelerator.unwrap_model(trainer.model)
    if final:
        model.merge_and_unload().save_pretrained(out, safe_serialization=True)
        return
    model.merge_adapter()
    try:
        base = model.get_base_model()
        sd = {k.replace(".base_layer.", "."): v for k, v in base.state_dict().items() if "lora_" not in k}
        base.save_pretrained(out, state_dict=sd, safe_serialization=True)
    finally:
        model.unmerge_adapter()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--schedule", default=None)
    ap.add_argument("--schedule-cmd", default=None)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--guard-algo", action="store_true")
    ap.add_argument("--token-cap", type=float, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout", default=None, help="rollout.py defining rollout(prompts, rows, trainer, ctx); default: "
                    "TRL's own sampling (every completion at the config temperature)")
    ap.add_argument("--checker", default=None, help="a .py file with check(row, response) -> bool to grade with (default: "
                    "the row's evalgen checker)")
    ap.add_argument("--checker-workers", type=int, default=8, help="processes that run the checker; use 1 for a "
                    "checker that is not safe to run from several processes at once (e.g. one that starts its own sandbox)")
    ap.add_argument("--max-prompt", type=int, default=MAX_PROMPT, help="skip rows whose rendered prompt is longer")
    ap.add_argument("--reward", choices=["checker", "format"], default="checker",
                    help="training reward: the evalgen checker (default) or answer-line format only; a rollout that returns "
                         "'env_reward' supplies the reward itself")
    ap.add_argument("--bf16", action="store_true", help="load the policy in bf16 (half the memory); pure-bf16 AdamW can "
                    "round away very small updates")
    ap.add_argument("--lora", type=int, default=0, metavar="R", help="train a rank-R LoRA (alpha 2R, all linear "
                    "projections); the merged full model is saved")
    ap.add_argument("--episodes-per-step", type=int, default=EPISODES_PER_STEP, help="completions (episodes) per "
                    "optimizer step, 32 to 256: num_generations per prompt, episodes / num_generations prompts")
    ap.add_argument("--micro-batch", type=int, default=16, help="completions per forward/backward pass; the rest is "
                    "gradient accumulation (lower it for long multi-turn episodes)")
    ap.add_argument("--max-steps", type=int, default=5000, help="stop after this many optimizer steps")
    ap.add_argument("--save-every", type=int, default=0, metavar="N", help="also save the policy to <out>_step<k> every "
                    "N steps (0: only the final save)")
    ap.add_argument("--stop-if-no-signal", action="store_true", help="stop (and save) once more than 80%% of groups "
                    "were all-correct or all-wrong for 3 consecutive steps")
    ap.add_argument("--attn", default="auto", choices=list(accel.ATTN_CHOICES), help="the policy's attention kernel "
                    "(auto: FA3 on Hopper, FA2 on other sm80+ GPUs, else SDPA)")
    ap.add_argument("--no-liger", action="store_true", help="no Liger kernels in the policy")
    a = ap.parse_args(argv)
    if a.lora < 0:
        ap.error("--lora R needs R > 0 (0: no LoRA)")
    if a.max_steps < 1 or a.save_every < 0:
        ap.error("--max-steps needs N >= 1 and --save-every N >= 0")
    if not 32 <= a.episodes_per_step <= 256 or a.micro_batch < 1 or a.episodes_per_step % a.micro_batch:
        ap.error("--episodes-per-step must be 32 to 256 and a multiple of --micro-batch")
    return a


def main():
    a = parse_args()
    t_start = time.time()
    if bool(a.schedule) == bool(a.schedule_cmd):
        sys.exit("give exactly one of --schedule / --schedule-cmd")
    if a.guard_algo:
        guard_algo(a.algo)
    log_dir = Path(a.log or a.out + "_log")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "pid").write_text(f"{os.getpid()}\n")
    checker = Checker(workers=a.checker_workers, path=a.checker)   # before torch/vLLM start: spawned workers stay light

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    algo = load_module(a.algo, "algo")
    cfg_over = dict(getattr(algo, "CONFIG", {}))
    check_config(cfg_over)
    group = int(cfg_over.pop("num_generations", 8))
    max_completion = int(cfg_over.pop("max_completion_length", 1024))
    if a.episodes_per_step % group:
        sys.exit(f"--episodes-per-step {a.episodes_per_step} is not a multiple of num_generations {group}")
    n_prompts = a.episodes_per_step // group

    tok = AutoTokenizer.from_pretrained(a.model)
    render = lambda p: chat.render(tok, [{"role": "user", "content": p}])
    pool, rng = {}, random.Random(a.seed)
    for f in sorted(Path(a.pool).glob("*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                r.setdefault("eval", f.stem); r.setdefault("difficulty", "all")   # plain {prompt, answer} rows work too
                pool.setdefault((r["eval"], r["difficulty"]), []).append(r)
    for k, v in pool.items():
        keep = [r for r in v if len(tok(render(r["prompt"]), add_special_tokens=False)["input_ids"]) <= a.max_prompt]
        rng.shuffle(keep)
        pool[k] = keep
    pool = {k: v for k, v in pool.items() if v}
    if not pool:
        sys.exit(f"no pool row has a prompt of at most {a.max_prompt} tokens: nothing to train on")
    envs = {}
    for e, d in sorted(pool):
        envs.setdefault(e, []).append(d)
    cursor = {k: 0 for k in pool}
    state = State(envs, a.minutes * 60, a.seed)
    arms = sorted(pool)
    if a.schedule:
        local = load_module(a.schedule, "schedule")
        schedule = type("S", (), {"next_batch": staticmethod(lambda st, n: list(local.next_batch(st, n))),
                                  "close": staticmethod(lambda: None)})
    else:
        schedule = RemoteSchedule(a.schedule_cmd, log_dir / "schedule.log")
    stats = {"fallback_batches": 0, "fallback_errors": [], "gen_tokens": 0, "stop": "budget", "check_timeouts": 0,
             "no_signal_warning": None, "checkpoints": []}
    guard = SignalGuard(stop=a.stop_if_no_signal)
    killed = []
    print(f"pool: {sum(map(len, pool.values()))} rows over {len(pool)} arms; {n_prompts} prompts x {group} per step, "
          f"completions up to {max_completion} tokens", flush=True)

    def picks_for(n):
        try:
            picks = schedule.next_batch(state, n)
            if len(picks) != n or any(tuple(p) not in pool for p in picks):
                raise ValueError(f"next_batch returned {len(picks)} pairs (wanted {n}) or an arm not in the pool")
            return [tuple(p) for p in picks]
        except Exception as e:   # the schedule's failure costs it this batch, not the run
            stats["fallback_batches"] += 1
            if len(stats["fallback_errors"]) < 5:
                stats["fallback_errors"].append(f"step {state.step}: {type(e).__name__}: {e}"[:300])
            return [state.rng.choice(arms) for _ in range(n)]

    def evalgen_reward(prompts, completions, row, completion_ids=None, env_reward=None, **kw):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        rows = [json.loads(x) for x in row]
        if env_reward is not None:   # a multi-turn rollout scored its own episodes
            true = [float(x) for x in env_reward]
        else:
            true = checker(list(zip(rows, texts)))   # the checker's verdicts: the reward by default, a diagnostic otherwise
        stats["check_timeouts"] = checker.timeouts
        if env_reward is not None:
            out = true
        elif a.reward == "format":
            out = [1.0 if re.search(r"ANSWER:\s*\S", t) else 0.0 for t in texts]
        else:
            out = true
        lens = [len(ids) for ids in completion_ids] if completion_ids is not None else [0] * len(out)
        stats["gen_tokens"] += sum(lens)
        for g in range(0, len(out), group):
            state.history.append({"step": state.step, "env": rows[g]["eval"], "difficulty": rows[g]["difficulty"],
                                  "rewards": out[g:g + group], "true": true[g:g + group], "lengths": lens[g:g + group],
                                  "truncated": sum(n >= max_completion for n in lens[g:g + group])})
        return out

    class Hooked(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            if self.model.training:
                n = len(inputs) // self.num_generations
                state.elapsed_s = time.time() - t_start
                rows = []
                for env, diff in picks_for(n):
                    key = (env, diff)
                    rows.append(pool[key][cursor[key] % len(pool[key])])
                    cursor[key] += 1
                inputs = group_inputs(rows, self.num_generations)
            self._rows = [json.loads(x["row"]) for x in inputs]   # for a rollout hook: rows aligned with the prompts
            out = super()._generate_and_score_completions(inputs)
            if self.model.training and hasattr(algo, "advantages"):
                groups = [dict(g) for g in state.history[-(len(inputs) // self.num_generations):]]
                rew = torch.tensor([g["rewards"] for g in groups], dtype=torch.float32)
                adv = algo.advantages(rew, {"groups": groups, "step": state.step})
                adv = torch.as_tensor(adv, dtype=torch.float32).reshape(-1).to(out["advantages"].device)
                if adv.shape != out["advantages"].shape or not torch.isfinite(adv).all():
                    raise ValueError(f"advantages() returned shape {tuple(adv.shape)} or non-finite values")
                out["advantages"] = adv
            return out

        def _compute_loss(self, model, inputs):
            if not hasattr(algo, "policy_loss"):
                return super()._compute_loss(model, inputs)
            ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
            att = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
            logp, _, _ = self._get_per_token_logps_and_entropies(model, ids, att, inputs["completion_ids"].size(1))
            old = inputs.get("old_per_token_logps")
            old = logp.detach() if old is None else old
            samp = inputs.get("sampling_per_token_logps")
            info = {"step": state.step, "sampling_logp": None if samp is None else samp.detach()}
            loss = algo.policy_loss(logp, old, inputs["advantages"], inputs["completion_mask"], info)
            if not (torch.is_tensor(loss) and loss.dim() == 0):
                raise ValueError("policy_loss() must return a scalar tensor")
            return loss / self.current_gradient_accumulation_steps

    class Clock(TrainerCallback):
        def on_step_end(self, args, st, control, **kw):
            state.step = st.global_step
            state.elapsed_s = time.time() - t_start
            recent = state.history[-n_prompts:]   # the latest generation batch: this step's groups
            acc = sum(sum(g["rewards"]) for g in recent) / max(1, group * len(recent))
            true_acc = sum(sum(g.get("true", g["rewards"])) for g in recent) / max(1, group * len(recent))
            fr = group_fractions(recent)
            left = (state.budget_s - state.elapsed_s) / 60
            with open(log_dir / "steps.jsonl", "a") as f:
                f.write(json.dumps({"step": st.global_step, "elapsed_s": round(state.elapsed_s, 1),
                                    "remaining_min": round(left, 1), "reward": round(acc, 4),
                                    "true_reward": round(true_acc, 4), "gen_tokens": stats["gen_tokens"], **fr}) + "\n")
            print(f"step {st.global_step}: {state.elapsed_s / 60:.1f} min elapsed, {left:.1f} of {a.minutes:g} min left, "
                  f"reward {acc:.3f}, groups all-correct {fr['all_correct']:.0%} all-wrong {fr['all_wrong']:.0%} "
                  f"mixed {fr['mixed']:.0%}", flush=True)
            warning, stop = guard.update(st.global_step, fr)
            if warning:
                stats["no_signal_warning"] = warning
                print("\n" + warning + "\n", flush=True)
            if a.save_every and st.global_step % a.save_every == 0:
                out_k = f"{a.out}_step{st.global_step}"
                save_policy(trainer, out_k, a.lora, final=False)
                tok.save_pretrained(out_k)
                stats["checkpoints"].append(out_k)
                print(f"saved {out_k}", flush=True)
            if state.elapsed_s > state.budget_s - min(300.0, max(60.0, 0.1 * state.budget_s)):   # leave room to save
                control.should_training_stop = True
            if a.token_cap and stats["gen_tokens"] >= a.token_cap:
                stats["stop"] = "token_cap"
                control.should_training_stop = True
            if stop:
                stats["stop"] = "no_signal"
                control.should_training_stop = True
            if killed:
                stats["stop"] = "signal"
                control.should_training_stop = True
            if st.global_step >= a.max_steps and not control.should_training_stop:
                stats["stop"] = "max_steps"
            return control

    cfg = dict(output_dir=str(log_dir / "trainer"), per_device_train_batch_size=a.micro_batch,
               gradient_accumulation_steps=a.episodes_per_step // a.micro_batch, num_generations=group,
               max_completion_length=max_completion, max_steps=a.max_steps, learning_rate=1e-6,
               lr_scheduler_type="constant_with_warmup", warmup_steps=5, bf16=True, gradient_checkpointing=True,
               use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35, vllm_enable_sleep_mode=False,
               vllm_max_model_length=a.max_prompt + max_completion + 64, logging_steps=1, save_strategy="no",
               report_to="none", seed=a.seed, shuffle_dataset=False, disable_tqdm=True,
               chat_template_kwargs=chat.template_kwargs(tok) or None)   # a template without a thinking switch gets none
    cfg.update(cfg_over)
    policy_cfg, peft_config = policy_options(a.bf16, a.lora, a.model)
    cfg.update(policy_cfg)
    attn, why = accel.attn_implementation(a.model, a.attn)
    cfg["model_init_kwargs"]["attn_implementation"] = attn
    liger_cfg, liger_note = accel.grpo_liger_options(a.model, enabled=not a.no_liger)
    cfg.update(liger_cfg)
    print(f"attention: {attn} ({why}); {liger_note}", flush=True)
    placeholder = Dataset.from_dict({"prompt": [[{"role": "user", "content": "."}]] * (a.max_steps * n_prompts),
                                     "row": ["{}"] * (a.max_steps * n_prompts)})
    rollout_func = None
    if a.rollout:
        rmod = load_module(a.rollout, "rollout")

        def rollout_func(prompts, trainer):
            ctx = {"tok": tok, "llm": trainer.vllm_generation.llm, "group": group, "max_completion": max_completion,
                   "step": state.step, "rng": state.rng, "temperature": trainer.args.temperature}
            out = rmod.rollout(prompts, trainer._rows, trainer, ctx)
            n = len(prompts)
            if any(len(out[k]) != n for k in ("prompt_ids", "completion_ids", "logprobs")):
                raise ValueError(f"rollout() must return one prompt_ids / completion_ids / logprobs entry per prompt ({n})")
            return out
    trainer = Hooked(model=a.model, reward_funcs=evalgen_reward, args=GRPOConfig(**cfg), train_dataset=placeholder,
                     processing_class=tok, callbacks=[Clock()], rollout_func=rollout_func, peft_config=peft_config)
    if trainer.args.use_vllm:
        skip_frozen_vision_sync(trainer)
    print(f"policy weights in {next(trainer.model.parameters()).dtype}", flush=True)
    if peft_config is not None:
        trainer.model.print_trainable_parameters()

    def on_term(*_):   # installed after vLLM starts, so it is ours; a second kill gets the default (exit at once)
        killed.append(time.time())
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print("SIGTERM: stopping after this step, then saving", flush=True)
    signal.signal(signal.SIGTERM, on_term)
    try:
        trainer.train()
    finally:
        schedule.close()
        checker.close()
    save_policy(trainer, a.out, a.lora)
    tok.save_pretrained(a.out)
    with open(log_dir / "history.jsonl", "w") as f:
        for g in state.history:
            f.write(json.dumps(g) + "\n")
    summary = {"steps": state.step, "elapsed_s": round(time.time() - t_start), "groups": len(state.history),
               "prompts_per_step": n_prompts, "group_size": group, "episodes_per_step": a.episodes_per_step, "max_completion_length": max_completion,
               "config": cfg_over, "bf16": a.bf16, "lora": a.lora, "attn": attn,
               "liger": bool(liger_cfg), **stats}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("TRAIN_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":   # vLLM spawns workers
    main()
