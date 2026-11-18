"""The fixed RL training run: train /opt/base on a pool of multi-file projects with GRPO, as the grader does.

  python3 train_rl.py --pool <dir with pool.jsonl [and config.json]> --out <model dir> [--minutes 120] [--check]

1. config.json (optional, plain JSON) is read; a value outside its range is reported and the default used instead:
     max_steps          int >= 1, or null (default): no step limit, the time cap stops training
     episodes_per_step  32 to 256, a multiple of 16 (default 64)
     num_generations    4, 8 or 16 (default 8): attempts per project in a step; episodes_per_step / num_generations
                        projects per step
     temperature        0.6 to 1.2 (default 1.0)
     learning_rate      1e-6 to 1e-4 (default 2e-5)
     schedule           optional {"stages": [{"from_step": k, "weights": {label: weight, ...}}, ...]}: from optimizer
                        step k on, projects are drawn from the labels (a row's "difficulty" field) with those weights.
                        Without a schedule (or before the first stage), every row is equally likely.
2. pool.jsonl is checked row by row and bad rows are dropped (reported with the reason). A row is a JSON object:
     params      {"files", "hidden", "max_calls", "timeout_s"} as in /app/data (graded optional: "api"; change
                 optional: the text of the new specification; task optional: "implement" or "port", with interface
                 (a file name) for "port"; these choose the environment's instructions); no fix needed
     prompt      optional str: the first message of the episode (default: the environment's instructions)
     difficulty  optional str label (at most 40 characters) for the schedule (default "all")
   Checks: at most MAX_ROWS rows; 1 to 40 files with flat .py names, each at most 30000 characters, that the sandbox
   accepts; 1 to 60 hidden functions, each of a module in files, with 1 to 100 inputs and as many expected values;
   max_calls 1 to 64; timeout_s 1 to 10; a change and a prompt of at most 30000 characters each; and the hidden tests
   run on the files as given within the grading time limit (whether they pass does not matter). A row whose first
   message, in the base model's chat template, is longer than 8192 tokens is dropped too (grpo.py would skip it).
   The grader takes pool.jsonl and config.json of at most 4 GB each.
3. GRPO from /opt/base with the kit's grpo.py: multi-turn episodes (rollouts/multiturn.py, the same loop as grading:
   at most 4096 tokens per reply, 64 replies, 32768 tokens per episode), reward 1 or 0 from the environment; loss
   "dapo" (token-level, clip 0.2 low / 0.28 high, no KL penalty, truncated episodes masked out, no std scaling);
   LoRA rank 64 (alpha 128) on every projection, base weights in fp32, merged into the saved model. Training stops at
   max_steps or when --minutes run out (whichever comes first; a few minutes are kept to save), and the merged model
   is saved to --out.

--check only checks the config and the pool (step 1 and 2) and prints what training would use.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULTS = {"max_steps": None, "episodes_per_step": 64, "num_generations": 8, "temperature": 1.0,
            "learning_rate": 2e-5, "schedule": None}
LORA_RANK = 64
MICRO_BATCH = 1               # episodes per forward/backward pass: two episodes near the 32768-token cap in one
                              # pass do not fit on an 80 GB H100 (the DAPO loss is normalized by the step's token
                              # count, so this changes the memory, not the gradient)
MAX_COMPLETION = 24576        # the whole episode after the prompt (8192): fills the 32768-token context
MAX_ROWS = 10000
MAX_LABEL = 40
MAX_PROMPT_CHARS = 30000
MAX_PROMPT_TOKENS = 8192      # grpo.py's --max-prompt: it skips rows whose rendered first message is longer
BASE = "/opt/base"
NO_LIMIT_STEPS = 100000


def kit_dir(arg):
    if arg:
        return Path(arg).resolve()
    for d in (HERE / "kit", Path("/app/posttrain-kit")):
        if (d / "grpo.py").exists():
            return d
    sys.exit("no kit directory with grpo.py found (give --kit)")


# --------------------------------------------------------------------------------------------------------------------
# config.json
# --------------------------------------------------------------------------------------------------------------------

def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_config(path):
    """(settings, problems): each bad value is replaced by its default and named in problems."""
    cfg, problems = dict(DEFAULTS), []
    if not path.exists():
        return cfg, problems
    try:
        raw = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError) as e:
        return cfg, [f"config.json is not JSON ({e}); all defaults used"]
    if not isinstance(raw, dict):
        return cfg, ["config.json is not a JSON object; all defaults used"]
    for k, v in raw.items():
        if k not in DEFAULTS:
            problems.append(f"{k}: not a setting (settings: {', '.join(DEFAULTS)}); ignored")
            continue
        ok = {
            "max_steps": v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 1),
            "episodes_per_step": isinstance(v, int) and not isinstance(v, bool) and 32 <= v <= 256 and v % 16 == 0,
            "num_generations": isinstance(v, int) and not isinstance(v, bool) and v in (4, 8, 16),
            "temperature": _num(v) and 0.6 <= v <= 1.2,
            "learning_rate": _num(v) and 1e-6 <= v <= 1e-4,
            "schedule": v is None or _schedule_ok(v) is None,
        }[k]
        if ok:
            cfg[k] = v
        else:
            why = _schedule_ok(v) if k == "schedule" else "out of range"
            problems.append(f"{k} = {json.dumps(v)[:200]}: {why}; default {json.dumps(DEFAULTS[k])} used")
    return cfg, problems


def _schedule_ok(s):
    """None if s is a valid schedule, else why not."""
    if not isinstance(s, dict) or set(s) != {"stages"} or not isinstance(s["stages"], list) or not s["stages"]:
        return 'a schedule is {"stages": [...]} with at least one stage'
    if len(s["stages"]) > 100:
        return "at most 100 stages"
    last = -1
    for st in s["stages"]:
        if not isinstance(st, dict) or set(st) != {"from_step", "weights"}:
            return 'each stage is {"from_step": int, "weights": {label: weight}}'
        k, w = st["from_step"], st["weights"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 0 or k <= last:
            return "from_step must be ints >= 0, increasing"
        last = k
        if not isinstance(w, dict) or not w or len(w) > 1000:
            return "weights must be a non-empty object of at most 1000 labels"
        if any(not isinstance(lb, str) or len(lb) > MAX_LABEL for lb in w):
            return f"labels are strings of at most {MAX_LABEL} characters"
        if any(not _num(x) or x < 0 for x in w.values()) or sum(w.values()) <= 0:
            return "weights are numbers >= 0 with a positive sum"
    return None


# --------------------------------------------------------------------------------------------------------------------
# pool.jsonl
# --------------------------------------------------------------------------------------------------------------------

PARAM_ORDER = ("files", "hidden", "max_calls", "timeout_s", "graded", "change", "task", "interface")
PARAM_KEYS = set(PARAM_ORDER)


def check_row(r, kit):
    """None if the row is usable, else why not. Runs the hidden tests once in the sandbox (on the files as given)."""
    from rl_evalgen.project_env import GRADE_TIMEOUT_S, MAX_FILE_CHARS
    from rl_evalgen.sandbox import FILE_NAME, MAX_PROJECT_FILES, call_project, check_project, decode
    if not isinstance(r, dict) or not isinstance(r.get("params"), dict):
        return "not an object with params"
    extra = set(r) - {"params", "prompt", "difficulty", "id", "eval", "env"}
    if extra:
        return f"unknown fields {sorted(extra)}"
    p = r["params"]
    if set(p) - PARAM_KEYS:
        return f"unknown params {sorted(set(p) - PARAM_KEYS)}"
    files, hidden = p.get("files"), p.get("hidden")
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_PROJECT_FILES:
        return f"files must be 1 to {MAX_PROJECT_FILES} files"
    for name, src in files.items():
        if not isinstance(name, str) or not FILE_NAME.match(name) or not isinstance(src, str):
            return "files: names are flat .py names and contents are strings"
        if len(src) > MAX_FILE_CHARS:
            return f"{name}: more than {MAX_FILE_CHARS} characters"
    why = check_project(files)
    if why:
        return f"the sandbox refuses the files: {why}"
    if not isinstance(hidden, list) or not 1 <= len(hidden) <= 60:
        return "hidden must be a list of 1 to 60 functions"
    for h in hidden:
        if not isinstance(h, dict) or set(h) != {"module", "func", "inputs", "expected"}:
            return 'each hidden entry is {"module", "func", "inputs", "expected"}'
        if not isinstance(h["module"], str) or f"{h['module']}.py" not in files or not isinstance(h["func"], str):
            return f"hidden: module {h.get('module')!r} is not one of the files"
        if not isinstance(h["inputs"], list) or not 1 <= len(h["inputs"]) <= 100 or \
                not isinstance(h["expected"], list) or len(h["expected"]) != len(h["inputs"]):
            return "hidden: 1 to 100 inputs per function, one expected value each"
        try:
            for args in h["inputs"]:
                if not isinstance(args, list):
                    return "hidden: each input is a list of encoded arguments"
                [decode(a) for a in args]
            [decode(v) for v in h["expected"]]
        except Exception as e:
            return f"hidden: a value is not in sandbox.encode's format ({type(e).__name__})"
    mc, ts = p.get("max_calls"), p.get("timeout_s")
    if not isinstance(mc, int) or isinstance(mc, bool) or not 1 <= mc <= 64:
        return "max_calls must be an int from 1 to 64"
    if not _num(ts) or not 1 <= ts <= 10:
        return "timeout_s must be a number from 1 to 10"
    if "graded" in p and p["graded"] != "api":
        return 'graded, when given, is "api"'
    if "change" in p and (not isinstance(p["change"], str) or not p["change"].strip() or len(p["change"]) > MAX_PROMPT_CHARS):
        return f"change must be a non-empty string of at most {MAX_PROMPT_CHARS} characters"
    if "task" in p and p["task"] not in ("implement", "port"):
        return 'task, when given, is "implement" or "port"'
    if (p.get("task") == "port") != ("interface" in p):
        return 'interface is given exactly when task is "port"'
    if "interface" in p and (not isinstance(p["interface"], str) or p["interface"] not in files):
        return "interface must be the name of one of the files"
    if "prompt" in r and (not isinstance(r["prompt"], str) or not r["prompt"].strip() or len(r["prompt"]) > MAX_PROMPT_CHARS):
        return f"prompt must be a non-empty string of at most {MAX_PROMPT_CHARS} characters"
    if "difficulty" in r and (not isinstance(r["difficulty"], str) or not 1 <= len(r["difficulty"]) <= MAX_LABEL):
        return f"difficulty must be a label of 1 to {MAX_LABEL} characters"
    calls = [{"module": h["module"], "func": h["func"], "inputs": h["inputs"]} for h in hidden]
    out, why = call_project(files, calls, timeout_s=GRADE_TIMEOUT_S)
    if out is None:
        return f"the hidden tests do not run on the files as given ({why})"
    return None


def read_pool(path, kit, workers):
    """(clean rows in the kit's pool format, problems)."""
    from rl_evalgen.project_env import instructions
    problems, raw = [], []
    if not path.exists():
        return [], [f"{path} does not exist"]
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            if len(raw) >= MAX_ROWS:
                problems.append(f"more than {MAX_ROWS} rows: the rest are dropped")
                break
            try:
                raw.append((n, json.loads(line)))
            except ValueError:
                problems.append(f"line {n}: not JSON; dropped")
    with ThreadPoolExecutor(workers) as ex:
        verdicts = list(ex.map(lambda nr: check_row(nr[1], kit), raw))
    rows = []
    for (n, r), why in zip(raw, verdicts):
        if why:
            problems.append(f"line {n}: {why}; dropped")
            continue
        params = {k: r["params"][k] for k in PARAM_ORDER if k in r["params"]}
        row = {"id": f"pool-{n}", "eval": "env_multifile_repair", "env": "multifile_repair",
               "difficulty": r.get("difficulty", "all"), "params": params}
        row["prompt"] = r["prompt"] if "prompt" in r else instructions(params)
        if "prompt" in r:
            row["train_prompt"] = r["prompt"]
        rows.append(row)
    return drop_long_prompts(rows, problems), problems


def drop_long_prompts(rows, problems):
    """The rows whose first message, rendered with the base model's chat template as grpo.py renders it, is at
    most MAX_PROMPT_TOKENS tokens. grpo.py would skip the others without a word, and a pool of only such rows
    would leave it nothing to train on."""
    if not rows:
        return rows
    if not os.path.isdir(BASE):
        problems.append(f"{BASE} not found: prompt lengths not checked")
        return rows
    import chat
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    keep = []
    for i in range(0, len(rows), 256):
        batch = rows[i:i + 256]
        texts = [chat.render(tok, [{"role": "user", "content": r["prompt"]}]) for r in batch]
        for r, ids in zip(batch, tok(texts, add_special_tokens=False)["input_ids"]):
            if len(ids) > MAX_PROMPT_TOKENS:
                problems.append(f"line {r['id'][len('pool-'):]}: the first message is {len(ids)} tokens, more "
                                f"than {MAX_PROMPT_TOKENS}; dropped")
            else:
                keep.append(r)
    return keep


# --------------------------------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------------------------------

ALGO = '''"""Written by train_rl.py: DAPO-style GRPO for multi-turn episodes, with the pool's settings."""
CONFIG = {{"loss_type": "dapo", "epsilon": 0.2, "epsilon_high": 0.28, "mask_truncated_completions": True,
          "scale_rewards": "none", "beta": 0.0, "learning_rate": {lr!r}, "num_generations": {g!r},
          "temperature": {t!r}, "max_completion_length": {mc!r}}}
'''

SCHEDULE = '''"""Written by train_rl.py: draw projects by label, with the weights of the stage in force (uniform over rows
before the first stage)."""
STAGES = {stages!r}
COUNTS = {counts!r}


def next_batch(state, n):
    arms = [(e, d) for e, ds in sorted(state.envs.items()) for d in ds]
    stage = None
    for st in STAGES:
        if state.step >= st["from_step"]:
            stage = st
    if stage is None:
        w = [COUNTS.get(d, 0) for _, d in arms]
    else:
        w = [stage["weights"].get(d, 0.0) for _, d in arms]
    if sum(w) <= 0:
        w = [COUNTS.get(d, 0) for _, d in arms]
    return state.rng.choices(arms, weights=w, k=n)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="directory with pool.jsonl and optionally config.json")
    ap.add_argument("--out", required=False, help="where the trained (merged) model is saved")
    ap.add_argument("--minutes", type=float, default=120, help="wall-clock cap of the training run (default 120)")
    ap.add_argument("--check", action="store_true", help="only check the config and the pool")
    ap.add_argument("--kit", default=None, help="the posttrain-kit directory (default: next to this file, or /app)")
    ap.add_argument("--work", default=None, help="where the checked pool, algo and logs go (default: <out>_work)")
    ap.add_argument("--workers", type=int, default=16, help="processes that check pool rows")
    a = ap.parse_args()
    kit = kit_dir(a.kit)
    sys.path.insert(0, str(kit))
    pool_dir = Path(a.pool)
    t0 = time.time()
    cfg, cfg_problems = read_config(pool_dir / "config.json")
    rows, pool_problems = read_pool(pool_dir / "pool.jsonl", kit, a.workers)
    labels = {}
    for r in rows:
        labels[r["difficulty"]] = labels.get(r["difficulty"], 0) + 1
    if cfg["episodes_per_step"] % cfg["num_generations"]:
        cfg_problems.append(f"episodes_per_step {cfg['episodes_per_step']} is not a multiple of num_generations "
                            f"{cfg['num_generations']}; num_generations {DEFAULTS['num_generations']} used")
        cfg["num_generations"] = DEFAULTS["num_generations"]
    report = {"settings": cfg, "config_problems": cfg_problems, "rows_kept": len(rows),
              "rows_dropped": sum(1 for p in pool_problems if p.endswith("dropped")),
              "labels": labels, "pool_problems": pool_problems, "check_seconds": round(time.time() - t0, 1)}
    print(json.dumps(report, indent=1), flush=True)
    if a.check:
        return
    if not a.out:
        sys.exit("--out is needed to train")
    work = Path(a.work or a.out + "_work")
    (work / "pool").mkdir(parents=True, exist_ok=True)
    status = work / "status.json"       # what happened, for the grader: never parsed from output the pool can reach
    if not rows:
        status.write_text(json.dumps({"trained": False, "reason": "the pool has no usable rows"}))
        print("No training: the pool has no usable rows", flush=True)
        return
    with open(work / "pool" / "env_multifile_repair.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (work / "algo.py").write_text(ALGO.format(lr=float(cfg["learning_rate"]), g=cfg["num_generations"],
                                              t=float(cfg["temperature"]), mc=MAX_COMPLETION))
    stages = cfg["schedule"]["stages"] if cfg["schedule"] else []
    (work / "schedule.py").write_text(SCHEDULE.format(stages=stages, counts=labels))
    (work / "report.json").write_text(json.dumps(report, indent=1))
    steps = cfg["max_steps"] or NO_LIMIT_STEPS
    cmd = [sys.executable, str(kit / "grpo.py"), "--model", "/opt/base", "--pool", str(work / "pool"), "--out", a.out,
           "--log", str(work / "log"), "--minutes", str(a.minutes), "--algo", str(work / "algo.py"),
           "--schedule", str(work / "schedule.py"), "--rollout", str(kit / "rollouts" / "multiturn.py"),
           "--lora", str(LORA_RANK), "--max-steps", str(steps), "--episodes-per-step", str(cfg["episodes_per_step"]),
           "--micro-batch", str(MICRO_BATCH), "--seed", "0"]
    print("train:", " ".join(cmd), flush=True)
    # expandable segments: PyTorch reuses memory it reserved but no longer uses (without them, long episodes could
    # run the GPU out of memory with 8-12 GiB reserved but unused)
    env = {**os.environ, "PYTHONPATH": str(kit), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    p = subprocess.Popen(cmd, env=env, cwd=str(kit))
    try:
        p.wait(timeout=a.minutes * 60 + 1200)          # grpo stops itself; this is a backstop
    except subprocess.TimeoutExpired:
        p.kill()
        sys.exit("training ran past its time cap and was stopped")
    if p.returncode:
        sys.exit(f"training failed (exit {p.returncode})")
    status.write_text(json.dumps({"trained": True}))
    print(f"Trained: {a.out}", flush=True)


if __name__ == "__main__":
    main()
