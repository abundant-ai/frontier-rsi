"""Final scoring (runs as root, isolated, after test.sh has stopped the agent session, emptied /logs/verifier and
locked /tests).

The hand-in is /app/rl_pool/: pool.jsonl (training projects) and config.json (settings). Both are copied, as regular
files only, into a root-only directory; every file the agent created or changed elsewhere is deleted (no planted
cache or module can reach the training or the scoring); the environment must pass a self-test (each sampled graded
project's fix, played as tool calls, scores 1; an immediate submit() scores 0), or no reward is written. Then
/tests/train_rl.py, with the pristine kit in /tests/kit, checks the pool (bad rows dropped) and the config (bad values
replaced by defaults) and trains /opt/base with its fixed GRPO run, capped at train_minutes of wall-clock time. The
trained model (its weights with /opt/base's config, tokenizer and chat template) works through every graded project
(/tests/held.jsonl) as one greedy, repeatable episode of the kit's multi-turn loop with the multifile_repair
environment; the environment grades each project with its hidden tests in the sandbox. Reward (raw) = the fraction
of projects solved. Without a pool (or with no usable row), the original model is scored without training.

The agent's scripts and logs (text files only, size-capped) are saved to /logs/verifier/agent_files.tar.gz first, the
training report and logs to /logs/verifier/train/, and every graded episode to /logs/verifier/held_rows.jsonl.gz. The
model's weights are not kept.

Only a finished scoring writes a reward. When the grading itself fails (the environment self-test, the training run,
playing the projects, any unexpected error), no reward is written and the trial is a verifier error, not a score of 0.
"""
import gzip, io, json, os, shutil, stat, subprocess, sys, tarfile, time, traceback

sys.path.insert(0, "/tests")
import evaluate  # noqa: E402

LOG = "/logs/verifier"
CFG = json.load(open("/tests/task_config.json"))
SUB = "/app/rl_pool"
HANDIN = ("pool.jsonl", "config.json")
MAX_POOL_BYTES = 4 << 30
WORK = "/tmp/gwork"
SCRATCH = ("/tmp", "/var/tmp", "/dev/shm")
TEXT = (".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".log", ".ipynb", ".jinja",
        ".csv", ".tsv", ".jsonl")
metrics = {"task": CFG["task_id"], "env": CFG["env"], "model": CFG["model"], "checks": {}}


def finish(acc=None, reason=""):
    reward = 0.0 if acc is None else min(1.0, max(0.0, acc))
    if acc is not None:
        metrics["solved"] = round(acc, 4)
    metrics.update({"reward": round(reward, 4), "reason": reason})
    json.dump({"reward": metrics["reward"]}, open(f"{LOG}/reward.json", "w"))
    open(f"{LOG}/reward.txt", "w").write(f"{metrics['reward']}\n")
    json.dump(metrics, open(f"{LOG}/metrics.json", "w"), indent=1)
    print(json.dumps(metrics, indent=1))
    sys.exit(0)


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def container_started():
    try:
        ticks = int(open("/proc/1/stat").read().rsplit(")", 1)[1].split()[19])
        btime = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK") - 2.0
    except Exception:  # noqa: BLE001
        return 0.0


def agent_uid():
    import pwd
    return pwd.getpwnam("agent").pw_uid


def submission_untouched_since_grading_started():
    """An entry of the hand-in changed after the session was over (test.sh's stamp) was written by a leftover
    process, not the session."""
    try:
        t0 = float(open(f"{LOG}/.grading_started").read().strip())
    except Exception:  # noqa: BLE001
        return True
    if not os.path.lexists(SUB):
        return True
    for root, dirs, files in os.walk(SUB):
        for name in [""] + dirs + files:
            st = os.lstat(os.path.join(root, name) if name else root)
            if max(st.st_mtime, st.st_ctime) > t0:
                return False
    return True


def collect_agent_files(limit_file=1 << 20, limit_total=30 << 20):
    """The agent's own scripts, configs and logs (text files it created, up to 1 MB each), to see how it trained."""
    uid, born, total = agent_uid(), container_started(), 0
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for top in ("/app", "/home/agent", "/tmp", "/var/tmp"):
            for d, dirs, files in os.walk(top):
                dirs[:] = [x for x in dirs if os.path.join(d, x) not in (SUB, "/app/data", WORK)
                           and not x.startswith((".cache", "__pycache__", ".triton", "torchinductor_"))]
                for n in files:
                    p = os.path.join(d, n)
                    try:
                        st = os.lstat(p)
                    except OSError:
                        continue
                    if (not n.endswith(TEXT) or st.st_uid != uid or st.st_ctime < born or not os.path.isfile(p)
                            or os.path.islink(p)):
                        continue
                    size = min(st.st_size, limit_file)
                    if total + size > limit_total:
                        continue
                    data = open(p, "rb").read(size)
                    ti = tarfile.TarInfo(p.lstrip("/"))
                    ti.size, ti.mtime = len(data), int(st.st_mtime)
                    tar.addfile(ti, io.BytesIO(data))
                    total += size
    open(f"{LOG}/agent_files.tar.gz", "wb").write(buf.getvalue())
    metrics["agent_files_bytes"] = total


def copy_submission():
    """Copy pool.jsonl and config.json from /app/rl_pool into a root-only directory (regular files only; anything
    else in the folder is ignored). None if there is no pool.jsonl."""
    if os.path.lexists(WORK):
        shutil.rmtree(WORK, ignore_errors=True) if os.path.isdir(WORK) and not os.path.islink(WORK) else os.remove(WORK)
    os.makedirs(WORK, mode=0o700)
    if not os.path.lexists(SUB):
        return None
    if os.path.islink(SUB) or not os.path.isdir(SUB):
        finish(reason="/app/rl_pool is not a directory")
    dst = os.path.join(WORK, "pool")
    os.makedirs(dst, mode=0o700)
    found = False
    for name in HANDIN:
        p = os.path.join(SUB, name)
        if not os.path.lexists(p):
            continue
        mode = os.lstat(p).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            finish(reason=f"/app/rl_pool/{name} is not a regular file")
        if os.path.getsize(p) > MAX_POOL_BYTES:
            finish(reason=f"/app/rl_pool/{name} is larger than {MAX_POOL_BYTES >> 30} GB")
        try:
            shutil.copyfile(p, os.path.join(dst, name))
        except OSError as e:
            finish(reason=f"/app/rl_pool/{name} could not be copied: {type(e).__name__}: {str(e)[:200]}")
        found = found or name == "pool.jsonl"
    metrics["checks"]["handin_bytes"] = {n: os.path.getsize(os.path.join(dst, n)) for n in HANDIN
                                        if os.path.exists(os.path.join(dst, n))}
    return dst if found else None


def clean_slate():
    """Delete every file the agent created or changed outside /app/rl_pool (ctime at or after the container
    started), and empty the scratch directories."""
    uid, born = agent_uid(), container_started()
    prune = ("/proc", "/sys", "/logs", "/tests", SUB, WORK)
    for d, dirs, files in os.walk("/", topdown=True):
        dirs[:] = [x for x in dirs if os.path.join(d, x) not in prune]
        for n in files:
            q = os.path.join(d, n)
            try:
                st = os.lstat(q)
                if st.st_uid == uid and st.st_ctime >= born:
                    os.remove(q)
            except OSError:
                pass
    for root in SCRATCH:
        for name in os.listdir(root) if os.path.isdir(root) else []:
            q = os.path.join(root, name)
            if q == WORK:
                continue
            try:
                shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) and not os.path.islink(q) else os.remove(q)
            except OSError:
                pass


def grader_error(reason):
    """The grading itself cannot be trusted: no reward is written, so the trial ends as a verifier error (retryable),
    not as a score."""
    metrics["reason"] = reason
    json.dump(metrics, open(f"{LOG}/metrics.json", "w"), indent=1)
    for f in ("reward.json", "reward.txt"):
        if os.path.exists(f"{LOG}/{f}"):
            os.remove(f"{LOG}/{f}")
    print(json.dumps(metrics, indent=1))
    sys.exit(1)


def env_selftest(items):
    """The environment works in this container: for a sample of graded projects (selftest_per_difficulty of each
    difficulty), the fix played as tool calls scores 1 and an immediate submit() scores 0. The sandbox turns any error
    into a failed test, so a broken sandbox would look like a model that solves nothing; if the self-test fails, the
    trial is a grader error, not a score."""
    n, bad = evaluate.replay(items, CFG, CFG["selftest_per_difficulty"])
    metrics["checks"]["env_selftest"] = f"{n - len({b[0] for b in bad})}/{n} projects: fix scores 1, submit() alone 0"
    if bad:
        metrics["checks"]["env_selftest_failures"] = [str(b)[:300] for b in bad[:5]]
        grader_error("the environment does not work in this container (self-test failed)")


def grader_env(invariant=True):
    """New, empty, root-only home, temp and cache directories for the scorer."""
    env = dict(os.environ)
    for var, sub in (("HOME", "home"), ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"), ("HF_HOME", "hf"),
                     ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
                     ("VLLM_CACHE_ROOT", "vllm"), ("CUDA_CACHE_PATH", "cuda")):
        env[var] = os.path.join(WORK, "env", sub)
        os.makedirs(env[var], mode=0o700, exist_ok=True)
    env["VLLM_BATCH_INVARIANT"] = "1" if invariant else "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"          # repeatable answers (evaluate.generate)
    return env


def run_training(pool):
    """train_rl.py on the copied pool: -> the trained model directory, or None when no row was usable."""
    out = os.path.join(WORK, "trained")
    work = os.path.join(WORK, "train")
    cmd = [sys.executable, "-I", "/tests/train_rl.py", "--pool", pool, "--out", out, "--work", work, "--kit",
           "/tests/kit", "--minutes", str(CFG["train_minutes"])]
    t = time.time()
    try:
        r = subprocess.run(cmd, timeout=CFG["train_minutes"] * 60 + CFG["train_overhead_s"], capture_output=True,
                           text=True, cwd=WORK, env=grader_env(invariant=False))
    except subprocess.TimeoutExpired:
        grader_error("the training run did not stop in time")
    os.makedirs(f"{LOG}/train", exist_ok=True)
    open(f"{LOG}/train/train.log", "w").write(r.stdout[-200000:] + "\n" + r.stderr[-200000:])
    for name, src in (("report.json", "report.json"), ("steps.jsonl", "log/steps.jsonl"),
                      ("summary.json", "log/summary.json")):
        p = os.path.join(work, src)
        if os.path.isfile(p):
            shutil.copyfile(p, f"{LOG}/train/{name}")
    try:
        metrics["training"] = {"report": json.load(open(os.path.join(work, "report.json")))}
        metrics["training"]["report"].pop("pool_problems", None)
    except Exception:  # noqa: BLE001
        metrics["training"] = {}
    metrics["training"]["wall_s"] = round(time.time() - t)
    try:                                  # train_rl.py's own status file, not its output (which shows pool text)
        status = json.load(open(os.path.join(work, "status.json")))
    except Exception:  # noqa: BLE001
        status = {}
    if r.returncode == 0 and status.get("trained") is False:
        metrics["training"]["result"] = "no usable row: the original model is scored"
        return None
    if r.returncode != 0 or status.get("trained") is not True or not os.path.isdir(out):
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else str(r.returncode)
        grader_error(f"the training run failed: {last[:300]}")
    try:
        summ = json.load(open(os.path.join(work, "log", "summary.json")))
        metrics["training"]["summary"] = {k: summ.get(k) for k in ("steps", "elapsed_s", "stop", "prompts_per_step",
                                                                   "group_size", "episodes_per_step", "gen_tokens")}
    except Exception:  # noqa: BLE001
        pass
    metrics["training"]["result"] = "trained"
    return out


def run_scorer(model):
    rows_path = os.path.join(WORK, "rows.jsonl")
    cmd = [sys.executable, "/tests/evaluate.py", "generate", "--model", model, "--items", "/tests/held.jsonl", "--config",
           "/tests/task_config.json", "--out", rows_path] + (["--repeat"] if CFG.get("repeat_check") else [])
    for invariant in (True, False):
        t = time.time()
        try:
            r = subprocess.run(cmd, timeout=CFG["score_timeout_s"], capture_output=True, text=True, cwd=WORK,
                               env=grader_env(invariant))
        except subprocess.TimeoutExpired:
            grader_error(f"playing the graded projects ran past {CFG['score_timeout_s']} s")
        open(f"{LOG}/score.log", "a").write(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
        if r.returncode == 0 and os.path.isfile(rows_path):
            summary = next((json.loads(l) for l in reversed(r.stdout.splitlines()) if l.startswith('{"n":')), {})
            summary["wall_s"] = round(time.time() - t)
            return rows_path, summary
        err = r.stderr.lower()
        if not (invariant and ("batch_invariant" in err or "batch invariant" in err or "batch-invariant" in err)):
            last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else str(r.returncode)
            grader_error(f"playing the graded projects failed: {last[:300]}")
        metrics["checks"]["batch_invariant_unsupported"] = True   # retried once without batch invariance
    grader_error("playing the graded projects failed")


def main():
    metrics["gpu"] = gpu()
    metrics["checks"]["session_end_hook"] = os.path.isfile("/var/lib/ptrl/session_end")
    if not submission_untouched_since_grading_started():
        finish(reason="/app/rl_pool was changed after the session was over")
    try:
        collect_agent_files()
    except Exception as e:  # noqa: BLE001 - never fail the scoring over the agent-file copy
        metrics["agent_files_error"] = f"{type(e).__name__}: {e}"[:300]
    pool = copy_submission()
    clean_slate()
    items = [json.loads(l) for l in open("/tests/held.jsonl")]
    env_selftest(items)
    trained = run_training(pool) if pool else None
    if pool is None:
        metrics["checks"]["handin"] = "no /app/rl_pool/pool.jsonl: the original model is scored"
    if trained is None:
        model = evaluate.BASE
    else:
        reason, info = evaluate.check_model(trained)
        metrics["checks"].update(info)
        if reason:
            grader_error(f"the trained model is not usable: {reason}")
        model = evaluate.assemble(trained, os.path.join(WORK, "model"), link=False)
    rows_path, summary = run_scorer(model)
    if CFG.get("repeat_check") and summary.get("repeat_identical") != 1.0:     # smoke copies: answers must repeat
        metrics["scoring"] = summary
        grader_error(f"the episodes did not repeat exactly (share identical {summary.get('repeat_identical')})")
    rows = [json.loads(l) for l in open(rows_path)]
    if len(rows) != len(items) or any(r["id"] != it["id"] for r, it in zip(rows, items)):
        grader_error("the episodes do not line up with the graded projects")
    with gzip.open(f"{LOG}/held_rows.jsonl.gz", "wt") as g:
        for r in rows:
            g.write(json.dumps(r) + "\n")
    metrics["scoring"] = dict(evaluate.summarize(rows), **summary)
    finish(sum(r["reward"] for r in rows) / len(rows))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - an unexpected error in the grading is a verifier error, never a score
        traceback.print_exc()
        grader_error(f"the grader failed: {type(e).__name__}: {str(e)[:300]}")
