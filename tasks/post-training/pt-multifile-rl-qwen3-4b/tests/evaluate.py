"""Check a trained model directory, let the model work through multi-file repair projects, and grade every project:
the code behind /app/score.py.

- check_model(dir): the directory must hold the model's weights as .safetensors files with exactly the tensors of the
  original model (/opt/base: the same names and shapes; bf16, fp16 or fp32), no code or pickle files, no symlinks and
  nothing but regular files and directories.
- assemble(dir, out): the model that works is those weights with /opt/base's own config, tokenizer, chat template
  and generation settings (anything else in the directory is not used).
- play(): every project is one episode of the posttrain-kit's multi-turn loop (rollouts/multiturn.py, run_episodes)
  with the multifile_repair environment, both taken from the kit directory in task_config.json: the project's
  instructions are the first user message in /opt/base's chat template, every reply of the model goes to the
  environment, which runs one tool call and answers with the next user message. Greedy (temperature 0), at most
  turn_tokens tokens per reply and context_tokens tokens for the prompt and the whole episode together. The project
  is graded when the model calls submit() or its tool calls run out (reward 1 if every function passes its hidden
  tests); an episode that ends any other way (a reply cut off at turn_tokens, the context full) scores 0.
- generate(): play() with vLLM, greedy and repeatable: the engine runs in this process, without prefix caching, and
  the projects are always played in the same order, all of them in each round of replies.

  python3 evaluate.py generate --model <assembled dir> --items <jsonl> --config <task_config.json> --out <rows.jsonl>
"""
import argparse, json, math, os, shutil, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/opt/base"
DTYPES = {"BF16", "F16", "F32"}
FORBIDDEN = (".py", ".pyc", ".pyo", ".so", ".dylib", ".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".npy",
             ".npz", ".h5", ".msgpack", ".gguf", ".onnx")


def read_header(path):
    """A .safetensors file's tensors: {name: (dtype, shape)}. The file must hold all the data its header declares."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100_000_000:
            raise ValueError("header too large")
        h = json.loads(f.read(n))
    t = {k: v for k, v in h.items() if k != "__metadata__"}
    end = max((v["data_offsets"][1] for v in t.values()), default=0)
    if os.path.getsize(path) < 8 + n + end:
        raise ValueError(f"the file is cut short: {os.path.getsize(path)} bytes, its header declares {8 + n + end}")
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in t.items()}


def core(name):
    """A tensor's name without the wrapper segments that differ between checkpoint layouts of the same model
    (`model.language_model.layers.0...` and `language_model.model.layers.0...` are the same tensor)."""
    return ".".join(p for p in name.split(".") if p not in ("model", "language_model"))


def shards(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".safetensors") and os.path.isfile(os.path.join(d, f)))


def tensors(d):
    out = {}
    for f in shards(d):
        for k, v in read_header(os.path.join(d, f)).items():
            if k in out:
                raise ValueError(f"tensor {k} appears in more than one .safetensors file")
            out[k] = v
    return out


def check_model(sub, base=BASE):
    """(None, info) if `sub` can be scored, else (reason, info)."""
    info = {}
    if os.path.islink(sub) or not os.path.isdir(sub):
        return "not a directory", info
    for root, dirs, files in os.walk(sub):
        for name in dirs + files:
            p = os.path.join(root, name)
            if os.path.islink(p):
                return f"{os.path.relpath(p, sub)} is a symlink", info
            if not (os.path.isfile(p) or os.path.isdir(p)):
                return f"{os.path.relpath(p, sub)} is not a regular file or directory", info
            if name.lower().endswith(FORBIDDEN):
                return f"{os.path.relpath(p, sub)}: code and pickle files are not allowed", info
    files = shards(sub)
    if not files:
        return "no .safetensors file at the top level", info
    size = sum(os.path.getsize(os.path.join(sub, f)) for f in files)
    numel = [math.prod(s) for _, s in tensors(base).values()]
    # an fp32 copy of the original tensors, plus a separately saved output layer of a tied model (as big as the
    # largest tensor), plus headers
    limit = 4 * (sum(numel) + max(numel, default=0)) + 2 ** 26
    info.update(files=len(files), bytes=size)
    if size > limit:
        return f"the weights take {size} bytes, more than an fp32 copy of the original model", info
    try:
        got = tensors(sub)
    except Exception as e:  # noqa: BLE001 - an unreadable file is the submission's problem
        return f"a .safetensors file cannot be read: {type(e).__name__}: {str(e)[:200]}", info
    want = tensors(base)
    dtypes = sorted({d for d, _ in got.values()})
    info["dtypes"] = dtypes
    if set(dtypes) - DTYPES:
        return f"tensor dtypes {dtypes}: only BF16, F16 and F32 are allowed", info
    unique = len({core(k) for k in want}) == len(want)
    key = core if unique else (lambda k: k)
    want_k = {key(k): s for k, (_, s) in want.items()}
    got_k = {}
    for k, (_, s) in got.items():
        if key(k) in got_k:
            return f"two tensors are the same tensor of the model: {k}", info
        got_k[key(k)] = s
    cfg = json.load(open(os.path.join(base, "config.json"))) if os.path.isfile(os.path.join(base, "config.json")) else {}
    tied = cfg.get("tie_word_embeddings", (cfg.get("text_config") or {}).get("tie_word_embeddings"))
    if tied or "lm_head.weight" not in want_k:      # tied input and output embeddings: the output layer is the input
        got_k.pop("lm_head.weight", None)           # embedding, and save_pretrained may or may not write it
        want_k.pop("lm_head.weight", None)
    missing = sorted(set(want_k) - set(got_k))
    extra = sorted(set(got_k) - set(want_k))
    shape = sorted(k for k in set(want_k) & set(got_k) if tuple(want_k[k]) != tuple(got_k[k]))
    info.update(tensors=len(got))
    if missing or extra or shape:
        parts = []
        if missing:
            parts.append(f"{len(missing)} tensor(s) of the original model missing (e.g. {missing[0]})")
        if extra:
            parts.append(f"{len(extra)} tensor(s) the original model does not have (e.g. {extra[0]})")
        if shape:
            parts.append(f"{len(shape)} tensor(s) with another shape (e.g. {shape[0]}: {got_k[shape[0]]} instead "
                         f"of {want_k[shape[0]]})")
        return "not the original model's tensors: " + "; ".join(parts), info
    return None, info


def assemble(sub, out, base=BASE, link=True):
    """The model that answers: `sub`'s weights + /opt/base's config, tokenizer, chat template and generation files."""
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in os.listdir(base):
        p = os.path.join(base, f)
        if os.path.isfile(p) and not f.endswith((".safetensors", ".safetensors.index.json")):
            shutil.copy(p, os.path.join(out, f))
    files = shards(sub)
    for f in files:
        src, dst = os.path.join(sub, f), os.path.join(out, f)
        os.symlink(os.path.abspath(src), dst) if link else shutil.move(src, dst)
    if len(files) > 1:
        wm = {k: f for f in files for k in read_header(os.path.join(sub if link else out, f))}
        json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    return out


def vllm_overrides(model):
    """vLLM config overrides so that vLLM computes the model as transformers (and the model's own reference config)
    does. vLLM 0.29 reads a YaRN config that has mscale and mscale_all_dim as plain YaRN and scales attention by
    0.1 ln(factor) + 1; transformers uses get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim), as does
    the model's own reference config (apply_scale false: no extra scale). vLLM also ignores llama_4_scaling_beta
    there. Without this, a model trained with transformers would be run as a different model by vLLM."""
    cfg = json.load(open(os.path.join(model, "config.json")))
    t = cfg.get("text_config") or cfg
    rp = t.get("rope_parameters") or t.get("rope_scaling") or {}
    if (rp.get("rope_type") or rp.get("type")) != "yarn" or "mscale_all_dim" not in rp:
        return {}

    def get_mscale(scale, m=1.0):
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
    f = rp["factor"]
    att = rp.get("attention_factor") or (get_mscale(f, rp["mscale"]) / get_mscale(f, rp["mscale_all_dim"])
                                         if rp.get("mscale") and rp.get("mscale_all_dim") else get_mscale(f))
    ov = {"rope_parameters": {"apply_yarn_scaling": False, "attn_factor": float(att)}}
    if rp.get("llama_4_scaling_beta") is not None:
        ov["llama_4_scaling"] = {"original_max_position_embeddings": rp["original_max_position_embeddings"],
                                 "beta": rp["llama_4_scaling_beta"]}
    return {"text_config": ov} if "text_config" in cfg else ov




def load_kit(cfg):
    """The kit's episode loop and environments from cfg["kit_dir"], with its limits checked against the config."""
    kit = cfg["kit_dir"] if os.path.isabs(cfg["kit_dir"]) else os.path.join(HERE, cfg["kit_dir"])
    os.environ["ENV_WORKERS"] = str(cfg["env_workers"])
    sys.path.insert(0, kit)
    import chat
    import envs
    from rollouts import multiturn
    limits = {"max_turns": multiturn.MAX_TURNS, "turn_tokens": multiturn.TURN_TOKENS,
              "context_tokens": multiturn.CONTEXT}
    for k, v in limits.items():
        if v != cfg[k]:
            raise RuntimeError(f"the kit at {kit} has {k} = {v}, the task config {cfg[k]}")
    return chat, envs, multiturn


def play(llm, tok, items, cfg):
    """Every project as one greedy episode: one row per project, in item order."""
    chat, envs, multiturn = load_kit(cfg)
    m = envs.load("multifile_repair")
    made = []

    class Recorded(m.Env):                       # keeps every episode's environment, in the order they are made
        def __init__(self, params):
            super().__init__(params)
            made.append(self)

    m.Env, orig = Recorded, m.Env
    try:
        rows = [{"id": it["id"], "env": "multifile_repair", "difficulty": it.get("difficulty"), "params": it["params"]}
                for it in items]
        out = multiturn.run_episodes(llm, tok, rows, 0.0, cfg["context_tokens"])
    finally:
        m.Env = orig
    if len(made) != len(items):
        raise RuntimeError("the episodes do not line up with the projects")
    res = []
    for it, env, reward, turns, comp in zip(items, made, out["env_reward"], out["turns"], out["completion_ids"]):
        graded = env.result is not None
        res.append({"id": it["id"], "difficulty": it.get("difficulty"), "reward": float(reward or 0.0),
                    "graded": graded, "calls": env.calls, "turns": turns, "tokens": len(comp),
                    "functions": env.result["functions"] if graded else None,
                    "grade_error": env.result["error"] if graded else None,
                    "transcript": tok.decode(comp)})
    return res


def generate(model, items, cfg, out_path, gpu_util=0.85, repeat=False):
    """play() with vLLM. With repeat, every project is played a second time and the share of identical episodes is
    returned too (it must be 1)."""
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    # the engine in this process: every round's requests are queued before its first step, so the batches (and the
    # replies) are the same on every run
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model)
    template = open(os.path.join(model, "chat_template.jinja")).read()
    if tok.chat_template != template:
        tok.chat_template = template                 # the pinned template, whatever the tokenizer config says
    kw = {}
    mcfg = json.load(open(os.path.join(model, "config.json")))
    if "vision_config" in mcfg:                      # an image-text (or image-audio-text) checkpoint: text prompts only
        kw["limit_mm_per_prompt"] = {"image": 0, **({"audio": 0} if mcfg.get("audio_config") else {})}
    if vllm_overrides(model):
        kw["hf_overrides"] = vllm_overrides(model)
    llm = LLM(model=model, tokenizer=model, max_model_len=cfg["context_tokens"], gpu_memory_utilization=gpu_util,
              seed=0, generation_config="vllm", trust_remote_code=False, enable_prefix_caching=False, **kw)
    rows = play(llm, tok, items, cfg)
    summary = {"n": len(rows), "gen_secs": round(time.time() - t0),
               "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1"}
    if repeat:
        again = play(llm, tok, items, cfg)
        summary["repeat_identical"] = round(sum(a["transcript"] == b["transcript"] and a["reward"] == b["reward"]
                                                for a, b in zip(rows, again)) / max(1, len(rows)), 4)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def replay(items, cfg, n_per_difficulty=None):
    """The environment without a model: each project's fix played as tool calls must score 1, an immediate submit()
    must score 0. -> (projects checked, failures)."""
    _, envs, _ = load_kit(cfg)
    m = envs.load("multifile_repair")
    picked, count = [], {}
    for it in items:
        d = it.get("difficulty")
        if n_per_difficulty is None or count.get(d, 0) < n_per_difficulty:
            picked.append(it)
            count[d] = count.get(d, 0) + 1
    bad = []
    for it in picked:
        for acts, want in ((m.expert(it["params"]), 1.0), (["submit()"], 0.0)):
            e = m.Env(it["params"])
            e.reset()
            r = 0.0
            for a in acts:
                _, r, done = e.step(a)
                if done:
                    break
            if r != want:
                bad.append((it["id"], acts[-1], r, e.result))
    return len(picked), bad


def summarize(rows):
    n = len(rows)
    by = {}
    for r in rows:
        by.setdefault(str(r["difficulty"]), []).append(r["reward"])
    return {"n": n, "solved": round(sum(r["reward"] for r in rows) / max(1, n), 4),
            "by_difficulty": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())},
            "not_graded": round(sum(not r["graded"] for r in rows) / max(1, n), 4),
            "mean_calls": round(sum(r["calls"] for r in rows) / max(1, n), 2),
            "mean_tokens": round(sum(r["tokens"] for r in rows) / max(1, n), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--repeat", action="store_true")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items) if l.strip()]
    generate(a.model, items, json.load(open(a.config)), a.out, a.gpu_util, a.repeat)


if __name__ == "__main__":
    main()
