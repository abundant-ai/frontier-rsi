"""Greedy evaluation of a model on environment episodes (the same episode loop as training, temperature 0).

  python3 -m envs.evaluate_env --model <dir> [--tokenizer <dir>] --rows <dir of env_*.jsonl> --out r.json [--limit 50]
Reports the success rate per environment and difficulty, mean turns, and keeps every transcript in <out>.episodes.jsonl.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768, help="tokens per episode (capped at CONTEXT minus the prompt)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM
    from rollouts.multiturn import run_episodes
    from rollouts.multiturn import CONTEXT
    rows = []
    for f in sorted(Path(a.rows).glob("env_*.jsonl")):
        rs = [json.loads(l) for l in open(f) if l.strip()]
        rows += rs[: a.limit] if a.limit else rs
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    llm = LLM(model=a.model, tokenizer=a.tokenizer or a.model, max_model_len=CONTEXT, gpu_memory_utilization=0.85, seed=0)
    out = run_episodes(llm, tok, rows, 0.0, a.max_tokens)
    by = {}
    for r, rew, t in zip(rows, out["env_reward"], out["turns"]):
        by.setdefault(f"{r['env']}/{r['difficulty']}", []).append((rew, t))
    res = {"success": {k: round(sum(x for x, _ in v) / len(v), 4) for k, v in sorted(by.items())},
           "turns": {k: round(sum(t for _, t in v) / len(v), 2) for k, v in sorted(by.items())},
           "overall": round(sum(out["env_reward"]) / max(1, len(rows)), 4), "n": len(rows)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    with open(Path(a.out).with_suffix(".episodes.jsonl"), "w") as f:
        for r, ids, rew in zip(rows, out["completion_ids"], out["env_reward"]):
            f.write(json.dumps({"id": r["id"], "env": r["env"], "reward": rew, "transcript": tok.decode(ids)}) + "\n")
    print("ENV_EVAL", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
