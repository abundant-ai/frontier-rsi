"""Measure a model directory (such as one /app/train_rl.py saved) on a file of projects, as the trained model is
measured at the end: the same prompts, episode loop, limits and decoding, the same grading.

  python3 /app/score.py --model <dir> [--items /app/data/samples.jsonl] [--limit 20] [--out rows.jsonl]

`--model /opt/base` measures the original model. `--out` keeps every episode (transcript, tool calls, which functions
passed). The episodes are played in a separate process, which frees the GPU again when it ends. The episode loop and
the environment are the ones in /app/posttrain-kit (rollouts/multiturn.py, envs/multifile_repair.py, rl_evalgen/), as
shipped; the final measurement uses its own unchanged copy of them.
"""
import argparse, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default=os.path.join(HERE, "data", "samples.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="write one row per project (with the whole episode) here")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    a = ap.parse_args()
    cfg_path = os.path.join(HERE, "task_config.json")
    items = [json.loads(l) for l in open(a.items) if l.strip()]
    items = items[:a.limit] if a.limit else items
    tmp = tempfile.mkdtemp(prefix="score_")
    if os.path.realpath(a.model) == os.path.realpath(evaluate.BASE):
        model = evaluate.BASE
    else:
        reason, info = evaluate.check_model(a.model)
        if reason:
            sys.exit(f"this directory would not be accepted: {reason}")
        model = evaluate.assemble(a.model, os.path.join(tmp, "model"))
    with open(os.path.join(tmp, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    rows_path = os.path.join(tmp, "rows.jsonl")
    r = subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "generate", "--model", model, "--items",
                        os.path.join(tmp, "items.jsonl"), "--config", cfg_path, "--out", rows_path,
                        "--gpu-util", str(a.gpu_util)])
    if r.returncode != 0:
        sys.exit("playing the projects failed (see the output above)")
    rows = [json.loads(l) for l in open(rows_path)]
    if a.out:
        with open(a.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(json.dumps(evaluate.summarize(rows)))


if __name__ == "__main__":
    main()
