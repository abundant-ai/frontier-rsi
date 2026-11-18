"""Check sample requests against the sequential program."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=(10, 100), default=10)
    parser.add_argument("--candidate", default="/app/submission/candidate.py")
    args = parser.parse_args()
    requests = json.loads(Path("/app/sample_prompts.json").read_text())[:args.count]
    expected_path = Path("/app/sample_expected.json")
    expected = json.loads(expected_path.read_text())[:args.count] if expected_path.exists() else None
    work = Path("/app/runs")
    work.mkdir(exist_ok=True)
    input_path, output_path = work / "sample-input.json", work / "sample-output.json"
    input_path.write_text(json.dumps(requests))
    output_path.unlink(missing_ok=True)
    if expected is None:
        baseline_path = work / "sample-baseline.json"
        subprocess.run([sys.executable, "/opt/runtime/run_candidate.py", "--candidate",
                        "/opt/runtime/baseline.py", "--input", str(input_path),
                        "--output", str(baseline_path)], check=True)
        expected = json.loads(baseline_path.read_text())
    start = time.monotonic()
    subprocess.run([sys.executable, "/opt/runtime/run_candidate.py", "--candidate", args.candidate,
                    "--input", str(input_path), "--output", str(output_path)], check=True)
    elapsed = time.monotonic() - start
    outputs = json.loads(output_path.read_text())
    if outputs != expected:
        bad = next((i for i, (a, b) in enumerate(zip(outputs, expected)) if a != b), None)
        raise SystemExit(f"Incorrect output at request {bad}; got {len(outputs)} requests, expected {len(expected)}")
    print(json.dumps({"correct": len(outputs), "seconds": elapsed,
                      "output_tokens_per_second": sum(x["max_new_tokens"] for x in requests) / elapsed}))


if __name__ == "__main__":
    main()
