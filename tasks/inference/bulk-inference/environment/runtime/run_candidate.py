"""Run the submitted pipeline as a separate, supervised process."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="/app/submission/candidate.py")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/opt/assets/model")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("submission", args.candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = json.loads(Path(args.input).read_text())
    outputs = module.generate_all(requests, args.model, "cuda:0")
    Path(args.output).write_text(json.dumps(outputs, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
