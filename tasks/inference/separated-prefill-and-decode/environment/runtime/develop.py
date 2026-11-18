import sys
sys.path.insert(0, '/opt/runtime')

from execution_guard import ensure_supervised
ensure_supervised()

import argparse
import json
from pathlib import Path
from policy import load
from prefix_service import frozen
from prefill_decode_service import workload, evaluate_pipeline


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='/app/submission/candidate.py')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--out', default='/app/runs/latest')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    devices = ['cpu', 'cpu'] if args.smoke else ['cuda:0', 'cuda:1']
    first, tokenizer = frozen(devices[0], smoke=args.smoke)
    second, _ = frozen(devices[1], smoke=args.smoke)
    jobs = workload(tokenizer, args.seed, quick=True, smoke=args.smoke)
    result = evaluate_pipeline((first, second), devices, load(args.candidate), jobs, quick=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'measurement.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
