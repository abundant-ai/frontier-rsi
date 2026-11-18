"""Trusted two-seed inference grading with supervised callbacks and raw evidence."""
import argparse
import json
import math
import shutil
import statistics
import hashlib
import secrets
import sys
from pathlib import Path

sys.path[:0] = ['/tests', '/opt/runtime']
from execution_guard import ensure_supervised
ensure_supervised()

from policy import CandidateError, load
from measurement_protocol import PROTOCOL, configure_cpu_threads, environment_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='/app/submission/candidate.py')
    parser.add_argument('--config', default='/tests/config.json')
    parser.add_argument('--output', default='/logs/verifier')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    candidate_sha256 = None
    try:
        for name in ('reward.json', 'reward.txt', 'failure.json', 'infrastructure_error.json',
                     'measurements.json', 'metrics.json', 'environment-before.json', 'progress.json'):
            path = out / name
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        if (config['quick'] or len(config['seeds']) != 2 or len(set(config['seeds'])) != 2
                or config['workloads_per_seed'] != 4 or config['repetitions_per_workload'] != 5):
            raise RuntimeError('full grading requires two seeds, four workloads and five repetitions')
        if config['warmup_seed'] in config['seeds']:
            raise RuntimeError('warmup must use separate inputs')
        workload_nonce = secrets.randbits(128)
        def request_seed(seed: int, workload_index: int) -> int:
            encoded = f'{workload_nonce}:{seed}:{workload_index}'.encode()
            return int.from_bytes(hashlib.sha256(encoded).digest(), 'big')
        configure_cpu_threads()
        from prefill_decode_service import evaluate_pipeline, workload
        from prefix_service import frozen

        devices = ['cpu', 'cpu'] if args.smoke else ['cuda:0', 'cuda:1']
        first, tokenizer = frozen(devices[0], smoke=args.smoke)
        second, _ = frozen(devices[1], smoke=args.smoke)
        policy = load(args.candidate, 'plan')
        candidate_sha256 = policy.source_sha256

        def evaluate(seed: int, workload_index: int, repetitions: int) -> dict:
            """Time the request stream and check every output and resource limit."""
            jobs = workload(tokenizer, request_seed(seed, workload_index), quick=args.smoke, smoke=args.smoke)
            digest = hashlib.sha256(repr(jobs).encode()).hexdigest()
            result = evaluate_pipeline((first, second), devices, policy, jobs, quick=args.smoke,
                                       repetitions=repetitions)
            return {'workload_sha256': digest, **result}
        before = environment_metadata()
        (out / 'environment-before.json').write_text(json.dumps(before, indent=2) + '\n')
        # Warm up on separate inputs before recording either seed.
        warmup = evaluate(config['warmup_seed'], 0, 1)
        measured = []
        all_timings = []
        workload_hashes = set()
        for seed in config['seeds']:
            suite = []
            for workload_index in range(config['workloads_per_seed']):
                result = evaluate(seed, workload_index, config['repetitions_per_workload'])
                if len(result['measurements']) != config['repetitions_per_workload']:
                    raise RuntimeError('missing candidate repetitions')
                digest = result['workload_sha256']
                if digest in workload_hashes:
                    raise RuntimeError('measured workloads must be distinct')
                workload_hashes.add(digest)
                timings = [float(row['elapsed_seconds']) for row in result['measurements']]
                if not all(math.isfinite(value) and value > 0 for value in timings):
                    raise RuntimeError('invalid candidate timing')
                all_timings.extend(timings)
                suite.append({'workload_index': workload_index, 'mean_seconds': statistics.mean(timings), **result})
            measured.append({'seed': seed, 'workloads': suite,
                             'mean_seconds': statistics.mean(row['mean_seconds'] for row in suite)})
            (out / 'progress.json').write_text(json.dumps({'seeds': measured}) + '\n')
        if len(all_timings) != 40 or not all(math.isfinite(value) and value > 0 for value in all_timings):
            raise RuntimeError('missing or invalid candidate timings')
        mean_seconds = statistics.mean(all_timings)
        raw_score = -mean_seconds
        record = {'schema_version': 1, 'protocol': PROTOCOL, 'valid': True,
                  'calibration_status': 'uncalibrated',
                  'candidate_sha256': policy.source_sha256,
                  'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                  'grader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'quick_validation': args.smoke, 'raw_score': raw_score,
                  'raw_metric': mean_seconds, 'unit': 'seconds per 24-request stream',
                  'direction': 'minimize',
                  'seeds': measured, 'workload_nonce': str(workload_nonce),
                  'workloads_per_seed': config['workloads_per_seed'],
                  'repetitions_per_workload': config['repetitions_per_workload'],
                  'aggregation': 'negative_mean_of_all_timed_streams',
                  'warmup': {'seed': config['warmup_seed'], **warmup},
                  'environment_before': before, 'environment_after': environment_metadata()}
        (out / 'measurements.json').write_text(json.dumps(record, indent=2) + '\n')
        (out / 'metrics.json').write_text(json.dumps(record, indent=2) + '\n')
        result = {'valid': 1, 'reward': raw_score, 'raw_score': raw_score}
    except CandidateError as exc:
        (out / 'failure.json').write_text(json.dumps({'type': type(exc).__name__, 'reason': str(exc)}) + '\n')
        (out / 'metrics.json').write_text(json.dumps({'schema_version': 1, 'valid': False,
            'calibration_status': 'uncalibrated', 'failure_reason': str(exc),
            'candidate_sha256': candidate_sha256,
            'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            'grader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}) + '\n')
        result = {'valid': 0, 'disqualified': 1, 'reward': -1e30}
    except Exception as exc:
        (out / 'infrastructure_error.json').write_text(json.dumps({'type': type(exc).__name__, 'reason': str(exc)}) + '\n')
        raise
    if args.smoke:
        print(json.dumps({'valid': result['valid'], 'smoke_only': True}), flush=True)
        return
    (out / 'reward.json').write_text(json.dumps(result) + '\n')
    if 'reward' in result:
        (out / 'reward.txt').write_text(str(result['reward']) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
