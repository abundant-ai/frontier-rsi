"""Trusted measurement finalization and offline reward calculation; no GPU imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SCHEMA = 'rsi-pretrain-measurements-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def score_metrics(metrics, config=None):
    """Recompute rewards from full validated measurements, never from cached reward."""
    config = {'mode': 'raw'} if config is None else config
    if metrics.get('measurement_schema') != SCHEMA:
        raise ValueError('unsupported measurement schema')
    if metrics.get('measurement_profile') != 'full':
        raise ValueError('diagnostics cannot be scored as full grading')
    if type(metrics.get('valid')) is not bool:
        raise ValueError('explicit validity required')
    if not metrics['valid']:
        return {'valid': 0, 'disqualified': 1}
    spec = metrics['measurement_spec']
    expected = metrics['provenance']['grading_seeds']
    records = metrics['seed_measurements']
    if (len(expected) != 2 or len(set(expected)) != 2 or len(records) != 2
            or [r['seed'] for r in records] != expected):
        raise ValueError('exactly two matching independent seeds required')
    if spec['direction'] not in ('minimize', 'maximize'):
        raise ValueError('unsupported direction')
    values = []
    for record in records:
        samples = record['values']
        if len(samples) != spec['measurements_per_seed'] or not all(finite(v) for v in samples):
            raise ValueError('missing or nonfinite measurements')
        if any(v < 0 for v in samples) or (spec['unit'] == 'fraction' and any(v > 1 for v in samples)):
            raise ValueError('measurement outside metric range')
        if spec['measurements_per_seed'] > 1 and any(v <= 0 for v in samples):
            raise ValueError('timings must be positive')
        values.extend(samples)
    raw = statistics.mean(values)
    mode = config.get('mode')
    if mode == 'raw':
        reward = -raw if spec['direction'] == 'minimize' else raw
    elif mode == 'linear':
        starter, reference = config.get('starter'), config.get('reference')
        if not finite(starter) or not finite(reference):
            raise ValueError('finite natural-unit calibration anchors required')
        if not (reference < starter if spec['direction'] == 'minimize' else reference > starter):
            raise ValueError('reference must improve on starter')
        reward = (raw - starter) / (reference - starter)
        if config.get('clip', True):
            reward = max(0.0, min(1.0, reward))
    else:
        raise ValueError('unsupported scoring mode')
    if not finite(reward):
        raise ValueError('nonfinite reward')
    return {'valid': 1, 'reward': reward}


def finalize_metrics(report, valid, tests, submission):
    """Attach immutable identities and extract measurements from trusted grader results."""
    tests = Path(tests)
    spec = json.loads((tests / 'measurement_spec.json').read_text())
    frozen = json.loads((tests / 'measurement_protocol.json').read_text())
    seeds = json.loads((tests / 'seeds.json').read_text())
    if digest(tests / 'measurement_spec.json') != frozen['files']['tests/measurement_spec.json']:
        raise ValueError('measurement specification differs from frozen package')
    provenance = {
        'task_id': spec['task_id'],
        'task_package_sha256': frozen['task_package_sha256'],
        'measurement_protocol_sha256': digest(tests / 'measurement_protocol.json'),
        'candidate_sha256': digest(submission),
        'grading_seeds': seeds,
        'checkpoint_retention': False,
    }
    evaluation = Path('/opt/assets/private/eval.npy')
    if evaluation.is_file():provenance['evaluation_dataset_sha256'] = digest(evaluation)
    result = dict(report, measurement_schema=SCHEMA, measurement_profile='full',
                  valid=bool(valid), measurement_spec=spec, provenance=provenance,
                  validity_checks={'grader_checks_passed': bool(valid),
                                   'failure_reason': report.get('failure_reason')})
    result['seed_measurements'] = []
    if not valid:
        return result
    runs = report.get('runs', [])
    if len(runs) != 2 or [r['seed'] for r in runs] != seeds:
        raise ValueError('incomplete grader seed evidence')
    for run in runs:
        if spec['measurements_per_seed'] == 1:
            values = [run['metric']]
        elif spec['metric'] == 'training_step_time':
            values = run['repeat_scores']
        else:
            values = run['research_validation']['repeated_timing']['seconds_by_sweep']
        values = [v * spec['source_unit_multiplier'] for v in values]
        result['seed_measurements'].append({'seed': run['seed'], 'values': values,
                                            'mean': statistics.mean(values)})
    # Validate completeness with the same routine used by offline scoring.
    reward = score_metrics(result)
    result['raw_reward'] = reward['reward']
    result['raw_score'] = statistics.mean(v for r in result['seed_measurements'] for v in r['values'])
    result['mean'] = result['raw_score']
    result['seed_scores'] = [record['mean'] for record in result['seed_measurements']]
    result['sample_stddev'] = statistics.stdev(result['seed_scores'])
    result['raw_score_unit'] = spec['unit']
    result['aggregation'] = 'arithmetic_mean_all_measurements_equal_seed_weight'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    config = json.loads(args.config.read_text()) if args.config else {'mode': 'raw'}
    result = score_metrics(metrics, config)
    encoded = json.dumps(result, allow_nan=False) + '\n'
    if args.out:
        args.out.write_text(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
