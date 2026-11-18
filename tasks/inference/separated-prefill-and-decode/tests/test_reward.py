"""Check the raw-score verifier using controlled, GPU-free timing evidence."""
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def module(name, attributes):
    value = ModuleType(name)
    value.__dict__.update(attributes)
    return value


def test_reward(task):
    tests = task / 'tests'
    config = json.loads((tests / 'config.json').read_text())
    branch = config['kind'] == 'branches'
    calls = []

    def replay(*args, repetitions, **kwargs):
        index = len(calls)
        calls.append(index)
        seconds = 1.0 + index / 10.0
        rows = [{'role': 'candidate', 'elapsed_seconds': seconds, 'repetition': repetition,
                 'workload_sha256': f'workload-{index}'} for repetition in range(repetitions)]
        return {'measurements': rows, 'latency_seconds': seconds}

    noop = lambda *args, **kwargs: None
    replacements = {
        'execution_guard': module('execution_guard', {'ensure_supervised': noop}),
        'measurement_protocol': module('measurement_protocol', {'PROTOCOL': 'test-protocol',
            'configure_cpu_threads': noop, 'environment_metadata': dict}),
        'policy': module('policy', {'CandidateError': ValueError, 'load': lambda *a: SimpleNamespace(source_sha256='a' * 64)}),
        'numeric_policy': module('numeric_policy', {'load': lambda *a: SimpleNamespace(source_sha256='a' * 64)}),
        'prefix_service': module('prefix_service', {'frozen': lambda *a, **k: (None, None)}),
        'branch_service': module('branch_service', {'workloads': noop, 'evaluate_branches': replay}),
        'prefill_decode_service': module('prefill_decode_service', {'workload': lambda _tokenizer, seed, **_kwargs: (seed,), 'evaluate_pipeline': replay}),
    }
    with TemporaryDirectory() as directory, patch.dict(sys.modules, replacements), patch.object(sys, 'argv',
            ['grade.py', '--candidate', str(task / 'environment/app/submission/candidate.py'),
             '--config', str(tests / 'config.json'), '--output', directory]), redirect_stdout(StringIO()):
        (Path(directory) / 'environment-before.json').mkdir()
        (Path(directory) / 'progress.json').mkdir()
        runpy.run_path(str(tests / 'grade.py'), run_name='__main__')
        reward = json.loads((Path(directory) / 'reward.json').read_text())
        metrics = json.loads((Path(directory) / 'metrics.json').read_text())
    expected_calls = 9
    expected_count = 40
    expected = -statistics.mean(1.0 + index / 10.0 for index in range(1, expected_calls))
    assert len(calls) == expected_calls
    assert reward == {'valid': 1, 'reward': expected, 'raw_score': expected}
    assert metrics['raw_score'] == expected
    assert metrics['calibration_status'] == 'uncalibrated'
    assert metrics['valid'] is True
    assert len(metrics['seeds']) == 2
    if branch:
        assert sum(len(seed['workloads']) for seed in metrics['seeds']) == 8
        assert sum(len(workload['measurements']) for seed in metrics['seeds'] for workload in seed['workloads']) == expected_count
    else:
        assert sum(len(seed['workloads']) for seed in metrics['seeds']) == 8
        assert sum(len(workload['measurements']) for seed in metrics['seeds'] for workload in seed['workloads']) == expected_count


def test_invalid_is_scoreless(task):
    tests = task / 'tests'
    def reject(*args, **kwargs):
        raise ValueError('candidate output mismatch')
    noop = lambda *args, **kwargs: None
    def remove_source_after_load(path, *_):
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source.unlink()
        return SimpleNamespace(source_sha256=digest)
    replacements = {
        'execution_guard': module('execution_guard', {'ensure_supervised': noop}),
        'measurement_protocol': module('measurement_protocol', {'PROTOCOL': 'test-protocol',
            'configure_cpu_threads': noop, 'environment_metadata': dict}),
        'policy': module('policy', {'CandidateError': ValueError, 'load': remove_source_after_load}),
        'numeric_policy': module('numeric_policy', {'load': lambda *a: SimpleNamespace(source_sha256='a' * 64)}),
        'prefix_service': module('prefix_service', {'frozen': lambda *a, **k: (None, None)}),
        'prefill_decode_service': module('prefill_decode_service', {'workload': noop, 'evaluate_pipeline': reject}),
    }
    with TemporaryDirectory() as directory, patch.dict(sys.modules, replacements), patch.object(sys, 'argv',
            ['grade.py', '--candidate', str(Path(directory) / 'candidate.py'),
             '--config', str(tests / 'config.json'), '--output', directory]), redirect_stdout(StringIO()):
        (Path(directory) / 'candidate.py').write_bytes((task / 'environment/app/submission/candidate.py').read_bytes())
        runpy.run_path(str(tests / 'grade.py'), run_name='__main__')
        reward = json.loads((Path(directory) / 'reward.json').read_text())
        metrics = json.loads((Path(directory) / 'metrics.json').read_text())
    assert reward == {'valid': 0, 'disqualified': 1, 'reward': -1e30}
    assert metrics['valid'] is False and 'raw_score' not in metrics
    assert metrics['candidate_sha256'] == hashlib.sha256((task / 'environment/app/submission/candidate.py').read_bytes()).hexdigest()


if __name__ == '__main__':
    test_reward(Path(__file__).resolve().parents[1])
    test_invalid_is_scoreless(Path(__file__).resolve().parents[1])
