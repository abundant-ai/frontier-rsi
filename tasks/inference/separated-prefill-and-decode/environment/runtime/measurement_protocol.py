"""Common repetition ordering and telemetry for frozen-model inference measurements."""
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import socket
import statistics
import subprocess
import time

PROTOCOL = 'inference-two-seed-raw-mean-v2'


def measurement_policies(policy, clone, *, quick=False, repetitions=None, baseline=None, pair_offset=0):
    count = (1 if quick else 100) if repetitions is None else repetitions
    if not isinstance(count, int) or not 1 <= count <= 1000:
        raise ValueError('invalid measured repetition count')
    for repetition in range(count):
        roles = [('candidate', policy)]
        if baseline is not None:
            roles.append(('baseline', baseline))
            if (repetition + pair_offset) % 2:
                roles.reverse()
        for role, original in roles:
            # Recompile the original immutable source snapshot: no function
            # object, module namespace or closure from a preceding workload.
            yield repetition, role, clone(original)


def aggregate(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    """Return the mean valid candidate duration and retain every observation.

    Args:
        rows: Trusted timing records, tagged with role and repetition number.
        metric: Field containing absolute elapsed seconds in each record.
    Returns:
        Mean candidate latency and raw observations; optional paired diagnostics.
    Raises:
        RuntimeError: Candidate timings are missing, nonfinite or nonpositive.
    """
    candidate = [row for row in rows if row['role'] == 'candidate']
    baseline = [row for row in rows if row['role'] == 'baseline']
    if not candidate:
        raise RuntimeError('no measured candidate repetitions')
    if not all(isinstance(row[metric], (int, float)) and not isinstance(row[metric], bool) and math.isfinite(row[metric]) and row[metric] > 0 for row in candidate):
        raise RuntimeError('every candidate duration must be finite and positive')
    result = {'latency_seconds': statistics.mean(row[metric] for row in candidate),
              'measurements': candidate,
              'measurement_order': [[row['repetition'], row['role']] for row in rows]}
    if baseline:
        if len(candidate) != len(baseline):
            raise RuntimeError('incomplete paired baseline measurements')
        candidate_by_repetition = {row['repetition']: row[metric] for row in candidate}
        baseline_by_repetition = {row['repetition']: row[metric] for row in baseline}
        if len(candidate_by_repetition) != len(candidate) or set(candidate_by_repetition) != set(baseline_by_repetition):
            raise RuntimeError('paired baseline repetitions do not match candidate repetitions')
        speedups = []
        for repetition in sorted(candidate_by_repetition):
            candidate_metric = candidate_by_repetition[repetition]
            baseline_metric = baseline_by_repetition[repetition]
            if not all(math.isfinite(value) and value > 0 for value in (candidate_metric, baseline_metric)):
                raise RuntimeError('paired metrics must be finite and positive')
            speedups.append({'repetition': repetition, 'baseline_over_candidate': baseline_metric / candidate_metric})
        result['paired_speedup'] = statistics.median(row['baseline_over_candidate'] for row in speedups)
        result['paired_speedup_repetitions'] = speedups
        result['comparison'] = {
            'latency_seconds': statistics.median(row[metric] for row in baseline), 'measurements': baseline}
    return result


def configure_cpu_threads():
    import torch
    # Use the same single-thread CPU tensor setting for baseline/candidate;
    # independent GPU worker threads remain part of the serving implementation.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def environment_metadata():
    import torch
    data = {'protocol': PROTOCOL, 'captured_at_unix': time.time(),
            'hostname': socket.gethostname(), 'platform': platform.platform(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'cpu_count': os.cpu_count(), 'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'gpus': [], 'runtime_sha256': {}}
    if hasattr(os, 'sched_getaffinity'):
        data['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        data['gpus'].append({'index': i, 'name': p.name, 'memory_bytes': p.total_memory,
                             'compute_capability': [p.major, p.minor], 'multiprocessors': p.multi_processor_count})
    for path in sorted(Path(__file__).parent.glob('*.py')):
        data['runtime_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    queries = {
        'gpu_telemetry': ['nvidia-smi', '--query-gpu=uuid,name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu', '--format=csv,noheader'],
        'gpu_topology': ['nvidia-smi', 'topo', '-m'],
        'gpu_processes': ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'],
        'cpu_description': ['lscpu'],
    }
    for key, command in queries.items():
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=5)
            data[key] = {'returncode': completed.returncode, 'stdout': completed.stdout[:16000],
                         'stderr': completed.stderr[:2000]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            data[key] = {'unavailable': type(exc).__name__}
    return data
