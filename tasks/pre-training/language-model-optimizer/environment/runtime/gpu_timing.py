"""Time GPU work after giving the host time to enqueue it.

A CUDA start event submitted to an idle stream can execute before Python has
submitted the candidate kernels. Its elapsed time then includes host dispatch
and scheduling gaps. A delay *before* that event keeps those gaps outside the
measurement. Calls remain eager: stateful kernels are not graph-replayed.
"""
import math
import os
import subprocess
import time

import torch

MAX_DELAY_CYCLES = 50_000_000


def queued_cuda_call(fn, delay_cycles=2_000_000, max_attempts=3):
    """Return the real result, GPU milliseconds, and queue diagnostics.

    Retry a timing sample if host submission outlasted the queueing allowance.
    Three GHz is a conservative upper clock bound for the pinned H100. No
    result is cached, and a slow/synchronizing implementation is not rejected:
    its final observed interval remains the measurement after bounded attempts.
    Submitted kernels must use max_attempts=1 and carry next_delay_cycles to
    their next fresh input so each timing sample uses distinct input.
    The allowance is capped because a synchronizing function waits for the
    queue delay itself; feeding that wait back without a cap grows forever.
    """
    if max_attempts not in (1, 2, 3):
        raise ValueError('one to three timing attempts required')
    delay_cycles = min(max(1, int(delay_cycles)), MAX_DELAY_CYCLES)
    for attempt in range(max_attempts):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(delay_cycles)
        host_start = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        host_seconds = time.perf_counter() - host_start
        torch.cuda.synchronize()
        milliseconds = start.elapsed_time(end)
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise ValueError('nonfinite or nonpositive GPU timing')
        prefilled = host_seconds <= delay_cycles / 3_000_000_000 / 2
        next_delay = delay_cycles if prefilled else min(MAX_DELAY_CYCLES, max(delay_cycles * 2, math.ceil(host_seconds * 3_000_000_000 * 2.5)))
        if prefilled or attempt == max_attempts - 1:
            return result, milliseconds, {
                'delay_cycles': delay_cycles,
                'next_delay_cycles': next_delay,
                'host_enqueue_seconds': host_seconds,
                'queue_prefilled': prefilled,
                'retries': attempt,
            }
        delay_cycles = next_delay


def hardware_snapshot(timing_protocol='queued-cuda-events-v1'):
    props = torch.cuda.get_device_properties(0)
    info = {
        'name': props.name, 'sm_count': props.multi_processor_count,
        'total_memory': props.total_memory, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'cpu_count': os.cpu_count(),
        'timing_protocol': timing_protocol,
        'gpu_count': torch.cuda.device_count(),
    }
    if hasattr(os, 'sched_getaffinity'):
        info['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.limit,power.draw,temperature.gpu',
            '--format=csv',
        ], capture_output=True, text=True, timeout=10)
        info['nvidia_smi'] = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        info['telemetry_error'] = str(error)
    return info
