"""Check custom attention cases or time a complete GPU-resident sweep."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from practice_cases import default_cases, read_cases


def run(command: list[str], log_path: Path, deadline: float) -> bool:
    """Run one phase within a shared deadline and kill its process group on timeout.

    Args:
        command: Worker or independent-checker invocation.
        log_path: Destination for stdout and stderr.
        deadline: Absolute monotonic deadline for the entire command.

    Returns:
        Whether the phase exited successfully before the deadline.
    """
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=max(.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return False
    return process.returncode == 0


def main() -> None:
    """Validate case specifications, run isolated phases, and summarize diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120,
                        help="Wall-clock budget for compilation, execution and checking together")
    parser.add_argument("--seed", type=int, default=101, help="Combined with each case's seed offset")
    parser.add_argument("--out", type=Path, default=Path("/app/runs/latest"))
    parser.add_argument("--cases", type=Path, help="JSON list of custom dimensions and optional seed offsets")
    parser.add_argument("--correctness-only", action="store_true", help="Check cases individually without timing a sweep")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, help="Number of evenly selected examples; defaults to all cases")
    selection.add_argument("--indices", help="Comma-separated case indices")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 0 <= args.seed < 2**63:
        parser.error("Use a positive time budget and a nonnegative seed below 2**63")
    try:
        cases = read_cases(args.cases) if args.cases else default_cases()
        count = args.count if args.count is not None else len(cases)
        if not 1 <= count <= len(cases):
            raise ValueError("Count exceeds the available case list")
        indices = (list(map(int, args.indices.split(","))) if args.indices is not None else
                   [0] if count == 1 else [round(i * (len(cases)-1) / (count-1)) for i in range(count)])
        if not indices or len(set(indices)) != len(indices) or any(not 0 <= i < len(cases) for i in indices):
            raise ValueError("Case indices must be distinct and within the supplied list")
        indices.sort()
        cases = [cases[index] for index in indices]
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=False)
    case_path, config_path = args.out / "cases.json", args.out / "contract.json"
    case_path.write_text(json.dumps(cases, indent=2))
    contract = json.loads(Path("/app/contract.json").read_text())
    contract["workloads"] = len(cases)
    config_path.write_text(json.dumps(contract, indent=2))
    common = ["--cases", str(case_path), "--config", str(config_path),
              "--seed", str(args.seed), "--out", str(args.out)]
    driver = str(Path(__file__).with_name("practice_driver.py"))
    deadline = time.monotonic() + args.seconds
    phase = "outputs" if args.correctness_only else "measure"
    computed = run([sys.executable, driver, phase, *common], args.out / "worker.log", deadline)
    checked = computed and run([sys.executable, driver, "check", *common], args.out / "check.log", deadline)
    summary: dict[str, object] = {
        "mode": "correctness_only" if args.correctness_only else "sweep",
        "requested_indices": indices, "case_count": len(cases),
        "all_requested_completed": bool(checked), "correctness_passed": bool(checked),
        "timed_out": time.monotonic() >= deadline,
        "correctness_report": str(args.out / "correctness.json"),
        "worker_log": str(args.out / "worker.log"), "check_log": str(args.out / "check.log"),
    }
    record = args.out / ("execution.json" if args.correctness_only else "measurements.json")
    if computed and record.exists():
        report = json.loads(record.read_text())
        summary["peak_allocated_bytes"] = report["peak_allocated_bytes"]
        if not args.correctness_only:
            summary.update(sweep_ms=report["sweep_ms"],
                           latency_note="Latency applies to the selected shapes and includes GPU input copies, forward, loss and backward.")
        else:
            summary["memory_scope"] = "one_case_at_a_time; not a whole-sweep memory measurement"
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not checked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
