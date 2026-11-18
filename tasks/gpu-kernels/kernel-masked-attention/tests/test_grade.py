"""Root orchestrator for the configured GPU sweeps and independent output checks."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from test_sweep_reward import MeasurementError, decide, validate_target


class InvalidSubmission(Exception):
    """The candidate failed a numerical, resource, or execution requirement."""


def read_json(path: Path) -> object:
    """Read a bounded regular JSON file without following symbolic links.

    Args:
        path: Expected private artifact or task configuration path.

    Returns:
        The decoded JSON value for validation by its caller.

    Raises:
        MeasurementError: The path is not a bounded regular file.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as file:
        metadata = os.fstat(file.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 8*1024**2:
            raise MeasurementError("Invalid measurement artifact")
        return json.load(file)


def read_object(path: Path) -> dict[str, object]:
    """Read an object-shaped artifact and validate its top-level keys.

    Args:
        path: Expected JSON object path.

    Returns:
        String-keyed JSON fields whose values require field-specific checking.

    Raises:
        MeasurementError: The artifact does not contain an object.
    """
    result = read_json(path)
    if not isinstance(result, dict) or any(not isinstance(key, str) for key in result):
        raise MeasurementError("Expected a JSON object")
    return result


def execute(command: list[str], log_path: Path, timeout: float, runner: bool,
            *, infrastructure_probe: bool = False) -> None:
    """Run an isolated measurement or trusted checker and retain diagnostics.

    Args:
        command: Fully qualified executable and arguments.
        log_path: Root-owned log output file.
        timeout: Remaining grading wall-clock budget.
        runner: Whether to drop to the submission runner UID.
        infrastructure_probe: Whether this is the root-owned environment probe,
            which never loads the candidate or reads its artifacts.

    Raises:
        InvalidSubmission: Candidate execution, correctness, or deadline failed.
        MeasurementError: GPU infrastructure prevented evaluation.
    """
    if infrastructure_probe and (runner or command != [sys.executable, "/tests/test_check_environment.py"]):
        raise MeasurementError("Invalid trusted environment probe")
    environment = {key: value for key, value in os.environ.items() if key in (
        "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")}
    environment.update(PYTHONPATH="/tests", PYTHONNOUSERSITE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    if runner:
        # /tests remains inaccessible; use the runtime staged after agent exit.
        environment.update(PYTHONPATH="/run/kernel-tests", HOME=command[-1],
                           TORCHINDUCTOR_CACHE_DIR=command[-1]+"/compile")
        command = ["setpriv", "--reuid=1001", "--regid=1001", "--clear-groups", "--no-new-privs", *command]
    with log_path.open("w+b") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=max(1., timeout))
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            if infrastructure_probe:
                raise MeasurementError("Environment probe exceeded its deadline") from error
            raise InvalidSubmission("Grading exceeded the execution deadline") from error
        finally:
            if runner:
                subprocess.run(["pkill", "-KILL", "-u", "1001"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if code:
        if infrastructure_probe:
            raise MeasurementError("Trusted environment probe failed; see probe log")
        # Candidate text and exit codes carry no authority to classify an outage.
        # Only an independent root-owned probe can establish an environment fault.
        check_environment(log_path.with_suffix(".environment.log"), min(60., timeout))
        raise InvalidSubmission("Candidate execution or numerical checking failed; see worker logs")


def check_environment(log_path: Path, timeout: float = 60.) -> None:
    """Check GPU health without importing candidate code or reading its outputs.

    Args:
        log_path: Root-owned destination for the probe's diagnostic output.
        timeout: Maximum time allowed for the separate trusted process.

    Raises:
        MeasurementError: The independent environment check fails or times out.
    """
    execute([sys.executable, "/tests/test_check_environment.py"], log_path, timeout,
            runner=False, infrastructure_probe=True)


def secure_outputs(directory: Path, workloads: int, memory_gib: float) -> None:
    """Transfer completed worker artifacts to root before numerical checking.

    Args:
        directory: Private output directory after all runner processes exit.
        workloads: Number of expected case output files.
        memory_gib: Upper bound used to reject oversized artifacts.

    Raises:
        InvalidSubmission: A required output is missing or is not a regular file.
    """
    os.chown(directory, 0, 0)
    directory.chmod(0o700)
    names = ["measurements.json", *(f"case-{index}.safetensors" for index in range(workloads))]
    for name in names:
        path = directory / name
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > memory_gib*1024**3:
            raise InvalidSubmission("Unsafe or oversized output artifact")
        os.chown(path, 0, 0, follow_symlinks=False)
        path.chmod(0o600)


def grade(tests: Path, logs: Path) -> tuple[dict[str, int | float | bool], dict[str, object]]:
    """Measure and check every seeded sweep required by the private contract.

    Args:
        tests: Private task files.
        logs: Root-owned verifier artifact directory.

    Returns:
        Continuous reward and all seed durations with variation statistics.

    Raises:
        MeasurementError: Task configuration or infrastructure is incomplete.
        InvalidSubmission: The candidate fails execution or correctness checks.
    """
    if os.geteuid() != 0:
        raise MeasurementError("The verifier must run as root")
    contract = read_object(tests / "test_contract.json")
    target = read_object(tests / "test_performance_target.json")
    seeds = read_json(tests / "test_seeds.json")
    count, memory = contract.get("workloads"), contract.get("memory_gib")
    seed_count = contract.get("grading_seed_count")
    if (type(count) is not int or type(memory) not in (int, float)
            or type(seed_count) is not int or seed_count < 1
            or not isinstance(seeds, list) or len(seeds) != seed_count
            or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != seed_count):
        raise MeasurementError("Invalid task or seed configuration")
    validate_target(target, count, seed_count=seed_count)
    check_environment(logs / "environment-before.log")
    workspace = Path("/run/rsi-kernel-workers")
    workspace.mkdir(mode=0o711, parents=True, exist_ok=True)
    worker_logs = logs / "worker-logs"
    worker_logs.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    deadline = time.monotonic() + 7100
    # Semantic cases are untimed and checked in a process that never imports
    # candidate code. Their shapes do not change the resident timing sweeps.
    from test_correctness_cases import CASE_COUNT
    output = workspace / f"structured-{os.getpid()}"
    output.mkdir(mode=0o700, exist_ok=False)
    os.chown(output, 1001, 1001)
    execute([sys.executable, "/run/kernel-tests/test_correctness_worker.py",
             "--candidate", "/opt/submission/candidate.py", "--config",
             "/run/kernel-tests/test_contract.json", "--out", str(output)],
            worker_logs / "structured.log", deadline-time.monotonic(), runner=True)
    secure_outputs(output, CASE_COUNT, float(memory))
    check_path = logs / "structured-correctness.json"
    execute([sys.executable, "/tests/test_check_correctness.py",
             "--directory", str(output), "--config", str(tests / "test_contract.json"), "--out", str(check_path)],
            worker_logs / "structured-check.log", deadline-time.monotonic(), runner=False)
    checked = read_object(check_path)
    if checked.get("correctness_passed") is not True or checked.get("case_count") != CASE_COUNT:
        raise MeasurementError("Structured check coverage is incomplete")
    for index in range(CASE_COUNT):
        (output / f"case-{index}.safetensors").unlink()
    # Added semantic cases do not participate in the timed CUDA graph.
    from test_additional_cases import LABELS
    additional_count = len(LABELS[contract["operation"]])
    additional_output = workspace / f"additional-{os.getpid()}"
    additional_output.mkdir(mode=0o700, exist_ok=False)
    os.chown(additional_output, 1001, 1001)
    execute([sys.executable, "/run/kernel-tests/test_additional_worker.py",
             "--candidate", "/opt/submission/candidate.py", "--config",
             "/run/kernel-tests/test_contract.json", "--out", str(additional_output)],
            worker_logs / "additional.log", deadline-time.monotonic(), runner=True)
    secure_outputs(additional_output, additional_count, float(memory))
    additional_check = logs / "additional-correctness.json"
    execute([sys.executable, "/tests/test_check_additional.py", "--directory", str(additional_output),
             "--config", str(tests / "test_contract.json"), "--out", str(additional_check)],
            worker_logs / "additional-check.log", deadline-time.monotonic(), runner=False)
    checked = read_object(additional_check)
    if checked.get("correctness_passed") is not True or checked.get("case_count") != additional_count:
        raise MeasurementError("Additional check coverage is incomplete")
    for index in range(additional_count):
        (additional_output / f"case-{index}.safetensors").unlink()
    for number, seed in enumerate(seeds, 1):
        output = workspace / f"seed-{number}-{os.getpid()}"
        output.mkdir(mode=0o700, exist_ok=False)
        os.chown(output, 1001, 1001)
        execute([sys.executable, "/run/kernel-tests/test_sweep_worker.py", "--candidate", "/opt/submission/candidate.py",
                 "--config", "/run/kernel-tests/test_contract.json", "--seed", str(seed), "--out", str(output)],
                worker_logs / f"seed-{number}.log", deadline-time.monotonic(), runner=True)
        secure_outputs(output, count, float(memory))
        report = read_object(output / "measurements.json")
        check_path = logs / f"seed-{number}-correctness.json"
        execute([sys.executable, "/tests/test_check_sweep.py", "--directory", str(output),
                 "--config", str(tests / "test_contract.json"), "--out", str(check_path)],
                worker_logs / f"seed-{number}-check.log", deadline-time.monotonic(), runner=False)
        checked = read_object(check_path)
        if (report.get("seed") != seed or checked.get("seed") != seed
                or checked.get("correctness_passed") is not True
                or checked.get("case_indices") != list(range(count))):
            raise MeasurementError("Independent check did not cover the complete sweep")
        report.update(correctness_passed=True, status="completed")
        records.append(report)
        (logs / "measurements.json").write_text(json.dumps(records, indent=2, allow_nan=False))
        for index in range(count):
            (output / f"case-{index}.safetensors").unlink()
    return decide(records, target, seeds, count, float(memory), seed_count=seed_count)


def main() -> None:
    """Write a reward only after valid evaluation; infrastructure errors get none."""
    tests, logs = Path("/tests"), Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("reward.json", "reward.txt", "infrastructure_error.json"):
        (logs / name).unlink(missing_ok=True)
    try:
        reward, metrics = grade(tests, logs)
    except InvalidSubmission as error:
        reward = {"valid": 0, "passed": False, "reward": 0}
        metrics = {"passed": False, "failure_reason": str(error)}
    except Exception as error:
        (logs / "infrastructure_error.json").write_text(json.dumps({"error": str(error)}))
        raise
    (logs / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (logs / "reward.json").write_text(json.dumps(reward, allow_nan=False))
    print(json.dumps(reward))


if __name__ == "__main__":
    main()
