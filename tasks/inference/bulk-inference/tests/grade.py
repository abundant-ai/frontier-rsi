"""Trusted final verifier for all 500 held-out requests."""

import hashlib
import json
import os
import pwd
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

INPUT = Path("/tests/holdout_prompts.json")
EXPECTED = Path("/tests/holdout_expected.json")
OUT = Path("/logs/verifier")
LIMIT_SECONDS = 3600


def set_agent_identity():
    agent = pwd.getpwnam("agent")
    os.setgroups([])
    os.setgid(agent.pw_gid)
    os.setuid(agent.pw_uid)
    resource.setrlimit(resource.RLIMIT_FSIZE, (8_000_000, 8_000_000))


def validate(outputs, expected, requests):
    if not isinstance(outputs, list) or len(outputs) != len(expected):
        return "Missing or extra responses"
    for index, (actual, reference, request) in enumerate(zip(outputs, expected, requests)):
        if not isinstance(actual, list) or len(actual) != request["max_new_tokens"]:
            return f"Response {index} has the wrong length"
        if any(type(token) is not int or token < 0 for token in actual):
            return f"Response {index} is not token IDs"
        if actual != reference:
            return f"Response {index} differs from reference"
    return None


def sample_gpu_memory(stop, samples):
    while not stop.wait(1.0):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            samples.extend(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass


def read_agent_output(path, max_bytes=4_000_000):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != pwd.getpwnam("agent").pw_uid
                or info.st_nlink != 1 or info.st_size > max_bytes):
            raise ValueError("response is not a regular agent-owned file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            return json.loads(stream.read(max_bytes + 1).decode("utf-8"))
    finally:
        os.close(fd)


def hash_agent_source(path, max_bytes=2_000_000):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != pwd.getpwnam("agent").pw_uid
                or info.st_nlink != 1 or info.st_size > max_bytes):
            raise ValueError("submission is not a regular agent-owned file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("submission exceeds the size limit")
        return hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)


def main():
    if OUT.is_symlink():
        raise RuntimeError("verifier output path is a symlink")
    OUT.mkdir(parents=True, exist_ok=True)
    os.chown(OUT, 0, 0)
    OUT.chmod(0o700)
    for filename in ("reward.json", "reward.txt", "metrics.json", "failure.json"):
        (OUT / filename).unlink(missing_ok=True)
    requests = json.loads(INPUT.read_text())
    expected = json.loads(EXPECTED.read_text())
    if len(requests) != 500 or len(expected) != 500:
        raise RuntimeError("verifier assets are incomplete")
    if any(len(tokens) != req["max_new_tokens"] for tokens, req in zip(expected, requests)):
        raise RuntimeError("verifier reference length mismatch")
    # The candidate is an untrusted process. Keep the answer file root-only.
    os.chown(EXPECTED, 0, 0)
    EXPECTED.chmod(0o600)
    candidate_path = Path("/app/submission/candidate.py")
    try:
        candidate_sha256 = hash_agent_source(candidate_path)
    except (OSError, ValueError):
        candidate_sha256 = None
    work = Path(tempfile.mkdtemp(prefix="bulk-inference-", dir="/tmp"))
    try:
        work.chmod(0o755)
        input_path, output_path = work / "input.json", work / "output.json"
        input_path.write_text(json.dumps(requests))
        input_path.chmod(0o444)
        output_path.parent.chmod(0o777)
        command = [sys.executable, "/opt/runtime/run_candidate.py", "--input", str(input_path),
                   "--output", str(output_path)]
        memory_samples = []
        stop_sampling = threading.Event()
        sampler = threading.Thread(target=sample_gpu_memory, args=(stop_sampling, memory_samples), daemon=True)
        sampler.start()
        started = time.monotonic()
        try:
            with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
                process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file,
                                           start_new_session=True, preexec_fn=set_agent_identity,
                                           env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
                try:
                    process.wait(timeout=LIMIT_SECONDS)
                    elapsed = time.monotonic() - started
                    if process.returncode != 0:
                        reason = "Candidate process failed"
                    elif not output_path.exists():
                        reason = "Candidate did not write responses"
                    else:
                        try:
                            outputs = read_agent_output(output_path)
                            reason = validate(outputs, expected, requests)
                        except (UnicodeError, ValueError, OSError):
                            reason = "Malformed response file"
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    elapsed = time.monotonic() - started
                    reason = "One-hour candidate time limit exceeded"
                stdout_file.seek(0, os.SEEK_END)
                stdout_file.seek(max(0, stdout_file.tell() - 4000))
                stderr_file.seek(0, os.SEEK_END)
                stderr_file.seek(max(0, stderr_file.tell() - 4000))
                stdout_tail = stdout_file.read().decode(errors="replace")[-3000:]
                stderr_tail = stderr_file.read().decode(errors="replace")[-3000:]
        finally:
            stop_sampling.set()
            sampler.join(timeout=6)
        if reason:
            reward = 0.0
            (OUT / "failure.json").write_text(json.dumps({"reason": reason,
                "stderr_tail": stderr_tail}) + "\n")
        else:
            reward = 1.0 / (1.0 + elapsed / 60.0)
        metrics = {"valid": reason is None, "elapsed_seconds": elapsed,
                   "output_tokens": sum(x["max_new_tokens"] for x in requests),
                   "output_tokens_per_second": sum(x["max_new_tokens"] for x in requests) / elapsed,
                   "request_count": len(requests), "time_limit_seconds": LIMIT_SECONDS,
                   "peak_gpu_memory_mb": max(memory_samples) if memory_samples else None,
                   "candidate_stdout_tail": stdout_tail,
                   "candidate_sha256": candidate_sha256,
                   "holdout_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                   "expected_sha256": hashlib.sha256(EXPECTED.read_bytes()).hexdigest()}
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (OUT / "reward.json").write_text(json.dumps({"reward": reward, "valid": int(reason is None)}) + "\n")
        (OUT / "reward.txt").write_text(f"{reward}\n")
        print(json.dumps({"reward": reward, "valid": int(reason is None), "seconds": elapsed}), flush=True)
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
