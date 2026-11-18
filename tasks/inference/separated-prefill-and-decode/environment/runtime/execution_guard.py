"""Trusted OS-process watchdog for candidate callbacks, including C-level stalls.

This enforces deadlines; the separate policy loaders enforce the restricted
language. It is not a general-purpose sandbox for arbitrary Python programs.
"""
import contextlib
import math
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

_CONNECTION = None
_LOCK = threading.Lock()


def _connection():
    global _CONNECTION
    if _CONNECTION is None:
        value = os.environ.get('RSI_SUPERVISION_FD')
        if value is None:
            raise RuntimeError('candidate execution requires the trusted supervisor')
        _CONNECTION = socket.socket(fileno=int(value))
    return _CONNECTION


def _receive_exact(sock, count):
    result = bytearray()
    while len(result) < count:
        part = sock.recv(count - len(result))
        if not part:
            raise RuntimeError('supervision channel closed')
        result.extend(part)
    return bytes(result)


@contextlib.contextmanager
def deadline(seconds):
    """Arm an independent watchdog before entering untrusted computation."""
    if not math.isfinite(seconds) or not 0 < seconds <= 180:
        raise ValueError('invalid candidate deadline')
    with _LOCK:
        sock = _connection()
        sock.sendall(b'B' + struct.pack('!d', seconds))
        if _receive_exact(sock, 1) != b'A':
            raise RuntimeError('watchdog did not acknowledge callback')
        try:
            yield
        finally:
            sock.sendall(b'E')
            if _receive_exact(sock, 1) != b'A':
                raise RuntimeError('watchdog did not acknowledge completion')


def supervise(command, *, overall_seconds=3550):
    parent, child = socket.socketpair()
    env = dict(os.environ, RSI_SUPERVISION_FD=str(child.fileno()))
    process = subprocess.Popen(command, env=env, pass_fds=(child.fileno(),), start_new_session=True)
    child.close()
    parent.setblocking(False)
    buffer = bytearray()
    active_deadline = None
    active_budget = None
    overall_deadline = time.monotonic() + overall_seconds
    timed_out = None
    try:
        while True:
            now = time.monotonic()
            cutoff = min(overall_deadline, active_deadline or overall_deadline)
            ready, _, _ = select.select([parent], [], [], max(0, min(.1, cutoff - now)))
            if ready:
                part = parent.recv(4096)
                if not part:
                    break
                buffer.extend(part)
                while buffer:
                    kind = buffer[0]
                    required = 9 if kind == ord('B') else 1
                    if len(buffer) < required:
                        break
                    if active_deadline is not None and time.monotonic() >= active_deadline:
                        timed_out = f'candidate callback exceeded {active_budget:g} seconds'
                        break
                    if kind == ord('B') and active_deadline is None:
                        seconds = struct.unpack('!d', bytes(buffer[1:9]))[0]
                        if not math.isfinite(seconds) or not 0 < seconds <= 180:
                            raise RuntimeError('invalid watchdog request')
                        active_budget = seconds
                        active_deadline = time.monotonic() + seconds
                    elif kind == ord('E') and active_deadline is not None:
                        active_deadline = active_budget = None
                    else:
                        raise RuntimeError('invalid watchdog protocol state')
                    del buffer[:required]
                    parent.sendall(b'A')
            if timed_out:
                break
            now = time.monotonic()
            if active_deadline is not None and now >= active_deadline:
                timed_out = f'candidate callback exceeded {active_budget:g} seconds'
                break
            if process.poll() is not None:
                break
            if now >= overall_deadline:
                timed_out = 'supervised process exceeded its overall time limit'
                break
        if timed_out:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            print(timed_out, file=sys.stderr)
            return 124
        return process.wait(timeout=10)
    finally:
        parent.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def ensure_supervised():
    """Launch a separate process to supervise candidate callback deadlines."""
    if 'RSI_SUPERVISION_FD' in os.environ:
        return
    raise SystemExit(supervise([sys.executable, '-I', *sys.argv]))
