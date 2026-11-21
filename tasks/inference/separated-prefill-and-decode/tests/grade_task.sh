#!/bin/bash
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH=/opt/runtime
export PYTHONNOUSERSITE=1
export HOME=/root
unset RSI_SUPERVISION_FD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /
# The agent and verifier share a container. Stop agent-owned background jobs and
# make the verdict directory root-only before launching trusted Python.
pkill -KILL -u agent 2>/dev/null || true
if [ -L /logs/verifier ] || [ ! -d /logs/verifier ]; then rm -f /logs/verifier; fi
install -d -o root -g root -m 700 /logs/verifier
chown root:root /logs
chmod go-w /logs
if python -I /tests/grade.py; then
    exit 0
else
    code=$?
fi
if [ "$code" -eq 124 ]; then
    python -I - <<'PY'
import hashlib
import json
from pathlib import Path

out = Path('/logs/verifier')
reason = 'supervised execution exceeded a callback or overall time limit'
failure = {'type': 'CandidateTimeout', 'reason': reason}
metrics = {'schema_version': 1, 'valid': False,
           'calibration_status': 'uncalibrated', 'failure_reason': reason,
           'candidate_sha256': None,
           'config_sha256': hashlib.sha256(Path('/tests/config.json').read_bytes()).hexdigest(),
           'grader_sha256': hashlib.sha256(Path('/tests/grade.py').read_bytes()).hexdigest()}
reward = {'valid': 0, 'disqualified': 1, 'reward': -1e30}
for name, payload in [('failure.json', failure), ('metrics.json', metrics),
                      ('reward.json', reward)]:
    (out / name).write_text(json.dumps(payload) + '\n')
(out / 'reward.txt').write_text(str(reward['reward']) + '\n')
PY
    exit 0
fi
exit "$code"
