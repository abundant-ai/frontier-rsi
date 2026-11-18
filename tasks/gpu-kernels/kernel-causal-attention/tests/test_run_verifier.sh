#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
cd /
python - <<'PYVERIFY'
import json
import math
from pathlib import Path
path = Path('/logs/verifier/reward.json')
if not path.is_file():
    raise RuntimeError('No grading result; inspect infrastructure_error.json')
reward = json.loads(path.read_text())
assert reward.get('valid') in (0, 1)
assert type(reward.get('passed')) is bool
score = reward.get('reward')
assert type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
assert reward['passed'] == (score == 1)
assert reward['valid'] == 1 or reward['reward'] == 0
print(json.dumps(reward))
PYVERIFY
