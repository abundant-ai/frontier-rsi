#!/bin/bash
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
cd /
python - <<'VERIFY'
import json,math
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
p=Path('/logs/verifier/reward.json')
if not p.is_file():raise RuntimeError('No completed grading reward; inspect infrastructure_error.json')
r=json.loads(p.read_text())
assert r.get('valid') in (0,1)
assert all(type(v) in (int,float) and math.isfinite(v) for v in r.values())
if r['valid']==0:assert 'reward' not in r
else:assert 'reward' in r
spec=spec_from_file_location('raw_scorer','/tests/score.py');scorer=module_from_spec(spec);spec.loader.exec_module(scorer)
m=json.loads(Path('/logs/verifier/metrics.json').read_text())
assert scorer.score_metrics(m,json.loads(Path('/tests/reward_config.json').read_text()))==r
print(json.dumps(r))
VERIFY
