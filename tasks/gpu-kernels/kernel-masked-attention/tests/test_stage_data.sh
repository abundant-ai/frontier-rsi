#!/bin/bash
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
cd /
# The harness starts this verifier in a fresh container and restores candidate.py.
# Private tests are baked into this image, which is never used by the agent.
chmod -R go-rwx /tests
# Reset the verifier log directory before creating trusted grading state.
python - <<'PYCLEAN'
import os,shutil,stat
from pathlib import Path
log=Path('/logs/verifier')
if log.is_symlink() or (log.exists() and not log.is_dir()):
    log.unlink()
log.mkdir(parents=True,exist_ok=True)
os.chown(log,0,0)
log.chmod(0o700)
for child in log.iterdir():
    if child.is_dir() and not child.is_symlink():shutil.rmtree(child)
    else:child.unlink()
PYCLEAN
install -d -m 700 /logs/verifier
install -d -m 755 /opt/submission
python - <<'PY'
import json,os,stat
from pathlib import Path
src=Path('/app/submission/candidate.py')
try:
    if not stat.S_ISREG(src.lstat().st_mode):
        raise ValueError('candidate.py is not a regular file')
    fd=os.open(src,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise ValueError('candidate.py is not a regular file')
        data=f.read(256*1024+1)
    if len(data)>256*1024:raise ValueError('candidate.py exceeds 256 KiB')
except (OSError,ValueError) as error:
    reason='invalid submission: '+str(error)
    log=Path('/logs/verifier')
    (log/'staging_invalid.json').write_text(json.dumps({'reason':reason}))
    (log/'metrics.json').write_text(json.dumps({'valid':False,'failure_reason':reason,'passed':False}))
    (log/'reward.json').write_text(json.dumps({'valid':0,'passed':False,'reward':0}))
else:
    dst=Path('/opt/submission/candidate.py');dst.write_bytes(data);dst.chmod(0o444)
PY

# Stage the grading worker after freezing the transferred submission.
# Only root and the numerical runner can read this tree.
install -d -o root -g runner -m 750 /run/kernel-tests
cp /tests/*.py /run/kernel-tests/
cp /tests/test_contract.json /run/kernel-tests/test_contract.json
cp /tests/numerical_policy.json /run/kernel-tests/numerical_policy.json
chown -R root:runner /run/kernel-tests
chmod 750 /run/kernel-tests
chmod 640 /run/kernel-tests/*
