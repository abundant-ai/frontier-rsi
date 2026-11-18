#!/bin/bash
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
cd /
# Harbor stages /tests only after the agent phase. Stop old agent/worker processes
# before inspecting private material or accepting a submission snapshot.
for account in agent runner runner2; do
    pkill -KILL -u "$account" 2>/dev/null || true
done
# SIGKILL is asynchronous, and container PID 1 may retain dead zombies.
# Zombies cannot execute or access files; any surviving live UID blocks staging.
python - <<'PYWAIT'
import os,pwd,signal,time
from pathlib import Path
uids={pwd.getpwnam(name).pw_uid for name in ('agent','runner','runner2')}
for attempt in range(51):
    live=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            status=(entry/'status').read_text()
            fields=dict(line.split(':',1) for line in status.splitlines() if ':' in line)
            if int(fields['Uid'].split()[0]) in uids and fields['State'].strip().split()[0] not in ('Z','X'):
                live.append(int(entry.name))
        except (FileNotFoundError,ProcessLookupError):continue
    if not live:break
    if attempt==50:raise SystemExit('Untrusted live processes remain alive')
    for pid in live:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    time.sleep(.1)
PYWAIT
chmod -R go-rwx /tests /opt/assets/private
install -d -m 700 /logs/verifier
install -d -m 755 /opt/submission
python - <<'PY'
import os,stat
from pathlib import Path
src=Path('/app/submission/candidate.py')
try:
    fd=os.open(src,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('not a regular candidate file')
        data=f.read(256*1024+1)
    if len(data)>256*1024:raise ValueError('candidate.py too large')
except (OSError,ValueError):
    # An absent, irregular or oversized deliverable is a candidate failure.
    # An empty snapshot lets the unprivileged worker report it as invalid.
    data=b''
dst=Path('/opt/submission/candidate.py');dst.write_bytes(data);dst.chmod(0o444)
PY

# The runner reads the immutable snapshot above. Hide agent-written research files
# so candidate code cannot load a precomputed checkpoint during graded training.
chmod 0700 /app/submission /app/runs /home/agent
# Harbor leaves agent research logs in world-writable directories. The graded
# worker must not read a checkpoint that the agent placed there before staging.
# Open without following symlinks and change the directory itself; root still
# writes /logs/verifier, while the worker UIDs cannot traverse these paths.
python - <<'PYSEALLOGS'
import os
for path in ('/logs/agent', '/logs/artifacts'):
    try:
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    except FileNotFoundError:
        continue
    try:os.fchmod(fd,0o700)
    finally:os.close(fd)
PYSEALLOGS
python - <<'PYSEAL'
import os,stat
from pathlib import Path
agent_uid=1000
for root in (Path('/tmp'),Path('/var/tmp'),Path('/dev/shm')):
    if not root.exists():continue
    for directory,dirs,files in os.walk(root,topdown=True,followlinks=False):
        for name in (*dirs,*files):
            path=Path(directory)/name
            try:
                info=path.lstat()
                if info.st_uid!=agent_uid or stat.S_ISLNK(info.st_mode):continue
                os.chmod(path,info.st_mode & ~0o077,follow_symlinks=False)
            except FileNotFoundError:continue
PYSEAL
