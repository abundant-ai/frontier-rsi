#!/bin/bash
# Verifier entrypoint (runs as root). Always exits 0: the harness reads reward.json / metrics.json, which grade.py
# writes only when it finishes; without a reward the trial is a verifier error, not a score.
#
# What the agent session may have left behind cannot reach the grading:
# - Before /tests is copied in, the task's collect hook (task.toml, [[verifier.collect]]) stops every process of the
#   agent user, empties /logs/verifier and notes when it ran (/var/lib/ptrl/session_end). This script does the same
#   again first, in case the hook did not run.
# - Only system paths and root's home are used, and Python runs isolated (-I: no user site-packages, no PYTHON*
#   variables, no script directory on the path), so no program or module the agent planted runs as root.
# - /logs/verifier, where the harness reads the reward, is writable by everyone while the agent works: everything in
#   it except the harness's own output is deleted before grading, so only what grade.py writes counts.
# - grade.py refuses a submission changed after the session was over (the stamp below).
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root PYTHONNOUSERSITE=1 \
       PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE LD_PRELOAD BASH_ENV ENV CDPATH
cd /
PY="/usr/bin/python3 -I"
# the stamp: the earliest moment the session was over that the container can tell -- when the collect hook started,
# or when the test files were copied in (only times after the container started count: files baked into the image
# say nothing about the copy); without either, now
STAMP=$($PY -c 'import os,time
now=time.time(); ts=[now]
try: ts.append(float(open("/var/lib/ptrl/session_end").read().split()[0]))
except Exception: pass
try:
    ticks=int(open("/proc/1/stat").read().rsplit(")",1)[1].split()[19])
    born=next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))+ticks/os.sysconf("SC_CLK_TCK")
except Exception:
    born=now
for d,_,fs in os.walk("/tests"):
    for p in [d]+[os.path.join(d,f) for f in fs]:
        try: c=os.lstat(p).st_ctime
        except OSError: continue
        if c>=born: ts.append(c)
print(repr(min(ts)))')
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
    [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null
done
for _k in $(seq 1 10); do
    pkill -KILL -u agent 2>/dev/null
    sleep 1
    ps -u agent -o stat= 2>/dev/null | grep -v Z | grep -q . || break
done
chmod -R go-rwx /tests 2>/dev/null || true
[ -L /logs/verifier ] && rm -f /logs/verifier
find /logs/verifier -mindepth 1 -maxdepth 1 ! -name test-stdout.txt ! -name test-stderr.txt -exec rm -rf {} + \
    2>/dev/null
mkdir -p /logs/verifier && chown root:root /logs/verifier && chmod 755 /logs/verifier
echo "$STAMP" > /logs/verifier/.grading_started
if ps -u agent -o stat= 2>/dev/null | grep -v Z | grep -q .; then
    echo "agent processes could not be stopped: no reward (verifier error)"
    exit 0
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
$PY /tests/grade.py 2>&1 | tee /logs/verifier/grade.log
echo "reward: $(cat /logs/verifier/reward.json 2>/dev/null || echo "none (verifier error)")"
exit 0
