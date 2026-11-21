#!/bin/bash
# rsi-bench reward. Runs the task's own grader (grade_task.sh, the previous test.sh, unchanged), then turns its
# raw metric into the benchmark reward with the 90% best-trial rule: the best valid model trial in the rsi-bench
# collection scores 1, 90% of its performance or worse scores 0, linear in between (reward_90pct.py; the best value
# is fixed in reward_90pct.json). The grader's own reward.json is kept as reward_grader.json.
bash /tests/grade_task.sh
code=$?
py=""
for c in /opt/conda/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then py="$c"; break; fi
done
if [ -n "$py" ] && "$py" -I /tests/reward_90pct.py /tests/reward_90pct.json /logs/verifier; then
    :
else
    echo "reward_90pct: could not apply the reward rule; reward 0" >&2
    mkdir -p /logs/verifier
    if [ -f /logs/verifier/reward.json ] && [ ! -f /logs/verifier/reward_grader.json ]; then
        cp /logs/verifier/reward.json /logs/verifier/reward_grader.json
    fi
    printf '{"reward": 0.0, "valid": 0}\n' > /logs/verifier/reward.json
    if [ -f /logs/verifier/reward.txt ]; then printf '0.0\n' > /logs/verifier/reward.txt; fi
fi
exit "$code"
