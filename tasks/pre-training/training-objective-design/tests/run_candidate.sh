#!/bin/bash
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
cd /
export PYTHONPATH=/opt/runtime
export HOME=/root HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONSTARTUP PYTHONUSERBASE HF_TOKEN HUGGING_FACE_HUB_TOKEN
# Root orchestrates; it never imports candidate.py. The runner UID executes it.
python /tests/grade.py
