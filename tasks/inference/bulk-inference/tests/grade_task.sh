#!/bin/bash
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH=/opt/runtime
cd /
python /tests/grade.py
