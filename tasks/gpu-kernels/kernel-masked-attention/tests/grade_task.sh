#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1
# Root verifier shells do not inherit the CUDA image's Conda PATH reliably.
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
bash /tests/test_stage_data.sh
if [ -f /logs/verifier/staging_invalid.json ]; then
    bash /tests/test_run_verifier.sh 2>&1 | tee /logs/verifier/test-stdout.txt
    exit 0
fi
bash /tests/test_run_candidate.sh
mkdir -p /logs/verifier
bash /tests/test_run_verifier.sh 2>&1 | tee /logs/verifier/test-stdout.txt
