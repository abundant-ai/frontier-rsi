#!/bin/bash
set -euo pipefail
# Root verifier shells do not inherit the CUDA image's Conda PATH reliably.
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
bash /tests/stage_data.sh
bash /tests/run_candidate.sh
bash /tests/run_verifier.sh
