#!/usr/bin/env bash
# Recompute every number of the paper from the archived predictions, without refitting any model.
# Needs: data/ (derived tables) and the prediction archives from the Zenodo record, unpacked by
# scripts/unzip_predictions.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${UFFCS_RAW_DIR:?Set UFFCS_RAW_DIR in .env}"
[ -z "${OMP_NUM_THREADS:-}" ] && unset OMP_NUM_THREADS
python src/fa_stats.py data results
python src/fa_report.py . "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV"
python src/fa_verify.py .
