#!/usr/bin/env bash
# Full pipeline, from the raw UFFCSData archive to the verified numbers of the paper.
# Refits every model: about 6 h (main run) plus about 30 h (revision-2 stages) on 2 CPU cores.
# All stages are resumable: re-running skips finished models.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${UFFCS_RAW_DIR:?Set UFFCS_RAW_DIR in .env (see .env.example)}"
: "${UFFCS_SELECTED_DIR:?Set UFFCS_SELECTED_DIR in .env}"
: "${UFFCS_DATASET_CSV:?Set UFFCS_DATASET_CSV in .env}"
[ -z "${OMP_NUM_THREADS:-}" ] && unset OMP_NUM_THREADS
mkdir -p data results/logs

echo "== 1/5 parse the archive, form inferred participant groups, build features"
python src/fa_data.py "$UFFCS_RAW_DIR" data "$UFFCS_SELECTED_DIR"

echo "== 2/5 fit all models (leakage gate first)"
python src/fa_run.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" data results
python src/fa_run.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" data results checks control hgbreg posctl_reg null500 null_lr500

echo "== 3/5 optional: VR-LENS re-implementation (needs xgboost)"
if python -c "import xgboost" 2>/dev/null; then
  python src/fa_vrlens.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV" data results
else
  echo "   xgboost not installed: skipped (pip install xgboost to run it)"
fi

echo "== 4/5 statistics (bootstrap, tests)"
python src/fa_stats.py data results

echo "== 5/5 numbers, tables, figures and claim checks"
python src/fa_report.py . "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV"
python src/fa_verify.py .
