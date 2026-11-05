#!/usr/bin/env bash
# Regenerate every printed number, table and figure from the shipped result files and predictions, and verify
# them, without refitting any model (about 10-15 min; 9 min measured on 4 cores). It runs the research code unchanged:
#   1 src/fa_stats.py          intervals and tests of the earlier analysis   -> results/stats.json
#   2 src/fa_report.py         values.json, all tables and figures, claims    -> results/paper/, figures/
#                              (calls src/fa_report_rev3.py for the present analysis)
#   3 src/fa_report_add3.py    addendum 3 (post hoc P1-P7)                     -> results/paper/values_add3.json
#   4 src/fa_report_rq3res.py  RQ3 resolution summary (Table S38)              -> results/paper/values_rq3res.json
#   5 src/fa_report_sample.py  sample description (Table S39)                  -> results/paper/values_sample.json
#   6 src/fa_verify.py         recomputes printed values independently; must end with "RESULT: PASS"
# Finally it lists every regenerated file that differs from the shipped version (expected: none, apart from the
# creation-date metadata inside the PDF figures).
#
# Step 1: src/fa_stats.py computes the statistics of the earlier analysis and skips the learners whose statistics come
# from src/fa_rev3_stats.py and src/fa_add3_stats.py. Until 5 November 2026 its skip list lacked the addendum-3 learner
# lroffs, so this script used to give it a filtered view of results/preds/ (see verification/README.md, "Corrected
# defect"). With lroffs on the skip list it reads results/preds/ directly and reproduces results/stats.json byte for byte.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d data/raw/UFFCSData/RAW_DATABASE/RAW_DATABASE ] || { echo "run scripts/prepare_data.sh first"; exit 1; }
[ -f .env ] || cp .env.example .env
set -a; . ./.env; set +a
[ -z "${OMP_NUM_THREADS:-}" ] && unset OMP_NUM_THREADS

echo "== 1/6 fa_stats.py (earlier analysis)"
python src/fa_stats.py data results

echo "== 2/6 fa_report.py (values, tables, figures, claims)"
python src/fa_report.py . "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV"
echo "== 3/6 fa_report_add3.py"
python src/fa_report_add3.py .
echo "== 4/6 fa_report_rq3res.py"
python src/fa_report_rq3res.py .
echo "== 5/6 fa_report_sample.py"
python src/fa_report_sample.py .
echo "== 6/6 fa_verify.py"
python src/fa_verify.py . | tee verification/fa_verify_output.txt

if git rev-parse --git-dir >/dev/null 2>&1; then
  echo "== regenerated files that differ from the shipped versions:"
  CHANGED=$(git status --porcelain -- results results_logo figures data)
  if [ -z "$CHANGED" ]; then echo "   none"; else echo "$CHANGED"; fi
fi
