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
# Why step 1 runs on a filtered view of results/preds/: src/fa_stats.py computes the statistics of the earlier
# analysis. It skips the learners of the present analysis (lroff, lroffp, xgboff, ctxb), whose statistics come from
# src/fa_rev3_stats.py, but it predates the strongly penalized learner of addendum 3 (lroffs, 14 prediction files,
# statistics in src/fa_add3_stats.py) and does not skip it. Given all prediction files it therefore adds 14 configs
# that src/fa_report.py cannot place (KeyError: 'CTXB+MOT'); this is also why scripts/check_results.sh, kept unchanged
# from development, fails. Given the files it was written for (all except *__lroffs.npz), it reproduces the shipped
# results/stats.json byte for byte. No research code is changed.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
[ -d data/raw/UFFCSData/RAW_DATABASE/RAW_DATABASE ] || { echo "run scripts/prepare_data.sh first"; exit 1; }
[ -f .env ] || cp .env.example .env
set -a; . ./.env; set +a
[ -z "${OMP_NUM_THREADS:-}" ] && unset OMP_NUM_THREADS

echo "== 1/6 fa_stats.py (earlier analysis; predictions of the learners it covers)"
VIEW=$(mktemp -d); trap 'rm -rf "$VIEW"' EXIT
mkdir -p "$VIEW/results/preds"
for f in "$ROOT"/results/preds/*.npz; do
  case "$f" in *__lroffs.npz) ;; *) ln -s "$f" "$VIEW/results/preds/" 2>/dev/null || cp "$f" "$VIEW/results/preds/";; esac
done
python src/fa_stats.py data "$VIEW/results"
cp "$VIEW/results/stats.json" results/stats.json

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
