#!/usr/bin/env bash
# Revision 3 (docs/ANALYSIS_PLAN_REVISION3.md): every new analysis, in order. Every stage is
# resumable (finished predictions and permutations are kept), so the script can be re-run after an
# interruption. Needs the revision-2 predictions (scripts/run_pipeline.sh, or the Zenodo archive)
# for the stages marked (*), and the Yalcin et al. archive for fa_yalcin.py:
#   git clone https://github.com/m1237/automatic-cybersickness-detection data/raw/yalcin
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
[ -z "${OMP_NUM_THREADS:-}" ] && export OMP_NUM_THREADS=1
: "${FA_NJOBS:=4}"; export FA_NJOBS
YALCIN_DIR="${YALCIN_DIR:-data/raw/yalcin}"
RUN3="python src/fa_rev3.py $UFFCS_RAW_DIR $UFFCS_SELECTED_DIR data results"

$RUN3 decode main control posctl within vrsq0      # stages 1-4 and 7 (VRSQ)
$RUN3 null_lr null_xgb                             # stage 3: label-transplant nulls (1,000 and 400 permutations)
$RUN3 hgb_control sens_group                       # stage 3 (HGB, HGB-reg) and stage 5   (*)
python src/fa_vrlens.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV" data results ablation   # stage 7
# addendum after the pre-submission review (docs/ANALYSIS_PLAN_REVISION3_ADDENDUM.md)
$RUN3 rotation pairs decode_nr main_nr pen         # B2 rotation channel, A3 pair composition, B5 scaled penalty
$RUN3 posctl_seeds                                 # B1 power of the decision rule (20 planted-channel seeds)
$RUN3 null_nm null_nr null_pen null_block          # B3, B2c, B5, B6: 200-permutation secondary nulls
# post hoc check (docs/ANALYSIS_PLAN_REVISION3_ADDENDUM2.md): VR-LENS with group-aware internal stacking folds
python src/fa_vrlens.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV" data results groupcv
python src/fa_yalcin.py "$YALCIN_DIR" results       # stage 6: second archive (B7 offset design)
python src/fa_sim.py results                        # stage 6: simulation (B4 episodic labels)
python src/fa_rev3_stats.py data results            # intervals, permutation p-values, power
