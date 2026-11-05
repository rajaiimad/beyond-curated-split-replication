# Reproduction guide

There are three tiers of reproduction. Tier 1 is enough to check every number in the manuscript. Tiers 2 and 3 re-run parts of the analysis, or all of it, from the raw data.

All commands are run from the repository root, after the installation steps in `README.md` (virtual environment, `pip install -r requirements.txt`, `bash scripts/prepare_data.sh`).

## Settings

**Configuration:** there are no separate configuration files.
- **In the code:** every analytical setting (feature windows, learners and their hyperparameters, folds, number of repeats, bootstrap replicates, permutations, seeds, decision rule) is a constant, fixed by the written analysis plans in `docs/`.
- **Environment variables:** the only run-time settings are in `.env` (copied from `.env.example`):

| Variable | Meaning | Default |
|---|---|---|
| `UFFCS_RAW_DIR` | Raw sessions | `data/raw/UFFCSData/RAW_DATABASE/RAW_DATABASE` |
| `UFFCS_SELECTED_DIR` | Curated sessions | `data/raw/UFFCSData/DATABASE_SELECTED_DATA/DATABASE` |
| `UFFCS_DATASET_CSV` | Curated table | `data/raw/UFFCSData/dataset.csv` |
| `FA_NJOBS` | Folds fitted in parallel | 2 (`run_rev3.sh`: 4) |
| `OMP_NUM_THREADS` | Threads per boosting fit | library default (`run_rev3.sh`: 1) |
| `FA_NSIM_LR`, `FA_NSIM_XGB`, `FA_NSEEDS` | Permutations of the LR+offset and boosting nulls, and planted-channel seeds | 1000, 400, 20, as in the paper |
| `FA_NULL_SPLIT` | Split point to run the 500-permutation null of the earlier analysis in two processes | 345 |

**Seeds:** every random seed is fixed in the code.
- **Earlier analysis:**
  - fold partitions: `numpy.random.default_rng(1000 + r)` for repeat r;
  - HGB and MLP: `random_state=0`;
  - label-shift control: seed 1;
  - label-transplant null: `700 + s`;
  - planted channel: `900 + t`;
  - bootstrap: 0.
- **Present analysis:**
  - boosting+offset (XGBoost): `seed=0`; HGB: `random_state=0`;
  - label-transplant nulls: `700 + s` (session transplant), `800 + s` (group-block transplant);
  - positive control: `900 + label`; its 20 planted-channel seeds: `5000 + 100 s + label`;
  - label-shift control: 1;
  - simulation: `2000 + rep` (step labels) and `3000 + rep` (episodic labels; episode lengths from `10000 + seed`);
  - VR-LENS: `random_state=0` for every learner and fold;
  - second archive: `random_state=0`, label-shift seed 1;
  - every other bootstrap and resampling step: 0.
- **Addendum 3:**
  - diffuse planted signal: `6000 + 100 s + label`;
  - VRSQ bootstrap and permutations: 0.

**Scripts kept from development:** `scripts/` holds the original entry points unchanged. Two of them need a word:
- **`check_results.sh` (and `.ps1`):** the original check. Since the correction of 5 November 2026 (`verification/README.md`, "Corrected defect") it runs to the end, but it does not regenerate the addendum-3, RQ3-resolution and sample files; `verification/verify_results.sh` does everything.
- **`unzip_predictions.sh`:** needed only for a zipped copy of the predictions; they are included here unzipped.

## Tier 1 — check every reported number (about 10–15 min, no refitting)

```bash
bash verification/verify_data.sh
bash verification/verify_results.sh
```

`verification/README.md` describes each step and its expected output. The run must end with `RESULT: PASS` and list no regenerated file as changed, apart from the creation-date metadata inside the PDF figures.

To regenerate only the tables and figures, run the report scripts directly:

```bash
set -a; . ./.env; set +a
python src/fa_report.py . "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV"   # values.json, tables, figures
python src/fa_report_add3.py .       # addendum 3: values_add3.json, tab_a3_*.tex
python src/fa_report_rq3res.py .     # Table S38: values_rq3res.json, tab_rq3_resolution.tex
python src/fa_report_sample.py .     # Table S39: values_sample.json, tab_sample.tex
python src/fa_verify.py .            # must end with RESULT: PASS
```

`docs/MANUSCRIPT_TRACEABILITY.md` names the file behind each table and figure.

## Tier 2 — partial re-runs from the raw data (minutes to hours)

First set up the shell with the thread settings of `scripts/run_rev3.sh`:

```bash
set -a; . ./.env; set +a
export OMP_NUM_THREADS=1 FA_NJOBS=4    # one thread per model, four models in parallel
```

Do not leave `OMP_NUM_THREADS` set but empty while models are fitted in parallel. Loading `.env` alone does exactly that, and the CPU can then be oversubscribed. In this repository's audit, the simulation took over an hour that way instead of under four minutes.

Each command below writes the file named on its right. Compare it with the shipped version using `git diff --stat` (or `git status`): an empty diff means the shipped result was reproduced exactly. Result files that record their run time (`secs`) differ in that field only; `src/fa_repro.py` compares them value by value without it (step 5 of Tier 3).

| Re-run | Command | Output | Time (4 cores) |
|---|---|---|---|
| Parse the archive, form the inferred groups, build the features | `python src/fa_data.py "$UFFCS_RAW_DIR" data "$UFFCS_SELECTED_DIR"` | `data/sessions.csv`, `data/features_w30.csv.gz` (then `bash verification/verify_data.sh`: identical, the feature table by content) | about 10 s |
| Second archive (needs `bash scripts/fetch_second_archive.sh`) | `python src/fa_yalcin.py data/raw/yalcin results` | `results/rev3_yalcin.json` | about 2 min |
| Simulation | `python src/fa_sim.py results` | `results/rev3_sim.json` | about 4 min |
| Present analysis, decoding and main fits | `python src/fa_rev3.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" data results decode main` | `results/rev3_decode.json`, `rev3_main.json` and their predictions | about 20 min |
| VR-LENS (needs xgboost) | `python src/fa_vrlens.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV" data results` | `results/vrlens.json` | about 15 min |

**Notes:**
- **Resumable stages:** the fitting stages skip models whose prediction file already exists. To refit one, delete or move its `.npz` file in `results/preds/` first; otherwise the stage only re-summarises the shipped predictions.
- **VR-LENS:** not reproducible bit for bit. Expect differences in the fourth to sixth decimal and no change in any printed value (Supplementary Note S2).
- **Leakage gate:** every fitting stage runs it and the mutation test first. Stage `checks` runs them alone: `python src/fa_run.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" data results checks`.

## Tier 3 — the full pipeline (several days of CPU time)

Run the stages in this order. Each stage is resumable, so an interrupted run can be restarted.

**1. Earlier analysis (revision 2):** `bash scripts/run_pipeline.sh`.
- The main run takes about 6 h on 2 cores.
- Its later stages:
  - HGB-reg: about 5 h;
  - HGB-reg positive control: about 2 h;
  - logistic-regression null: about 2.5 h;
  - 500-permutation HGB null: about 30 h on one core, or about 15 h split over two processes with `null500_lo` and `null500_hi`, then `null_merge`.
- VR-LENS runs if xgboost is installed (about 15 min).

**2. Present analysis (revision 3) and the first and second addenda:** `bash scripts/run_rev3.sh` (4 cores).
- Stage times:
  - `main`: about 20 min;
  - `null_lr`: about 3 h;
  - `null_xgb`: about 1.5 h;
  - `sens_group`: about 1 h;
  - `posctl_seeds` and the secondary nulls: several hours;
  - the rest: minutes.
- It runs, in order:
  - `decode main control posctl within vrsq0`;
  - `null_lr null_xgb`;
  - `hgb_control sens_group`;
  - VR-LENS `ablation`;
  - the first addendum (`rotation pairs decode_nr main_nr pen`, `posctl_seeds`, `null_nm null_nr null_pen null_block`);
  - the second addendum (VR-LENS `groupcv`);
  - the second archive and the simulation;
  - `src/fa_rev3_stats.py`.
- It needs the second archive (`bash scripts/fetch_second_archive.sh`) and the predictions of step 1.

**3. Third addendum (post hoc P1–P7):** about 5 h on 4 cores.

```bash
python src/fa_add3.py "$UFFCS_RAW_DIR" "$UFFCS_SELECTED_DIR" "$UFFCS_DATASET_CSV" results \
    vrsq vrlens_rw main_s null_p null_s null_block null_nr null_nm null_sess posctl6 posctl_sess diffuse
python src/fa_add3_stats.py data results       # results/add3_stats.json
python src/fa_add3_outputs.py results          # results/add3_family6.json, add3_sess.json
```

**4. Statistics, values, tables, figures and verification:** `bash verification/verify_results.sh`.

**5. Re-run comparison (optional):** compare re-computed result files with the shipped ones, value by value (run times excluded):
1. Copy each re-computed file to `<name>.rerun.json`.
2. Run `python src/fa_repro.py results HEAD results/<name>.rerun.json ...`.

This is how `results/reproducibility.json` was made: 14,933 values compared, 5 differing beyond 10⁻⁹, all in the VR-LENS file. The script overwrites that file; restore the shipped version with `git checkout results/reproducibility.json`.

**Not re-run by the pipeline:** the first, leave-one-group-out run of the earlier analysis (Supplementary Table S4). Its results are read from `results_logo/`.

## The written plans and what each stage implements

The plans are copied unchanged. The revision-2 plan names the scripts by their location at the time (`code/`); they are now in `src/`.

| Plan | Stages |
|---|---|
| `docs/ANALYSIS_PLAN_REVISION2.md` (earlier analysis) | `fa_run.py`: `checks`, `main`, `ctx`, `mitig`, `cell`, `control`, `rowwise`, `vrsq`, `sens`, `logo_ctx`, `null`, `null_lr`, `posctl`, `es_check`, `hgbreg`, `posctl_reg`, `null500`, `null_lr500` |
| `docs/ANALYSIS_PLAN_REVISION3.md` (present analysis) | `fa_rev3.py`: `decode`, `main`, `control`, `hgb_control`, `posctl`, `within`, `vrsq0`, `null_lr`, `null_xgb`, `sens_group`; `fa_yalcin.py`; `fa_sim.py`; `fa_vrlens.py ablation`; `fa_rev3_stats.py` |
| `docs/ANALYSIS_PLAN_REVISION3_ADDENDUM.md` (first addendum) | `fa_rev3.py`: `rotation`, `decode_nr`, `main_nr`, `pairs`, `pen`, `null_pen`, `posctl_seeds`, `null_nm`, `null_nr`, `null_block` |
| `docs/ANALYSIS_PLAN_REVISION3_ADDENDUM2.md` (second, post hoc) | `fa_vrlens.py groupcv` |
| `docs/addendum_3.md` (third, post hoc) | `fa_add3.py` stages P1–P7, `fa_add3_stats.py`, `fa_add3_outputs.py`, `fa_report_add3.py` |
| Written after all results (post hoc, descriptive) | `fa_report_rq3res.py` (Table S38), `fa_report_sample.py` (Table S39) |
