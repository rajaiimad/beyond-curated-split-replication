# Beyond the Curated Split — reproducibility package

Data, code, result files and verification scripts for the manuscript:

> Rajai Darabee, Manoranjitham Muniandy, Cheng Wai Khuen, Ebrahim Hamid Sumiea and Omar Shindi,
> *Beyond the Curated Split: Context Shortcuts and Evaluation Design in Cybersickness Detection on the Full
> UFFCSData Archive* (submitted; under review).

This repository is meant for reviewers and readers. With it you can:
- inspect the exact code that produced every number, table and figure in the manuscript and its supplement;
- check each printed value against the result files and the saved model predictions (about 10–15 minutes, no model refitting);
- re-run any part of the analysis, or all of it, from the raw data.

The manuscript and supplement themselves are not included.

**Contents at a glance**

| | |
|---|---|
| Dataset | UFFCSData (Porcino et al.), the 74-session raw archive and the 35-session curated release, at the upstream commit used in the paper. The files are unmodified and redistributed under their AGPL-3.0 licence. |
| Second archive | Yalcin et al. (2024), used for three evaluation-design findings. It is fetched from its authors at a fixed commit and checked by checksums; it is not redistributed (see [Data](#3-data)). |
| Code | The 20 analysis scripts in `src/`: the 18 that produced the results of version 9, with two later corrections that change no reported value (see [Corrections](#corrections-5-november-2026)), and the two scripts added for version 10 (see [Version 10](#version-10-13-november-2026)); plus one check script that reads stored predictions only (see [Check of 14 November 2026](#check-of-14-november-2026-post-hoc)) |
| Results | Every result file (JSON), the 1,756 saved out-of-group predictions, the generated value files and tables, and the figures |
| Verification | `verification/verify_data.sh` (data checksums) and `verification/verify_results.sh` (regenerate and verify everything) |
| Documentation | `docs/REPRODUCTION_GUIDE.md`, `docs/MANUSCRIPT_TRACEABILITY.md`, `docs/DATA_DICTIONARY.md`, `docs/PROVENANCE.md`, `docs/REPRODUCIBILITY_AUDIT.md` (how this repository was tested before publication) and the written analysis plans |

---

## 1. Purpose

The manuscript audits the full public UFFCSData cybersickness archive. It asks three questions:
- **RQ1:** how context and data-quality problems shape the labels and the telemetry;
- **RQ2:** how evaluation design changes the measured performance of telemetry detectors, including a re-implemented published detector (VR-LENS);
- **RQ3:** whether a planned test detects a telemetry gain over a baseline of game, campaign and exposure time, and how large a gain it could have detected.

This repository makes every reported result inspectable and re-computable. It also keeps the distinction between planned and post hoc analyses visible: the written plans are in `docs/`, and their commit dates are in `docs/PROVENANCE.md`.

## 2. The research it supports

| Part of the manuscript | Where to look |
|---|---|
| Every table, figure and supplementary note | `docs/MANUSCRIPT_TRACEABILITY.md` maps each one to the script that generates it, the result files it reads and the generated output |
| Every number printed in the text | `results/paper/values*.json` hold each printed value under a key. `src/fa_verify.py` checks them; `results/paper/paper_keys*.json` list the keys the paper prints. |
| Analysis plans: what was planned and what was post hoc | `docs/ANALYSIS_PLAN_REVISION2.md` (the paper's "earlier analysis"), `docs/ANALYSIS_PLAN_REVISION3.md` (the "present analysis"), `docs/ANALYSIS_PLAN_REVISION3_ADDENDUM.md` (first addendum), `docs/ANALYSIS_PLAN_REVISION3_ADDENDUM2.md` (second, post hoc), `docs/addendum_3.md` (third, post hoc), with its results memo `docs/results_addendum3.md` and audit `docs/audit_addendum3.md` |
| When each plan was committed | `docs/PROVENANCE.md` |

**Naming:** "Revision 2" and "revision 3" in file names are internal rounds of the analysis, not rounds of journal review. In the paper, revision 2 is the *earlier analysis* and revision 3 the *present analysis*.

## 3. Data

| Data | Where | Source and version | Licence | In this repository? |
|---|---|---|---|---|
| UFFCSData raw archive (74 sessions; telemetry, in-game ratings, VRSQ files) | `data/raw/UFFCSData/RAW_DATABASE.zip` | github.com/tmp1986/UFFCSData, commit `b5128e6` (1 May 2020) | GNU AGPL-3.0 (`data/raw/UFFCSData/LICENSE`) | Yes, unmodified |
| UFFCSData curated release (35 sessions) | `data/raw/UFFCSData/DATABASE_SELECTED_DATA.zip`, `dataset.csv` | same commit | AGPL-3.0 | Yes, unmodified |
| Derived tables (per session; per record with features) | `data/sessions.csv`, `data/features_w30.csv.gz`, `results/session_level.csv` | made by `src/fa_data.py` and `src/fa_run.py` from the archive | AGPL-3.0, as derived works | Yes |
| Second archive (eye tracking and button presses of 20 participants) | `data/raw/yalcin/` after `scripts/fetch_second_archive.sh` | github.com/m1237/automatic-cybersickness-detection, commit `14a242a` | MIT (its authors) | **No.** It is fetched from its authors and the 40 files used are checked against `data/SHA256SUMS.yalcin`. |

- **Redistribution:**
  - AGPL-3.0 permits verbatim copies of UFFCSData with its licence and notices; they are included here unmodified.
  - The second archive's MIT licence would also permit copies. It is not re-hosted because it contains other researchers' physiological recordings of human participants: it stays with its owners, and checksums make the fetched copy exact.
- **Personal data:** UFFCSData records seven self-reported demographic answers per session (gender, age band, VR experience and others) but no names or identifiers. Its authors publish it openly. Nothing beyond the published archives is included.
- **Preparation:** `scripts/prepare_data.sh` unpacks the two zips into the layout the code expects, checks all 179 session files against `data/SHA256SUMS.unpacked` and creates `.env`.
- **Details:** `data/README.md` describes the files and `docs/DATA_DICTIONARY.md` every field.

## 4. Software requirements

- **Python 3.11 or 3.13.**
  - The earlier analysis (revision 2) was run with Python 3.11.15; the present analysis and the re-run check with Python 3.13.16.
  - The verification in this repository was tested with Python 3.13.16.
- **Packages:** pinned in `requirements.txt`:
  - numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9 and joblib 1.5.3;
  - xgboost 3.4.1 (`requirements-optional.txt`), needed only to refit the boosting learner and VR-LENS.
- **Tools:** `bash`, `unzip`, `sha256sum` and `git`. On Windows, use WSL or Git Bash; PowerShell versions of the original run scripts are in `scripts/`.
- **Hardware:** 2–4 CPU cores and 4 GB RAM. No GPU is needed.
- **Disk:** about 1.3 GB after cloning (the files plus their compressed copy in `.git`); about 1.5 GB with the unpacked data and the second archive.

## 5. Installation

```bash
git clone https://github.com/rajaiimad/beyond-curated-split-replication.git
cd beyond-curated-split-replication
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-optional.txt   # xgboost: only for refitting the boosting learner and VR-LENS
pip install pytest                     # only for the smoke tests (step 2)
bash scripts/prepare_data.sh           # unpack UFFCSData, check every file, create .env
```

`.env` holds only data paths and compute settings (`.env.example` explains each variable). The project needs no API keys or passwords.

## 6. Order of execution

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `bash scripts/prepare_data.sh` | Unpack and check the data | seconds |
| 2 | `python -m pytest -q tests` | Smoke tests of the statistics engine (no data needed) | seconds |
| 3 | `bash verification/verify_results.sh` | Regenerate every value, table and figure from the shipped result files and predictions, and verify them independently | about 10–15 min (9 min measured on 4 cores) |
| 4 (optional) | `bash scripts/fetch_second_archive.sh` | Fetch and check the second archive | about 1 min |
| 5 (optional) | the partial re-runs in `docs/REPRODUCTION_GUIDE.md` | Re-run selected stages from the raw data and compare with the shipped results | minutes to hours |
| 6 (optional) | `bash scripts/run_pipeline.sh`, then `bash scripts/run_rev3.sh`, then the addendum-3 stages | Refit every model from the raw data | several days of CPU time |

`docs/REPRODUCTION_GUIDE.md` gives the full order, every stage, the runtimes, the seeds and how to compare a re-run with the shipped files.

**Before refitting any model (steps 5 and 6),** set the threads as `scripts/run_rev3.sh` does: `export OMP_NUM_THREADS=1 FA_NJOBS=4`. Loading `.env` alone leaves `OMP_NUM_THREADS` set but empty, and parallel fits can then oversubscribe the CPU and run many times slower.

## 7. Reproducing the manuscript's tables and figures

`verification/verify_results.sh` regenerates every table and figure into `results/paper/` (LaTeX tables, from which the manuscript's tables were typeset) and `figures/` (PDF). It runs:
- `src/fa_stats.py`;
- `src/fa_report.py` (which calls `src/fa_report_rev3.py`);
- `src/fa_report_add3.py`, `src/fa_report_rq3res.py` and `src/fa_report_sample.py`.

Table, figure and section numbers here follow the submitted journal version of the manuscript (Tables 1–9, Sections 1–8; notes, tables and figures numbered S in Online Resource 1, except Table S1 in Online Resource 2). The dated documents in `docs/` keep the numbering of the manuscript version they were written for.

`docs/MANUSCRIPT_TRACEABILITY.md` gives, for each table and figure:
- the generated file;
- the script and function that write it;
- the result files it reads;
- the value keys it prints.

| Manuscript | Generated file |
|---|---|
| Fig. 1 (pipeline schematic) | `figures/fig_pipeline.pdf` (source `figures/fig_pipeline.tex`) |
| Fig. 2 | `figures/fig_archive.pdf` |
| Fig. 3 | `figures/fig_vrsq.pdf` |
| Fig. 4 (Supplementary Fig. S2 up to version 9) | `figures/fig_gain_null.pdf` |
| Fig. S1 | `figures/fig_null.pdf` |
| Table S1, Online Resource 2 (not printed in Online Resource 1 since version 10) | `results/paper/supp_table.tex` |
| Not in the manuscript | `figures/fig_auc.pdf` (generated, not printed) |

## 8. Verifying the reported results

```bash
bash verification/verify_data.sh      # every data file against its SHA-256 checksum
bash verification/verify_results.sh   # regenerate, recompute and compare; must end with "RESULT: PASS"
```

`src/fa_verify.py`, run by `verify_results.sh`, does the following:
- **[1]** checks that every value key the paper prints exists;
- **[2]** recomputes the printed AUCs of the earlier analysis (25) from the saved predictions, independently with scikit-learn;
- **[2b]** does the same for 2,399 AUCs and gains of the present analysis;
- **[2c]** checks the claims of the present analysis;
- **[2d]** recomputes the addendum-3 values, permutation tests, Holm adjustments and power;
- **[2e]** recomputes the RQ3 resolution summary by a separate code path;
- **[2f]** recounts the sample description from the raw session files;
- **[4]** re-checks the qualitative statements at full precision.

Bootstrap intervals and the permutation p-values of the primary families are regenerated from the shipped result files, not recomputed by a second implementation.

At the end, `verify_results.sh` lists any regenerated file that differs from the shipped one. Apart from the creation-date metadata inside the PDF figures, none should differ. `verification/README.md` explains each step and the expected output.

## 9. Limitations and restrictions

- **Full re-run cost.** Refitting every model takes several days of CPU time, mostly in the permutation nulls (for example, the 500-permutation null of the earlier analysis alone takes about 30 h on one core). The standard check (step 3) needs no refitting.
- **Bit-for-bit reproducibility.**
  - With the pinned versions, the shipped result files are expected to be reproduced exactly. `docs/REPRODUCIBILITY_AUDIT.md` records what was reproduced, and how, before publication.
  - One stage is not reproducible bit for bit: the VR-LENS re-implementation. Two re-runs on the same machine with identical versions differed in the sixth decimal of the record-wise AUCs, and no printed value changed (Online Resource 1, Note S2). `results/reproducibility.json` documents the first re-run comparison: 14,933 values, 5 differing beyond 10⁻⁹, all in the VR-LENS file.
  - With other library versions, small numerical differences are possible.
- **The first, leave-one-group-out run** (Online Resource 1, Table S4) is read from its archived output in `results_logo/`; its predictions are not included.
- **The second archive** must be fetched from its authors (step 4). If that repository becomes unavailable, `src/fa_yalcin.py` cannot be re-run, but its results remain in `results/rev3_yalcin.json`.
- **Timing of the analysis plans.** The plans were committed in the authors' private development repository. `docs/PROVENANCE.md` lists the commit dates. They are self-reported, not an independent registration; the full history is available to the editor on request.
- **VR-LENS** is a re-implementation from its paper. No public code was found, and its authors were not contacted (Online Resource 1, Note S2).

## Corrections (5 November 2026)

The first release of this repository is commit `c9f1b0f` (4 November 2026). Two defects found in its audit were then corrected, in this repository and in the development repository. Neither changes any reported value:

1. **`scripts/check_results.sh` stopped with `KeyError: 'CTXB+MOT'`.** `src/fa_stats.py`, which computes the statistics of the earlier analysis, predated the addendum-3 learner `lroffs` and did not skip its 14 prediction files (their statistics come from `src/fa_add3_stats.py`). `lroffs` is now on its skip list, as `lroffp` was on 10 October 2026. `results/stats.json` is reproduced byte for byte, and `verification/verify_results.sh` no longer needs its earlier workaround.
2. **The generated caption of Table 7 printed a range over both labels.** The caption of `results/paper/r3_tab_gain.tex` gave the share of between-session pairs over both labels (80–98%), although the table is for severity ≥ 2. `src/fa_report_rev3.py` now stores the range for each label (`r3pair.s1.btw.min/max`, `r3pair.s2.btw.min/max`), and the caption prints the severity ≥ 2 range (80–97%), as the manuscript does. `src/fa_report_rq3res.py` lists the two keys as printed in Table 7, so `src/fa_verify.py` checks them.

What changed in the generated files: four new keys in `results/paper/values.json` and `values.tex`, the two key names in the caption of `r3_tab_gain.tex`, and two entries in `paper_keys_rq3res.json`. Every other value, table, result file and figure is unchanged. `docs/REPRODUCIBILITY_AUDIT.md` (section 7) records the re-verification.

## Version 10 (13 November 2026)

After a further internal pre-submission review of the manuscript (13 November 2026), the description of the archive was checked against the creators' own papers (Porcino et al. 2022; its conference version at ICEC 2021, Porcino et al. 2021b; and the XR in Games Workshop paper, Porcino et al. 2021a). Two runs were then added. Both are **post hoc**: they were fixed after that review and are not part of any analysis plan.

- **`src/fa_v10.py`** repeats two curated-release results with every curated session as its own unit, which is participant-level under the creators' account. It runs (1) the re-implemented VR-LENS detector with the inferred groups, only to keep its per-record predictions; it reproduces the stored accuracy exactly. It also runs (2) VR-LENS and (3) HGB-reg on TIME, CTX+TIME, MOT+TIME and MOT+CTX+TIME with sessions as units. The predictions are in `results/v10/preds/`, apart from `results/preds/`, so `src/fa_stats.py` and every earlier result file are unchanged. Summary: `results/v10/v10_runs.json`. About 15 minutes on 4 cores.
- **`src/fa_report_v10.py`** writes `results/paper/values_v10.json` with three kinds of value:
  - the archive facts of the manuscript's Section 3.2, counted from `data/sessions.csv` and the raw files;
  - the numbers the manuscript quotes from the creators' papers and the VR-LENS paper, each with its source in `values_v10_sources.json`;
  - the values of the two runs.

  It needs no model fit. `verification/verify_results.sh` runs it as step 6, and `src/fa_verify.py` step [2g] recounts the archive facts and recomputes the run values from the stored predictions.

The manuscript's Section 3.2 states which links between the archive and the creators' protocols are inferences from matching counts. The flight-game questionnaire files do not reproduce the creators' per-participant scores (Online Resource 1, Note S8).

## Check of 14 November 2026 (post hoc)

`src/fa_check_rowwise_ctx.py` explains why the record-wise CTX+TIME model (default HGB, curated release, severity ≥ 2) has a context-stratified AUC below 0.5: within a stratum its score follows the share of high-rated training records at the same second of exposure, which is lower for a held-out high-rated record because its own label is absent from training. It fits no model and writes no file; `python src/fa_check_rowwise_ctx.py .` prints the two AUCs and their correlation.

## Licences

| What | Licence |
|---|---|
| Code and documentation written for this project (`src/`, `scripts/`, `verification/`, `tests/`, `docs/`, figures) | MIT (`LICENSE`), the five authors |
| UFFCSData files in `data/raw/UFFCSData/` | GNU AGPL-3.0 (`data/raw/UFFCSData/LICENSE`), © their authors; redistributed unmodified |
| Tables derived from UFFCSData (`data/sessions.csv`, `data/features_w30.csv.gz`, `results/session_level.csv`) and the model predictions (`results/preds/`) | GNU AGPL-3.0, as works derived from the archive; please credit UFFCSData [Porcino et al.] |
| Second archive (not included) | MIT, © its authors |

## How to cite

See `CITATION.cff`. The paper cites the release tagged `v1.0-submission`; its commit is given in the paper's data availability statement. Please also cite UFFCSData (Porcino et al., 2022; github.com/tmp1986/UFFCSData) and, if you use the second archive, Yalcin et al. (2024).

## Contact

- Code and repository: Rajai Darabee — rajai_24006920@utp.edu.my
- Corresponding author of the manuscript: Manoranjitham Muniandy — manoranjitham@utp.edu.my
