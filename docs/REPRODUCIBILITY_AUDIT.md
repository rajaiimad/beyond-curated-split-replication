# Reproducibility audit of this repository (4 November 2026)

Before publication, the repository was tested as a reviewer would use it: from a fresh clone, in a new environment built only from `requirements.txt` and `requirements-optional.txt`.

**Environment:**
- Linux (x86-64), 4 CPU cores;
- Python 3.13.16;
- numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, joblib 1.5.3, xgboost 3.4.1.

## 1. Data

| Check | Result |
|---|---|
| `scripts/prepare_data.sh` | 74 raw and 35 curated sessions unpacked |
| The five UFFCSData files against upstream commit `b5128e6` | byte-identical (git blob hashes compared with the upstream repository) |
| The 179 unpacked session files against the files the analysis used | byte-identical |
| `verification/verify_data.sh` | all checksums OK |
| `scripts/fetch_second_archive.sh` | second archive at commit `14a242a`; all 40 files used match `data/SHA256SUMS.yalcin` |
| Smoke tests (`python -m pytest -q tests`, with pytest installed separately) | 6 passed |

## 2. Tier 1: regenerate and verify every reported number

Run with `bash verification/verify_results.sh`. It took 9 min 14 s.

| Step | Result |
|---|---|
| `fa_stats.py` (on the prediction files it covers) | `results/stats.json`: 690 configurations, 648 tests; **byte-identical** to the shipped file |
| `fa_report.py` (with `fa_report_rev3.py`) | 45,205 values, all tables and figures; "CLAIM CHECKS: all passed" and "REV3 CLAIM CHECKS: all passed" |
| `fa_report_add3.py` | 2,819 values; claims 27 true, 0 false |
| `fa_report_rq3res.py` | 263 values; claims 17 true, 0 false |
| `fa_report_sample.py` | 84 values; claims 6 true, 0 false |
| `fa_verify.py` [1] | 736 printed keys, none missing |
| [2] | 25 AUCs of the earlier analysis recomputed with scikit-learn: 0 mismatches |
| [2b] | 2,399 AUCs and gains of the present analysis recomputed: 0 mismatches |
| [2c] | 97 claim checks true, 0 false |
| [2d] | addendum 3: 2,819 values; gains, family tests, power and VR-LENS AUCs recomputed: 0 mismatches |
| [2e] | RQ3 resolution summary recomputed by a separate code path: 0 mismatches |
| [2f] | sample description recounted from the 74 raw session files: 0 mismatches |
| [4a] | qualitative claim checks at full precision: 90 true, 0 false |
| Final line | **`RESULT: PASS`** |

**Regenerated files compared with the shipped ones:**
- **Value files, generated tables and result files:** every one is byte-identical.
- **The five PDF figures:** their bytes differ only in the embedded creation date. Rendered at 100 dpi, every figure is pixel-for-pixel identical to the shipped one.

## 3. Tier 2: stages re-run from the raw data

| Stage | Result |
|---|---|
| `src/fa_data.py`: parse the raw archive, form the inferred groups, build the features | `data/sessions.csv` byte-identical. `data/features_w30.csv.gz` has identical content; only the timestamp in its gzip header differs, which is why `verify_data.sh` checks it by content. 38 groups (6 in both campaigns); flags: 1 gap, 1 configuration, 6 short; 70 VRSQ files. |
| `src/fa_yalcin.py`: second archive | `results/rev3_yalcin.json` reproduced: 343 of 343 values identical (`fa_repro.py`, run time excluded); 2 min 14 s |
| `src/fa_sim.py`: simulation | `results/rev3_sim.json` reproduced: 1,909 of 1,909 values identical; 3 min 47 s with `FA_NJOBS=4`. An earlier attempt with 2 workers and an empty `OMP_NUM_THREADS` (as loaded from `.env`) was still running after an hour, most likely because of thread oversubscription, and was stopped. The guide now sets the threads as `scripts/run_rev3.sh` does. |

**Not re-run in this audit:**
- **The full pipeline:** several days of CPU time.
- **The VR-LENS refit:** not reproducible bit for bit; see Supplementary Note S2.
- **The permutation nulls:** these, and the other model-fitting stages, are the stages behind the shipped predictions, and the predictions themselves were verified in Tier 1.

## 4. Manuscript cross-check

- **Tables:** every table of the submitted manuscript and supplement was matched number by number with its generated table file. Every multi-digit result number is present in the generated file, except in the four tables composed by hand (Table IX, S28, S29, S37), whose numbers come from the value files.
- **Running text:** 1,045 multi-digit numbers were checked. 27 occurrences have no generated source: design constants, numbers quoted from other papers, identifiers, and two items listed in `docs/MANUSCRIPT_TRACEABILITY.md` (both resolved on 5 November 2026; section 7):
  - the "80–97%" range in the caption of Table VII, which is computed from the value file but not stored under its own key;
  - the "at most 0.0013" in Supplementary Note S2, which comes from a manual re-run check and cannot be regenerated from this repository.

## 5. Defect found and how it was handled

`scripts/check_results.sh`, kept unchanged from development, failed with `KeyError: 'CTXB+MOT'`.
- **Cause:** `src/fa_stats.py` predated the addendum-3 learner `lroffs` and did not skip its 14 prediction files.
- **Workaround in the first release:** `verification/verify_results.sh` gave `fa_stats.py` all prediction files except these 14. That reproduced the shipped `results/stats.json` byte for byte.
- **Correction (5 November 2026):** `lroffs` is now on the skip list of `fa_stats.py`, and the workaround is no longer needed (section 7).
- **Effect on results:** none.

## 6. Security and content

- **Secrets and private data:** no credentials, tokens or private file paths.
- **Contact details:** only the two contact addresses of the authors.
- **Personal data:** none beyond the published UFFCSData archive.
- **File sizes:** the largest file is 12 MB.
- **Development history:** the development repository was not modified while this repository was built; the corrections of 5 November 2026 were made in both (section 7).

## 7. Re-check and corrections (5 November 2026)

### The first release, checked again

The published repository was checked again from a fresh clone of `main`, in a new environment built from `requirements.txt` and `requirements-optional.txt` (Python 3.13.16).

| Check | Result |
|---|---|
| State of `main` | Commit `f8f9bd7`, which had removed the `docs/` folder; every other file was identical to the first release `c9f1b0f`. The folder has been restored unchanged and then updated as listed below. |
| Data | `scripts/prepare_data.sh` and `verification/verify_data.sh`: all checksums OK. The five UFFCSData files have the same git blob hashes as upstream commit `b5128e6`. |
| Smoke tests | The README's step 2 failed as written (`No module named pytest`): pytest is not in the requirement files, and the GitHub workflow installs it separately. With pytest installed: 6 passed. The README now installs it. |
| `verification/verify_results.sh` | `RESULT: PASS` in 7 min 46 s. The only regenerated files that differed were the five PDF figures, by their creation date; rendered at 100 dpi, each is pixel-identical to the shipped one. |
| Traceability map | For every table of the submitted manuscript (Tables I–IX) and supplement (Tables S1–S39), every multi-digit number of the Word table was found in the generated file that `docs/MANUSCRIPT_TRACEABILITY.md` names; the numbers of the four hand-composed tables were found among the printed values. Each figure embedded in the manuscript and supplement matched the PDF the map names, and every script, function and input file the map names exists. |
| Content | No credentials, private paths, manuscripts, logs or AI attribution. The only e-mail addresses are the two author contacts. |

### Corrections

Made in the development repository (commit `2b0aed9`) and copied here; `src/` now matches that commit.

1. **`scripts/check_results.sh`:** `src/fa_stats.py` skips `lroffs` (section 5). In the development repository, `check_results.sh` now ends with `RESULT: PASS` (8 min 28 s).
2. **Table VII caption:** `src/fa_report_rev3.py` stores the share of between-session pairs per label (`r3pair.s1.btw.min/max` = 92/98, `r3pair.s2.btw.min/max` = 80/97), and the caption of `results/paper/r3_tab_gain.tex` prints the severity ≥ 2 range, 80–97%, as the manuscript does; before, it printed 80–98% over both labels. `src/fa_report_rq3res.py` lists the two printed keys, so `fa_verify.py` step [1] checks them.
3. **Note S2's "at most 0.0013":** removed from the revised supplement, because no shipped file records it (`docs/MANUSCRIPT_TRACEABILITY.md`).
4. **README:** installs pytest for the smoke tests, and states the thread setting needed before any refit (its sections 5 and 6).

### The corrected repository, verified

| Check | Result |
|---|---|
| `verification/verify_results.sh` in the working copy | `RESULT: PASS` in 8 min 7 s; step [1] checks 738 printed keys (before: 736), none missing; every recomputation step 0 mismatches. |
| Fresh clone of commit `cdb30e1`, new environment set up as the README says | Install (with the pytest line), `prepare_data.sh`, smoke tests (6 passed), `fetch_second_archive.sh` (40 of 40 files match) and `verify_data.sh` all pass. `verify_results.sh`: `RESULT: PASS` in 7 min 39 s, 738 printed keys, 0 mismatches. `scripts/check_results.sh`: `RESULT: PASS` in 7 min 38 s. After both, no file differs from the commit except the five PDF figures, by their creation date. This row was added afterwards, in the next commit, which changes only this file. |
| Regenerated files compared with the first release | `results/stats.json` byte-identical. Changed, exactly as intended: `values.json` (4 new keys), `values.tex` (4 new definitions), `r3_tab_gain.tex` (the two key names in its caption) and `paper_keys_rq3res.json` (2 new entries). Every other value file, table and result file is byte-identical; the PDF figures differ only in their creation date. |

## 8. Version 10 (13 November 2026)

After a further internal pre-submission review of the manuscript (13 November 2026), the description of the archive was checked against the creators' papers, and two post hoc runs were added (README, "Version 10").

| Check | Result |
|---|---|
| New runs (`src/fa_v10.py`, 4 cores) | VR-LENS with the inferred groups reproduced the stored four-class accuracy exactly (53.1%) and the stored AUCs to the fifth decimal; VR-LENS and HGB-reg with sessions as units ran without error. About 13 minutes in all. |
| `verification/verify_results.sh` with the new step 6 (`fa_report_v10.py`) | `RESULT: PASS`; step [1] checks 835 printed keys, none missing; step [2g] recounts 38 archive facts and recomputes 14 values of the new runs from their predictions, 0 mismatches. |
| Earlier result files | Unchanged: `results/stats.json` and every value file, table and result file of version 9 are byte-identical after regeneration; the new predictions are stored in `results/v10/preds/`, which `src/fa_stats.py` does not read. |

