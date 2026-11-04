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
| Smoke tests (`python -m pytest -q tests`) | 6 passed |

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
- **Running text:** 1,045 multi-digit numbers were checked. 27 occurrences have no generated source: design constants, numbers quoted from other papers, identifiers, and two items listed in `docs/MANUSCRIPT_TRACEABILITY.md`:
  - the "80–97%" range in the caption of Table VII, which is computed from the value file but not stored under its own key;
  - the "at most 0.0013" in Supplementary Note S2, which comes from a manual re-run check and cannot be regenerated from this repository.

## 5. Defect found and how it is handled

`scripts/check_results.sh`, kept unchanged from development, fails with `KeyError: 'CTXB+MOT'`.
- **Cause:** `src/fa_stats.py` predates the addendum-3 learner `lroffs` and does not skip its 14 prediction files.
- **Workaround:** `verification/verify_results.sh` gives `fa_stats.py` all prediction files except these 14. That reproduces the shipped `results/stats.json` byte for byte.
- **Effect on results:** no research code or result was changed.

## 6. Security and content

- **Secrets and private data:** no credentials, tokens or private file paths.
- **Contact details:** only the two contact addresses of the authors.
- **Personal data:** none beyond the published UFFCSData archive.
- **File sizes:** the largest file is 12 MB.
- **Development history:** the development repository was not modified by this work.
