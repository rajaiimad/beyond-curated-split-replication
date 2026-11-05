# Verification

There are two scripts. Both run the research code in `src/` as it is; neither modifies it.

## `verify_data.sh`: are the data exact? (seconds)

Checks SHA-256 checksums:
- `data/SHA256SUMS`: the five UFFCSData files are byte-identical to upstream commit `b5128e6`, and `data/sessions.csv` and `results/session_level.csv` are unchanged;
- `data/SHA256SUMS.decompressed`: the content of `data/features_w30.csv.gz` is unchanged (checked after decompression, because a gzip header records the time of writing);
- `data/SHA256SUMS.unpacked`: the 179 session files unpacked by `scripts/prepare_data.sh`;
- `data/SHA256SUMS.yalcin`: the 40 files of the second archive, if `scripts/fetch_second_archive.sh` was run.

Expected output:

```
data: 7 shipped files OK
data: features_w30.csv.gz content OK
data: 179 unpacked session files OK
data: 40 second-archive files OK            (or: second archive not fetched (optional; ...))
```

## `verify_results.sh`: do the results reproduce? (about 10–15 min; 9 min measured on 4 cores)

It regenerates every printed value, table and figure from the shipped result files and predictions, and verifies them. No model is refitted.

| Step | Program | What it does |
|---|---|---|
| 1 | `src/fa_stats.py` | Group-bootstrap intervals and tests of the earlier analysis (`results/stats.json`); see "Corrected defect" below. |
| 2 | `src/fa_report.py` (calls `src/fa_report_rev3.py`) | Every printed value (`results/paper/values.json`), all tables (`results/paper/*.tex`), the figures and the automatic claim checks |
| 3 | `src/fa_report_add3.py` | Addendum 3 (post hoc P1–P7): `values_add3.json`, `tab_a3_*.tex` |
| 4 | `src/fa_report_rq3res.py` | RQ3 resolution summary (Table S38): `values_rq3res.json` |
| 5 | `src/fa_report_sample.py` | Sample description (Table S39): `values_sample.json` |
| 6 | `src/fa_verify.py` | Independent check; its output is saved to `verification/fa_verify_output.txt` |

`fa_verify.py` prints one line per check:
- **[1]** every value key the paper prints exists (`missing: 0`);
- **[1b]** no generated table contains a missing value;
- **[2]** printed AUCs of the earlier analysis are recomputed from the predictions with scikit-learn (`mismatches: 0`);
- **[2b]** 2,399 AUCs and gains of the present analysis are recomputed independently (`mismatches: 0`);
- **[2c]** claim checks of the present analysis (`0 false`);
- **[2d]** addendum-3 values, permutation tests, Holm adjustments and power are recomputed (`mismatches 0`);
- **[2e]** the RQ3 resolution summary is recomputed by a separate code path (`mismatches 0`);
- **[2f]** the sample description is recounted from the 74 raw session files (`mismatches 0`);
- **[4a]–[4c]** qualitative statements are re-checked at full precision; [4c] only lists printed intervals with a bound shown as 0.50, for a wording check;
- **the last line must be `RESULT: PASS`**.

Then the script lists every regenerated file that differs from the shipped version. Expected:
- **Data and results:** no file differs.
- **PDF figures:** they can show as modified, because matplotlib writes the creation date into the file. Their drawn content is the same, which you can confirm by rendering them, for example with `pdftoppm`.

## Corrected defect (5 November 2026)

In the first release (commit `c9f1b0f`), `scripts/check_results.sh` ran `fa_stats.py` on the full `results/preds/` folder and then stopped in `fa_report.py` with `KeyError: 'CTXB+MOT'`.

**Cause:**
- `fa_stats.py` computes the statistics of the earlier analysis.
- It skips the learners of the present analysis, but it was written before the addendum-3 learner `lroffs` (LR+offset with a ten-times-stronger penalty), whose 14 prediction files it therefore picked up.
- `fa_report.py` has no place for these configurations.

**Correction:** `lroffs` is now on the skip list of `fa_stats.py`, as `lroffp` was on 10 October 2026. The statistics of `lroffs` are computed by `fa_add3_stats.py` (`results/add3_stats.json`) and checked by `fa_verify.py` step [2d]. In the first release, `verify_results.sh` worked around the defect by giving `fa_stats.py` all prediction files except `*__lroffs.npz`; both ways give the same `results/stats.json`, byte for byte.

**Effect on results:** none. `scripts/check_results.sh` now runs to the end. It runs only `fa_stats.py`, `fa_report.py` and `fa_verify.py`, so `verify_results.sh`, which also regenerates the addendum-3, RQ3-resolution and sample files, remains the complete check.

## What the verification does not cover

- **Intervals:** bootstrap intervals and the permutation p-values of the primary families are read from the result files and regenerated (steps 1–5), not recomputed by a second implementation.
- **The first run (Table S4):** read from `results_logo/`; its predictions are not included.
- **Refitting:** no model is refitted. Refitting is Tier 2 and Tier 3 in `docs/REPRODUCTION_GUIDE.md`.
