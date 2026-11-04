# Audit of addendum 3 (P1–P7)

Written on 16 October 2026, after all analyses of `docs/addendum_3.md` had been run and reported. This note records an independent re-check of the code, the result files, the statistics, the repository history and the manuscript references. All analyses of addendum 3 remain **post hoc**. No decision rule was changed, nothing was tuned, and no existing result file was changed by the audit.

## Timeline

| Event (UTC) | Time |
|---|---|
| Addendum committed (`21dea29`) | 15 Oct 21:07:36 |
| Amendment 1 committed (`b045311`) | 15 Oct 21:16:34 |
| Analysis and statistics code committed (`d41190b`) | 15 Oct 21:16:51 |
| SHA-256 list of the 1,831 existing result files written | 15 Oct 21:18:15 |
| First stage started (queue log) | 15 Oct 21:18:15 |
| Report step and verifier step [2d] committed (`b7ed1d5`) | 15 Oct 21:42:31 |
| Last stage and statistics finished (resume log) | 16 Oct 02:29:59 |
| Results committed (`01b1b2f`) | 16 Oct 02:41:21 |

The addendum, its amendment and the analysis code were committed before the first run. The report step only formats the statistics and was committed while the runs were in progress.

## Re-checks and their results

| Check | Result |
|---|---|
| The 1,831 result files present before the runs (SHA-256 list of 21:18:15) | All identical now |
| Leakage gate and mutation test | Passed before every stage, in both queue logs, including the resumed P4 null |
| Re-run from scratch of 13 randomly drawn permutations of the six new nulls (rng 2026 and 2027; one from the resumed part of the P4 null) | Identical to the saved rows (largest difference 0.0, 1,631 fields) |
| Re-run from scratch of one randomly drawn realization of each planted-signal stage (P2, P4, P7) | Identical (largest difference 0.0, 666 fields; P7 also the same five features) |
| P1 penalty: refit of one outer fold in two protocols | The specified C reproduces the saved predictions (largest difference 6.4e-16); C = 1 or the other learner's C differ by 0.15–0.39 |
| Bootstrap intervals, re-implemented independently (weighted Mann–Whitney per stratum; same draws): both P1 learners, seated racing cell and within standing, both labels, overall, between and within sessions | 24 of 24 gains and intervals identical |
| `add3_stats.json`, `values_add3.json`/`.tex`, `claims_add3.json` and the 11 `tab_a3_*.tex`, regenerated from the saved files in a scratch copy | Byte-identical |
| P5, recomputed independently (counts, Fisher test, bootstrap interval of the difference, ρ, Holm, both clustering tests) | Identical: Fisher p 0.631061, interval [−0.342659, 0.129545], date clustering p 0.015798, group clustering p 0.472153. The bootstrap p-values of the correlations were not re-drawn; Holm was re-applied to them |
| P6 record-wise run without demographics, re-run in a scratch directory with the same settings | **Not identical.** Predictions differ by up to 0.008; four-class accuracy 73.29 % against 73.30 %; AUCs agree to three decimals; macro F1 0.53485 against 0.53500, which prints as 0.53 against 0.54. As for the other VR-LENS runs, this stage is not reproducible bit for bit |
| `src/fa_verify.py` | PASS (514 printed keys; [2d]: 2,818 values, 0 mismatches) |
| `pytest tests` | 6 passed |
| Manuscript sources (outside this repository) | 511 printed keys, all listed in `paper_keys*.json` and defined. No analysis of addendum 3 is called pre-specified. The numbers typed by hand in the new text are design constants (signal levels, family size, numbers of permutations, table numbers, dates) |

## Deviations from addendum 3

1. **Two named outputs were missing.** P2 names `results/add3_family6.json` and P4 names `results/add3_sess.json`. Their content was written only inside `results/add3_stats.json`. Both files are now written by `src/fa_add3_outputs.py` as extractions from `add3_stats.json`, with no recomputation. `fa_verify.py` [2d] checks that they agree with it.
2. **Values file.** The addendum says the `a3.` values are "added to `results/paper/values.json`". They are in `results/paper/values_add3.json` instead, because the addendum also requires that every existing result file stays unchanged. No existing key is redefined; the report step and `fa_verify.py` check this.
3. **Overall-AUC gain.** P1 specifies the overall gain "without a decision". `add3_stats.json` and `values_add3.json` nevertheless hold a Holm-adjusted p and a decision flag for it (all Holm p 1.000, none declared). They are not printed: the manuscript table shows the overall gain without a p, and the memo calls it descriptive. The files are not overwritten. The named files of item 1 leave the overall gain out, and `fa_verify.py` now fails if any of these keys is printed.
4. **Gate file of the resumed stage.** The gate file of the first P4 null run (`results/add3_gate/null_sess.json`) was overwritten by the gate of the resumed run. Both passes are in the queue logs.
5. **Addendum outputs regenerated once.** In the manuscript commit (`fae6bb5`), `add3_stats.json`, `values_add3.json` and `claims_add3.json` were regenerated. A field-by-field comparison with the results commit (`01b1b2f`) shows only additions (10 statistics fields, 38 values, 15 claim checks) and four values re-formatted from `0.0e+00` to `0`; no number changed.
6. **Timing of the n/e change (correction).** The memo and the addendum say that the n/e change in the statistics code was made "before the statistics were computed". This is true of the statistics step, but not of interim previews. While the queue ran, real-data permutation p-values and Holm-adjusted p-values were previewed for each null as it finished. These previews ran at 22:11 (scaled penalty), 23:07 (stronger penalty), 23:52 (group-block), 00:22 (without rotation), 01:05 (falling sessions), and 01:41 and 01:45 (session grouping); the power stages were previewed as well. The change was made at 23:52:31. That was after the group-block preview, which listed the stored two-group intervals of this test (seated flight-game cell, severity ≥ 2). It is the only change to the analysis or statistics code after the first run apart from item 5: `src/fa_add3.py` and `src/fa_rev3.py` are unchanged since `d41190b`. The change can only remove a declaration, never create one. The test's Holm-adjusted p is above 0.05 in all 15 families that contain it (smallest 0.059: stronger penalty, falling sessions). No decision changed.
7. **Memo, "Not done".** The memo lists "any edit of the manuscript" as not done. The manuscript was edited later, at the author's request. Its sources are not part of this repository.
8. **Number of demographic answers.** P6(c) of the addendum says the run leaves out "the four `User*` demographic answers". The reduced model has five `User*` columns (`src/fa_vrlens.py`), and the run used ablation A, which leaves out all five; five inputs remain. The memo said "four" and "six inputs", and the generated caption of Supplementary Table S36 said "four". The memo is corrected, and the caption now prints the count from the value file.

## Verification of 16 October 2026 (second pass)

- `src/fa_verify.py` was run on the committed state `197ff2a` (clean working tree, identical to the remote branch).
- Chronology: Git history and the session's server-side timestamps agree. The addendum was committed and pushed at 21:07:36–21:07:38, and the amendment and the code at 21:16:34–21:16:54. The queue was launched at 21:17:03, and its first stage started at 21:18:15. Every `results/add3_*` file was first committed in `01b1b2f` or later. After `21dea29`, the addendum only received appended sections, with no deleted line.
- Memo: every number was checked against the result files. Three statements were wrong and are corrected (see item 8 and the note at the end of the memo).
- VR-LENS code: new web searches on 16 October 2026 again located no publicly available code for VR-LENS; the arXiv and NSF-PAR pages remained unreachable (network policy of the analysis machine). The wording in the memo and the manuscript was changed from "no public code" to "could not locate publicly available code".
- Macro F1 of the P6 run: the two-decimal value lies at a rounding boundary (stored run 0.53485, repeated run 0.53500). A three-decimal value key (`a3.vrl.rwnd.f1x`, 0.535 for both runs) was added, and the supplement prints it; no existing key changed.

## Compute time

From the queue logs `results/logs/add3_queue.log` and `results/logs/add3_queue_resume.log`. These logs are not tracked by git, so this table is the repository's record of them.

| Stage | Seconds |
|---|---|
| VRSQ files (P5) | 241 |
| VR-LENS record-wise (P6) | 295 |
| Stronger-penalty predictions (P1) | 9 |
| Scaled-penalty null, extension to 1000 (P1/P3) | 2,699 |
| Stronger-penalty null (P1) | 3,163 |
| Group-block null (P3) | 2,665 |
| Null without rotation (P3) | 2,026 |
| Falling-session null (P3) | 2,709 |
| Session-grouping null, first run, stopped by the worker restart at 912 of 1000 (P4) | about 1,968 |
| Session-grouping null, resumed (P4) | 199 |
| 6-test power (P2) | 761 |
| Session-grouping power (P4) | 629 |
| Diffuse control (P7) | 955 |
| Statistics | 13 |

The wall-clock time was 4 h 26 min for the first queue (21:18:15 to the last log line at 01:44:07) and 44 min for the resumed queue (01:45:36 to 02:29:59). That makes about 5 h 10 min on 4 cores, including the leakage gates (11–18 s each). The longest stage took 53 min, below the cap of 8 h per stage.

## Analyses not run

- The HGB-reg null (excluded by the addendum).
- A positive control for the stronger-penalty learner (not specified), so its power is unknown.
- A P7 comparison at matched realized AUC and with the same learner as Table IX (not specified). The single-channel and diffuse results are therefore not like for like.
- A re-reading of the VR-LENS paper's own code-availability statement and of the 81 % accuracy it reports for the reduced model: the paper could not be opened from the analysis machine.
