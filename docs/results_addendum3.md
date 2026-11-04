# Results of addendum 3 (post hoc analyses P1–P7)

The power table of the paper (the request calls it Table VIII) is Table IX in the current manuscript; it is called Table IX below.

Everything here is **post hoc**: specified in `docs/addendum_3.md` after all earlier results were known, and committed before the runs. The decision rule is unchanged: Holm-adjusted one-sided permutation p < 0.05 within the family **and** a 95 % group-bootstrap interval of the gain above 0.

- **Run dates:** 15–16 October 2026, on 4 CPU cores.
- **Results:** the numbers below are in `results/add3_stats.json` and `results/paper/values_add3.json` (keys `a3.*`). The tables are `results/paper/tab_a3_*.tex`.
- **Verification:** `src/fa_verify.py` passes. Its new step [2d] recomputes, independently of the statistics code and with no mismatch:
  - 110 printed gains from the predictions;
  - 277 family tests (observed gains, permutation p and Holm p) from the null files;
  - 284 power entries.
- **No existing file was changed.** All 1,831 result files present before the runs are byte-identical afterwards, including `results/paper/values.json` and `results/leakage_checks.json`.

## Summary

| Part | What was run | Result | Does a conclusion of the paper change? |
|---|---|---|---|
| P1 | LR+offset with a session-scaled penalty and with a 10× stronger penalty; 1,000-permutation nulls | No test met the rule with either penalty (14- or 6-test families) | No |
| P2 | Post hoc 6-test family (within seated, within standing, pooled × 2 labels) | No real-data test met the rule, as was known in advance from the existing p-values. Power rose between sessions at AUC 0.80 (LR+offset: 6/6 vs 4/6) but changed little at 0.70 or within sessions | No. The family-size effect is now quantified |
| P3 | The four secondary nulls with 1,000 permutations | None met the rule. Smallest Holm p 0.168, previously 0.279 | Partly. The four analyses are now testable, and negative; the statement that they "could not pass" must be replaced |
| P4 | Null and positive control with every session as its own group | No test met the rule. Power almost unchanged | No |
| P5 | VRSQ all-zero files: by rating, Fisher test, bootstrap, Holm, listing, clustering | All-zero files are not more common in high-rated sessions (16/38 vs 16/32, p = 0.63). They cluster by recording date (p = 0.016), not by group | No; the existing caveat is strengthened |
| P6 | Search for the VR-LENS code; record-wise run without demographic answers | No publicly available code located. Record-wise accuracy without demographics was 73.3 %, against 86 % with them and 57.2 % group-disjoint | No; the record-wise inflation is partly, not wholly, due to the demographic answers |
| P7 | Diffuse planted signal (5 shifted features), LR+offset with session-scaled penalty and boosting+offset | Detected far more often than the single planted channel. Its realized within-session AUC exceeded the calibration target | **Yes**: the paper's statement that power for diffuse signals is "likely lower" is not supported, and the power characterization depends on the signal model |

## P1. LR+offset with a session-scaled and a stronger penalty

**Setup.**

- Learners:
  - `lroffp`: C = training sessions / training records, 0.0035–0.0051 over the outer folds;
  - `lroffs`: C = 0.1 × that.
- CTXB+MOT, the seven protocols of the primary families, both labels.
- Inferred-group bootstrap (B = 10,000).
- Nulls: 1,000 permutations each, session transplant, seeds 700 + s, refitting the baseline and the learner.
- The `lroffp` gains and intervals reproduce those in `results/rev3_stats.json` exactly (largest difference 0.0).

**Gains** (Table `tab_a3_p1.tex`):

| | Overall | Between sessions | Within sessions |
|---|---|---|---|
| Scaled penalty, severity ≥ 2 | −0.05 to +0.02 | −0.04 to +0.08 | −0.15 to +0.00 |
| Scaled penalty, severity ≥ 1 | −0.11 to +0.01 | −0.22 to +0.10 | −0.11 to +0.08 |
| 10× stronger penalty, severity ≥ 2 | −0.03 to +0.01 | −0.05 to +0.07 | −0.05 to +0.01 |
| 10× stronger penalty, severity ≥ 1 | −0.12 to +0.01 | −0.23 to +0.09 | −0.04 to +0.01 |

The stronger penalty pulls the within-session gains towards 0, in both directions.

**Families** (Holm within the family; smallest attainable Holm p 0.014 with 14 tests and 0.006 with 6):

| Learner | Estimand | 14 tests: unadjusted p < 0.05 | 14 tests: smallest Holm p | 6 tests: smallest Holm p | Declared |
|---|---|---|---|---|---|
| Scaled penalty | between | 0 | 1.000 | 0.671 | 0 |
| Scaled penalty | within | 2 | 0.168 | 0.132 | 0 |
| 10× stronger | between | 0 | 1.000 | 1.000 | 0 |
| 10× stronger | within | 1 | 0.587 | 0.252 | 0 |

- The scaled penalty's two unadjusted results within sessions are severity ≥ 2 in the seated flight-game cell (+0.002, p = 0.012; only two groups with a positive record, so no interval) and severity ≥ 1 pooled (+0.005, p = 0.022).
- The overall-AUC gains are descriptive: the smallest unadjusted p is 0.191 (scaled) and 0.272 (stronger).
- Not specified in the addendum but computed: falling-session families for both learners. Smallest Holm p 0.176 (scaled) and 0.059 (stronger). The latter is the seated flight-game cell for severity ≥ 2 with a gain of +0.000, a cell without an interval.

**Conclusion.** No telemetry gain is established with either penalty. The penalty does not rescue the telemetry. A stronger penalty mainly shrinks the gains towards the baseline.

## P2. A post hoc 6-test family

**Real data** (Table `tab_a3_family6.tex`). As stated in the addendum before the run, the outcome was fixed by the known p-values. Smallest Holm p within the family of 6 (between / within):

| Learner, features | Smallest Holm p, between / within |
|---|---|
| LR+offset, MOT | 0.575 / 0.144 |
| LR+offset, SELFZ | 0.216 / 0.881 |
| Boosting+offset, MOT | 0.838 / 0.254 |

Nothing is declared.

**Power** (Table `tab_a3_power6.tex`; the realizations of addendum 1, B1). The refitted planted gains reproduced the stored B1 gains exactly (largest difference 0.0). The table gives the number of the six label–protocol combinations declared in at least half of 20 realizations, with the 14-test family on the same six combinations in parentheses:

| Learner | AUC | Between sessions | Within sessions |
|---|---|---|---|
| LR+offset | 0.80 | 6 (4) | 3 (3) |
| LR+offset | 0.70 | 2 (1) | 0 (0) |
| LR+offset | 0.65 | 1 (0) | 0 (0) |
| Boosting+offset | 0.80 | 4 (1) | 5 (5) |
| Boosting+offset | 0.70 | 1 (0) | 0 (0) |
| Boosting+offset | 0.65 | 0 (0) | 0 (0) |

**Conclusion.** A smaller family raises power between sessions for large signals. It does not make moderate signals (AUC 0.65–0.70) detectable, and it does not help within sessions. The paper's explanation that the 14-test families lower power is confirmed for large signals only.

## P3. The secondary analyses with 1,000 permutations

The four nulls of addendum 1 were extended from 200 to 1,000 permutations into new files. The HGB-reg null was not re-run.

**Reproducibility checks.**

- Before extending, permutations 0 and 199 of each stored null were re-run and matched the stored rows exactly.
- The falling-session null re-runs the transplants and model of the main 1,000-permutation null. It reproduced all 1,000 permutations of `results/rev3_null_lroff.json` exactly (14,000 rows, largest difference 0.0).

**Results** (Table `tab_a3_secondary.tex`, the new version of Table S20). With 1,000 permutations the smallest attainable Holm p is 0.014 (14 tests) or 0.013 (13 tests), so every family could now meet the rule. None did:

| Analysis | Unadjusted p < 0.05 (between / within) | Smallest Holm p (between / within) | Declared |
|---|---|---|---|
| Group-block null | 0 / 1 (before: 1 / 2) | 0.923 / 0.591 | 0 |
| Falling sessions (13 tests) | 2 (before: 1) | 0.412 | 0 |
| Scaled penalty | 0 / 2 (before: 0 / 3) | 1.000 / 0.168 | 0 |
| Without rotation | 0 / 1 (before: 0 / 1) | 0.979 / 0.671 | 0 |

- The smallest Holm-adjusted p across the four analyses is now 0.168 (before: 0.279).
- **Severity ≥ 1, standing racing-game cell, within sessions:**
  - Holm p 0.779 against the group-block null;
  - 0.563 on falling sessions;
  - 0.671 without rotation;
  - 1.000 with the scaled penalty.

  The best Holm p for it (0.244, boosting+offset) is unchanged. It remains a lead, not a finding.

**Conclusion.** The secondary analyses are now inferential, and negative. The paper's statement that "none of them could meet the decision rule whatever the data" is outdated and must be replaced (see the change list).

## P4. Every session as its own group

- **Setup:**
  - grouping: every session is its own group, used for the folds, the transplant's donor rule and the bootstrap clusters;
  - learner and protocols: LR+offset, CTXB+MOT, within seated, within standing and pooled, both labels;
  - null: 1,000 permutations.
- **Real data:** no test met the rule. Smallest Holm p 0.342 between and 0.216 within sessions (severity ≥ 1 pooled, gain −0.002, unadjusted p 0.036).
- **Power** (combinations out of 6 declared in at least half of 20 realizations):

  | AUC | Between, session grouping | Within, session grouping | Between, inferred groups (P2) | Within, inferred groups (P2) |
  |---|---|---|---|---|
  | 0.80 | 6 | 4 | 6 | 3 |
  | 0.70 | 1 | 0 | 2 | 0 |
  | 0.65 | 0 | 0 | 1 | 0 |
  | 0.55, 0.60 | 0 | 0 | — | — |

- **Conclusion:** unchanged. The finer grouping changes neither the real-data decisions nor the power materially.

## P5. VRSQ files

The code reproduced the paper's stored correlations and intervals exactly.

**(a) All-zero share by highest in-game rating.**

- By rating: 0: 7/16; 1: 9/16; 2: 5/13; 3: 11/25.
- Rating ≥ 2: 16/38 (42 %), against 16/32 (50 %) for a rating below 2.
- Difference −8 percentage points, 95 % group-bootstrap interval [−34, +13]. Fisher's exact test: two-sided p = 0.63, odds ratio 0.73.
- The all-zero files are not more frequent among highly rated sessions.

**(b) Without the all-zero files** (38 sessions; **this selects on a VRSQ total above 0**):

- ρ ranges from 0.13 to 0.59.
- 4 of 8 correlations have Holm p < 0.05 (two-sided group-bootstrap p): highest rating, and share of records rated ≥ 2, each with the post-exposure total and with the pre-to-post change. Holm p 0.011, 0.001, < 0.001 and < 0.001.

**(c) Not conditioning on VRSQ > 0** (all 70 sessions):

- ρ between the all-zero indicator and the four in-game summaries ranges from −0.16 to −0.03.
- Every interval includes 0. Smallest Holm p 0.36.

**(d) Multiplicity.** With all 70 files, none of the 8 correlations has Holm p < 0.05 (ρ 0.18–0.34, smallest Holm p 0.051). All correlations are labelled exploratory.

**(e) Listing and clustering** (Table `tab_a3_allzero.tex`).

- The 32 files are 15 standing-campaign sessions (October 2019) and 17 seated-campaign sessions (December 2019).
- By date: 28 Oct 8/24, 29 Oct 4/9, 30 Oct 3/3, 6 Dec 8/15, 9 Dec 0/5, 10 Dec 9/14.
- Two of the six dates have all or none of their files all-zero, against 0.31 expected under random assignment (one-sided permutation p = 0.016).
- Among the 15 groups with two or more sessions, 6 are homogeneous, against 5.4 expected (p = 0.47).
- No origin is inferred.

**Conclusion.** The paper's caveat stands and is strengthened:

- the all-zero files are about as common in high-rated as in low-rated sessions;
- the higher correlations without them come from selecting on VRSQ > 0;
- the files show some clustering by recording date.

## P6. VR-LENS

**(a) Code availability.** Web searches on 15 and 16 October 2026 (paper title; authors; "VR-LENS" with "github" or "code") located no publicly available repository for VR-LENS. They returned the paper's arXiv, ACM, NSF-PAR and ResearchGate pages and other papers of the same group; a later testbed of the same group (Adversarial VR, 2025) is described in its title as open source, but it is not VR-LENS. This does not establish that no code exists. The arXiv, NSF-PAR and Papers-with-Code pages could not be opened from the analysis machine (network policy), so the paper's own code statement was not re-read. The 81 % accuracy that the manuscript quotes for the reduced model could not be re-checked from this machine either; it should be checked against the paper before submission.

**(b) Where the re-implementation could differ from the authors' code.** We could not locate the authors' code, so the re-implementation follows the paper and can differ from the authors' implementation in seven places:

1. **Preprocessing.** We standardize the inputs of the SVC and KNN learners inside their pipelines, fitted on training records only, and pass the demographic answers as the numeric codes of the released table. The authors may have scaled or encoded them differently.
2. **Data.** We use the released curated table (9,390 records). Any row filtering, de-duplication, resampling or class balancing by the authors is not described.
3. **Stacking.** We use scikit-learn's stacking with shuffled, stratified internal 5-fold folds, predicted probabilities passed to a logistic-regression meta-learner with default regularization. Another super-learner implementation, number of internal folds or kind of passed prediction could change the result.
4. **Outer folds.** Ours are shuffled and stratified, with seed 0.
5. **Unstated hyperparameters.** For example: the SVC's degree, C and probability calibration; KNN weighting; tree depths; and XGBoost's defaults, which change between versions (we used scikit-learn 1.8.0 and XGBoost 3.4.1).
6. **Random states.**
7. **The ten input columns.** We take them from the paper's table of the reduced model, by name.

Any of these could move the record-wise accuracy by a few points. None is likely to remove the gap between record-wise and group-disjoint evaluation, which also appears with group-aware internal folds and without the demographic answers.

**(c) Record-wise 10-fold without the five demographic answers** (five inputs; 294 s):

| Measure | Without demographics | With demographics (stored run) |
|---|---|---|
| Four-class accuracy | 73.3 % | 86 % |
| Macro F1 | 0.53 | not reported here |
| AUC, severity ≥ 2 | 0.90 | 0.95 |
| AUC, severity ≥ 1 | 0.87 | 0.97 |
| Context-stratified AUC, severity ≥ 2 | 0.70 | 0.87 (computed here from the stored predictions) |
| Context-stratified AUC, severity ≥ 1 | 0.84 | 0.96 |

The majority class is 55.8 %, and group-disjoint accuracy without demographics is 57.2 %.

**Conclusion.** The demographic answers account for about 13 points of the record-wise accuracy. The remaining record-wise advantage over group-disjoint evaluation (16 points) comes from inputs that still identify sessions: player position, speed, one rotation field and `StaticFrame`.

## P7. A diffuse positive control

**Setup** (as amended before the run; see Amendment 1 of the addendum):

- In each of 20 realizations and for each label, five of the 14 non-rotation features are drawn.
- The positive records of each feature are shifted by the same amount d on the normal-score scale of their context stratum and mapped back to observed values.
- d is calibrated so that the within-stratum AUC of the sum of the five shifted features is 0.70 or 0.80 over all records. d ranged from 0.20 to 0.63 (target 0.70) and from 0.38 to 0.91 (target 0.80); no calibration failed.
- Learners: LR+offset with the session-scaled penalty, scored against its 1,000-permutation null from P1, and boosting+offset, scored against its 400-permutation null.

**Power** (Table `tab_a3_diffuse.tex`): label–protocol combinations declared in at least half of 20 realizations, out of 13 estimable (family of 14):

| Learner | AUC | Between sessions | Within sessions |
|---|---|---|---|
| LR+offset, scaled penalty | 0.80 | 10 | 11 |
| LR+offset, scaled penalty | 0.70 | 3 | 7 |
| Boosting+offset | 0.80 | 7 | 11 |
| Boosting+offset | 0.70 | 1 | 7 |

For comparison, the single planted channel (Table IX, B1) at 0.80 gave 4 and 5 for LR+offset (unpenalized) and 1 and 6 for boosting+offset. With the 6-test family, LR+offset with the scaled penalty declared the diffuse signal at 0.70 in 5 of 6 combinations, both between and within sessions.

**This comparison is not like for like:**

1. **The realized signal differs.** The single channel has about the same AUC within strata and within sessions in every protocol: 0.67–0.71 at the 0.70 target, and 0.64 in the seated flight-game cell for severity ≥ 2. The diffuse shift reaches its target within-stratum AUC in the pooled protocol (0.700 and 0.800), but:
   - between sessions it varies by protocol (0.64–0.78 at the 0.70 target and 0.76–0.86 at the 0.80 target; lower in standing, higher in seated);
   - within sessions it is higher, 0.70–0.87 at the 0.70 target and 0.81–0.93 at the 0.80 target. Within-session variation of the features is smaller than within-stratum variation, so the same shift is larger relative to it.
2. **The LR learner differs:** session-scaled penalty here, unpenalized in B1.
3. **The signal sits in existing features** rather than in an added 29th feature.

**Conclusion (unfavourable to a statement in the paper).**

- The manuscript says that power for signals more diffuse than a single channel "is likely lower" (Methods) and "may be lower" (Limitations). Under this construction the opposite was observed.
- The paper's characterization of power — at least even odds for large signals only, and in a minority of combinations — holds for the single planted channel with the unpenalized learner. It does not hold for a diffuse shift of existing features with the penalized learner.
- Power therefore depends strongly on how the signal is planted and on the learner, and the paper should say so instead of guessing a direction.
- No "evidence against" rule was specified for P7, so these results do not by themselves show that diffuse signals of these sizes are absent from the real data.

## Checks, deviations and choices

- **Leakage gate and mutation test.** Run and passed before every stage: `results/add3_gate/*.json`. The session-grouping folds were also checked for P4.
- **Amendment 1.** Before any run, a smoke test showed that the P7 shift as first specified (x + δσy) moved positive records of binary and zero-inflated features to values no negative record has. Boosting could then isolate them; a seed-0 smoke test gave between-session gains of about +0.49. The shift was changed to one on the observed values, and committed before any run. The smoke-test values are recorded in the addendum.
- **Worker restart.** The session's worker restarted during the P4 null, after 912 of its 1,000 permutations had been saved. The stage was resumed with the same seeds, and the gate was run again before the resumed run. No permutation was lost or duplicated; resumption is by permutation seed.
- **Post-run correction in the statistics code (before the statistics were computed).** In a test with fewer than three groups holding a positive record (severity ≥ 2, seated flight-game cell), the paper reports no interval. The new family code would have used that test's stored bootstrap interval for the interval part of the rule. It was changed to treat such a test as having no interval, as in the paper. The test stays in its Holm family, as planned. No decision changed: its Holm p is above 0.05 in every family.
- **Choices not fixed by the request.**
  - The stronger penalty is a fixed multiple (0.1) of the session-scaled C, not a fixed number.
  - The overall-AUC gain gets no decision.
  - The P1 learners were also scored in the 6-test family.
  - P2 covers LR+offset (MOT and SELFZ) and boosting+offset.
  - P4 uses all five planted sizes.
  - P5 uses two-sided group-bootstrap p-values for Holm, with a Holm adjustment per set of correlations, plus a homogeneity statistic for clustering.
  - P7 draws from the 14 non-rotation features, uses shifts on the normal-score scale within strata (Amendment 1), and reports the 6-test family as secondary.
  - All are recorded in the addendum.
- **Not done.**
  - The HGB-reg null (as instructed).
  - A power estimate for the stronger-penalty learner, which the request did not ask for.
  - Any edit of the manuscript (as instructed).

## Compute

Total compute about 5 h 10 min (4 cores), plus about 6 min for verification:

| Stage | Wall-clock time |
|---|---|
| VRSQ | 4 min |
| VR-LENS record-wise | 5 min |
| Stronger-penalty predictions | 0.2 min |
| Scaled-penalty null (800 added permutations) | 45 min |
| Stronger-penalty null (1,000) | 53 min |
| Group-block null (800) | 44 min |
| Without-rotation null (800) | 34 min |
| Falling-session null (800) | 45 min |
| Session-grouping null (1,000) | 37 min, in two parts |
| 6-test power | 13 min |
| Session-grouping power | 10.5 min |
| Diffuse control | 16 min |
| Statistics | 0.2 min |
| Leakage gates | about 15 s each |

## Audit (16 October 2026)

See `docs/audit_addendum3.md`. It corrects "before the statistics were computed" above (real-data p-values of the finished nulls had been previewed before that change), notes that the manuscript was later edited at the author's request, and adds `results/add3_family6.json` and `results/add3_sess.json`.

Corrected in this memo on 16 October 2026, after checking every number against the result files: the P6 run has five inputs and leaves out five demographic answers (it said six and four); the P7 shift range at the 0.80 target is 0.38–0.91 (it said 0.39–0.92); the code-availability wording now says that no code could be located, not that none exists. All other numbers in this memo agree with the result files.
