# Addendum 3: further post hoc analyses (P1–P7)

Written on 15 October 2026, before any of the analyses below was run. It was committed to the repository before the first run.

## Status of these analyses

Everything in this addendum is **post hoc**. It was specified after all results of the analysis plan (`docs/ANALYSIS_PLAN_REVISION3.md`) and of addenda 1 and 2 were known. In the paper these analyses are to be called "post hoc" or "specified after the main results", never "prespecified".

They do not replace anything:

- The primary families A and B of the analysis plan remain the primary result, with their 1,000-permutation null.
- The secondary families A′/B′ and A″/B″ are kept as they are.
- Every existing result file stays unchanged. New results go to new files only (listed under "Outputs").

## What was known when this addendum was written

All results in the repository at commit `9cb2eac` and in the current manuscript were known, including the following.

**Primary and secondary families** (14 tests each; rule: Holm-adjusted permutation p < 0.05 and a bootstrap interval above 0):

- LR+offset, CTXB+MOT, 1,000 permutations: no test met the rule (0/14 between sessions, 0/14 within sessions). Unadjusted p < 0.05 in 0 between-session and 3 within-session tests: severity ≥ 1 pooled, p = 0.024; severity ≥ 2 seated flight cell, p = 0.028; severity ≥ 1 standing racing cell, p = 0.043.
- Boosting+offset, 400 permutations: 0/14 and 0/14. Smallest Holm-adjusted p 1.000 between and 0.244 within sessions.
- Self-normalized features: 0/14 and 0/14.

**Power of the decision rule** (addendum 1, B1; 20 realizations of a planted channel, seeds 5000 + 100·s + label):

- At a planted within-stratum AUC of 0.80, LR+offset declared the signal in at least half of the realizations in 4 of 13 between-session and 5 of 13 within-session label–protocol combinations.
- At 0.70: 1 and 0. At 0.65: 0 between sessions.
- Boosting+offset at 0.80: 1 of 13 and 6 of 13.
- The permutation part of the rule alone, at 0.80 within sessions: 8.
- For every realization the stored rows (`results/rev3_posctl_seeds.json`) contain the planted gains and their one-sided permutation p-values. The rescoring of these p-values with a 6-test Holm family (P2 below) has **not** been computed or inspected before this addendum.

**Secondary nulls of addendum 1** (200 permutations each; smallest attainable Holm-adjusted p 0.070 with 14 tests, so none could meet the rule):

- group-block null;
- session-scaled penalty for LR+offset (C = training sessions / training records);
- the 14 features not derived from rotation;
- within-session AUC on sessions whose rating also falls.

Smallest Holm-adjusted p across them: 0.279. Counts of unadjusted p < 0.05 (between / within):

- group-block: 1 / 2;
- scaled penalty: 0 / 3;
- without rotation: 0 / 1;
- falling sessions: 1 of 13.

**Grouping.** With every session as its own group, the gains of LR+offset ranged from −0.02 to +0.11 between sessions and from −0.14 to +0.18 within sessions. No null or positive control was run under that grouping.

**Questionnaire (VRSQ).**

- 32 of 70 files are zero in every item.
- 16 of the 38 sessions whose highest in-game rating was 2 or 3 have an all-zero file, against 16 of the 32 sessions with a highest rating below 2.
- Spearman correlations (four in-game summaries × post-exposure total and pre-to-post change): 0.18 to 0.34 with all 70 files; 0.13 to 0.59 on the 38 sessions without all-zero files.
- No test of the difference in all-zero shares, no multiplicity correction and no listing of the all-zero files by date had been made.

**VR-LENS re-implementation.**

- Record-wise 10-fold accuracy: 86 %.
- Group-disjoint accuracy: 53 %, 52.2 % with group-aware internal stacking folds; majority-class rate 55.8 %.
- Group-disjoint without the demographic answers: 57.2 %.
- No record-wise run without the demographic answers had been made.

**Checks made while writing this addendum, before any run.**

1. The first 200 permutations of `results/rev3_null_lroff_nm.json` reproduce rows 0–199 of the main null `results/rev3_null_lroff.json` exactly: identical gains in context-stratified and within-session AUC, largest absolute difference 0.0 over 2,800 rows. This supports extending the stored 200-permutation nulls from their saved permutations instead of re-running them.
2. The session-scaled penalty C (training sessions / training records) ranges from 0.0035 to 0.0051 over the outer folds of the 7 family protocols (median 0.0040).
3. A web search for public code of VR-LENS (Kundu et al., IUI 2023) found no repository. The arXiv, NSF-PAR and Papers-with-Code pages could not be opened from the analysis machine, so the paper's own code-availability statement was not re-read.

**Consequence for P2.** The real-data p-values of the six tests in the new 6-test family are already known, because they are the same tests as in the 14-test families. For LR+offset (CTXB+MOT) the smallest are 0.0959 between sessions and 0.024 within sessions. For boosting+offset they are 0.140 and 0.042. The Holm-adjusted p is therefore at least 6 × 0.024 = 0.144 within sessions and higher between sessions. **No real-data test of the 6-test family can meet the decision rule**, and this was known before the family was defined. Only the power of the 6-test family is not yet known.

## Rules common to all parts

**Decision rule (unchanged).** A gain is declared when both hold:

- the one-sided permutation p, p = (1 + k)/(1 + n), is below 0.05 after Holm adjustment within its family;
- the 95 % group-bootstrap interval of the gain lies above 0 (lower bound > 0).

The bootstrap uses B = 10,000 draws with the fitted models fixed (`fa_rev3_stats.arrays`, nested weights within sessions).

- The rule is not changed after any result is seen.
- Nothing is tuned.
- Every result is reported, favourable or not.

**Leakage gate.** Before every stage, the leakage gate and the mutation test of Supplementary Note S1 are run:

- `fa_data.assert_no_leakage` on the grouped folds of all 14 protocols;
- for P4, also on the session-grouped folds;
- `fa_run.leak_mutation_test`.

The result goes to a new file, `results/add3_gate/<stage>.json`; `results/leakage_checks.json` is not overwritten. A stage does not start unless both checks pass. As before, every fold also asserts that its training and test groups are disjoint.

**Unchanged setup.** Unless stated otherwise:

- the folds are the study's grouped 5-fold × 5 repeats (`make_folds`, seeds 1000 + r);
- the 38 inferred groups, the context strata (fold × campaign × game × 30-s bin), the label thresholds, the baseline and the learners' settings are those of the plan;
- no operating threshold is computed (`inner=False`, as in the addendum-1 stages).

**Compute.** The cap is the plan's 8 h per stage. The expected total is about 5 h on 4 cores. If a stage would exceed the cap, it stops and the shortfall is reported.

**Extended nulls.** Where a stored 200-permutation null is extended to 1,000:

- its rows are copied into a new file and the stage resumes with the same seeds;
- before extending, permutations 0 and 199 are re-run into a scratch file and must reproduce the stored rows (absolute difference ≤ 1e-12);
- if they do not, the extension stops and the discrepancy is reported.

## P1. LR+offset with a session-scaled and a stronger penalty

**Learners.**

- `lroffp` (addendum 1, B5): C = training sessions / training records of each outer fold (0.0035–0.0051).
- `lroffs` (new): C = 0.1 × training sessions / training records, so 0.00035–0.00051. This is a ten-times-stronger ridge penalty. The factor 0.1 is fixed here and is not varied.

Both use the same baseline, offset, folds, groups and strata as `lroff`, and the CTXB+MOT features (28). The 7 family protocols are run for both labels.

**Estimands.** For each of the 14 label–protocol combinations, the gain over the context baseline in:

- overall AUC;
- context-stratified (between-session) AUC;
- within-session AUC.

Each has a 95 % group-bootstrap interval.

**Nulls.** Label-transplant null with refitting (baseline and learner refitted on every permutation), session transplant, seeds 700 + s, 1,000 permutations (s = 0…999):

- `lroffp`: the 200 stored permutations (`results/rev3_null_lroffp.json`) are copied to `results/add3_null_lroffp.json` and extended;
- `lroffs`: new, `results/add3_null_lroffs.json`.

**Families and rule.** Between and within sessions are separate families of 14 tests each, per learner, with Holm and the common rule. In addition, each learner is scored in the post hoc 6-test family of P2. The overall-AUC gain is not a tested estimand of the plan: it is reported with its interval and an unadjusted one-sided permutation p only, without a decision.

## P2. A post hoc 6-test family

**Family.** The protocols within seated, within standing and pooled, for both labels: 6 tests per estimand. Between sessions and within sessions are separate families.

It is applied to:

- LR+offset with CTXB+MOT and with CTXB+SELFZ (1,000-permutation null, `results/rev3_null_lroff.json`);
- boosting+offset with CTXB+MOT (400 permutations, `results/rev3_null_xgboff.json`).

Holm is applied within each family, with the common rule. The smallest attainable Holm-adjusted p is 6/1001 = 0.006 with 1,000 permutations and 6/401 = 0.015 with 400. The 14-test families are kept, and both families are reported. As stated above, the real-data outcome of this family is already determined by known p-values.

**Power.**

- Realizations: the 20 planted-channel realizations of addendum 1, B1 (seeds 5000 + 100·s + label, s = 0…19), at planted within-stratum AUCs 0.65, 0.70 and 0.80.
- Models: CTXB+MOT+SPK with LR+offset and with boosting+offset.
- Refitting: the models are refitted for the three protocols, because the per-realization predictions were not stored. The refitted planted gains must equal the stored B1 gains (absolute difference ≤ 1e-9); otherwise the stage stops.
- Scoring: each planted gain is scored against the real-telemetry null of its learner, with Holm within the 6 tests of its level, learner and estimand. Its bootstrap interval is computed where the Holm-adjusted p < 0.05. A signal is declared under the common rule.
- Table (in the format of Table VIII): the percentage of the 20 realizations in which the signal was declared, for each protocol, label, estimand (B, W) and level. Also reported: the number of label–protocol combinations, out of 6, declared in at least half of the realizations.

**Output.** `results/add3_family6.json` (real-data families) and `results/add3_posctl6.json` (power).

## P3. The secondary analyses with 1,000 permutations

The four secondary nulls of addendum 1 are extended from 200 to 1,000 permutations (s = 0…999), with refitting, into new files:

| Analysis | Learner, features | Transplant, seeds | New file |
|---|---|---|---|
| Group-block null | LR+offset, CTXB+MOT | whole groups (`transplant_labels_block`), 800 + s | `results/add3_null_lroff_block.json` |
| Session-scaled penalty | `lroffp`, CTXB+MOT | session, 700 + s | `results/add3_null_lroffp.json` (shared with P1) |
| Without rotation | LR+offset, CTXB+MOTNR (14 features) | session, 700 + s | `results/add3_null_lroff_nr.json` |
| Falling sessions | LR+offset, CTXB+MOT, within-session AUC on sessions whose rating also falls | session, 700 + s | `results/add3_null_lroff_nm.json` |

- The families are those of Supplementary Table S20, with Holm and the common rule: 14 tests each, and 13 for falling sessions (for severity ≥ 1 no session of the seated flight cell has a falling rating).
- The HGB-reg null is **not** re-run.
- Consistency check: the falling-session null re-runs the transplant and model of the main null. For s = 0…999 its gains in context-stratified and within-session AUC must equal those of `results/rev3_null_lroff.json`. Any difference is reported.
- Reporting: a new version of Table S20 (`results/paper/tab_a3_secondary.tex`) and a list of the changes to Table S20 and to the paragraph "Secondary analyses are descriptive" of Section V-E. The manuscript is not edited.

## P4. Nulls and positive control with every session as its own group

**Grouping.** Every session is its own group, exactly as in `stage_sens_group` (group_id = session codes). This grouping is used for:

- the folds (`make_folds`, same seeds);
- the transplant's donor rule;
- the bootstrap clusters.

**Real gains.** The stored session-grouping predictions of LR+offset (`results/preds/sess_*__CTXB+MOT__lroff.npz` and `sess_*__CTXB__ctxb.npz`), with session-cluster bootstrap intervals.

**Null.** LR+offset, CTXB+MOT, protocols within seated, within standing and pooled, both labels, 1,000 permutations (seeds 700 + s, refitting): `results/add3_null_sess.json`.

**Family and rule.** 6 tests per estimand (between, within), with Holm and the common rule.

**Positive control.**

- The B1 channel (seeds 5000 + 100·s + label, s = 0…19) at planted AUCs 0.55, 0.60, 0.65, 0.70 and 0.80.
- Model: CTXB+MOT+SPK with LR+offset under session grouping, for the same 3 protocols and both labels.
- Each planted gain is scored against the P4 null, with Holm within 6. The interval uses session clusters and is computed where the Holm-adjusted p < 0.05.
- The power table uses the format of Table VIII.

**Comparison.** The result is compared with the inferred-group 6-test family of P2 (real-data decisions and power counts). We state whether the conclusions change.

**Output.** `results/add3_sess.json`.

## P5. VRSQ files

Sessions: the 70 sessions with a VRSQ file. All-zero: every pre- and post-exposure item is zero (`rev3_vrsq0` definition).

**(a) All-zero share by highest in-game rating.**

- Counts and shares of all-zero files for a highest in-game rating of 0, 1, 2 and 3.
- For ≥ 2 against < 2 (16/38 against 16/32):
  - Fisher's exact test, two-sided;
  - the difference in shares with a 95 % percentile interval from a group bootstrap (resampling the inferred groups, sessions of a drawn group enter together; B = 10,000, seed 0).

**(b) Correlations without all-zero files.** The 8 correlations on the 38 remaining sessions (4 in-game summaries × post-exposure total and pre-to-post change). These are recomputed with the existing method and reported with the explicit statement that they select on a VRSQ total above 0.

**(c) A model that does not condition on VRSQ > 0.** Among all 70 sessions, the Spearman correlation of each of the 4 in-game summaries with the all-zero indicator. Each has a group-bootstrap interval (B = 10,000, seed 0) and a two-sided group-bootstrap p, 2·min(share ≤ 0, share ≥ 0), as in `fa_stats.p_two_sided`.

**(d) Multiplicity.** Each set of correlations gets its own Holm adjustment, applied to the two-sided group-bootstrap p-values:

- the 8 correlations with all files;
- the 8 without all-zero files;
- the 4 in (c).

All correlations are also labelled exploratory. Unadjusted values are reported next to the adjusted ones.

**(e) Listing and clustering.**

- A table of the 32 all-zero files: session, date, inferred group, game, campaign.
- Their counts per recording date and per group.
- Two descriptive permutation checks of clustering (10,000 random reassignments of the 32 all-zero flags among the 70 sessions, seed 0): the number of dates, and the number of groups with more than one session, in which all sessions or no session are all-zero, compared with the observed number. One-sided p for more homogeneity than chance.
- No origin of the all-zero files is inferred or guessed.

**Output.** `results/add3_vrsq.json`.

## P6. VR-LENS

**(a) Code availability.** The search described above is reported as made, with its limits: no public code was found, and the paper's own statement could not be re-read from the analysis machine. No code is invented or assumed.

**(b) Possible differences.** A one-paragraph note lists where the re-implementation could differ from the authors' code. It is taken from the assumptions recorded in `src/fa_vrlens.py`:

- scaling;
- stacking internal folds;
- outer folds and their seed;
- random states;
- library defaults and versions, including XGBoost;
- encoding of the demographic columns;
- the curated table used;
- how probabilities are passed to the meta-learner.

**(c) Record-wise run without demographic answers.** The record-wise 10-fold protocol of the original evaluation (`StratifiedKFold(10, shuffle=True, random_state=0)`, the same stacking settings) is run with the features of ablation A (the ten features without the four `User*` demographic answers).

- Reported: four-class accuracy, macro F1, record-wise AUC for severity ≥ 1 and ≥ 2, and the context-stratified AUC (fold × context strata).
- Comparison: with the record-wise run with demographics (86 %) and the group-disjoint run without them (57.2 %).
- Output: `results/add3_vrlens_rw_nodemo.json` and `results/preds/a3rw_sensVRLENSA_sev{1,2}__within_curated__VRLENS__stack.npz`.

## P7. A diffuse positive control

**Planted signal.** In each realization s = 0…19 and for each label, five telemetry features are drawn without replacement from the 14 non-rotation features (MOTNR), with `numpy.random.default_rng(6000 + 100·s + label)`. The rotation-derived features are excluded because they are constant within standing sessions (some identically zero), so a shift would create a noiseless channel there.

Each selected feature j is shifted on the positive records:

x_j ← x_j + δ·σ_j·y

- σ_j is the pooled within-stratum standard deviation of feature j over all records (strata: campaign × game × 30-s bin);
- y is the label (severity ≥ 1 or ≥ 2).

**Calibration.**

- δ ≥ 0 is found by bisection on [0, 10] (60 iterations).
- Target: the within-stratum AUC (same strata, all 18,384 records) of the combined score S = Σ_j x_j/σ_j, the equally weighted sum of the five shifted standardized features, equals 0.70 or 0.80.
- If the AUC of S at δ = 0 already exceeds the target, δ = 0 and the realization is flagged.
- Recorded: the selected features, δ, the AUC of S at δ = 0, and the realized AUC of S on each protocol's target records.

**Models and families.**

- Models: CTXB+MOT with the shifted features (no extra channel), with `lroffp` and with boosting+offset; the 7 family protocols, both labels.
- Each planted gain is scored against the real-telemetry null of its learner: the 1,000-permutation `lroffp` null of P1, and the 400-permutation boosting+offset null.
- Families: Holm within the 14 tests of each level, learner and estimand (the rule of Table VIII). Secondary: Holm within the 6-test family of P2.
- Bootstrap intervals where the Holm-adjusted p < 0.05 in either family. Declared under the common rule.

**Output.** `results/add3_posctl_diffuse.json`, and a new supplementary table in the format of Table VIII.

## Outputs and reporting

**New files only.**

- `results/add3_*.json`;
- `results/add3_gate/*.json`;
- new prediction files with new keys in `results/preds/`;
- tables `results/paper/tab_a3_*.tex`;
- values with the prefix `a3.`, added to `results/paper/values.json` by a new report step (`src/fa_report_add3.py`). No existing key may change; this is checked by comparison with the values before the step.

**Code.** New stages are in `src/fa_add3.py`. The existing modules receive only additions that leave their current behaviour unchanged: the learner `lroffs`, and a protocol argument for the null stage.

**Verification.** `src/fa_verify.py` gains a step for this addendum:

- it recomputes the new AUC gains from the saved predictions;
- it recomputes the permutation p-values and Holm adjustments of every new family from the null files, independently of the statistics code;
- it checks the qualitative claims (`results/claims_add3.json`).

The existing steps are unchanged and must still pass.

**Memo.** `docs/results_addendum3.md` reports, for each part:

- what was run;
- the numbers;
- whether a conclusion of the paper changes.

Unfavourable results are stated plainly. A separate list of every manuscript sentence or table that would change (old and new values) is prepared for the authors; the manuscript itself is not edited.

## Amendment 1 (15 October 2026, before any stage was run)

This amendment was written and committed after a smoke test of the new code and before any stage of this addendum was run. It changes one element of P7. The smoke test wrote no result file.

**What the smoke test ran and showed.** The test checked that the code runs. It ran:

1. One permutation (s = 0) of the `lroffs` null in the seated racing-game cell (severity ≥ 1: gains +0.102 between and −0.005 within sessions).
2. Permutation 0 of the stored `lroffp` and group-block nulls in that cell. Both reproduced the stored rows exactly (largest difference 0.0).
3. Realization 0 of the B1 channel at 0.80 in that cell. It reproduced the stored B1 gains exactly (+0.155 / +0.007 for severity ≥ 1, +0.168 / +0.068 for severity ≥ 2), and the scoring code ran.
4. Realization 0 of the B1 channel at 0.80 under session grouping in the within-standing protocol (gains +0.226 / +0.099 for severity ≥ 1, +0.214 / +0.045 for severity ≥ 2).
5. Realization 0 of the P7 shift as first specified, at 0.80 in the seated racing-game cell. The planted gains between / within sessions were:
   - severity ≥ 1: `lroffp` +0.163 / +0.096, boosting+offset +0.488 / +0.234;
   - severity ≥ 2: `lroffp` +0.238 / +0.104, boosting+offset +0.457 / +0.226.

No real-label result of any analysis in this addendum was computed.

**Why P7 is changed.** A combined within-stratum AUC of 0.80 cannot produce a between-session gain of nearly +0.5 through a distributional shift. The cause is the support of the features:

- 3 of the 14 non-rotation features are binary: `roi_background`, `roi_middleground` and `roi_foreground`.
- Several others have large point masses at 0 (`PlayerSpeed_diff` 50 %, `PlayerAcceleration` 49 %, `PlayerAcceleration_diff` 35 %), and `PlayerSpeed` takes 147 distinct values.

A fixed additive shift δ·σ_j on the positive records moves them to values that no negative record has (for a binary feature, 0 + δσ and 1 + δσ). A tree learner can isolate such values exactly. The planted signal would then be detected through the shift's artefact, not through its size, and the power of boosting+offset would be overstated. This change is made because of that mechanism, not because of any significance result.

**The amended P7 shift.** Everything else in P7 is unchanged:

- the five features and their seeds;
- calibration to a within-stratum AUC of 0.70 and 0.80 of the equally weighted sum of the five shifted features, each divided by its pooled within-stratum SD, with bisection on [0, 10];
- the learners, nulls, families, rule and outputs.

The shift itself is changed. Within each stratum (campaign × game × 30-s bin), every record of a selected feature gets its mid-quantile in that stratum:

- ties are broken by uniform random numbers drawn from the same generator, after the features;
- the quantile of each positive record is moved by d on the normal-score scale, q′ = Φ(Φ⁻¹(q) + d);
- the record takes the stratum's empirical quantile at q′.

Consequences:

- Every shifted value is a value observed in that stratum, so a binary feature stays binary: the shift flips some positive records from 0 to 1.
- Negative records are unchanged, and d = 0 returns the data exactly.
- The shift is monotone in d, so the bisection is valid.
- d is in normal-score units, not in units of σ_j. It is recorded with the share of positive records whose values changed.

A calibration-only check, with no learner fitted, gave a planted-score AUC of exactly 0.70 and 0.80 for realizations 0 and 1 and both labels, and no negative record changed. P7 is reported as amended. The original specification and the smoke-test values above are kept here.

## After the run (recorded 16 October 2026)

All stages ran as specified, with Amendment 1 applied to P7. The results are reported in `docs/results_addendum3.md`. Three events are recorded here.

1. **Worker restart.** The worker process of the analysis session restarted during the P4 null, after 912 of its 1,000 permutations had been saved. The stage was resumed with the same seeds. Resumption is by permutation seed, so none was lost or duplicated. The leakage gate and mutation test were run again before the resumed run, and passed.
2. **Statistics code.** Before the statistics were computed, one detail of the new statistics code was aligned with the paper. A test with fewer than three groups holding a positive record (severity ≥ 2, seated flight-game cell) has no reported interval, so it cannot meet the interval part of the rule. It stays in its Holm family as planned. No decision depends on this: its Holm-adjusted p is above 0.05 in every family.
3. **Extra families.** Computed but not specified above, and reported as descriptive:
   - the falling-session families of the two P1 learners;
   - the 6-test family of the P1 learners, which is specified in P1.

## Audit (recorded 16 October 2026)

An audit of this addendum, its runs and its reporting is in `docs/audit_addendum3.md`. It corrects item 2 above: the change was made before the statistics step, but after the real-data p-values of the first three finished nulls had been previewed. The change can only remove a declaration, and no decision changed. The audit also adds the two named outputs of P2 and P4 (`results/add3_family6.json`, `results/add3_sess.json`) as extractions from `results/add3_stats.json`.
