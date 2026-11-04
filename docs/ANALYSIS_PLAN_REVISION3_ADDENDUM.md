# Addendum to the revision-3 analysis plan (pre-submission review)

Written on 9 October 2026, after an internal pre-submission review and **before any of the analyses
below were run**. `ANALYSIS_PLAN_REVISION3.md` is unchanged. Its primary families are unchanged:
family A (context-stratified gain) and family B (within-session gain) of LR+offset with MOT, seven
protocols × two labels, label-transplant null, Holm. This addendum covers three things only: bug fixes,
secondary analyses that answer the review, and the decision rules for them.

## What was known when this was written

- **Power under the real decision rule.** The positive control counted a planted signal as recovered
  when its fixed-model bootstrap interval was above 0. The real data must also pass the Holm-adjusted
  permutation test. Scoring the stored single-seed planted gains against the existing null gave these
  between-session counts (of 13 estimable combinations):

  | Planted AUC | Declared, real rule | Recovered, bootstrap-only rule |
  |---|---|---|
  | 0.80 | 4 | 10 |
  | 0.70 | 1 | 6 |
  | 0.65 | 1 | 3 |

  The null 95 % range of the gain is about 1.5 times wider than the bootstrap interval (median).
- **Frozen rotation in the standing campaign.** In all 37 standing sessions the three camera-rotation
  fields are constant (0 of 37 seated sessions). The 14 rotation-derived MOT features are therefore
  zero or session constants there.
- **Within-session bootstrap weights.** For within-session strata, the group bootstrap weighted pairs
  by k² instead of k, so those intervals are too wide.
- **Yalcin timestamps.** Eye-tracking timestamps that fall on a whole second are mis-parsed. A scratch
  re-run changes the second-archive numbers in the second decimal; the conclusions do not change.
- **Within-session saturation.** Sessions whose rating is a single upward step:
  - severity ≥ 1: 32 of 54 sessions with both classes, 65 % of within-session pairs;
  - severity ≥ 2: 11 of 38 sessions, 37 % of pairs.
  In the simulation every label is a single step, so a within-session gain is impossible.
- **Penalty.** A 40-permutation check with C = 0.001 moved the mean null within-session gain from about
  −0.045 to −0.006 (within seated, severity ≥ 2).
- **Cell medians without all-zero VRSQ files.** Post-exposure VRSQ medians:

  | Cell | Median |
  |---|---|
  | Seated racing | 50.0 |
  | Standing racing | 18.3 |
  | Standing flight | 14.6 |
  | Seated flight | 5.8 |

## A. Corrections (no analytic choice involved)

1. **Bootstrap for nested strata.** When every stratum lies inside one resampling unit (within-session
   and within-participant AUCs), a unit drawn k times enters with weight k: num = Σ_g k_g A_gg,
   den = Σ_g k_g D_gg. Overall and context-stratified AUCs are unchanged.
2. **Yalcin timestamps.** Parse the first 14 characters as date and time and the remainder as a
   right-padded fraction of a second. Sort eye-tracking samples by time before computing gaze speed.
   Button timestamps use the same parser. Count button-press episodes (timestamps less than 1 s apart
   form one episode) as well as timestamps.
3. **Between-session pair share.** Compute it on the estimand's own strata (fold × campaign × game ×
   30-s bin, averaged over repeats). Also report the share of between-session pairs whose two sessions
   belong to the same inferred group.

## B. Secondary analyses

**B1. Power of the pre-specified rule** (replaces the single-seed "minimum detectable effect" as the
basis of every power statement)

- The planted channel (d·y + AR(1) noise, ρ = 0.9) is regenerated with 20 noise seeds,
  seed = 5000 + 100·s + label for s = 0, …, 19.
- Grid: the five levels (0.55, 0.60, 0.65, 0.70, 0.80) × the seven family protocols × two labels.
- Models:
  - LR+offset with CTXB+MOT+SPK and with CTXB+SPK;
  - boosting+offset with CTXB+MOT+SPK.
  The baseline is the study's CTXB, the same fit as for real labels.
- Scoring:
  - Each planted gain (context-stratified and within-session) is scored against the label-transplant
    null of the corresponding real-telemetry model (LR+offset: 1000 permutations; boosting+offset: 400).
  - A planted signal is **declared** when two conditions hold:
    1. the Holm-adjusted p is below 0.05 within its family of 14 tests, all planted at the same level
       and seed;
    2. the bootstrap interval of the gain (corrected as in A1) excludes 0.
- Power is the share of the 20 seeds in which the signal is declared.
- Also reported:
  - the stricter single-protocol criterion p ≤ 0.05/14;
  - the realised within-stratum and within-session AUC of the channel (mean over seeds).
- **Evidence against.** A real gain counts as evidence against a planted size only if it is smaller
  than at least 19 of the 20 planted gains of that size in the same protocol and label. This is
  descriptive; it replaces the earlier comparison of two fixed-model intervals, which ignores the
  variability between channel realisations and refits.
- The original single-seed recovery table stays in the supplement, labelled as recovery under the
  bootstrap-only rule.

**B2. Rotation channel**

- (a) Count, per cell, the sessions with constant rotation fields.
- (b) Repeat the context decoding (same learners, folds and targets) without the 14 rotation-derived
  features: MOTNR, which keeps speed, acceleration and region of interest (14 features).
- (c) LR+offset with CTXB+MOTNR in the seven family protocols, for both labels:
  - gains with bootstrap intervals;
  - a 200-permutation label-transplant null (sims 0–199, the main null's seeds);
  - Holm families of 14 as in the main analysis.

**B3. Within-session saturation**

- New estimand: within-session AUC restricted to sessions whose labels are not non-decreasing (the
  rating also falls at least once), computed on the labels being evaluated.
- Reported, with bootstrap intervals, for the context baseline, LR+offset (MOT) and boosting+offset
  in all protocols.
- Null: a 200-permutation re-run of the main transplant null (sims 0–199, identical permutations),
  LR+offset MOT, recording this estimand as well. It yields a Holm family of 14 (secondary).

**B4. Simulation with episodic labels**

- Labels: after the onset time τ, each session's label alternates between on-episodes and
  off-episodes, each of length U[20, 60] s, so exposure time no longer orders the records within a
  session.
- Scenarios:
  - Ne: no information;
  - Me: feature 0 shifted by 0.5 SD while the label is on;
  - Me2: shifted by 1.0 SD.
- 20 replications each, seeds 3000 + rep.
- Outputs:
  - the same outputs as before;
  - the within-session AUC of the time-only baseline;
  - the realised within-session AUC of feature 0.
- Expectation written now: the within-session gain is near or below 0 in Ne and positive in Me2.
- The step-label scenarios N, S and M are kept unchanged.

**B5. Penalty sensitivity**

- LR+offset with C = (training sessions) / (training records): one session's weight per unit of
  penalty, instead of C = 1.
- Run with CTXB+MOT in the seven family protocols, both labels.
- Outputs: gains, a 200-permutation null (sims 0–199) with Holm families, and the positive control
  with the original seed.

**B6. Group-block null**

- Transplant whole groups within a cell: a derangement over the cell's groups, so a group's sessions
  all receive sequences from one other group.
- Within the pair of groups, recipient and donor sessions are matched by length rank; donor sessions
  are re-used cyclically if the donor group has fewer sessions.
- Run with 200 permutations, seeds 800 + sim, LR+offset MOT, seven protocols, both labels.
- Report null quantiles and p-values next to those of the session-level null.

**B7. Second archive, offset design**

- LR+offset on the 18 eye-tracking features, with a time-only baseline (t, t²) as the offset.
- Grouped 5-fold × 5 repeats.
- Report the overall and within-participant gains with bootstrap intervals (A1).

**B8. Label-shift table.** Add the existing default-HGB results on MOT alone (correct and shifted labels)
to the label-shift table.

**B9. Boosting+offset positive control.** Report the stored single-seed recovery table and B1's power for
boosting+offset.

## C. Decision rules

- All statements in the paper about power or "evidence against" rest on B1.
- B2–B7 are secondary (sensitivity or descriptive). They can qualify the primary result but do not
  replace it.
- If a secondary null (B2c, B3, B5, B6) yields a Holm-adjusted p below 0.05 with an interval that
  excludes 0, it is reported as a secondary finding, with its family and number of permutations, and
  the abstract says so.
- Families A and B are not re-tested. Their permutation p-values do not depend on the bootstrap, so A1
  changes only their intervals.

## D. Deviation recorded after B1 was run (9 October 2026)

The "evidence against" rule of B1 turned out to be uninformative, so it is not used in the paper.
The rule counted a real gain as evidence against a planted size when it was smaller than at least 19
of the 20 planted gains of that size. It flagged planted within-stratum AUCs as small as 0.55. The
reason is that the planted gains are computed on the same records as the real gain, with one channel
added, and vary only with that channel's noise, not with the sample of participants. The real gain
is therefore below almost every planted gain whenever the planted channel adds anything. The interval
comparison of the original plan has the same weakness.

The paper reports only the power of the decision rule, which B1 defines without this problem.
Supplementary Note S2 describes the deviation.

Three further parts of the plan were not carried out or not reported as written (recorded on
9 October 2026, after all addendum analyses had been run):

- **B1, channel alone.** The 20-seed replication was run for LR+offset and boosting+offset with
  CTXB+MOT+SPK, not for LR+offset with CTXB+SPK. The channel alone appears only with the original
  single seed (Supplementary Table S19).
- **B3, other models.** The within-session AUC on falling sessions was computed for the context
  baseline, LR+offset and boosting+offset in all protocols (`results/rev3_stats.json`), but the paper
  reports it only for LR+offset in the family protocols.
- **B2c, intervals.** The bootstrap intervals of the gains without the rotation-derived features were
  computed (`results/rev3_stats.json`) but are not tabulated; Supplementary Table S20 gives the gains
  and Holm-adjusted p-values only.

Update after the final independent check (9 October 2026). No analysis was re-run; the following were
read from the existing result files and added to the Supplement:

- B3, other models: the within-session AUC on falling sessions for the context baseline, LR+offset and
  boosting+offset (Supplementary Table S22). Boosting+offset was not tested against a null of this
  estimand; two of its gains for severity ≥ 1 have intervals above 0, which the paper reports as such.
- B2c, intervals: the gains without the rotation-derived features with their intervals (Table S23).
- Plan outputs that had been computed but not reported: the label-shift control for severity ≥ 1 and
  for the context baseline (Table S24), the location of the gain under the group-block null (Table S25),
  the within-session AUC of the earlier learners (Table S26), session-level decoding (Table S27) and the
  single-protocol criterion of B1 (Table S18).
- B1, channel alone: still not run; the paper says so (Note S2, Section IV-F).
- Grouping sensitivity: only the gains were recomputed under the other groupings; no null or control.

Note recorded after the final review of the Results (9 October 2026). No analysis was re-run.

- **Secondary nulls cannot meet the rule of section C.** B2c, B3, B5 and B6 were run with 200
  permutations. With Holm over 14 tests (13 for falling sessions) the smallest attainable adjusted p
  is 14/201 = 0.070 (13/201 = 0.065), so the rule "Holm-adjusted p below 0.05 with an interval that
  excludes 0" could not be met whatever the data. The smallest observed adjusted p in these families
  is 0.209. The paper reports this and treats these analyses as descriptive (Section V-E,
  Supplementary Table S20 and Note S2). Making them confirmatory would need at least 280
  permutations each, better 1000 (LR+offset; about 3 h per 1000 permutations on 4 cores).
- **Label-shift control.** The circular displacement of the training labels is a single draw
  (seed 1) and was not repeated; the paper says so.

Note recorded during the final pre-submission audit (10 October 2026). No analysis was re-run.

- **What the record shows about timing.** Run start times were not logged. Git shows that this
  addendum (commit ff68a1c, 9 October 2026, 08:49 UTC) was committed before any of its result files
  (first at 09:11 UTC); the paper therefore says "committed before any of their result files"
  rather than "before they were run".
