# Analysis plan for revision 3 — fixed before any new result was computed

Written 2026-10-08, before any real-label result of the new learners, controls or estimands
below was computed. It answers the referee report on revision 2
(points 3.1–3.7, Q1–Q5, S1–S7). The code in `src/fa_rev3.py`, `src/fa_yalcin.py` and
`src/fa_sim.py` implements exactly these choices. Nothing below may be changed after results
are seen; if a run fails, the failure is reported, not replaced by a different design.

**What was already known when this plan was written.** All revision-2 results (in particular the
logistic-regression results, which were the second learner), and three diagnostics computed
while the plan was drafted:

- on held-out participant groups, the 28 telemetry features recognise the campaign with
  ROC-AUC 1.00 and the game with 0.99 (logistic regression and default HGB);
- 0.9 % (severity ≥ 2) and 0.7 % (severity ≥ 1) of the positive–negative pairs inside the
  context strata come from the same session;
- 38 (severity ≥ 2) and 54 (severity ≥ 1) of the 74 sessions contain both classes; exposure time
  alone ranks the records inside these sessions with AUC 0.75 and 0.83.

The new learners were timed on transplanted (null) labels only.

**Reproducibility check.** Because the archived predictions were not reachable from the
revision machine, every revision-2 stage that writes predictions is re-run with the released code
(`checks main ctx mitig cell control rowwise vrsq sens logo_ctx posctl es_check hgbreg
posctl_reg`). The re-computed result files are compared number by number with the committed
ones; every difference is reported. The 500-permutation nulls of revision 2 (`null.json`,
`null_lr.json`) are kept as committed and are **not** re-run.

**Compute cap.** No stage may need more than 8 hours on the 4-core revision machine. The
500-permutation HGB-reg null is not run (more than 200 hours).

---

## Stage 1 — Does the telemetry encode the context? (referee 3.1, Q2)

- Targets: game (racing = 1), campaign (standing = 1), and the 4 game × campaign cells.
- Features: MOT (28), and separately SELFZ (28).
- Learners: logistic regression (C = 1) and default HGB (study settings), standardised inside
  the fold.
- Folds: the pooled protocol's folds (grouped 5-fold × 5 repeats over inferred groups).
- Metric: within-fold ROC-AUC (one-vs-rest macro AUC for the cells), with group-bootstrap 95 %
  intervals (B = 10,000); session-level AUC (mean score per session) as a second number.

## Stage 2 — Main learners and the telemetry-gain estimand (3.1, 3.2)

**Context baseline (`ctxb`).** Logistic regression (C = 1, class-balanced, unpenalised
intercept) on exposure time t (minutes), t², the game indicator r, the campaign indicator s,
r·s (so each of the four cells has its own level), r·t and s·t. Columns standardised on the
training records of the fold. In a single cell, r and s are constant and the baseline is a
quadratic in exposure time.

**Main learner (`lroff`).** Logistic regression on the telemetry features with the baseline's
logit as a fixed offset: logit p = offset + b + x′β, objective
C·Σ wᵢ·log(1 + exp(−sᵢηᵢ)) + ‖β‖²/2 with C = 1 and class-balanced weights (the study's LR
objective). The baseline is fitted first on the same training records and then held fixed.
Solved by exact Newton steps (unique optimum). Nothing is tuned.

**Second learner (`xgboff`).** xgboost 3.4.1 started from the same offset (base_margin), fixed
settings: 150 rounds, learning rate 0.05, max depth 2, min_child_weight 25, λ = 10, hist, no
subsampling, class-balanced weights, seed 0, one thread. Nothing is tuned; the same settings are
used for real and null labels.

**Feature sets.** CTXB (baseline only), CTXB+MOT, CTXB+SELFZ. Exposure time and context enter
only through the offset.

**Protocols.** The 10 protocols of Table VI and the 4 single game × campaign cells; both labels;
the study's folds; operating threshold as before (10 % training false-alarm budget).

**Estimands** (all on identical target records, pairs counted only within a fold model):

1. overall AUC;
2. context-stratified AUC (fold × campaign × game × 30-s bin);
3. **within-session AUC** (fold × session: only positive and negative records of the same session
   are compared; sessions without both classes do not contribute).

**Telemetry gain** = metric(CTXB+telemetry) − metric(CTXB), with paired group-bootstrap 95 %
intervals (B = 10,000, fitted models fixed).

**Role of the learners.** `lroff` carries the main results (the referee's S1, second option:
a learner that passes every control). `xgboff` checks whether a non-linear learner changes the
conclusion. HGB-reg and default HGB are reported in the supplement; HGB-reg is no longer used for
any claim. This change of main learner is disclosed in the response letter.

## Stage 3 — Controls for the main learners (3.2, S1)

**Label-transplant null.** Same transplant rule and seeds (700 + s) as revision 2.

- Protocols: within seated, within standing, pooled and the 4 cells (7); both labels.
- `lroff` (with its own baseline refitted on the null labels): CTXB+MOT and CTXB+SELFZ,
  **1,000 permutations** (s = 0…999).
- `xgboff`: CTXB+MOT, **400 permutations** (s = 0…399; chosen so that its cost stays under the
  compute cap and its smallest Holm-adjusted p, 14/401 = 0.035, is below 0.05).
- Statistics per permutation: gain in context-stratified AUC and gain in within-session AUC, and
  the overall and context-stratified AUC of every model.
- One-sided permutation p = (1 + k)/(1 + n), k = permutations with a gain ≥ the observed gain.

**Families (Holm within each, 14 tests = 7 protocols × 2 labels):**

- **A (primary):** `lroff`, CTXB+MOT, gain in context-stratified AUC.
- **B (primary):** `lroff`, CTXB+MOT, gain in within-session AUC.
- A′/B′: `lroff`, CTXB+SELFZ. A″/B″: `xgboff`, CTXB+MOT (secondary).

**Decision rules, fixed now.**

- *Telemetry signal reported:* Holm-adjusted p < 0.05 in family A or B **and** the bootstrap
  interval of the gain excludes 0. Reported plainly with size and interval.
- *Evidence against a signal of size X:* in that protocol and label, the positive control below
  recovers a planted signal of size X (interval of its gain excludes 0), and the upper bound of
  the real gain's interval is below the lower bound of the recovered planted gain.
- Otherwise: "no evidence either way at the detectable size", with the detectable size stated.

**Label-shift control.** Same offsets as before (seed 1, uniform in [n/4, 3n/4] records per
session); training labels shifted, test labels untouched.

- Learners: `lroff` (CTXB and CTXB+MOT), `xgboff` (CTXB+MOT), default HGB (MOT and MOT+TIME),
  HGB-reg (MOT+TIME).
- Protocols: within seated, within standing, pooled and the 4 cells; both labels.
- Within a single cell the game and campaign are constant, so a model that stays predictive under
  the shift there must use session-level (not game- or campaign-level) differences. This separates
  the two explanations the referee raised.

**Positive control and minimum detectable effect.** The study's planted channel
(d·y + AR(1) noise, ρ = 0.9, seed 900 + label), at planted within-stratum AUCs
**0.55, 0.60, 0.65, 0.70 and 0.80**.

- Added to: CTXB alone (`lroff` with the channel only; ceiling), CTXB+MOT (`lroff` and `xgboff`).
- Protocols: the 7 null protocols; both labels.
- **Minimum detectable effect** (per protocol, label and learner): the smallest planted AUC whose
  recovered gain in context-stratified AUC has a 95 % interval above 0. The same is reported for
  within-session AUC.

**HGB-reg.** Label-shift control only (above) and its revision-2 positive control. No null.

## Stage 4 — Momentary state (3.3, S3)

- Within-session AUC (Stage 2) for every learner, including default HGB, HGB-reg and LR, from
  the saved predictions.
- **Pair decomposition:** for every context-stratified AUC in Table VI, the share of its pairs that
  come from the same session.
- **Event-locked analysis.** Sessions whose label switches from 0 to 1 at least once (first onset
  τ ≥ 30 s and at least 30 s of record after τ). For `lroff` CTXB+MOT, the telemetry part of the
  score (logit p − offset) is averaged over [τ − 30 s, τ) and [τ, τ + 30 s); the statistic is
  the mean over onset sessions of (after − before), minus the same difference at the same clock
  times in the sessions of the same cell whose label does not change in that window. Group
  bootstrap interval; the same statistic under the transplant null (first 200 permutations of the
  `lroff` null, which store the telemetry part) gives its reference distribution.
- **Null location.** The mean context-stratified AUC of every learner under its transplant null
  is reported next to 0.5, so values below 0.5 can be read against the right reference.

## Stage 5 — Grouping robustness (3.6, Q4)

- **Session grouping:** every session its own group (most lenient).
- **Merged grouping (stricter):** connected components of inferred groups that differ in at most
  one of the seven demographic answers.
- Re-run: `lroff` and `xgboff` (all feature sets) in the 10 protocols and 4 cells; HGB-reg
  (MOT+TIME, CTX+TIME) within seated and pooled. Both labels. Bootstrap clusters follow the
  grouping.
- Report the number of groups (all, curated release, per cell) under each grouping.
- Q4 (asking the archive's authors for participant identifiers) is done by the authors of the
  paper; its outcome is reported as it is.

## Stage 6 — Second dataset and simulation (3.5, S2)

**Second dataset.** Yalcin et al. (2024), Frontiers in Virtual Reality 5:1364207, public at
github.com/m1237/automatic-cybersickness-detection (MIT licence), commit as cloned
(recorded in the result file). 20 participant folders, one VR roller-coaster ride each; a button
press while feeling sick is logged with a timestamp (`cs_timestamps.csv`).

Pre-processing, fixed now:

- **Duplicate:** folders 13 and 16 contain byte-identical recordings and button files (different
  SSQ answers). Folder 16 is excluded; 19 recordings remain. Reported as a finding.
- **Clock:** the eye-tracking file and the button file share one clock; the wearables (Empatica,
  Polar) run on clocks offset by about one hour and cover only part of the ride for some
  participants. **Only the eye-tracking stream is used**; the wearables are not used.
- **Records:** 1-s bins from the first to the last eye-tracking sample. Label = 1 if at least one
  button timestamp falls in the bin; button timestamps outside the eye-tracking span are dropped
  (counted and reported). Participants without any in-span press stay in as all-negative.
- **Features (past-only, per 1-s bin):** mean pupil diameter (valid samples, mean of both eyes),
  within-bin SD of pupil diameter, mean angular speed of the combined gaze direction, within-bin SD
  of the gaze direction (sum of the three component SDs), blink fraction, validity fraction; each
  as the current bin, its trailing 10-s mean and its trailing 10-s SD (18 features), plus elapsed
  time.
- Flag: recordings longer than twice the median ride; analyses are repeated without them.

Tests (participant = group; default HGB as in the study, and LR):

- **D1 record-wise vs participant-disjoint:** record-wise stratified 5-fold against grouped
  5-fold × 5 repeats; plus participant identification (19-class accuracy) from the features under
  the record-wise split.
- **D2 prevalence under grouped CV:** a training-prevalence score and default HGB evaluated
  leave-one-participant-out with scores pooled across folds versus within-fold pairing.
- **D3 between-session learning:** label-shift control (same offset rule) under participant-
  disjoint folds; overall and within-session AUC with true and shifted training labels.
- Exposure time alone as the context baseline (one ride, one setting).

**Simulation (known ground truth).** 60 sessions of 300 records (1 Hz) from 30 participants
(2 sessions each); session susceptibility u ~ N(0, 1); each session's label switches from 0 to 1
once, at a time that is earlier for larger u, and never if u < −0.5. Ten telemetry features =
participant fingerprint (fixed per participant) + AR(1) noise (ρ = 0.9). Scenarios:
**N** no link between telemetry and labels; **S** one feature's session level is correlated with
u (between-session information only); **M** one feature shifts by δ = 0.5 SD after onset
(momentary information). 20 replications per scenario, seeds 2000 + rep. For each: D1, D2, D3,
and the context-stratified (time bins) and within-session AUC of `lroff` with a time-only offset.
Expected by construction: S raises the stratified but not the within-session gain; M raises both.

## Stage 7 — Follow-ups (Q3, Q5)

- **VRSQ zeros:** number of VRSQ files whose pre- and post-exposure items are all 0, per cell;
  Table IX recomputed without them.
- **VR-LENS ablation** (curated release, inferred-participant-disjoint protocol as before):
  (i) without the five demographic answers; (ii) also without player position;
  (iii) also without StaticFrame. Overall and context-stratified AUC with intervals.

## Reporting

Every number in the manuscript is generated from result files by `fa_report.py` and checked by
`fa_verify.py`. Results are reported whether or not they support the revision-2 conclusions.
