# Analysis plan for revision 2 — fixed before any new result was computed

Written 2026-09-27 05:23 UTC, before the new analyses were run. The code in `code/fa_run.py` and `code/fa_stats.py` implements exactly these choices. Nothing below may be changed after results are seen. If a run fails, the failure is reported; it is not replaced by a different design.

## Stage 2a — Label-transplant null, extended

- **Permutations.** n = 500, with seeds 700 + s for s = 0…499.
  - s = 0…39 are the 40 existing permutations, which are reused unchanged.
  - 500 is the largest number the available compute (2 CPU cores) allows within the revision.
- **Learners.** Default HGB and logistic regression, using the same permuted labels.
- **Protocols.** Within seated, within standing and pooled, for both labels.
- **Feature sets.** MOT+TIME and SELFZ+TIME.
  - The CTX model was fitted in s = 0…39 only.
  - Its context-stratified AUC is 0.5 by construction; this was verified in all 40 of those permutations.
- **p-value.** One-sided permutation p = (1 + k)/(1 + n), where k counts permutations with a context-stratified AUC ≥ the observed value.
- **Multiplicity.** Holm correction over the 12 comparisons, separately for each learner.
- **Smallest attainable Holm-adjusted p.** 12/501 = 0.024.

## Stage 2b — Label-shift control, within standing

- **Setup.** Same offsets as before (seed 1, uniform in [n/4, 3n/4] records per session).
- **Runs.** Default HGB; feature sets MOT and MOT+TIME; both labels.
- **Protocol added.** within_standing.

## Stage 2d — Strata table

This uses the existing predictions; nothing is refitted. For every context-stratified AUC in Table VII (main table) and its supplementary counterparts, report, averaged over the 5 repeats:

- the number of strata (fold × campaign × game × 30-s bin) that contain both positive and negative records;
- the number of positive–negative pairs.

## Stage 3 — Regularised, group-validated HGB ("HGB-reg")

**Model settings.**
- Grid, ordered from most to least regularised. Ties in the selection score go to the earlier entry.
  - max_depth ∈ {2, 3};
  - min_samples_leaf ∈ {300, 100};
  - l2_regularization ∈ {10, 1}.
  - That gives 8 configurations. Order: depth, then leaf, then l2.
- Fixed settings: learning_rate 0.05, at most 300 iterations, class_weight "balanced", random_state 0, scikit-learn's built-in early stopping off.

**Outer loop.** The study's folds, unchanged: grouped 5-fold × 5 repeats over inferred participant groups.

**Inner loop.** GroupKFold(3) over the training groups of each outer fold, using training records only. For cross-setting protocols this means source-domain groups only.

**Early stopping.**
- In each inner fold, every configuration is fitted for up to 300 iterations.
- The number of iterations n\* minimises the class-balanced log-loss on the held-out inner groups.

**Selection.**
- For each configuration, compute the mean over inner folds of the inner context-stratified AUC at n\*. The inner strata are campaign × game × 30-s bin, weighted by positive–negative pairs.
- If no inner fold has an estimable context-stratified AUC, the overall inner AUC is used instead.
- The configuration with the highest mean is selected.

**Refit.**
- The selected configuration is refitted on all training records.
- Its number of iterations is the median of the n\* values in its inner folds.

**Operating threshold.** The 90th percentile of the negative records' inner out-of-fold scores of the selected configuration. This is the same 10 % false-alarm rule used by the other learners.

**Runs.**
- Feature sets TIME, CTX, CTX+TIME, MOT+TIME, MOT+CTX+TIME and SELFZ+TIME.
- The 10 protocols of Table VII: curated release; within seated, standing, flight game and racing game; pooled; and the four cross-setting protocols. Both labels.
- The positive control with the same planted channels (same seeds): SPK+TIME, MOT+SPK+TIME and SELFZ+SPK+TIME; within seated, within standing and pooled; both labels; planted AUC 0.60 and 0.70.

**Comparison family.** The 13 comparisons of Table VIII are computed with HGB-reg.
- **HGB-reg family (main):** Holm over its 13 comparisons.
- **Default HGB and LR:** each Holm-adjusted within its own family, reported as sensitivity and second learner.
- All families are described as exploratory. The family was fixed after the first run had been seen.

**Reporting rule, fixed now.**
- HGB-reg is the main learner of the comparison family. Default HGB is reported as a sensitivity result, and its session-memorisation behaviour is reported as a finding.
- **Evidence against a signal.** This applies where, in a protocol, HGB-reg recovers the planted 0.70 channel among the real features (lower bound of the 95 % interval of its context-stratified AUC > 0.5). If, in that protocol, the upper bound of the real-label context-stratified AUC interval is below the lower bound of the recovered planted-0.70 interval, the text states: evidence against a within-context signal of that size, for these 28 features, in this protocol.
- **Reporting a signal.** If a real-label context-stratified interval of HGB-reg lies entirely above 0.5, the text reports a within-context signal plainly, with its size and interval.

## Stage 4 — Re-implementation of a published detector

This is decided after checking whether the published papers give enough detail. If a faithful re-implementation is not possible, the authors are told before any approximation is built.

## Leakage checks

- The leakage gate runs before every new stage: one group per session; disjoint groups and sessions in every fold; past-only features; one row per record.
- The mutation test must fail as intended.
- The inner folds of HGB-reg are asserted to be group-disjoint.
