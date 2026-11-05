# Addendum 2: VR-LENS with group-aware internal stacking folds (post hoc)

Recorded on 11 October 2026, before the run. It concerns one robustness check only.

## Why this check is being added

A pre-submission self-review of the manuscript found a gap. In the group-disjoint evaluation of the re-implemented VR-LENS detector (`src/fa_vrlens.py`, Section IV-I of the paper), the outer folds keep inferred participant groups apart. The internal stacking folds, however, are record-wise (`StratifiedKFold(5, shuffle=True)`). The meta-learner is therefore trained on base-learner predictions for records whose sessions also appear in the base learners' training folds. That could make the stacked detector look worse on new groups than a group-aware implementation would. The paper names this limitation but had not tested it.

## What was known when this note was written

Every result of the paper was known, including the following VR-LENS results:

- group-disjoint 4-class accuracy of 53.1 %;
- severity ≥ 2 AUC of 0.77 [0.65, 0.88];
- context-stratified AUC of 0.61;
- the three ablations.

This check is therefore post hoc. It was added in response to the review, not planned in advance.

## The check

The run is identical to the existing group-disjoint VR-LENS run, with one change:

- **Same as before:** curated release, the study's grouped 5-fold × 5 repeats (`make_folds`), all ten features, the same base learners, meta-learner and hyperparameters.
- **The change:** the stacking classifier's internal folds are `StratifiedGroupKFold(5, shuffle=True, random_state=0)` over the inferred groups of the training records. No internal fold then holds records of a group that is also in its training part.

Outputs: `results/rev3_vrlens_groupcv.json` and `results/preds/sensVRLENSG_sev{1,2}__within_curated__VRLENS__stack.npz`. The predictions are scored by `src/fa_stats.py` exactly as the other VR-LENS runs are.

## How it will be reported

The 4-class accuracy, the AUC and the context-stratified AUC (both labels, group-bootstrap intervals) will be reported next to the current result, whatever they show. The existing result stays the main VR-LENS result, because it follows the published method. The new run is described as a post hoc robustness check. There is no decision rule. If the accuracy changes materially, the paper's statements about the detector will be revised to match.

## Result (recorded after the run, 11 October 2026)

The check ran once, as specified, and took 388 s. Group-disjoint four-class accuracy was 52.2 %, against 53.1 % with record-wise internal folds. For severity ≥ 2:

- AUC 0.71 [0.59, 0.80], against 0.77 with record-wise internal folds;
- context-stratified AUC 0.52 [0.45, 0.65].

The record-wise internal folds therefore do not explain the drop on held-out groups. The paper reports the run in Section V-C and Supplementary Note S2, and keeps the run that follows the published method as its main VR-LENS result. No other result changed: `results/stats.json` gained the 18 entries of the two new configurations, and every existing value is identical.
