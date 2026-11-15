# Provenance: when the analysis plans and result files were committed

The analysis was developed in the authors' **private** development repository. This reproducibility repository is a
fresh snapshot of the files the submitted manuscript relies on, taken from that repository at commit `d1eb045`
(4 November 2026). The paper's own figures, tables and numbers were produced at commit `a5579de`; the only later
changes are two lists recording where values are printed (`results/paper/paper_keys_rq3res.json`,
`results/paper/paper_keys_sample.json`). No analysis code, result file or value changed between the two commits.

**Version 10 (13 November 2026), post hoc.** After a further internal pre-submission review of the manuscript, two runs were added (`src/fa_v10.py`, `src/fa_report_v10.py`; results in `results/v10/` and `results/paper/values_v10.json`). They were fixed after that review and are not part of any analysis plan. No earlier result file changed.

**Check of 14 November 2026, post hoc.** `src/fa_check_rowwise_ctx.py` explains a value already printed: the context-stratified AUC below 0.5 of the record-wise CTX+TIME model (manuscript Section 5.4; Online Resource 1, Table S40). It reads stored predictions only, fits no model and writes no file (`README.md`, "Check of 14 November 2026").

**Corrections after the first release (5 November 2026).** The first release of this repository is commit `c9f1b0f`.
Two defects found in its audit were then corrected in the development repository (commit `2b0aed9`, 5 November 2026)
and copied here; `src/` now matches that commit. `src/fa_stats.py` skips the addendum-3 learner `lroffs`, so that
`scripts/check_results.sh` runs again, and `src/fa_report_rev3.py` stores the share of between-session pairs per label,
which the caption of Table 7 prints (`README.md`, "Corrections"). No result changed: `results/stats.json` is
reproduced byte for byte, and the value files only gain the new keys.

The manuscript distinguishes planned analyses from analyses specified after results were known (post hoc). That
distinction rests on **when each analysis plan was committed relative to its results**. The commit history itself
is not public, so this file lists the relevant commits: their short hash, their date (UTC) and their subject.

**Limits of this record:**
- The dates come from the authors' own repository. They are not an independent registration (such as OSF or AsPredicted) and cannot be verified independently of the authors.
- The full commit history can be made available to the editor on request.
- Several result files record the development-repository commit at which they were produced (for example, the addendum-3 files). Those hashes refer to the development repository, not to this one.
- A commit date shows when a file was committed, not when it was computed. Every plan below was committed before the first result file of the analyses it specifies; the revision-2 plan is the exception (see below).

## Analysis plans and memos: every commit

A plan was sometimes edited after its results were known. Those edits are listed below, and the plans themselves state what was changed and why. The manuscript reports the deviations (for example, the evidence-against rule of the first addendum was dropped as uninformative).

| Document (in `docs/`) | Commit | Date (UTC) | Commit subject |
|---|---|---|---|
| `ANALYSIS_PLAN_REVISION2.md` | `3f1f25a` | 2026-09-28 05:36 | Initial commit: code, results and manuscript of the UFFCSData full-archive audit |
| `ANALYSIS_PLAN_REVISION3.md` | `a3ad679` | 2026-10-08 20:57 | Revision 3: freeze the analysis plan before any new result |
| `ANALYSIS_PLAN_REVISION3_ADDENDUM.md` | `ff68a1c` | 2026-10-09 08:49 | Plan addendum: corrections and secondary analyses after the pre-submission review |
| `ANALYSIS_PLAN_REVISION3_ADDENDUM.md` | `91a1f55` | 2026-10-09 10:16 | Regenerated statistics, tables, figures, paper, supplement and letter from the addendum results; evidence-against rule dropped as uninformative (deviation recorded) |
| `ANALYSIS_PLAN_REVISION3_ADDENDUM.md` | `db510dc` | 2026-10-09 13:53 | Re-check fixes: claims match the evidence, deviations disclosed, generated phrasing repaired |
| `ANALYSIS_PLAN_REVISION3_ADDENDUM.md` | `bf21239` | 2026-10-09 18:43 | Complete the reporting from existing results; fix the issues of the independent final review |
| `ANALYSIS_PLAN_REVISION3_ADDENDUM.md` | `7b0b370` | 2026-10-09 20:16 | Results audit, second pass: disclose Holm floors and nominal results |
| `ANALYSIS_PLAN_REVISION3_ADDENDUM.md` | `a3e6d01` | 2026-10-10 10:40 | Final pre-submission audit: reproducibility fixes, chronology wording, editorial fixes |
| `ANALYSIS_PLAN_REVISION3_ADDENDUM2.md` | `3d5e2a5` | 2026-10-11 07:04 | Addendum 2 (post hoc): VR-LENS with group-aware internal stacking folds, recorded before the run |
| `ANALYSIS_PLAN_REVISION3_ADDENDUM2.md` | `eddb194` | 2026-10-11 07:25 | Pre-submission revision: author prose in IEEE source, review fixes, post hoc VR-LENS check |
| `addendum_3.md` | `21dea29` | 2026-10-15 21:07 | Addendum 3: post hoc analyses P1-P7, specified before any run |
| `addendum_3.md` | `b045311` | 2026-10-15 21:16 | Addendum 3, amendment 1: P7 shift kept on the observed values, before any run |
| `addendum_3.md` | `01b1b2f` | 2026-10-16 02:41 | Addendum 3: results of the post hoc analyses P1-P7, values, memo; no existing result file changed |
| `addendum_3.md` | `197ff2a` | 2026-10-16 08:52 | Audit of addendum 3: named P2 and P4 outputs extracted from add3_stats.json, verifier checks them and that no overall-gain decision is printed; audit note with re-runs, deviations, corrected timing of the n/e change and compute time; no existing result file changed |
| `results_addendum3.md` | `01b1b2f` | 2026-10-16 02:41 | Addendum 3: results of the post hoc analyses P1-P7, values, memo; no existing result file changed |
| `results_addendum3.md` | `fae6bb5` | 2026-10-16 07:21 | Addendum 3 in the manuscript: value keys and claim checks for every new statement, 1000-permutation versions of Tables S20, S22, S23 and S25, printed-key list for the new text; verifier reads both value files |
| `results_addendum3.md` | `197ff2a` | 2026-10-16 08:52 | Audit of addendum 3: named P2 and P4 outputs extracted from add3_stats.json, verifier checks them and that no overall-gain decision is printed; audit note with re-runs, deviations, corrected timing of the n/e change and compute time; no existing result file changed |
| `results_addendum3.md` | `98136a3` | 2026-10-16 16:25 | Verification of addendum 3: memo numbers checked against the result files (P6 five inputs, P7 shift range corrected), code-availability wording, three-decimal macro F1 key, Table S36 caption prints the number of demographic answers; verifier passes |
| `audit_addendum3.md` | `197ff2a` | 2026-10-16 08:52 | Audit of addendum 3: named P2 and P4 outputs extracted from add3_stats.json, verifier checks them and that no overall-gain decision is printed; audit note with re-runs, deviations, corrected timing of the n/e change and compute time; no existing result file changed |
| `audit_addendum3.md` | `98136a3` | 2026-10-16 16:25 | Verification of addendum 3: memo numbers checked against the result files (P6 five inputs, P7 shift range corrected), code-availability wording, three-decimal macro F1 key, Table S36 caption prints the number of demographic answers; verifier passes |

**Revision 2 (the paper's "earlier analysis"):** `ANALYSIS_PLAN_REVISION2.md` arrived in the repository's initial commit (28 September 2026), together with the code and the results of that analysis. The history therefore cannot show that this plan preceded its results. The manuscript treats the comparison family of the earlier analysis as exploratory (Section 4.5; Online Resource 1, Notes S2 and S4).

## Result files: date of first commit

The table groups the result files of `results/` and `results_logo/` by the commit that first added them.

| First committed (UTC) | Commit | Files |
|---|---|---|
| 2026-09-28 05:36 | `3f1f25a` | `results/cell.json`, `results/claims.json`, `results/context.json`, `results/control.json`, `results/es_check.json`, `results/hgbreg.json`, `results/leakage_checks.json`, `results/logo_ctx.json` and 19 more |
| 2026-10-08 22:03 | `0124d56` | `results/rev3_decode.json`, `results/rev3_yalcin.json` |
| 2026-10-08 22:08 | `5c337fc` | `results/rev3_main.json` |
| 2026-10-08 22:11 | `730dd61` | `results/rev3_control.json` |
| 2026-10-08 22:17 | `05e3d09` | `results/rev3_posctl.json`, `results/rev3_vrsq0.json`, `results/rev3_within.json` |
| 2026-10-09 01:11 | `9e0e1ed` | `results/rev3_null_lroff.json`, `results/rev3_sim.json` |
| 2026-10-09 02:34 | `c370e18` | `results/rev3_null_xgboff.json` |
| 2026-10-09 05:17 | `38c81ce` | `results/rev3_hgb_control.json` |
| 2026-10-09 05:53 | `05f42ec` | `results/rev3_sens_group.json` |
| 2026-10-09 07:53 | `e50b34b` | `results/claims_rev3.json`, `results/reproducibility.json`, `results/rev3_stats.json`, `results/rev3_vrlens_ablation.json` |
| 2026-10-09 09:11 | `51eff87` | `results/rev3_decode_nr.json`, `results/rev3_main_nr.json`, `results/rev3_pairs.json`, `results/rev3_pen.json`, `results/rev3_rotation.json` |
| 2026-10-09 09:20 | `66380e3` | `results/rev3_null_lroff_nm.json` |
| 2026-10-09 09:21 | `743ee03` | `results/rev3_null_lroff_nr.json` |
| 2026-10-09 09:39 | `c9b45ae` | `results/rev3_null_lroff_block.json`, `results/rev3_null_lroffp.json` |
| 2026-10-09 10:11 | `884aca9` | `results/rev3_posctl_seeds.json` |
| 2026-10-11 07:25 | `eddb194` | `results/rev3_vrlens_groupcv.json` |
| 2026-10-12 11:42 | `72dbf94` | `results/paper/paper_keys.json`, `results/paper/values.json` |
| 2026-10-16 02:41 | `01b1b2f` | `results/add3_gate/diffuse.json`, `results/add3_gate/main_s.json`, `results/add3_gate/null_block.json`, `results/add3_gate/null_nm.json`, `results/add3_gate/null_nr.json`, `results/add3_gate/null_p.json`, `results/add3_gate/null_s.json`, `results/add3_gate/null_sess.json` and 23 more |
| 2026-10-16 07:21 | `fae6bb5` | `results/paper/paper_keys_add3.json` |
| 2026-10-16 08:52 | `197ff2a` | `results/add3_family6.json`, `results/add3_sess.json` |
| 2026-10-29 19:18 | `bbb089c` | `results/claims_rq3res.json`, `results/paper/paper_keys_rq3res.json`, `results/paper/values_rq3res.json`, `results/rq3_resolution.json` |
| 2026-10-31 09:07 | `e31915f` | `results/claims_sample.json`, `results/paper/paper_keys_sample.json`, `results/paper/values_sample.json` |

## Order of plans and first results

| Plan | Plan committed | First result files of its analyses committed |
|---|---|---|
| Revision 3 plan (`ANALYSIS_PLAN_REVISION3.md`) | 2026-10-08 20:57 | 2026-10-08 22:03 (`rev3_*` files) |
| First addendum (`ANALYSIS_PLAN_REVISION3_ADDENDUM.md`) | 2026-10-09 08:49 | 2026-10-09 09:11 |
| Second addendum, post hoc (`ANALYSIS_PLAN_REVISION3_ADDENDUM2.md`) | 2026-10-11 07:04 | 2026-10-11 07:25 (`rev3_vrlens_groupcv.json`) |
| Third addendum, post hoc (`addendum_3.md`), amendment 1 at 21:16 | 2026-10-15 21:07 | 2026-10-16 02:41 (`add3_*` files) |
| RQ3 resolution summary (post hoc, descriptive; `src/fa_report_rq3res.py`) | written after all results | 2026-10-29 |
| Sample description (descriptive; `src/fa_report_sample.py`) | written after all results | 2026-10-31 |

`docs/audit_addendum3.md` gives the minute-by-minute timeline of the third addendum. It cites the run logs `results/logs/add3_queue.log` and `add3_queue_resume.log`, which are kept in the development repository and not included here, because they record file paths of the development machine; they are available on request. In that timeline, the plan, its amendment and the analysis code were committed, the SHA-256 list of the existing result files was written, and only then did the first stage start.
