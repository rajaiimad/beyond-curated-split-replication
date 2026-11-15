# Manuscript traceability

This file maps every table, figure and supplementary note of the submitted manuscript to the code and files behind it.

**Numbering.** This file follows the numbering of the submitted journal version of the manuscript: Tables 1–9, Figs. 1–4 and Sections 1–8 of the main text; Notes S1–S8, Tables S2–S40 and Fig. S1 of Online Resource 1 (the supplementary PDF); and Table S1 as Online Resource 2 (a CSV export of `results/paper/supp_table.tex`). The two check records dated 4 November 2026 below, and the dated documents in `docs/` (analysis plans, addenda and audits), keep the numbering of the manuscript version they were written for: Roman table numbers (Table VII), lettered sections (Section III-B) and numbered references.

**How the outputs are made.** Every number printed in the manuscript and its supplement is generated from result files:
- **Printed values:** the report scripts write each value under a key in `results/paper/values*.json`.
- **Tables:** they write each table as a LaTeX file in `results/paper/*.tex`; the manuscript's tables were typeset from these files.
- **Figures:** they write each figure as a PDF in `figures/`.
- **Checking:** `verification/verify_results.sh` regenerates all of them, and `src/fa_verify.py` recomputes the printed values independently from the saved predictions and the raw data.

**How this map was checked (4 November 2026).**
- **Tables:** every number in every table of the submitted manuscript (Tables I–IX) and supplement (Tables S1–S39) was compared with the numbers in the generated table files.
  - Each generated table below contains **all** the multi-digit result numbers of its manuscript table. The only exception is a scale label: the "0–100" in Table I's footnote.
  - The exceptions are the tables composed by hand from printed values (Table IX, S28, S29 and S37), which are marked.
  - No number in any table lacks a generated source.
- **Figures and value files:** each figure was compared visually with its generated PDF, and every number of the text is held in a value file.

**Status column:**
- **Planned:** fixed in the analysis plan before any real-label result of the new learners.
- **Addendum 1:** fixed after an internal review of the main results, before the addendum's own results.
- **Post hoc:** specified with all results known.
- **Descriptive:** counts and estimates without a test.

`docs/PROVENANCE.md` lists when each plan was committed.

**Stages:**
- `fa_run.py` is the earlier analysis;
- `fa_rev3.py` the present analysis (with `fa_yalcin.py`, `fa_sim.py` and `fa_vrlens.py`);
- `fa_add3.py` the third addendum.

The stage that writes a result file has the same name, for example stage `posctl_seeds` → `results/rev3_posctl_seeds.json`.

## Version 10

- **Section 3.2 (data collection, curated release, recording day and field of view, other sessions):** archive facts in `values_v10.json` (`v10a.*`, from `src/fa_report_v10.py`); numbers quoted from the creators' papers and the VR-LENS paper in `values_v10.json` (`v10l.*`), each with its source in `values_v10_sources.json`.
- **Section 5.6 (sessions as participant-level units):** `values_v10.json` (`v10r.*`, from `results/v10/v10_runs.json`), together with the existing session-level results of Tables S8, S11, S31 and S32 of Online Resource 1.
- **Every key printed by the v10 text:** `results/paper/paper_keys_v10.json`; `src/fa_verify.py` step [1] checks that each exists and step [2g] recounts or recomputes the v10 values.

## Main manuscript

| Item | Content | Generated file | Written by (function) | Principal inputs | Status |
|---|---|---|---|---|---|
| Table 1 | The raw archive by game–campaign cell | `results/paper/tab_archive.tex` | `fa_report.py` (`audit`) | raw archive; `data/sessions.csv`; `data/features_w30.csv.gz` | Descriptive |
| Table 2 | Evaluation protocols | `tab_protocols.tex` | `fa_report.py` (`table_protocols`) | `data/features_w30.csv.gz`; protocols in `fa_run.domains` | — |
| Table 3 | Session-level condition fields | `tab_fields.tex` | `fa_report.py` (`table_fields`, `field_table`) | raw `FILE.xml` files | Descriptive |
| Table 4 | What the telemetry encodes | `r3_tab_decode.tex` | `fa_report_rev3.py` (`table_decode`) | `results/rev3_decode.json`, `rev3_decode_nr.json` (stages `decode`, `decode_nr`) | Planned; the no-rotation columns are from addendum 1 |
| Table 5 | Rating–VRSQ correlations | `tab_vrsq.tex` | `fa_report.py` (`table_vrsq`) | `results/vrsq.json`, `results/session_level.csv`, `rev3_vrsq0.json` | Descriptive; the tests in Table S34 are post hoc |
| Table 6 | Curated release under different protocols; VR-LENS | `tab_curated.tex` | `fa_report.py` (`table_curated`) | `results/stats.json`, `hgbreg.json`, `main.json`, `rowwise_reference.json`, `vrlens.json` | Earlier analysis (exploratory); VR-LENS post hoc there; session-disjoint rows (version 10, post hoc): `results/v10/v10_runs.json` → `values_v10.json` (`v10r.t6s.*`, `v10r.vrl.sess.*`), entered in the table from the value file |
| Table 7 | Telemetry gain over the context baseline (severity ≥ 2) | `r3_tab_gain.tex` | `fa_report_rev3.py` (`table_gain`) | `rev3_main.json`, `rev3_stats.json`, `rev3_null_lroff.json`; caption range of between-session pairs: `rev3_pairs.json` → `r3pair.s2.btw.min/max` in `values.json` | Planned |
| Table 8 | Power of the decision rule | `r3_tab_mde.tex` | `fa_report_rev3.py` (`table_power`) | `rev3_posctl_seeds.json`, `rev3_stats.json` | Addendum 1 |
| Table 9 | Principal claims, evidence and status | none: composed by the authors | — | The values it prints are in `values.json` (`cur.raceshare`, `vrl.rw.acc`, `vrl.grp.acc`, `vrl.majority`, `r3pwn.lo.80.*`) and `values_rq3res.json` (`r3r.c.b14.min/max`) | — |
| Fig. 1 | Study pipeline (schematic) | `figures/fig_pipeline.pdf` (TikZ source `fig_pipeline.tex`) | drawn by hand | `aud.nsess`, `aud.nrec`, `aud.nvrsq`, `aud.ngroups` in `values.json` | — |
| Fig. 2 | The four game–campaign cells | `figures/fig_archive.pdf` | `fa_report.py` (`fig_archive`) | `data/features_w30.csv.gz` | Descriptive |
| Fig. 3 | Highest in-game rating vs post-exposure VRSQ | `figures/fig_vrsq.pdf` | `fa_report.py` (`fig_vrsq`) | `results/session_level.csv` | Descriptive |
| Fig. 4 | Telemetry gain against its null (Supplementary Fig. S2 up to version 9) | `figures/fig_gain_null.pdf` | `fa_report_rev3.py` (`fig_gain_null`) | `rev3_main.json`, `rev3_null_lroff.json`, `rev3_stats.json` | Specified before being run |

**Text of the main manuscript.** Every number in the Abstract and Sections 1–8 is held under a key in:
- `results/paper/values.json` (written by `fa_report.py` and `fa_report_rev3.py`);
- `values_add3.json` (`fa_report_add3.py`);
- `values_rq3res.json` (`fa_report_rq3res.py`);
- `values_sample.json` (`fa_report_sample.py`).

`docs/DATA_DICTIONARY.md` (section 8) explains the key prefixes. The section-by-section sources are as follows:
- **Section 3.2, Participants** (58, 61, 47 of 74 sessions): `values_sample.json` (`smp.*`), from `data/sessions.csv`. `fa_verify.py` step [2f] recounts them from the raw files.
- **Section 5.1, audit:** `aud.*`, `cur.*` (`fa_report.py`).
- **Section 5.2, decoding:** `r3d.*` (`rev3_decode*.json`).
- **Section 5.3, questionnaire:** `vrsq.*`, `r3v.*`, `a3.vrsq.*` (`vrsq.json`, `rev3_vrsq0.json`, `add3_vrsq.json`).
- **Section 5.4, evaluation design:**
  - earlier-analysis models: `t.*`, `s1.*`, `s2.*`;
  - VR-LENS: `vrl*.*`;
  - second archive: `y.*` (`rev3_yalcin.json`);
  - simulation: `sim.*` (`rev3_sim.json`).
- **Section 5.5, telemetry beyond context:**
  - AUCs and gains: `r3.*`, `r3g.*`;
  - nulls: `r3n.*`;
  - power: `r3pw.*`, `r3pwn.*`;
  - resolution: `r3r.*`;
  - secondary and post hoc: `a3.*`.
- **Sections 6–8:** the same keys as the results they summarise.

## Online Resources 1 and 2 (supplementary material)

| Item | Content | Generated file | Written by (function) | Principal inputs | Status |
|---|---|---|---|---|---|
| Table S1 | Every model of the earlier learners (Online Resource 2; not printed in Online Resource 1 since version 10) | `supp_table.tex` | `fa_report.py` (`supp_table`) | `results/stats.json` | Earlier analysis |
| Table S2 | Secondary label, HGB-reg | `supp_main_s1.tex` | `fa_report.py` (`table_main`) | `stats.json`, `hgbreg.json` | Earlier analysis |
| Table S3 | Primary label, logistic regression | `supp_main_lr.tex` | `fa_report.py` (`table_main`) | `stats.json`, `main.json` | Earlier analysis |
| Table S4 | Comparison family, first (leave-one-group-out) run | `supp_firstrun.tex` | `fa_report.py` (`supp_firstrun`) | `results_logo/stats.json`, `stats.json` | Earlier analysis (exploratory) |
| Table S5 | Strata behind the context-stratified AUCs | `supp_strata.tex` | `fa_report.py` (`supp_strata`) | `results/preds/` | Descriptive |
| Table S6 | Primary label, default HGB | `supp_main_hgb.tex` | `fa_report.py` (`table_main`) | `stats.json`, `main.json` | Earlier analysis |
| Table S7 | Training and testing inside one cell | `tab_cells.tex` | `fa_report.py` (`table_cells`) | `cell.json`, `stats.json` | Earlier analysis |
| Table S8 | Sensitivity analyses | `tab_sens.tex` | `fa_report.py` (`table_sens`) | `sensitivity.json`, `stats.json` | Earlier analysis |
| Table S9 | Remedies under shift | `tab_mitig.tex` | `fa_report.py` (`table_mitig`) | `mitigation.json` | Earlier analysis |
| Table S10 | Earlier analysis, main table (HGB-reg) | `tab_main.tex` (same numbers in `supp_main_hgbr.tex`) | `fa_report.py` (`table_main`) | `stats.json`, `hgbreg.json` | Earlier analysis |
| Table S11 | Sensitivity to the grouping | `r3_supp_groups.tex` | `fa_report_rev3.py` (`table_groups`) | `rev3_sens_group.json`, `rev3_stats.json` | Planned |
| Table S12 | The second archive in full | `r3_supp_second.tex` | `fa_report_rev3.py` (`table_second_full`) | `rev3_yalcin.json` | Planned |
| Table S13 | Exploratory comparison family | `tab_tests.tex` (same numbers in `supp_tests.tex`) | `fa_report.py` (`table_tests`) | `stats.json` (tests) | Earlier analysis (exploratory) |
| Table S14 | As Table 7, severity ≥ 1 | `r3_supp_gain_s1.tex` | `fa_report_rev3.py` (`table_gain`) | `rev3_main.json`, `rev3_stats.json`, `rev3_null_lroff.json` | Planned |
| Table S15 | As Table 7, self-normalized features | `r3_supp_gain_z.tex` | `fa_report_rev3.py` (`table_gain`) | same | Planned (secondary) |
| Table S16 | As Table 7, boosting+offset | `r3_supp_gain_xgb.tex` | `fa_report_rev3.py` (`table_gain`) | `rev3_main.json`, `rev3_stats.json`, `rev3_null_xgboff.json` | Planned (secondary) |
| Table S17 | Positive control of the earlier analysis | `supp_posctl.tex` (from `tab_posctl.tex`; its numbers are filled in through value macros) | `fa_report.py` (`table_posctl`) | `posctl.json`, `posctl_reg.json` | Post hoc in the earlier analysis |
| Table S18 | Power of the decision rule in full | `r3_supp_power.tex` | `fa_report_rev3.py` (`table_power_full`) | `rev3_posctl_seeds.json`, `rev3_posctl.json` | Addendum 1 |
| Table S19 | Recovery of one realization of the planted channel | `r3_supp_recovery.tex` | `fa_report_rev3.py` (`table_recovery`) | `rev3_posctl.json` | Planned |
| Table S20 | Secondary analyses with 1,000-permutation nulls | `tab_a3_secondary.tex` | `fa_report_add3.py` (`table_secondary`) | `add3_stats.json` (from the `add3_null_*.json` files) | Addendum 1 analyses; 1,000-permutation nulls post hoc (P3) |
| Table S21 | The simulation in full | `r3_supp_sim.tex` | `fa_report_rev3.py` (`table_sim_full`) | `rev3_sim.json` | Planned |
| Table S22 | Within-session AUC on sessions whose rating also falls | `tab_a3_falling.tex` | `fa_report_add3.py` (`table_falling`) | `add3_stats.json` (`add3_null_lroff_nm.json`) | Addendum 1; null extended post hoc |
| Table S23 | LR+offset with the 14 non-rotation features | `tab_a3_nr.tex` | `fa_report_add3.py` (`table_nr`) | `add3_stats.json` (`add3_null_lroff_nr.json`), `rev3_main_nr.json` | Addendum 1; null extended post hoc |
| Table S24 | Label-shift control in full | `r3_supp_control.tex` | `fa_report_rev3.py` (`table_control_full`) | `rev3_control.json`, `rev3_hgb_control.json` | Planned |
| Table S25 | Gain under the two label-transplant nulls | `tab_a3_nullranges.tex` | `fa_report_add3.py` (`table_nullranges`) | `add3_stats.json` (`add3_null_lroff_block.json`), `rev3_null_lroff.json` | Addendum 1; null extended post hoc |
| Table S26 | Earlier learners within sessions | `r3_supp_oldwithin.tex` | `fa_report_rev3.py` (`table_old_within`) | `rev3_within.json`, `rev3_stats.json` | Planned |
| Table S27 | What the telemetry encodes, by session | `r3_supp_decode_session.tex` | `fa_report_rev3.py` (`table_decode_session`) | `rev3_decode.json` | Planned |
| Table S28 | Public cybersickness datasets | none: literature table, composed by the authors | — | the references cited in it | — |
| Table S29 | Scope of the findings | none: composed by the authors | — | printed values of the tables it summarises | — |
| Table S30 | P1: LR+offset with stronger penalties | `tab_a3_p1.tex` | `fa_report_add3.py` (`table_p1`) | `add3_stats.json` (`add3_main_lroffs.json`, `add3_null_lroffs.json`, `add3_null_lroffp.json`) | Post hoc |
| Table S31 | P2 and P4: family of six tests; session grouping | `tab_a3_family6.tex` | `fa_report_add3.py` (`table_family6`) | `add3_family6.json`, `add3_sess.json` | Post hoc |
| Table S32 | P2 and P4: power | `tab_a3_power6.tex` | `fa_report_add3.py` (`table_power6`) | `add3_posctl6.json`, `add3_posctl_sess.json`, `add3_stats.json` | Post hoc |
| Table S33 | P7: diffuse planted signal | `tab_a3_diffuse.tex` | `fa_report_add3.py` (`table_diffuse`) | `add3_posctl_diffuse.json` | Post hoc |
| Table S34 | P5: VRSQ correlations and tests | `tab_a3_vrsq.tex` | `fa_report_add3.py` (`table_vrsq`) | `add3_vrsq.json` | Post hoc |
| Table S35 | P5: the 32 all-zero questionnaire files | `tab_a3_allzero.tex` | `fa_report_add3.py` (`table_allzero`) | `add3_vrsq.json` | Post hoc |
| Table S36 | P6: VR-LENS without the demographic answers | `tab_a3_vrlens.tex` | `fa_report_add3.py` (`table_vrlens`) | `add3_vrlens_rw_nodemo.json`, `vrlens.json`, `rev3_vrlens_ablation.json` | Post hoc |
| Table S37 | Evaluation-design findings D1–D3 | none: composed by the authors | — | printed values from `rev3_sim.json` (`sim.*`), `rev3_yalcin.json` (`y.*`) and the UFFCSData results | Planned on the second archive and in simulation |
| Table S38 | Resolution of the primary RQ3 tests | `tab_rq3_resolution.tex` | `fa_report_rq3res.py` (`table`) | `rev3_null_lroff.json`, `rev3_posctl_seeds.json`, `rev3_stats.json` → `results/rq3_resolution.json` | Post hoc, descriptive |
| Table S39 | Sample description | `tab_sample.tex` | `fa_report_sample.py` (`table`) | `data/sessions.csv` (recounted from the raw files by `fa_verify.py` [2f]) | Descriptive |
| Table S40 | Provenance of the analyses: date, document, what was known, what was specified or run, status | none: composed by the authors | — | commit dates in `docs/PROVENANCE.md`; the plan documents in `docs/` | — |
| Fig. S1 | Earlier analysis: null and positive control | `figures/fig_null.pdf` | `fa_report.py` (`fig_null`) | `null.json`, `null_lr.json`, `posctl.json`, `stats.json` | Earlier analysis |
| Fig. S2 | Moved to the main text as Fig. 4 in version 10 | — | — | — | — |
| — | AUCs of the earlier learners | `figures/fig_auc.pdf` | `fa_report.py` (`fig_auc`) | `stats.json` | Generated but **not** in the manuscript |

**Notes of Online Resource 1:**

| Note | Title | Main sources |
|---|---|---|
| S1 | Leakage gate and mutation test | `results/leakage_checks.json`, `results/add3_gate/*.json`; code: `fa_data.py` (checks), `fa_run.py` and `fa_add3.py` (gate before every stage) |
| S2 | Analysis history, earlier learners and reproducibility | `docs/ANALYSIS_PLAN_*.md`, `docs/addendum_3.md`, `docs/PROVENANCE.md`; `results/reproducibility.json` (re-run comparison), `vrlens.json`, `vrlens_attempt1.json`, `rev3_vrlens_ablation.json`, `rev3_vrlens_groupcv.json`; `fa_repro.py` |
| S3 | Sensitivity analyses | `sensitivity.json`, `es_check.json`, `logo_ctx.json` |
| S4 | Extended threats to validity | `data/sessions.csv` (inferred groups), `rev3_sens_group.json`, `rev3_rotation.json`; values `smp.*` |
| S5 | Second archive and simulation | `rev3_yalcin.json`, `rev3_sim.json`; code: `fa_yalcin.py`, `fa_sim.py` |
| S6 | Secondary analyses of the telemetry gain in detail | `rev3_control.json`, `rev3_within.json`, `rev3_pairs.json`, `rev3_pen.json`, `rev3_posctl_seeds.json`, `rev3_null_lroff_*.json`, `rq3_resolution.json` |
| S7 | Secondary label, other feature sets and learners | `rev3_main.json`, `rev3_stats.json`, `rev3_null_*.json` |
| S8 | Questionnaire files | `vrsq.json`, `rev3_vrsq0.json`, `add3_vrsq.json` |

## Cross-check of the running text (4 November 2026)

**What was checked:** every multi-digit number in the running text of the submitted manuscript and supplement (1,045 occurrences; references, citation numbers, section and table labels and years excluded) was matched against the value files and generated tables.

**Result:** 27 occurrences (17 distinct numbers) had no generated source. None is a hidden result. The last two rows of the table were resolved on 5 November 2026 (see the end of this section):

| Kind | Numbers | Source |
|---|---|---|
| Design constants written in the code | 300 records per simulated session; at most 300 iterations; leaf-size grid {100, 300}; 10,000 bootstrap replicates; MLP layers of 64 and 32 units; simulated onset at 30 + 240 Φ(−u) s; the "150–180-s" exposure bin | `src/fa_sim.py`, `src/fa_run.py`, `src/fa_stats.py`, `src/fa_report.py` |
| Numbers quoted from other papers | 87 or 88 volunteers; accuracies of 82–94% and 87–93% | the cited references [12], [13], [32] |
| Identifiers | postal code 32610; SHA-256; arXiv:2302.01985 | — |
| Software versions of the first VR-LENS run | scikit-learn 1.6.0, NumPy 2.2.1, pandas 2.2.3 | `results/vrlens_attempt1.json` (`versions`) |
| Table VII caption, "80–97% of its pairs from different sessions" | 97 | **Resolved:** now generated. The range over the severity ≥ 2 protocols (`r3pair.s2.*.btw`: 80.2–97.3) is stored as `r3pair.s2.btw.min/max` = 80–97 and printed in the generated caption, which before printed the range over both labels (80–98%). |
| Supplementary Note S2, VR-LENS re-runs differing "by at most 0.0013" | 0.0013 | **Resolved:** removed from the revised supplement. It came from a manual re-run check made on 10 October 2026 and recorded only in the authors' development notes, so it could not be regenerated from this repository. The revised Note S2 keeps the qualitative description of that check (record-wise AUCs differing at the sixth decimal, the predictions of one of the ten record-wise folds changed, group-disjoint AUCs agreeing to three decimals) without the magnitude; that description, too, rests on the development notes. |

**After the corrections of 5 November 2026,** every result number in the manuscript and supplement has a generated source. The 25 remaining occurrences without one are design constants, numbers quoted from other papers, identifiers and software versions.

## What is not reproduced from code

- **Hand-composed elements:**
  - **Composed by the authors:** Table 9, Tables S28, S29, S37 and S40, and the Fig. 1 schematic. Their numbers come from the value files above; Table S28 summarises the literature, and Table S40 lists dates and documents.
- **The first run (Table S4):** read from its archived output in `results_logo/`. Its predictions are not included, so `fa_verify.py` cannot recompute those AUCs. Re-running it is part of the full pipeline.
- **VR-LENS stage:** not reproducible bit for bit. Re-runs differ in the sixth decimal of the record-wise AUCs, and no printed value changes (Online Resource 1, Note S2).
