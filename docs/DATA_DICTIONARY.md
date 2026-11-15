# Data dictionary

All definitions below come from the archive files and from `src/fa_data.py`, which parses them. Portuguese answer labels are translated; the raw value is given in the first column of each list.

## 1. Raw session file: `RAW_DATABASE/RAW_DATABASE/<session>/FILE.xml`

- **Session folder names:** `DATA_EXPORT_<Game>_<YYYY>_<MM>_<DD>_<hh>_<mm>_<ss>`. `<Game>` is `Flight` or `Race`, and the date and time are when recording started.
- **Records:** one `<Data>` element per logging tick (median interval 1.00 s).
- **Number format:** decimal numbers use a **comma** (for example `95,63988`); `fa_data.py` converts them.

### Demographic answers

There are seven per session, the same on every record, self-reported, with no names or identifiers. Sessions with identical answers form the paper's 38 inferred groups (`group_id`).

| Field | Values (raw → meaning) | Sessions (of 74) |
|---|---|---|
| `UserGenere` | `masculino` → male; `feminino` → female | 58; 16 |
| `UserAge` | `18 A 36` → 18–36 years; `37 A 50` → 37–50; `+50` → over 50 | 61; 6; 7 |
| `UserExperience` | `NENHUMA` → none; `ALGUMA` → some; `MUITA` → much. Read as previous VR experience; the archive does not document the exact question. | 47; 18; 9 |
| `UserFlicker` | `NENHUM` → none; `ALGUM` → some (sensitivity to flicker) | 69; 5 |
| `UserGlassesUse` | `NÃO` → no; `SIM` → yes | 52; 22 |
| `UserVisionProblems` | `NENHUM` (none), `MIOPIA` (myopia), `ASTIGMATISMO` (astigmatism), `HIPERMETROPIA` (hyperopia), and combinations such as `ASTIGMATISMO/MIOPIA` | 44 none |
| `UserEyeDominance` | `DIREITO` → right; `ESQUERDO` → left | 43; 31 |

### Session conditions

These fields are the same on every record of a session (see the manuscript's Table 3).

| Field | Values | Meaning in this archive |
|---|---|---|
| `UserPosture` | `SENTADO` (seated), `DE PÉ` (standing) | The recording campaign (37 sessions each) |
| `UserSymptoms` | `NENHUM` in every session | Constant; carries no information |
| `StaticFrame` | `True` / `False` | Alias of the game (racing = True) |
| `CameraAutoMovement` | `True` / `False` | Alias of the game (flight = True) |
| `DegreeOfControl` | `total` / `TOTAL` | Alias of the game (letter case only) |
| `CameraFieldOfView` | 111.8 / 96.0 (seated flight / racing); 95.6 (standing); 60 in one session | Separates the games in the seated campaign |
| `HapticFeedback`, `DofSimulation`, `Locomotion` | `False` in every session | Constant |

### Telemetry and labels

| Field | Meaning |
|---|---|
| `TimeStamp` | Seconds since the start of the session |
| `PlayerSpeed`, `PlayerAcceleration` | Player speed and acceleration, as logged by the game |
| `CameraRotationX/Y/Z` | Vector part of a unit quaternion: never above 1 in norm, and **not** angles. They follow the vehicle in the racing game. They are constant within every standing-campaign session. |
| `PlayerPositionX/Y/Z` | Player position. Logged in the seated campaign only, so not used by the main analyses. |
| `RegionOfInterest` | `background`, `middleground` or `foreground` |
| `GameFps` | Rendering frame rate. It identifies a game–campaign cell, so it is excluded from the main feature sets. |
| `DiscomfortLevel` | The current in-game discomfort rating, 0–3: the **label** |

## 2. Questionnaire file: `RAW_DATABASE/RAW_DATABASE/<session>/VRSQ_FILE.xml`

- **Coverage:** present for 70 of the 74 sessions.
- **Structure:** `<Session 0>` is before exposure and `<Session 1>` after; two files repeat the pair identically. There are nine items, each scored 0–3:
  - oculomotor items: `GeneralDiscomfort`, `Fatigue`, `EyeStraing`, `DifficultyFocusing`;
  - disorientation items: `Headache`, `FulnessOfHead`, `BlurredVision`, `DizzyEyesClosed`, `Vertigo`.
- **Scoring** (Kim et al., 2018; `fa_data.vrsq_scores`):
  - oculomotor = sum of the four oculomotor items / 12 × 100;
  - disorientation = sum of the five disorientation items / 15 × 100;
  - total = mean of the two (0–100).
- **All-zero files:** 32 of the 70 files are zero in every item, both before and after.

## 3. Curated release: `DATABASE_SELECTED_DATA/DATABASE/<1..35>/FILE.xml` and `dataset.csv`

- **Contents:** the 35 seated sessions released by the archive's authors, with the same fields as the raw files.
- **Mapping:** `fa_data.curated_sessions` maps each curated session to exactly one raw session by its camera-rotation series (`curated_id` in `sessions.csv`).
- **`dataset.csv`:** the same release as one table; read by `src/fa_vrlens.py`.

## 4. Derived table: `data/sessions.csv` (one row per session, 74 rows)

| Column | Meaning |
|---|---|
| `session_id` | Raw session folder name |
| `game` | `Flight` or `Race` |
| `campaign` | `seated` or `standing` (from `UserPosture`) |
| `date`, `start` | Recording date and start time (from the folder name) |
| `fingerprint` | The seven demographic answers joined by `\|` |
| `group_id` | Inferred participant group: sessions with identical `fingerprint` (38 groups) |
| `n_rows` | Number of records |
| `duration_s` | Last minus first timestamp |
| `max_gap_s` | Largest gap between consecutive records |
| `fov` | Camera field of view |
| `fps_mean` | Mean frame rate |
| `sha16` | First 16 hex digits of the SHA-256 of the session's `FILE.xml` |
| `vrsq_pre_*`, `vrsq_post_*` | VRSQ oculomotor, disorientation and total, before and after exposure (empty if the session has no VRSQ file) |
| `vrsq_post_raw_sum` | Sum of the nine post-exposure items |
| `flag_gap` | `max_gap_s` > 5 |
| `flag_config` | `fov` < 80 or `fps_mean` < 40 (one session) |
| `flag_short` | `duration_s` < 60 |
| `curated_release`, `curated_id` | Whether the session is in the curated release, and its curated folder number |

`results/session_level.csv` adds in-game rating summaries per session:
- `ing_max`, `ing_mean`: the session's highest and mean rating;
- `ing_frac_ge1`, `ing_frac_ge2`: the share of records rated 1 or more, and 2 or more;
- `onset_ge1_s`, `onset_ge2_s`: the first time a rating of 1 or more, or 2 or more, appears;
- `det_mean_risk_sev1`, `det_mean_risk_sev2`: the earlier analysis's pooled MOT+TIME HGB detector's mean out-of-group risk in the session (mean over repeats).

## 5. Derived table: `data/features_w30.csv.gz` (one row per raw record, 18,384 rows)

All features use only the current and earlier records of the same session. Windows are 30 s of recorded time (not rows), so logging gaps do not stretch them.

| Columns | Meaning |
|---|---|
| `t_elapsed` | Exposure time in seconds (TIME) |
| `<x>`, `<x>_mean`, `<x>_std`, `<x>_diff` for `PlayerSpeed`, `PlayerAcceleration`, `CameraRotationX/Y/Z` | Current value, trailing 30-s mean and standard deviation, and first difference (20 features) |
| `rot_step`, `rot_instability` | Norm of the change of the rotation vector between consecutive records, and its trailing 30-s standard deviation (2) |
| `roi_<level>`, `roi_<level>_frac` for `background`, `middleground`, `foreground` | Region-of-interest indicator and its trailing 30-s share (6) |
| `GameFps*` | Frame-rate features (sensitivity analysis only, `MOT+FPS+TIME`) |
| `q_theta*`, `q_angstep*` | Quaternion angle features (sensitivity analysis only, `MOTQ+TIME`) |
| `z_<feature>` | Each of the 28 MOT features as a z-score against the session's own history: expanding mean and SD, 0 while undefined, clipped to ±10 (SELFZ) |
| `session_id`, `campaign`, `game`, `group_id` | Keys and context |
| `UserGenere` … `UserEyeDominance` | The seven demographic answers (used only in sensitivity analyses and by VR-LENS) |
| `severity` | The label: `DiscomfortLevel` 0–3. The paper's binary labels are severity ≥ 2 (primary) and severity ≥ 1 (secondary). |

**Feature sets** (`fa_data.build_features`):
- MOT: the 28 features above (20 + 2 + 6);
- MOTNR: the 14 MOT features not derived from the rotation fields;
- SELFZ: `z_` versions of the 28;
- TIME: `t_elapsed`;
- CTX: game and campaign indicators;
- CTXB: the context baseline of the present analysis, a logistic regression on exposure time, its square, game, campaign and their products. It is fitted in `src/fa_rev3.py`.

## 6. Predictions: `results/preds/` (1,756 files)

| File | Contents |
|---|---|
| `<tag><label>__<protocol>__<features>__<learner>.npz` | `p`: out-of-group score per record, shape (5 repeats, 18,384 records), NaN where the record is not a test record of the protocol; `thr`: the operating threshold (set on training groups) applied to that record; `fold`: its test fold (−1 where it is not tested). Offset learners also have `off`, the context-baseline logit used as offset. |
| `rowwise_...npz` | Record-wise reference split: `p`, `idx` and `fold` for one repeat |
| `*.selection.json` | HGB-reg only: the hyperparameters chosen by the inner group-validated search in each outer fold |

- **`<label>`:** `sev1` or `sev2`.
- **`<protocol>`:** for example `within_seated`, `pooled`, `cell_standing_race` or `seated_to_standing` (manuscript Table 2).
- **`<learner>`:**
  - `hgb`, `hgbreg`, `logreg` and `mlp`: the earlier analysis;
  - `ctxb`: the context baseline;
  - `lroff`: LR+offset;
  - `xgboff`: boosting+offset;
  - `lroffp` and `lroffs`: LR+offset with the session-scaled and the ten-times-stronger penalty.
- **`<tag>`:** names a control or sensitivity run, for example:
  - `control_`: label-shift control;
  - `r3pos80_`: planted channel of AUC 0.80;
  - `sess_`: every session as its own group;
  - `mrg_`: merged groups;
  - `sensDEMO`: with the demographic answers.

## 7. Result files: `results/*.json`, `results_logo/*.json`

Each analysis stage writes one JSON file. For example:
- `main.json` (earlier analysis);
- `rev3_main.json`, `rev3_null_lroff.json`, `rev3_posctl_seeds.json` (present analysis);
- `add3_*.json` (addendum 3);
- `stats.json` and `rev3_stats.json` (intervals and tests).

`docs/MANUSCRIPT_TRACEABILITY.md` names the file behind each table.

## 8. Printed values: `results/paper/values*.json`

Every number printed in the manuscript and supplement is stored as a string under a key. The prefix says what produced it:

| Value file (script) | Key prefixes |
|---|---|
| `values.json` (`fa_report.py` with `fa_report_rev3.py`) | `aud.` audit; `cur.` curated release; `t.`, `s1.`, `s2.` earlier-analysis models; `r3.` present-analysis AUCs; `r3g.` telemetry gains; `r3n.` label-transplant nulls; `r3pw.`, `r3pwn.` power; `r3d.` context decoding; `vrl*.` VR-LENS; `y.` second archive; `sim.` simulation; `vrsq.`, `r3v.` questionnaire; and others |
| `values_add3.json` (`fa_report_add3.py`) | `a3.` addendum 3 (post hoc P1–P7) |
| `values_rq3res.json` (`fa_report_rq3res.py`) | `r3r.` RQ3 resolution summary (Table S38) |
| `values_sample.json` (`fa_report_sample.py`) | `smp.` sample description (Table S39) |

- **Location lists:** `results/paper/paper_keys*.json` list the keys the paper prints and where.
  - `paper_keys.json` and `paper_keys_add3.json` name the LaTeX source files of the time (for example `results_telemetry.tex`).
  - `paper_keys_rq3res.json` and `paper_keys_sample.json` name manuscript sections and tables.
- **Generated tables:** `results/paper/*.tex` are the generated tables; `values*.tex` define the same values as LaTeX macros (`\V{key}`).
