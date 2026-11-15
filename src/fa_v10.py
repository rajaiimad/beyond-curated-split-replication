"""
fa_v10.py -- two runs added after a further internal pre-submission review (13 November 2026), once the
creators' account of the data collection had been checked. They are NOT part of any analysis plan or addendum: post hoc.

Under the creators' account [9], [ICEC 2021] every curated session is one participant who played one game,
so a split by session is a split by participant. The inferred groups of the paper merge sessions with
identical demographic answers; they remain the pre-specified unit of every primary analysis. These runs
repeat two curated-release results with every curated session as its own unit:

  vrlens    the re-implemented VR-LENS detector (src/fa_vrlens.py, unchanged), group-disjoint 5-fold x
            5 repeats as in the paper, (a) with the inferred groups as units -- a re-run of the stored run,
            made only to keep the per-record four-class predictions, which the stored run did not save --
            and (b) with every curated session as its own unit. Four-class accuracy with a percentile
            bootstrap interval over the units (B = 10,000, the study's resampling scheme), and the overall
            and context-stratified AUCs for severity >= 2 and >= 1 with intervals.
  tab6sess  the group-disjoint rows of Table VI (HGB-reg on TIME, CTX+TIME, MOT+TIME, MOT+CTX+TIME) with
            every curated session as its own unit (folds and bootstrap over sessions).

Nothing existing is changed: predictions go to <results>/v10/preds/ (not results/preds, which fa_stats.py
reads), and the summary to <results>/v10/v10_runs.json. Stochastic learners keep the study's seeds; the
VR-LENS stage is not reproducible bit for bit (Supplementary Note S2), so a re-run can differ slightly.

Usage: python src/fa_v10.py <RAW_DATABASE dir> <DATABASE_SELECTED_DATA dir> <dataset.csv> <data dir> <results dir> [vrlens] [tab6sess]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fa_data import add_context, build_features, curated_sessions, load_archive   # noqa: E402
from fa_run import R_REP, domains, make_folds, run_grid                          # noqa: E402
from fa_stats import Boot, ci, load_frame, score_arrays, summary                  # noqa: E402
from fa_vrlens import FEATURES, curated_rows, vrlens_reduced                      # noqa: E402

TAB6_FSETS = ["TIME", "CTX+TIME", "MOT+TIME", "MOT+CTX+TIME"]


def frames(RAWDIR, SELDIR, DATA):
    raw, _ = load_archive(RAWDIR)
    cur = set(curated_sessions(SELDIR, RAWDIR).values()); assert len(cur) == 35
    raw["curated"] = raw.session_id.isin(cur)
    F = add_context(build_features(raw)); F["curated"] = raw["curated"].to_numpy()
    Fs, _ = load_frame(DATA)                                   # the frame fa_stats.py scores on (s_ctx strata)
    assert (Fs.session_id.to_numpy() == F.session_id.to_numpy()).all(), "record order differs from features_w30"
    assert (Fs.group_id.to_numpy() == F.group_id.to_numpy()).all()
    return raw, F, Fs


def session_units(F):
    """Every session its own unit; the same coding as fa_run's sensSESSGROUP_ and fa_rev3's sess_ runs."""
    return F.session_id.astype("category").cat.codes.to_numpy()


def accuracy_boot(bt, y4, proba, fold):
    """Four-class accuracy, mean over repeats of the record-level accuracy (as in fa_vrlens.py), for the
    observed sample (row 0) and every bootstrap draw of the units (a unit drawn k times enters k times)."""
    G = len(bt.ug); out = []
    for r in range(proba.shape[0]):
        v = fold[r] >= 0
        correct = (proba[r].argmax(1) == y4) & v
        K = np.bincount(bt.gi[v], weights=correct[v].astype(float), minlength=G)
        N = np.bincount(bt.gi[v], minlength=G).astype(float)
        out.append((bt.C @ K) / (bt.C @ N))
    return np.mean(out, axis=0)


def stage_vrlens(RAWDIR, SELDIR, CSV, DATA, R, res):
    raw, F, Fs = frames(RAWDIR, SELDIR, DATA)
    D, rows = curated_rows(raw, CSV)
    X = D[FEATURES].to_numpy(float); y4_all = D.DiscomfortLevel.to_numpy().astype(int)
    src, tgt = domains(F)["within_curated"]
    pos_of = np.full(len(F), -1); pos_of[rows] = np.arange(len(rows))
    tidx = np.where(tgt)[0]; y4 = y4_all[pos_of[tidx]]
    assert (y4 == F.severity.to_numpy()[tidx]).all(), "dataset.csv labels differ from the archive"
    ctx = Fs.s_ctx.to_numpy()[tidx].astype(str)
    P = R / "v10" / "preds"; P.mkdir(parents=True, exist_ok=True)
    out = {"features": FEATURES, "n_records": int(len(tidx)),
           "majority_class_share": float(np.bincount(y4).max() / len(y4))}
    for unit, g in (("groups", F.group_id.to_numpy()), ("sessions", session_units(F))):
        t0 = time.time(); PR = np.full((R_REP, len(tidx), 4), np.nan); FOLD = np.full((R_REP, len(tidx)), -1, np.int8)
        loc = np.full(len(F), -1); loc[tidx] = np.arange(len(tidx))
        for r, f, tr, te in make_folds(g, src, tgt):
            assert not (set(g[tr]) & set(g[te])), "unit leakage"
            a, b = pos_of[tr], pos_of[te]; assert (a >= 0).all() and (b >= 0).all()
            PR[r, loc[te]] = vrlens_reduced().fit(X[a], y4_all[a]).predict_proba(X[b]); FOLD[r, loc[te]] = f
            print(f"vrlens [{unit}] repeat {r + 1}/{R_REP} fold {f + 1} ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(P / f"v10vrl_{unit}__within_curated.npz", proba=PR, fold=FOLD, idx=tidx)
        units = g[tidx]; bt = Boot(units)
        acc = accuracy_boot(bt, y4, PR, FOLD)
        d = {"units": int(len(bt.ug)), "accuracy_4class": float(acc[0]), "accuracy_4class_ci": ci(acc),
             "accuracy_4class_repeats": [float(((PR[r].argmax(1) == y4)).mean()) for r in range(R_REP)],
             "secs": round(time.time() - t0)}
        for lab in (1, 2):
            yy = (y4 >= lab).astype(int); sc = PR[:, :, lab:].sum(2)
            d[f"sev{lab}"] = summary(score_arrays(bt, yy, sc, FOLD, ctx), len(yy), len(bt.ug), yy.mean())
        out[unit] = d; print(unit, json.dumps(d), flush=True)
    res["vrlens"] = out


def stage_tab6sess(RAWDIR, SELDIR, CSV, DATA, R, res):
    raw, F, Fs = frames(RAWDIR, SELDIR, DATA)
    G = F.copy(); G["group_id"] = session_units(F)
    P = R / "v10" / "preds"; P.mkdir(parents=True, exist_ok=True)
    runs = run_grid(G, ["within_curated"], TAB6_FSETS, ["hgbreg"], [1, 2], P, tag="v10sess_")
    tgt = domains(F)["within_curated"][1]; units = G.group_id.to_numpy()[tgt]; bt = Boot(units)
    ctx = Fs.s_ctx.to_numpy()[tgt].astype(str); out = {"units": int(len(bt.ug)), "configs": {}}
    for run in runs:
        z = np.load(P / f"{run['key']}.npz"); lab = int(run["label"][-1])
        yy = (F.severity.to_numpy()[tgt] >= lab).astype(int)
        out["configs"][run["key"]] = summary(score_arrays(bt, yy, z["p"][:, tgt], z["fold"][:, tgt], ctx), len(yy), len(bt.ug), yy.mean())
    res["tab6sess"] = out


if __name__ == "__main__":
    RAWDIR, SELDIR, CSV, DATA, R = map(Path, sys.argv[1:6]); stages = sys.argv[6:] or ["vrlens", "tab6sess"]
    f = R / "v10" / "v10_runs.json"; f.parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(f)) if f.exists() else {}
    res["status"] = ("post hoc: added after a further internal pre-submission review (13 November 2026), once the "
                     "creators' protocol had been checked; not part of any analysis plan")
    for st in stages:
        t0 = time.time()
        {"vrlens": stage_vrlens, "tab6sess": stage_tab6sess}[st](RAWDIR, SELDIR, CSV, DATA, R, res)
        json.dump(res, open(f, "w"), indent=1); print(f"== {st} done in {time.time() - t0:.0f}s", flush=True)
