"""
fa_run.py -- all model fits of the full-archive study.

Design (identical for every protocol):
  * repeated grouped K-fold over the participant groups of the TARGET domain
    (K = 5 folds, R = 5 repeats with different random partitions);
  * for a test fold, the model is trained on SOURCE records of all groups that are
    NOT in the fold, and applied to the TARGET records of the groups in the fold;
  * standardisation is fitted on the training records of the fold;
  * the operating threshold is chosen on training groups only (inner 3-fold group
    split, 10 % false-alarm budget), then applied once to the test fold.

Every test record therefore gets R out-of-group scores, one per repeat, together
with the id of the fold (i.e. the model) that produced it. Metrics compare only
records scored by the same model (fa_stats.py): pooling scores of different fold
models across a leave-one-group-out design biases the AUC of prevalence-like
predictors downward (Airola et al., 2011), which we observed in a first run.

Usage: python fa_run.py <RAW_DATABASE dir> <DATABASE_SELECTED_DATA dir> <data dir> <results dir> [stage ...]
stages: checks main ctx mitig cell control rowwise vrsq sens logo_ctx null null_lr posctl es_check   (default: all)
revision 2 (run explicitly): hgbreg posctl_reg null500 null_lr500   -- see docs/ANALYSIS_PLAN_REVISION2.md
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from fa_data import (DEMO, add_context, assert_no_leakage, build_features, curated_sessions,  # noqa: E402
                     feature_sets, load_archive, session_features)

BUDGET, K, R_REP, SEED = 0.10, 5, 5, 0
NJOBS = int(os.environ.get("FA_NJOBS", "2"))     # parallel folds; set FA_NJOBS=1 to run two stages side by side
MODELS = {
    "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "hgb": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, class_weight="balanced", random_state=0),
    "mlp": MLPClassifier((64, 32), max_iter=300, early_stopping=True, n_iter_no_change=10, random_state=0),
    "hgbreg": "hgbreg",       # regularised HGB chosen by nested grouped CV (fit_hgbreg); see docs/ANALYSIS_PLAN_REVISION2.md
}

# ----------------------------------------------------------------------------- HGB-reg (revision 2)
# Fixed before any HGB-reg result was computed (docs/ANALYSIS_PLAN_REVISION2.md). Grid ordered from
# most to least regularised; ties in the selection score go to the earlier entry.
HGBREG_GRID = [dict(max_depth=d, min_samples_leaf=l, l2_regularization=r)
               for d in (2, 3) for l in (300, 100) for r in (10.0, 1.0)]
HGBREG_FIXED = dict(learning_rate=0.05, class_weight="balanced", early_stopping=False, random_state=0)
HGBREG_MAX_ITER, HGBREG_INNER_K = 300, 3
LAST_SELECTION = []            # per-fold selection record of the last hgbreg transfer() call


def strat_auc(y, s, strata):
    """AUC over positive-negative pairs of the same stratum, weighted by pairs (nan if none)."""
    num = den = 0.0
    for st in np.unique(strata):
        m = strata == st; yy = y[m]; npos = int(yy.sum()); nneg = len(yy) - npos
        if npos and nneg:
            r = rankdata(s[m]); num += (r[yy == 1].sum() - npos * (npos + 1) / 2); den += npos * nneg
    return num / den if den else np.nan


def _staged_best(model, Xv, yv):
    """Iteration count minimising the class-balanced log-loss on held-out groups, and the
    held-out logits at that iteration."""
    w = np.where(yv == 1, len(yv) / (2 * yv.sum()), len(yv) / (2 * (len(yv) - yv.sum())))
    best, best_loss, best_z = 0, np.inf, None
    for i, z in enumerate(model.staged_decision_function(Xv)):
        z = np.ravel(z); loss = float(np.average(np.logaddexp(0, np.where(yv == 1, -z, z)), weights=w))
        if loss < best_loss:
            best, best_loss, best_z = i + 1, loss, z
    return best, best_z


def fit_hgbreg(X, y, groups, strata, tr):
    """Nested selection inside the training records tr of one outer fold. Returns the refitted
    model, the operating threshold and a selection record."""
    inner = [(tr[a], tr[b]) for a, b in GroupKFold(HGBREG_INNER_K).split(tr, groups=groups[tr])]
    for a, b in inner:
        assert not (set(groups[a]) & set(groups[b]))            # inner folds are group-disjoint
    rows = []
    for ci, cfg in enumerate(HGBREG_GRID):
        oof = np.full(len(y), np.nan); nstar, sc, ov = [], [], []
        for a, b in inner:
            if len(set(y[a])) < 2 or len(set(y[b])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=HGBREG_MAX_ITER, **cfg, **HGBREG_FIXED).fit(X[a], y[a])
            n_best, z = _staged_best(m, X[b], y[b])
            oof[b] = 1 / (1 + np.exp(-z)); nstar.append(n_best)
            sc.append(strat_auc(y[b], z, strata[b])); ov.append(roc_auc_score(y[b], z))
        ctx_score = np.nanmean(sc) if sc and not np.isnan(sc).all() else np.nan
        rows.append({"cfg": ci, "ctx": ctx_score, "overall": float(np.mean(ov)) if ov else np.nan,
                     "nstar": nstar, "oof": oof})
    use_ctx = any(not np.isnan(r["ctx"]) for r in rows)
    score = [(r["ctx"] if use_ctx else r["overall"]) for r in rows]
    score = [-np.inf if (s_ is None or np.isnan(s_)) else s_ for s_ in score]
    ci = int(np.argmax(score))                                    # first maximum = most regularised tie
    ch = rows[ci]; n_iter = int(round(np.median(ch["nstar"]))) if ch["nstar"] else HGBREG_MAX_ITER
    n_iter = max(n_iter, 1)
    final = HistGradientBoostingClassifier(max_iter=n_iter, **HGBREG_GRID[ci], **HGBREG_FIXED).fit(X[tr], y[tr])
    neg = ch["oof"][tr][(y[tr] == 0) & ~np.isnan(ch["oof"][tr])]
    thr = float(np.quantile(neg, 1 - BUDGET)) if len(neg) else 0.5
    return final, thr, {"cfg": ci, **HGBREG_GRID[ci], "n_iter": n_iter, "criterion": "ctx" if use_ctx else "overall",
                        "inner_scores": [None if not np.isfinite(v) else float(v) for v in score]}


def _one_fold_reg(X, y, groups, strata, tr, te, inner):
    if len(te) == 0 or len(set(y[tr])) < 2:
        return None
    assert not (set(groups[tr]) & set(groups[te]))              # leakage guard, every fold
    m, thr, sel = fit_hgbreg(X, y, groups, strata, tr)
    return m.predict_proba(X[te])[:, 1], (thr if inner else np.nan), sel


# ----------------------------------------------------------------------------- folds
def make_folds(groups, src, tgt, k=K, reps=R_REP):
    """[(repeat, fold, train_idx, test_idx)] -- target groups split into k folds per repeat."""
    ug = np.unique(groups[tgt]); k = min(k, len(ug)); out = []
    for r in range(reps):
        perm = np.random.default_rng(1000 + r).permutation(ug)
        fold_of = {g: i % k for i, g in enumerate(perm)}          # balanced number of groups per fold
        for f in range(k):
            test_groups = np.array([g for g in ug if fold_of[g] == f])
            te = np.where(tgt & np.isin(groups, test_groups))[0]
            tr = np.where(src & ~np.isin(groups, test_groups))[0]
            out.append((r, f, tr, te))
    return out


# ----------------------------------------------------------------------------- core
def _fit_predict(model, X, y, tr, te):
    pipe = make_pipeline(StandardScaler(), clone(model)).fit(X[tr], y[tr])
    return pipe.predict_proba(X[te])[:, 1]


def _one_fold(model, X, y, groups, tr, te, inner):
    if len(te) == 0 or len(set(y[tr])) < 2:
        return None
    assert not (set(groups[tr]) & set(groups[te]))              # leakage guard, every fold
    p = _fit_predict(model, X, y, tr, te)
    thr = np.nan
    if inner:
        s = np.full(len(tr), np.nan)
        for a, b in GroupKFold(3).split(tr, groups=groups[tr]):
            if len(set(y[tr[a]])) == 2:
                s[b] = _fit_predict(model, X, y, tr[a], tr[b])
        neg = s[(y[tr] == 0) & ~np.isnan(s)]
        thr = float(np.quantile(neg, 1 - BUDGET)) if len(neg) else 0.5
    return p, thr


def transfer(F, cols, y_train, src, tgt, model, inner=True, n_jobs=None):
    """Returns P, THR (R x n float, nan outside the target) and FOLD (R x n int8, -1 outside)."""
    X = F[cols].to_numpy(float); groups = F.group_id.to_numpy(); n = len(F)
    n_jobs = NJOBS if n_jobs is None else n_jobs
    folds = make_folds(groups, src, tgt)
    if isinstance(model, str) and model == "hgbreg":
        tbin = np.minimum((F.t_elapsed.to_numpy() // 30).astype(int), 9)
        strata = (F.campaign + "|" + F.game).to_numpy().astype(str)
        strata = np.char.add(np.char.add(strata, "|"), tbin.astype(str))
        out = Parallel(n_jobs=n_jobs)(delayed(_one_fold_reg)(X, y_train, groups, strata, tr, te, inner) for _, _, tr, te in folds)
        LAST_SELECTION.clear()
        LAST_SELECTION.extend([{"repeat": r, "fold": f, **res[2]} for (r, f, _, _), res in zip(folds, out) if res is not None])
    else:
        out = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(model, X, y_train, groups, tr, te, inner) for _, _, tr, te in folds)
    P = np.full((R_REP, n), np.nan); THR = np.full((R_REP, n), np.nan); FOLD = np.full((R_REP, n), -1, np.int8)
    for (r, f, tr, te), res in zip(folds, out):
        if res is None:
            continue
        P[r, te], THR[r, te], FOLD[r, te] = res[0], res[1], f
    return P, THR, FOLD


# ----------------------------------------------------------------------------- point summaries
def _auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) == 2 else np.nan


def within_fold_auc(y, p, fold):
    """AUC over positive-negative pairs scored by the same model (same fold)."""
    num = den = 0.0
    for f in np.unique(fold[fold >= 0]):
        m = fold == f
        npos = y[m].sum(); nneg = m.sum() - npos
        if npos and nneg:
            num += roc_auc_score(y[m], p[m]) * npos * nneg; den += npos * nneg
    return num / den if den else np.nan


def summarise(y, P, THR, FOLD, g):
    """Point estimates (intervals come from fa_stats.py). Averages over repeats."""
    aucs, prs, ops, pg = [], [], [], {}
    for r in range(P.shape[0]):
        m = FOLD[r] >= 0
        yy, pp, tt, ff, gg = y[m], P[r][m], THR[r][m], FOLD[r][m], g[m]
        aucs.append(within_fold_auc(yy, pp, ff)); prs.append(average_precision_score(yy, pp))
        for k in np.unique(gg):
            a = _auc(yy[gg == k], pp[gg == k])
            if not np.isnan(a):
                pg.setdefault(k, []).append(a)
        if not np.isnan(tt).all():
            yh = (pp >= tt).astype(int)
            ops.append([yh[yy == 0].mean(), yh[yy == 1].mean(), yy[yh == 1].mean() if yh.any() else 0.0,
                        matthews_corrcoef(yy, yh)])
    per_group = [np.mean(v) for v in pg.values()]
    m = FOLD[0] >= 0
    out = {"n_rows": int(m.sum()), "n_groups": int(len(np.unique(g[m]))), "prevalence": float(y[m].mean()),
           "roc_auc": float(np.mean(aucs)), "roc_auc_repeats": [float(a) for a in aucs],
           "pr_auc": float(np.mean(prs)), "coverage": float((FOLD >= 0).sum() / (P.shape[0] * len(y))),
           "per_group_auc_median": float(np.median(per_group)) if per_group else float("nan"),
           "per_group_auc_iqr": [float(np.percentile(per_group, 25)), float(np.percentile(per_group, 75))] if per_group else None,
           "n_groups_both_classes": len(per_group)}
    if ops:
        o = np.mean(ops, axis=0)
        out.update({"op_fpr": float(o[0]), "op_recall": float(o[1]), "op_precision": float(o[2]), "op_mcc": float(o[3])})
    return out


# ----------------------------------------------------------------------------- setup
def domains(F):
    c, gm = F.campaign.to_numpy(), F.game.to_numpy(); allm = np.ones(len(F), bool)
    cur = F.curated.to_numpy().astype(bool) if "curated" in F else np.zeros(len(F), bool)
    return {"within_seated": (c == "seated", c == "seated"),
            "within_curated": (cur, cur),
            "within_standing": (c == "standing", c == "standing"),
            "pooled": (allm, allm),
            "seated_to_standing": (c == "seated", c == "standing"),
            "standing_to_seated": (c == "standing", c == "seated"),
            "flight_to_race": (gm == "Flight", gm == "Race"),
            "race_to_flight": (gm == "Race", gm == "Flight"),
            "within_flight": (gm == "Flight", gm == "Flight"),
            "within_race": (gm == "Race", gm == "Race"),
            # single game x posture cells: no between-cell shortcut is available in training
            "cell_seated_flight": ((c == "seated") & (gm == "Flight"),) * 2,
            "cell_seated_race": ((c == "seated") & (gm == "Race"),) * 2,
            "cell_standing_flight": ((c == "standing") & (gm == "Flight"),) * 2,
            "cell_standing_race": ((c == "standing") & (gm == "Race"),) * 2}


def run_grid(F, protocols, fsets, models, labels, preds_dir, tag="", inner=True, y_train_fn=None):
    res = []
    D = domains(F); FS = feature_sets(F); FS.update(fsets if isinstance(fsets, dict) else {})
    names = fsets if isinstance(fsets, list) else list(fsets)
    for lab in labels:
        y = (F.severity.to_numpy() >= lab).astype(int)
        y_train = y_train_fn(y) if y_train_fn else y
        for pr in protocols:
            src, tgt = D[pr]
            for fs in names:
                for mn in models:
                    t0 = time.time()
                    key = f"{tag}sev{lab}__{pr}__{fs}__{mn}"
                    f = preds_dir / f"{key}.npz"; P = None
                    if f.exists():                      # resume: reuse finished predictions
                        try:
                            z = np.load(f); P, THR, FOLD = z["p"], z["thr"], z["fold"]
                            assert P.shape == (R_REP, len(y)) and (FOLD[:, tgt] >= 0).any()
                        except Exception:
                            P = None
                    if P is None:
                        P, THR, FOLD = transfer(F, FS[fs], y_train, src, tgt, MODELS[mn], inner=inner)
                        if mn == "hgbreg":
                            json.dump(LAST_SELECTION, open(preds_dir / f"{key}.selection.json", "w"))
                        tmp = f.with_suffix(".tmp.npz")
                        np.savez_compressed(tmp, p=P, thr=THR, fold=FOLD); tmp.replace(f)   # atomic write
                    s = summarise(y[tgt], P[:, tgt], THR[:, tgt], FOLD[:, tgt], F.group_id.to_numpy()[tgt])
                    s.update(key=key, label=f"sev>={lab}", protocol=pr, features=fs, model=mn, secs=round(time.time() - t0, 1))
                    res.append(s)
                    print(f"{key:62s} AUC {s['roc_auc']:.3f} (reps {min(s['roc_auc_repeats']):.3f}-{max(s['roc_auc_repeats']):.3f}) "
                          f"PR {s['pr_auc']:.3f} prev {s['prevalence']:.3f}"
                          + (f" | FPR {s['op_fpr']:.3f} rec {s['op_recall']:.3f} MCC {s['op_mcc']:.3f}" if 'op_fpr' in s else "")
                          + f" ({s['secs']}s)", flush=True)
    return res


def load_pred(preds_dir, key):
    z = np.load(preds_dir / f"{key}.npz"); return z["p"], z["thr"], z["fold"]


# ----------------------------------------------------------------------------- stages
MAIN_PROTOCOLS = ["within_seated", "within_standing", "pooled", "seated_to_standing",
                  "standing_to_seated", "flight_to_race", "race_to_flight"]
CTX_PROTOCOLS = MAIN_PROTOCOLS + ["within_flight", "within_race", "within_curated"]
CELL_PROTOCOLS = ["cell_seated_flight", "cell_seated_race", "cell_standing_flight", "cell_standing_race"]


def stage_main(F, R, P):
    res = run_grid(F, CTX_PROTOCOLS, ["TIME", "MOT", "MOT+TIME"], ["logreg", "hgb"], [1, 2], P)
    json.dump(res, open(R / "main.json", "w"), indent=1)


def stage_ctx(F, R, P):
    """Context baselines. The archive audit showed that the share of high-severity
    records differs strongly between game x posture cells, so a detector can score
    well by recognising the cell. CTX = game and posture indicators; CTX+TIME adds
    exposure time; MOT+CTX+TIME asks whether telemetry adds to context."""
    res = run_grid(F, CTX_PROTOCOLS, ["CTX"], ["hgb"], [1, 2], P)
    res += run_grid(F, CTX_PROTOCOLS, ["CTX+TIME", "MOT+CTX+TIME"], ["logreg", "hgb"], [1, 2], P)
    json.dump(res, open(R / "context.json", "w"), indent=1)


def stage_mitig(F, R, P):
    """Per-session self-normalised telemetry in every protocol (same-setting, cross-setting,
    pooled, curated release and single cells)."""
    res = run_grid(F, ["within_seated", "within_standing", "within_flight", "within_race", "seated_to_standing",
                       "standing_to_seated", "flight_to_race", "race_to_flight", "pooled", "within_curated"] + CELL_PROTOCOLS,
                   ["SELFZ+TIME"], ["logreg", "hgb"], [1, 2], P)
    json.dump(res, open(R / "mitigation.json", "w"), indent=1)


def stage_cell(F, R, P):
    """Train and test inside one game x posture cell. Within a cell, game and posture
    are constant, so discrimination beyond exposure time must come from telemetry."""
    res = run_grid(F, CELL_PROTOCOLS, ["TIME", "MOT+TIME"], ["logreg", "hgb"], [1, 2], P)
    json.dump(res, open(R / "cell.json", "w"), indent=1)


def stage_control(F, R, P):
    """Label-shift control: circularly shift each session's labels before TRAINING (test
    labels untouched). The shifted label keeps each session's share of high ratings and
    its autocorrelation, but loses its alignment with the records."""
    rng = np.random.default_rng(1)
    # offset between a quarter and three quarters of the session, so no session is left
    # close to its original alignment
    shift = {sid: int(rng.integers(len(idx) // 4, max(len(idx) // 4 + 1, (3 * len(idx)) // 4 + 1)))
             for sid, idx in sorted(F.groupby("session_id").indices.items())}
    idx_of = F.groupby("session_id").indices

    def shifted(y):
        ys = y.copy()
        for sid, idx in idx_of.items():
            ys[idx] = np.roll(y[idx], shift[sid])
        return ys
    res = run_grid(F, ["within_seated", "within_curated", "pooled", "seated_to_standing", "within_standing"], ["MOT", "MOT+TIME"],
                   ["hgb"], [1, 2], P, tag="control_", inner=False, y_train_fn=shifted)
    agree = {}
    for lab in (1, 2):
        y = (F.severity.to_numpy() >= lab).astype(int); ys = shifted(y)
        agree[f"sev{lab}"] = {"record_agreement": float((y == ys).mean()),
                              "positives_still_positive": float(ys[y == 1].mean()),
                              "negatives_now_positive": float(ys[y == 0].mean())}
    json.dump({"runs": res, "label_agreement_after_shift": agree,
               "offset_rule": "uniform integer in [n/4, 3n/4] records, one per session, seed 1"},
              open(R / "control.json", "w"), indent=1)
    print(agree, flush=True)


def stage_rowwise(F, R, P):
    """For reference only: the record-wise stratified 5-fold split common in the literature."""
    res = []
    for lab in (1, 2):
        y = (F.severity.to_numpy() >= lab).astype(int)
        for pr in ("within_curated", "within_seated", "pooled"):
            m = domains(F)[pr][0]; idx = np.where(m)[0]; yy = y[idx]
            for fs in ("MOT+TIME", "CTX+TIME"):
                X = F[feature_sets(F)[fs]].to_numpy(float)[idx]; p = np.zeros(len(idx)); fold = np.zeros(len(idx), np.int8)
                for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, yy)):
                    p[b] = make_pipeline(StandardScaler(), clone(MODELS["hgb"])).fit(X[a], yy[a]).predict_proba(X[b])[:, 1]
                    fold[b] = k
                np.savez_compressed(P / f"rowwise_sev{lab}__{pr}__{fs}__hgb.npz", p=p, idx=idx, fold=fold)
                res.append({"label": f"sev>={lab}", "protocol": pr, "features": fs, "model": "hgb",
                            "split": "record-wise stratified 5-fold (participants on both sides)",
                            "roc_auc": float(within_fold_auc(yy, p, fold)), "pr_auc": float(average_precision_score(yy, p)),
                            "prevalence": float(yy.mean()), "n_rows": int(len(yy))})
                print(res[-1], flush=True)
    json.dump(res, open(R / "rowwise_reference.json", "w"), indent=1)


def stage_sens(F, R, P, raw):
    res = []
    base_prot = ["within_seated", "pooled", "seated_to_standing", "standing_to_seated"]
    res += run_grid(F, base_prot, ["MOT+FPS+TIME"], ["hgb"], [1, 2], P, tag="sensFPS_")
    res += run_grid(F, base_prot, ["MOT+TIME"], ["mlp"], [1, 2], P, tag="sensMLP_", inner=False)
    oh = pd.get_dummies(F[DEMO].astype(str), prefix=DEMO).astype(float)
    Fd = pd.concat([F, oh], axis=1)
    res += run_grid(Fd, base_prot, {"MOT+TIME+DEMO": feature_sets(F)["MOT+TIME"] + list(oh.columns)}, ["hgb"], [1, 2], P, tag="sensDEMO_")
    S = pd.read_csv(DATA / "sessions.csv")
    bad = set(S[S.flag_gap | S.flag_config | S.flag_short].session_id)
    Fc = F[~F.session_id.isin(bad)].reset_index(drop=True)
    res += run_grid(Fc, base_prot, ["MOT+TIME"], ["hgb"], [1, 2], P, tag="sensCLEAN_")
    Fs = F.copy(); Fs["group_id"] = Fs.session_id.astype("category").cat.codes
    res += run_grid(Fs, base_prot, ["MOT+TIME"], ["hgb"], [1, 2], P, tag="sensSESSGROUP_")
    for w in (10, 60):
        Fw = add_context(build_features(raw, window=w)); Fw["curated"] = F["curated"].to_numpy()
        res += run_grid(Fw, base_prot, ["MOT+TIME"], ["hgb"], [1, 2], P, tag=f"sensW{w}_")
    res += run_grid(F, base_prot, ["MOTQ+TIME"], ["hgb"], [1, 2], P, tag="sensQUAT_")
    json.dump(res, open(R / "sensitivity.json", "w"), indent=1)


def stage_logo_ctx(F, R, P):
    """Reproduces the first-run observation that motivated the change of fold design:
    a context-only model evaluated leave-one-group-out, with scores POOLED across folds."""
    out = {}
    D = domains(F); X = F[feature_sets(F)["CTX"]].to_numpy(float); g = F.group_id.to_numpy()
    for lab in (1, 2):
        y = (F.severity.to_numpy() >= lab).astype(int)
        for pr in ("within_seated", "within_curated", "pooled"):
            src, tgt = D[pr]; p = np.full(len(y), np.nan)
            for k in np.unique(g[tgt]):
                tr = np.where(src & (g != k))[0]; te = np.where(tgt & (g == k))[0]
                p[te] = _fit_predict(MODELS["hgb"], X, y, tr, te)
            out[f"sev{lab}__{pr}__CTX__hgb"] = {"logo_pooled_auc": float(roc_auc_score(y[tgt], p[tgt]))}
    json.dump(out, open(R / "logo_ctx.json", "w"), indent=1); print(out, flush=True)


def stage_vrsq(F, R, P, bb=10_000, seed=SEED):
    """Label validity: session summaries of the in-game ratings against the VRSQ.
    Spearman's rho with a participant-group bootstrap (sessions of a drawn group enter
    together). The detector's mean risk per session (pooled MOT+TIME, HGB, mean over the
    repeats' out-of-group scores) is included for completeness."""
    S = pd.read_csv(DATA / "sessions.csv").set_index("session_id")
    sev = F.groupby("session_id").severity
    S["ing_max"] = sev.max(); S["ing_mean"] = sev.mean()
    S["ing_frac_ge1"] = sev.apply(lambda s: (s >= 1).mean()); S["ing_frac_ge2"] = sev.apply(lambda s: (s >= 2).mean())
    for lv in (1, 2):
        S[f"onset_ge{lv}_s"] = F.groupby("session_id").apply(
            lambda d: d.t_elapsed[d.severity >= lv].min() if (d.severity >= lv).any() else np.nan)
    for lab in (1, 2):
        p, _, _ = load_pred(P, f"sev{lab}__pooled__MOT+TIME__hgb")
        S[f"det_mean_risk_sev{lab}"] = pd.Series(np.nanmean(p, axis=0), index=F.index).groupby(F.session_id).mean()
    V = S.dropna(subset=["vrsq_post_total"]).copy()
    V["vrsq_delta_total"] = V.vrsq_post_total - V.vrsq_pre_total

    def spear(a, b, dfr):
        rng = np.random.default_rng(seed); r = dfr[a].rank().corr(dfr[b].rank())
        gs = dfr.group_id.unique(); by = {k: dfr.index[dfr.group_id == k] for k in gs}; v = []
        for _ in range(bb):
            d = dfr.loc[np.concatenate([by[k] for k in rng.choice(gs, len(gs))])]
            v.append(d[a].rank().corr(d[b].rank()))
        return {"rho": float(r), "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))], "n": int(len(dfr))}

    out = {"n_sessions_with_vrsq": int(len(V)), "B": bb, "correlations": {}, "by_campaign": {}}
    for a in ("ing_max", "ing_mean", "ing_frac_ge1", "ing_frac_ge2", "det_mean_risk_sev1", "det_mean_risk_sev2"):
        for b in ("vrsq_post_total", "vrsq_delta_total"):
            out["correlations"][f"{a}~{b}"] = spear(a, b, V)
    for camp, d in V.groupby("campaign"):
        out["by_campaign"][camp] = {k: spear(k, "vrsq_post_total", d) for k in ("ing_max", "ing_mean")}
    ct = pd.crosstab(V.ing_max, V.vrsq_post_total > 0)
    out["crosstab"] = {"rows_ingame_max": ct.index.tolist(),
                       "vrsq_post_zero": ct.get(False, pd.Series(0, ct.index)).tolist(),
                       "vrsq_post_positive": ct.get(True, pd.Series(0, ct.index)).tolist()}
    pos_med = V.vrsq_post_total[V.vrsq_post_total > 0].median()
    out["discordant"] = {"ingame_ge2_but_vrsq_post_0": int(((V.ing_max >= 2) & (V.vrsq_post_total == 0)).sum()),
                         "ingame_0_but_vrsq_post_ge_median_pos": int(((V.ing_max == 0) & (V.vrsq_post_total >= pos_med)).sum())}
    out["onset"] = {camp: {"ge1_median_s": float(d.onset_ge1_s.median()), "ge1_n": int(d.onset_ge1_s.notna().sum()),
                           "ge2_median_s": float(d.onset_ge2_s.median()), "ge2_n": int(d.onset_ge2_s.notna().sum()),
                           "n_sessions": int(len(d))} for camp, d in S.groupby("campaign")}
    out["vrsq_summary"] = {camp: {"pre_mean": float(d.vrsq_pre_total.mean()), "post_mean": float(d.vrsq_post_total.mean()),
                                  "post_median": float(d.vrsq_post_total.median()),
                                  "post_zero_frac": float((d.vrsq_post_total == 0).mean())}
                           for camp, d in V.groupby("campaign")}
    S.to_csv(R / "session_level.csv")
    json.dump(out, open(R / "vrsq.json", "w"), indent=1)
    print({k: round(v["rho"], 3) for k, v in out["correlations"].items()}, flush=True)


def spike_feature(F, y, d, seed, rho=0.9):
    """Positive control: a synthetic telemetry channel that carries a within-context signal
    of known size. Noise is a stationary AR(1) process per session (N(0,1) marginals,
    lag-1 correlation rho, like a slowly varying 1-Hz signal); the channel is d*y + noise,
    so its AUC is Phi(d / sqrt 2) in every stratum."""
    rng = np.random.default_rng(seed); e = np.empty(len(F))
    for sid, idx in sorted(F.groupby("session_id").indices.items()):
        z = rng.standard_normal(len(idx)); a = np.empty(len(idx)); a[0] = z[0]
        for i in range(1, len(idx)):
            a[i] = rho * a[i - 1] + np.sqrt(1 - rho ** 2) * z[i]
        e[idx] = a
    return d * y + e


def stage_posctl(F, R, P, models=("hgb", "logreg"), fname="posctl.json"):
    """Positive control (sensitivity of the design). A synthetic channel with a planted
    within-stratum AUC (0.60 or 0.70) is run through the study's folds in three ways:
      SPK+TIME        the channel alone with exposure time -- what folds and metric recover
                      when nothing else competes (ceiling);
      MOT+SPK+TIME    the channel among the 28 logged telemetry features -- what the study's
                      raw-telemetry models would recover if the logs held such a signal;
      SELFZ+SPK+TIME  the same with self-normalised features (the channel self-normalised too).
    Both learners; no threshold is set (inner=False)."""
    from scipy.stats import norm
    D = domains(F); FSd = feature_sets(F); mot, selfz = FSd["MOT"], [c for c in FSd["SELFZ+TIME"] if c != "t_elapsed"]
    out = {"runs": [], "planted": {}}
    prots = ["within_seated", "within_standing", "pooled"]
    for target_auc in (0.60, 0.70):
        d = float(np.sqrt(2) * norm.ppf(target_auc)); tag = f"sensSPK{int(round(100 * target_auc))}_"
        for lab in (1, 2):
            y = (F.severity.to_numpy() >= lab).astype(int)
            spk = spike_feature(F, y, d, seed=900 + lab)
            zs = np.empty(len(F))
            for sid, idx in F.groupby("session_id").indices.items():      # same rule as the z_ features
                x = pd.Series(spk[idx]); mu = x.expanding().mean(); sd = x.expanding().std()
                zs[idx] = ((x - mu) / sd.where(sd > 1e-9)).fillna(0).clip(-10, 10).to_numpy()
            Fs = F.assign(spk=spk, z_spk=zs)
            for pr in prots:
                t = D[pr][1]
                out["planted"][f"{tag}sev{lab}__{pr}"] = {"d": d, "target_auc": target_auc,
                                                          "channel_auc_on_target": float(roc_auc_score(y[t], spk[t]))}
            fsets = {"SPK+TIME": ["spk", "t_elapsed"], "MOT+SPK+TIME": mot + ["spk", "t_elapsed"],
                     "SELFZ+SPK+TIME": selfz + ["z_spk", "t_elapsed"]}
            out["runs"] += run_grid(Fs, prots, fsets, list(models), [lab], P, tag=tag, inner=False)
    json.dump(out, open(R / fname, "w"), indent=1)


def stage_hgbreg(F, R, P):
    """Revision 2: regularised HGB chosen by nested grouped CV, on the protocols and feature sets
    of the main table and the comparison family."""
    res = run_grid(F, CTX_PROTOCOLS, ["TIME", "CTX", "CTX+TIME", "MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME"],
                   ["hgbreg"], [1, 2], P)
    json.dump(res, open(R / "hgbreg.json", "w"), indent=1)


def stage_es_check(F, R, P):
    """HGB uses scikit-learn's automatic early stopping when a training set exceeds 10,000
    records (validation = a random 10 % of the training records). Records how many of the
    150 iterations each fold actually used (repeat 0, MOT+TIME, both labels)."""
    D = domains(F); X = F[feature_sets(F)["MOT+TIME"]].to_numpy(float); g = F.group_id.to_numpy(); out = {}
    for lab in (1, 2):
        y = (F.severity.to_numpy() >= lab).astype(int)
        for pr in MAIN_PROTOCOLS + ["within_flight", "within_race"]:
            src, tgt = D[pr]; rows = []
            for r, f, tr, te in make_folds(g, src, tgt):
                if r:
                    break
                m = make_pipeline(StandardScaler(), clone(MODELS["hgb"])).fit(X[tr], y[tr])[-1]
                rows.append({"n_train": int(len(tr)), "early_stopping": bool(m.do_early_stopping_), "n_iter": int(m.n_iter_)})
            out[f"sev{lab}__{pr}"] = rows
    json.dump(out, open(R / "es_check.json", "w"), indent=1)
    es = [r for v in out.values() for r in v if r["early_stopping"]]
    print(f"early stopping active in {len(es)} folds; stopped before 150 in {sum(r['n_iter'] < 150 for r in es)}", flush=True)


def transplant_labels(F, y, rng):
    """Null labels: the rating sequences of the sessions of each game x posture cell are
    permuted among those sessions (no session keeps its own; a different participant group
    where possible) and aligned by elapsed time. Label persistence, session-level prevalence and the cell's time course
    are kept; any link between a session's own telemetry and its labels is broken."""
    t = F.t_elapsed.to_numpy(); cell = (F.campaign + "|" + F.game).to_numpy()
    idx = F.groupby("session_id").indices; grp = F.groupby("session_id").group_id.first()
    out = np.empty_like(y)
    for c in np.unique(cell):
        sids = sorted(F.session_id[cell == c].unique())
        # a derangement within the cell: every session gives and receives exactly one sequence;
        # prefer donors from another participant group
        for strict in (True, False):
            for _ in range(2000):
                perm = list(rng.permutation(sids))
                if all(d != s_ and (not strict or grp[d] != grp[s_]) for s_, d in zip(sids, perm)):
                    break
            else:
                continue
            break
        for sid, don in zip(sids, perm):
            a, b = idx[sid], idx[don]
            k = np.clip(np.searchsorted(t[b], t[a], side="right") - 1, 0, len(b) - 1)
            out[a] = y[b][k]
    return out


def stage_null(F, R, P, n_sim=40, model="hgb", fname="null.json", sims=None):
    """Reference distribution of the AUCs when telemetry carries no information about the
    labels: label transplant between sessions of the same cell (see transplant_labels),
    scored with the exact folds of the study. Resumable: a simulation is identified by its
    seed (700 + sim), so finished simulations are kept and not refitted. The same seeds give
    the same transplanted labels for every learner (null.json: HGB; null_lr.json: LR)."""
    from fa_stats import Boot
    tbin = np.minimum((F.t_elapsed.to_numpy() // 30).astype(int), 9)
    cell = (F.campaign + "|" + F.game).to_numpy(); ctx = np.char.add(np.char.add(cell.astype(str), "|"), tbin.astype(str))
    D = domains(F); FS = feature_sets(F)
    fsets = lambda sim: ("CTX", "MOT+TIME", "SELFZ+TIME") if sim < 40 else ("MOT+TIME", "SELFZ+TIME")
    per_sim = lambda sim: 2 * 3 * len(fsets(sim))        # labels x protocols x feature sets
    old = json.load(open(R / fname)) if (R / fname).exists() else []
    done = {s for s in {r["sim"] for r in old} if sum(r["sim"] == s for r in old) == per_sim(s)}
    res = [r for r in old if r["sim"] in done]
    for sim in (range(n_sim) if sims is None else sims):   # sims: a sub-range, to split the work over processes
        if sim in done:
            continue
        rng = np.random.default_rng(700 + sim)
        for lab in (1, 2):
            ysim = transplant_labels(F, (F.severity.to_numpy() >= lab).astype(int), rng)
            for pr in ("within_seated", "within_standing", "pooled"):
                src, tgt = D[pr]
                bt = Boot(F.group_id.to_numpy()[tgt], b=0)
                for fs in fsets(sim):      # CTX only in the first 40 (its stratified AUC is 0.5 by construction)
                    Pm, _, Fo = transfer(F, FS[fs], ysim, src, tgt, MODELS[model], inner=False)
                    ov, cx = [], []
                    for r in range(Pm.shape[0]):
                        v = Fo[r, tgt] >= 0; fo = Fo[r, tgt].astype(str)
                        ov.append(bt.auc(ysim[tgt], Pm[r, tgt], fo, v)[0])
                        cx.append(bt.auc(ysim[tgt], Pm[r, tgt], np.char.add(np.char.add(fo, "|"), ctx[tgt]), v)[0])
                    res.append({"sim": sim, "label": f"sev{lab}", "protocol": pr, "features": fs,
                                "overall_auc": float(np.nanmean(ov)), "context_stratified_auc": float(np.nanmean(cx))})
        print(f"null ({model}) sim {sim} done", flush=True)
        json.dump(res, open(R / fname, "w"), indent=1)


NULL_SPLIT = int(os.environ.get("FA_NULL_SPLIT", "345"))


def merge_null(R, main, part, n_sim):
    """Merge the permutations computed by a second process into the main null file; every
    permutation must be complete and present exactly once."""
    a, b = json.load(open(R / main)), json.load(open(R / part))
    sims_a, sims_b = {r["sim"] for r in a}, {r["sim"] for r in b}
    assert not (sims_a & sims_b), "a permutation was computed twice"
    res = sorted(a + b, key=lambda r: (r["sim"], r["label"], r["protocol"], r["features"]))
    per = lambda s: 18 if s < 40 else 12
    cnt = {}
    for r in res:
        cnt[r["sim"]] = cnt.get(r["sim"], 0) + 1
    assert sorted(cnt) == list(range(n_sim)) and all(cnt[s] == per(s) for s in cnt), "incomplete permutations"
    json.dump(res, open(R / main, "w"), indent=1)
    print(f"null merged: {len(cnt)} permutations", flush=True)


def leak_mutation_test(raw):
    """The past-only check must FAIL on a deliberately leaky (centred-window) feature."""
    import fa_data
    orig = fa_data.session_features

    def leaky(g, window=30):
        f = orig(g, window)
        f["PlayerSpeed_mean"] = g.PlayerSpeed.astype(float).rolling(window, center=True, min_periods=1).mean().values
        return f
    fa_data.session_features = leaky
    try:
        fa_data.assert_no_leakage(raw, fa_data.build_features(raw), {})
        caught = False
    except AssertionError:
        caught = True
    finally:
        fa_data.session_features = orig
    assert caught, "leakage check did not catch a centred window"
    return caught


if __name__ == "__main__":
    RAWDIR, SELDIR, DATA, R = map(Path, sys.argv[1:5]); P = R / "preds"; P.mkdir(parents=True, exist_ok=True)
    stages = sys.argv[5:] or ["checks", "main", "ctx", "mitig", "cell", "control", "rowwise", "vrsq", "sens",
                              "logo_ctx", "null", "null_lr", "posctl", "es_check"]
    raw, S = load_archive(RAWDIR)
    cur = set(curated_sessions(SELDIR, RAWDIR).values()); assert len(cur) == 35
    raw["curated"] = raw.session_id.isin(cur)
    F = add_context(build_features(raw)); F["curated"] = raw["curated"].to_numpy()
    if "checks" in stages:
        g = F.group_id.to_numpy()
        folds = {pr: [(tr, te) for _, _, tr, te in make_folds(g, src, tgt)] for pr, (src, tgt) in domains(F).items()}
        checks = assert_no_leakage(raw, F, folds)
        checks["n_folds_checked"] = int(sum(len(v) for v in folds.values()))
        checks["mutation_test_centred_window_caught"] = leak_mutation_test(raw)
        checks["curated_release_sessions_found_in_raw"] = len(cur)
        json.dump(checks, open(R / "leakage_checks.json", "w"), indent=1); print(checks, flush=True)
    for st in stages:
        t0 = time.time()
        if st == "main": stage_main(F, R, P)
        elif st == "ctx": stage_ctx(F, R, P)
        elif st == "mitig": stage_mitig(F, R, P)
        elif st == "cell": stage_cell(F, R, P)
        elif st == "control": stage_control(F, R, P)
        elif st == "rowwise": stage_rowwise(F, R, P)
        elif st == "vrsq": stage_vrsq(F, R, P)
        elif st == "sens": stage_sens(F, R, P, raw)
        elif st == "logo_ctx": stage_logo_ctx(F, R, P)
        elif st == "null": stage_null(F, R, P)
        elif st == "null_lr": stage_null(F, R, P, model="logreg", fname="null_lr.json")
        elif st == "null500": stage_null(F, R, P, n_sim=500)
        # the same 500 permutations split over two processes (identical seeds, so identical results),
        # then merged into null.json; used because the revision had two CPU cores
        elif st == "null500_lo": stage_null(F, R, P, sims=range(0, NULL_SPLIT))
        elif st == "null500_hi": stage_null(F, R, P, sims=range(NULL_SPLIT, 500), fname="null_hi.json")
        elif st == "null_merge": merge_null(R, "null.json", "null_hi.json", 500)
        elif st == "null_lr500": stage_null(F, R, P, n_sim=500, model="logreg", fname="null_lr.json")
        elif st == "hgbreg": stage_hgbreg(F, R, P)
        elif st == "posctl_reg": stage_posctl(F, R, P, models=("hgbreg",), fname="posctl_reg.json")
        elif st == "posctl": stage_posctl(F, R, P)
        elif st == "es_check": stage_es_check(F, R, P)
        else:
            continue
        print(f"== stage {st} done in {time.time()-t0:.0f}s", flush=True)
