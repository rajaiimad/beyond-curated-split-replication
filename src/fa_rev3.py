"""
fa_rev3.py -- revision 3 (docs/ANALYSIS_PLAN_REVISION3.md): context-offset learners, their
controls, and the new estimands. Every choice below was fixed in the plan before any
real-label result of these learners was computed.

Learners (no hyperparameter is tuned; everything is fixed in advance):
  ctxb    the CONTEXT BASELINE. Logistic regression (C = 1, class-balanced) on exposure time,
          its square, the game and campaign indicators, their product (one coefficient per
          game x campaign cell) and the products of time with game and with campaign.
          Fitted on the training records of each outer fold.
  lroff   logistic regression on telemetry features WITH THE BASELINE'S LOGIT AS A FIXED OFFSET:
              logit p = offset(context, time) + b + x' beta,
          with the same penalty (C = 1, class-balanced weights) as the study's LR. The baseline is
          fitted first, on the same training records, and is then held fixed, so beta can only
          describe what the telemetry adds to context and exposure time.
  xgboff  gradient-boosted trees (xgboost, fixed settings XGB_PARAMS) started from the same
          offset (base_margin); the trees can only add to the baseline.
Telemetry gain = AUC(ctxb + telemetry) - AUC(ctxb) on identical records, folds and models.

Estimands: overall within-fold AUC; context-stratified AUC (fold x campaign x game x 30-s bin);
within-session AUC (fold x session: positive and negative records of the SAME session only).

Usage: python fa_rev3.py <RAW_DATABASE dir> <DATABASE_SELECTED_DATA dir> <data dir> <results dir> [stage ...]
stages: decode main control posctl null_lr null_xgb within sens_group hgb_control
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from scipy.stats import norm, rankdata
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from fa_data import add_context, build_features, curated_sessions, feature_sets, load_archive  # noqa: E402
import fa_run  # noqa: E402
from fa_run import BUDGET, R_REP, make_folds, domains, summarise, transplant_labels, spike_feature  # noqa: E402

NJOBS = int(os.environ.get("FA_NJOBS", "4"))
C_LR = 1.0
PEN_STRONG = 0.1     # addendum 3, P1: 'lroffs' uses C = 0.1 x training sessions / training records
XGB_PARAMS = dict(objective="binary:logistic", eta=0.05, max_depth=2, min_child_weight=25.0, reg_lambda=10.0,
                  subsample=1.0, colsample_bytree=1.0, tree_method="hist", max_bin=256, nthread=1, seed=0,
                  verbosity=0)
XGB_ROUNDS = 150
PROTOCOLS = fa_run.CTX_PROTOCOLS + fa_run.CELL_PROTOCOLS            # 10 + 4 single game x campaign cells
NULL_PROTOCOLS = ["within_seated", "within_standing", "pooled"] + fa_run.CELL_PROTOCOLS
CONTROL_PROTOCOLS = ["within_seated", "within_standing", "pooled"] + fa_run.CELL_PROTOCOLS
STRATUM_S = 30


# ----------------------------------------------------------------------------- design matrices
def baseline_matrix(F: pd.DataFrame) -> np.ndarray:
    """Context and exposure-time design of the baseline (raw; standardised inside each fold)."""
    t = F.t_elapsed.to_numpy(float) / 60.0
    r = F.ctx_race.to_numpy(float); s = F.ctx_standing.to_numpy(float)
    return np.c_[t, t ** 2, r, s, r * s, r * t, s * t]


def tele_cols(F: pd.DataFrame, kind: str) -> list:
    fs = feature_sets(F)
    if kind == "MOT":
        return fs["MOT"]
    if kind == "MOTNR":                       # addendum B2: MOT without the 14 rotation-derived features
        return [c for c in fs["MOT"] if c.startswith(("PlayerSpeed", "PlayerAcceleration", "roi_"))]
    if kind == "SELFZ":
        return [c for c in fs["SELFZ+TIME"] if c != "t_elapsed"]
    raise KeyError(kind)


def _std(Xtr, *Xs):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    return [(X - mu) / sd for X in (Xtr,) + Xs]


def _balanced_w(y):
    n, npos = len(y), y.sum()
    return np.where(y == 1, n / (2 * npos), n / (2 * (n - npos)))


# ----------------------------------------------------------------------------- learners
def fit_baseline(Btr, ytr):
    """Returns a function raw-design -> baseline logit, fitted on (Btr, ytr) (same objective as
    fit_offset_lr with a zero offset, on standardised columns)."""
    mu = Btr.mean(0); sd = Btr.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    b, be = fit_offset_lr((Btr - mu) / sd, ytr, np.zeros(len(ytr)))
    return lambda B: b + ((B - mu) / sd) @ be


def fit_offset_lr(X, y, off, C=C_LR, tol=1e-9, max_iter=100):
    """argmin  C * sum_i w_i log(1 + exp(-s_i (off_i + b + x_i'beta)))  +  ||beta||^2 / 2
    (scikit-learn's L2 logistic-regression objective, class-balanced weights, unpenalised
    intercept), with the offset held fixed. Exact Newton steps with step halving; the
    objective is strictly convex, so the solution is unique. Returns (b, beta)."""
    w = _balanced_w(y); s = 2.0 * y - 1.0
    Xa = np.c_[np.ones(len(y)), X]; pen = np.r_[0.0, np.ones(X.shape[1])]
    obj = lambda th: C * float(w @ np.logaddexp(0, -s * (off + Xa @ th))) + 0.5 * float((pen * th) @ th)
    th = np.zeros(Xa.shape[1]); f0 = obj(th)
    for _ in range(max_iter):
        p = expit(off + Xa @ th)
        g = C * (Xa.T @ (w * (p - y))) + pen * th
        H = C * (Xa.T * (w * p * (1 - p))) @ Xa + np.diag(pen)
        step = np.linalg.solve(H, g); t = 1.0
        while True:
            f1 = obj(th - t * step)
            if f1 <= f0 or t < 1e-10:
                break
            t /= 2
        th = th - t * step; done = abs(f0 - f1) <= tol * max(1.0, abs(f0)) and np.max(np.abs(t * step)) < 1e-8
        f0 = f1
        if done:
            break
    return th[0], th[1:]


def fit_offset_xgb(X, y, off):
    import xgboost as xgb
    d = xgb.DMatrix(X, label=y, weight=_balanced_w(y), base_margin=off)
    return xgb.train(XGB_PARAMS, d, num_boost_round=XGB_ROUNDS)


def _fit_score(kind, Xtr, Btr, ytr, Xte, Bte, C=C_LR):
    """Fit one learner on training records; return (test probability, test baseline logit).
    'lroffp' (addendum B5) is LR+offset with the penalty scale C passed in (training sessions /
    training records); 'lroffs' (addendum 3, P1) the same with C ten times smaller; the baseline is
    the same in every learner."""
    base = fit_baseline(Btr, ytr); otr, ote = base(Btr), base(Bte)
    if kind == "ctxb":
        z = ote
    elif kind in ("lroff", "lroffp", "lroffs"):
        Xs, Xt = _std(Xtr, Xte); b, be = fit_offset_lr(Xs, ytr, otr, C=C if kind in ("lroffp", "lroffs") else C_LR)
        z = ote + b + Xt @ be
    elif kind == "xgboff":
        import xgboost as xgb
        m = fit_offset_xgb(Xtr, ytr, otr)
        z = m.predict(xgb.DMatrix(Xte, base_margin=ote), output_margin=True)
    else:
        raise KeyError(kind)
    return 1 / (1 + np.exp(-z)), ote


def _pen(sess, idx):
    """Addendum B5: penalty scale that weighs one training session, not one record, against the
    ridge term (C = training sessions / training records)."""
    return len(np.unique(sess[idx])) / len(idx)


def _one_fold(kind, X, B, y, groups, tr, te, inner, sess=None):
    if len(te) == 0 or len(set(y[tr])) < 2:
        return None
    assert not (set(groups[tr]) & set(groups[te]))              # leakage guard, every fold
    cp = (lambda idx: _pen(sess, idx)) if kind == "lroffp" else \
         (lambda idx: PEN_STRONG * _pen(sess, idx)) if kind == "lroffs" else (lambda idx: C_LR)
    p, off = _fit_score(kind, X[tr] if X is not None else None, B[tr], y[tr],
                        X[te] if X is not None else None, B[te], C=cp(tr))
    thr = np.nan
    n_tr_groups = len(np.unique(groups[tr]))
    if inner and n_tr_groups >= 2:                               # operating threshold on training groups only
        s = np.full(len(tr), np.nan)                             # (3 inner folds; fewer if fewer groups)
        for a, b in GroupKFold(min(3, n_tr_groups)).split(tr, groups=groups[tr]):
            if len(set(y[tr[a]])) == 2:
                s[b] = _fit_score(kind, X[tr[a]] if X is not None else None, B[tr[a]], y[tr[a]],
                                  X[tr[b]] if X is not None else None, B[tr[b]], C=cp(tr[a]))[0]
        neg = s[(y[tr] == 0) & ~np.isnan(s)]
        thr = float(np.quantile(neg, 1 - BUDGET)) if len(neg) else 0.5
    return p, thr, off


def transfer(F, kind, cols, y_train, src, tgt, inner=True, n_jobs=None, groups=None):
    """Same folds as fa_run.transfer. Returns P, THR, FOLD, OFF (R x n; nan / -1 outside)."""
    X = F[cols].to_numpy(float) if cols else None; B = baseline_matrix(F)
    groups = F.group_id.to_numpy() if groups is None else groups; n = len(F)
    folds = make_folds(groups, src, tgt)
    n_jobs = NJOBS if n_jobs is None else n_jobs
    sess = F.session_id.to_numpy()
    out = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(kind, X, B, y_train, groups, tr, te, inner, sess)
                                  for _, _, tr, te in folds)
    P = np.full((R_REP, n), np.nan); THR = np.full((R_REP, n), np.nan)
    OFF = np.full((R_REP, n), np.nan); FOLD = np.full((R_REP, n), -1, np.int8)
    for (r, f, tr, te), res in zip(folds, out):
        if res is not None:
            P[r, te], THR[r, te], OFF[r, te] = res[0], res[1], res[2]; FOLD[r, te] = f
    return P, THR, FOLD, OFF


# ----------------------------------------------------------------------------- fast point metrics
def strat_auc(y, s, strata):
    """Pair-weighted AUC over positive-negative pairs of the same stratum (ties 1/2); nan if none.
    Identical to fa_stats.Boot.auc(...)[0] (checked in tests)."""
    o = np.argsort(strata, kind="stable"); y, s, st = y[o], s[o], strata[o]
    cut = np.flatnonzero(st[1:] != st[:-1]) + 1; num = den = 0.0
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(st)]):
        yy = y[a:b]; npos = int(yy.sum()); nneg = len(yy) - npos
        if npos and nneg:
            r = rankdata(s[a:b]); num += r[yy == 1].sum() - npos * (npos + 1) / 2; den += npos * nneg
    return num / den if den else np.nan


def ctx_strata(F):
    tbin = np.minimum((F.t_elapsed.to_numpy() // STRATUM_S).astype(int), 300 // STRATUM_S - 1)
    return (F.campaign + "|" + F.game + "|" + pd.Series(tbin, index=F.index).astype(str)).to_numpy().astype(str)


def falls_mask(F, y):
    """Addendum B3: records of sessions whose labels are not non-decreasing in time (the rating also
    falls at least once). In a session whose label is a single upward step, exposure time already
    ranks every within-session pair correctly, so telemetry cannot add there."""
    keep = np.zeros(len(F), bool); t = F.t_elapsed.to_numpy()
    for _, idx in F.groupby("session_id").indices.items():
        o = idx[np.argsort(t[idx], kind="stable")]
        keep[idx] = bool((np.diff(y[o]) < 0).any())
    return keep


def within_nm(y, P, FOLD, sess, keep):
    """Within-session AUC on the records of 'falls' sessions only, mean over repeats."""
    ws = []
    for r in range(P.shape[0]):
        v = (FOLD[r] >= 0) & ~np.isnan(P[r]) & keep
        ws.append(strat_auc(y[v], P[r][v], np.char.add(np.char.add(FOLD[r][v].astype(str), "|"), sess[v])))
    return float(np.nanmean(ws)) if np.isfinite(ws).any() else float("nan")


def point_metrics(y, P, FOLD, ctx, sess):
    """Overall (within-fold), context-stratified and within-session AUC, mean over repeats."""
    ov, cx, ws = [], [], []
    for r in range(P.shape[0]):
        v = (FOLD[r] >= 0) & ~np.isnan(P[r]); f = FOLD[r][v].astype(str)
        yy, pp = y[v], P[r][v]
        ov.append(strat_auc(yy, pp, f))
        cx.append(strat_auc(yy, pp, np.char.add(np.char.add(f, "|"), ctx[v])))
        ws.append(strat_auc(yy, pp, np.char.add(np.char.add(f, "|"), sess[v])))
    return float(np.nanmean(ov)), float(np.nanmean(cx)), float(np.nanmean(ws))


# ----------------------------------------------------------------------------- grid runner
def cols_for(F, fs):
    """'CTXB' -> []; 'CTXB+MOT+SPK' -> MOT columns + the planted channel, etc."""
    parts = fs.split("+"); assert parts[0] == "CTXB", fs; cols = []
    for k in parts[1:]:
        cols += ["spk"] if k == "SPK" else tele_cols(F, k)
    return cols


def run_grid(F, protocols, fsets, learners, labels, preds_dir, tag="", inner=True, y_train_fn=None, quiet=False):
    """Saves <tag>sev<lab>__<protocol>__<fs>__<learner>.npz (p, thr, fold, off); resumes from
    finished files. The baseline (fs 'CTXB') is learner-independent and saved once as 'ctxb'."""
    res = []; D = domains(F); sess = F.session_id.to_numpy().astype(str); ctx = ctx_strata(F)
    g = F.group_id.to_numpy()
    for lab in labels:
        y = (F.severity.to_numpy() >= lab).astype(int)
        y_train = y_train_fn(y) if y_train_fn else y
        for pr in protocols:
            src, tgt = D[pr]
            for fs in fsets:
                for ln in (["ctxb"] if fs == "CTXB" else learners):
                    key = f"{tag}sev{lab}__{pr}__{fs}__{ln}"; f = preds_dir / f"{key}.npz"; t0 = time.time()
                    if f.exists():
                        z = np.load(f); P, THR, FOLD, OFF = z["p"], z["thr"], z["fold"], z["off"]
                    else:
                        P, THR, FOLD, OFF = transfer(F, ln, cols_for(F, fs), y_train, src, tgt, inner=inner)
                        tmp = f.with_suffix(".tmp.npz"); np.savez_compressed(tmp, p=P, thr=THR, fold=FOLD, off=OFF)
                        tmp.replace(f)
                    if not (FOLD[:, tgt] >= 0).any(axis=1).all():     # a repeat without any scored record:
                        s = {"roc_auc": float("nan"), "n_rows": int(tgt.sum()), "estimable": False}   # e.g. one group only
                        ov = cx = ws = float("nan")
                    else:
                        s = summarise(y[tgt], P[:, tgt], THR[:, tgt], FOLD[:, tgt], g[tgt])
                        ov, cx, ws = point_metrics(y[tgt], P[:, tgt], FOLD[:, tgt], ctx[tgt], sess[tgt])
                    s.update(key=key, label=f"sev>={lab}", protocol=pr, features=fs, model=ln,
                             ctx_auc=cx, within_session_auc=ws, secs=round(time.time() - t0, 1))
                    res.append(s)
                    if not quiet:
                        print(f"{key:60s} AUC {s['roc_auc']:.3f} ctx {cx:.3f} sess {ws:.3f} ({s['secs']}s)", flush=True)
    return res


def shifted_labels_fn(F):
    """The revision-2 label-shift control: each session's training labels circularly shifted by
    an offset uniform in [n/4, 3n/4] records (seed 1, sessions in sorted order) -- identical
    offsets to fa_run.stage_control."""
    rng = np.random.default_rng(1)
    idx_of = F.groupby("session_id").indices
    shift = {sid: int(rng.integers(len(idx) // 4, max(len(idx) // 4 + 1, (3 * len(idx)) // 4 + 1)))
             for sid, idx in sorted(idx_of.items())}

    def shifted(y):
        ys = y.copy()
        for sid, idx in idx_of.items():
            ys[idx] = np.roll(y[idx], shift[sid])
        return ys
    return shifted


# ----------------------------------------------------------------------------- stage 1: context decoding
def stage_decode(F, R, P, fsets=("MOT", "SELFZ"), fname="rev3_decode.json"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from fa_stats import Boot, ci
    learners = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
                "hgb": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=0)}
    g = F.group_id.to_numpy(); allm = np.ones(len(F), bool); folds = make_folds(g, allm, allm)
    cell = (F.campaign + "|" + F.game).to_numpy(); cells = sorted(np.unique(cell))
    targets = {"game_race": F.ctx_race.to_numpy().astype(int), "campaign_standing": F.ctx_standing.to_numpy().astype(int)}
    sess = F.session_id.to_numpy(); bt = Boot(g); out = {"folds": "pooled protocol (grouped 5-fold x 5 repeats)", "runs": {}}
    for fsn in fsets:
        X = F[tele_cols(F, fsn)].to_numpy(float)
        for ln, mk in learners.items():
            # one multiclass fit per fold gives the game, campaign and cell scores
            Pc = np.full((R_REP, len(F), len(cells)), np.nan); FO = np.full((R_REP, len(F)), -1)
            yc = np.searchsorted(cells, cell)

            def fit(tr, te):
                m = mk().fit(X[tr], yc[tr]); return m.predict_proba(X[te])
            res = Parallel(n_jobs=NJOBS)(delayed(fit)(tr, te) for _, _, tr, te in folds)
            for (r, f, tr, te), pr in zip(folds, res):
                Pc[r, te] = pr; FO[r, te] = f
            ci_ = {c: i for i, c in enumerate(cells)}
            scores = {"game_race": Pc[..., [ci_[c] for c in cells if c.endswith("Race")]].sum(-1),
                      "campaign_standing": Pc[..., [ci_[c] for c in cells if c.startswith("standing")]].sum(-1)}
            for c in cells:
                targets[f"cell_{c}"] = (cell == c).astype(int); scores[f"cell_{c}"] = Pc[..., ci_[c]]
            for tn, sc in scores.items():
                y = targets[tn]; arr = []
                for r in range(R_REP):
                    v = FO[r] >= 0; arr.append(bt.auc(y, sc[r], FO[r].astype(str), v))
                a = np.nanmean(arr, axis=0)
                # session level: mean score per session, pairs within a fold model
                sl = []
                for r in range(R_REP):
                    d = pd.DataFrame({"s": sess, "p": sc[r], "y": y, "f": FO[r]}).groupby("s").agg(p=("p", "mean"), y=("y", "first"), f=("f", "first"))
                    sl.append(strat_auc(d.y.to_numpy(), d.p.to_numpy(), d.f.to_numpy().astype(str)))
                out["runs"][f"{fsn}__{ln}__{tn}"] = {"auc": float(a[0]), "ci": ci(a), "session_auc": float(np.mean(sl))}
                print(f"decode {fsn:5s} {ln:6s} {tn:28s} AUC {a[0]:.3f} {ci(a)} session {np.mean(sl):.3f}", flush=True)
            mac = np.mean([out["runs"][f"{fsn}__{ln}__cell_{c}"]["auc"] for c in cells])
            out["runs"][f"{fsn}__{ln}__cell_macro"] = {"auc": float(mac)}
    json.dump(out, open(R / fname, "w"), indent=1)


# ----------------------------------------------------------------------------- stage 2: main runs
FS_MAIN = ["CTXB", "CTXB+MOT", "CTXB+SELFZ"]


def stage_main(F, R, P):
    res = run_grid(F, PROTOCOLS, FS_MAIN, ["lroff", "xgboff"], [1, 2], P)
    json.dump(res, open(R / "rev3_main.json", "w"), indent=1)


# ----------------------------------------------------------------------------- stage 3: controls
def stage_control(F, R, P):
    sh = shifted_labels_fn(F)
    res = run_grid(F, CONTROL_PROTOCOLS, ["CTXB", "CTXB+MOT"], ["lroff", "xgboff"], [1, 2], P,
                   tag="r3control_", inner=False, y_train_fn=sh)
    json.dump(res, open(R / "rev3_control.json", "w"), indent=1)


def stage_hgb_control(F, R, P):
    """Default HGB (MOT, MOT+TIME) and HGB-reg (MOT+TIME) under the label-shift control, in the
    7 control protocols. Uses fa_run's learners and keys (tag control_), so the default-HGB runs
    of revision 2 are reused where they exist."""
    sh = shifted_labels_fn(F)
    res = fa_run.run_grid(F, CONTROL_PROTOCOLS, ["MOT", "MOT+TIME"], ["hgb"], [1, 2], P, tag="control_",
                          inner=False, y_train_fn=sh)
    res += fa_run.run_grid(F, CONTROL_PROTOCOLS, ["MOT+TIME"], ["hgbreg"], [1, 2], P, tag="control_",
                           inner=False, y_train_fn=sh)
    # the same models with true labels in the cells (revision 2 has hgb/hgbreg MOT+TIME only there)
    res += fa_run.run_grid(F, fa_run.CELL_PROTOCOLS, ["MOT"], ["hgb"], [1, 2], P)
    res += fa_run.run_grid(F, fa_run.CELL_PROTOCOLS + ["within_seated", "within_standing", "pooled"], ["MOT+TIME"],
                           ["hgbreg"], [1, 2], P)
    json.dump(res, open(R / "rev3_hgb_control.json", "w"), indent=1)


POS_LEVELS = (0.55, 0.60, 0.65, 0.70, 0.80)


def stage_posctl(F, R, P):
    out = {"runs": [], "planted": {}}
    for target_auc in POS_LEVELS:
        d = float(np.sqrt(2) * norm.ppf(target_auc)); tag = f"r3pos{int(round(100 * target_auc))}_"
        for lab in (1, 2):
            y = (F.severity.to_numpy() >= lab).astype(int)
            spk = spike_feature(F, y, d, seed=900 + lab)
            Fs = F.assign(spk=spk)
            for pr in NULL_PROTOCOLS:
                t = domains(F)[pr][1]
                out["planted"][f"{tag}sev{lab}__{pr}"] = {"d": d, "target_auc": target_auc,
                                                          "channel_ctx_auc_on_target": float(strat_auc(y[t], spk[t], ctx_strata(F)[t]))}
            out["runs"] += run_grid(Fs, NULL_PROTOCOLS, ["CTXB", "CTXB+SPK"], ["lroff"], [lab], P, tag=tag, inner=False)
            out["runs"] += run_grid(Fs, NULL_PROTOCOLS, ["CTXB+MOT+SPK"], ["lroff", "xgboff"], [lab], P, tag=tag, inner=False)
    json.dump(out, open(R / "rev3_posctl.json", "w"), indent=1)


# ----------------------------------------------------------------------------- stage 4: event-locked
def onset_sessions(F, y, mask, lead=30.0, lag=30.0):
    """{session: tau} for sessions in mask whose label first switches 0 -> 1 at tau >= lead with
    at least lag seconds of record after tau."""
    out = {}
    t = F.t_elapsed.to_numpy()
    for sid, idx in F.groupby("session_id").indices.items():
        if not mask[idx[0]]:
            continue
        yy, tt = y[idx], t[idx]
        on = np.flatnonzero((yy[1:] == 1) & (yy[:-1] == 0))
        if len(on) and yy[0] == 0:
            tau = tt[on[0] + 1]
            if tau >= lead and tt[-1] >= tau + lag:
                out[sid] = tau
    return out


def event_locked(F, y, tele, mask, win=30.0):
    """Difference-in-differences of the telemetry part of the score around the first onset.
    tele: per-record telemetry part (nan where unscored). Returns (statistic, per-session values,
    session ids)."""
    t = F.t_elapsed.to_numpy(); cell = (F.campaign + "|" + F.game).to_numpy()
    idx_of = F.groupby("session_id").indices; on = onset_sessions(F, y, mask, win, win)
    vals, sids = [], []
    for sid, tau in on.items():
        idx = idx_of[sid]
        def d_of(ix):
            tt, zz = t[ix], tele[ix]
            b = zz[(tt >= tau - win) & (tt < tau)]; a = zz[(tt >= tau) & (tt < tau + win)]
            return np.nanmean(a) - np.nanmean(b) if np.isfinite(a).any() and np.isfinite(b).any() else np.nan
        d = d_of(idx); ctrl = []
        for s2, ix2 in idx_of.items():
            if s2 == sid or cell[ix2[0]] != cell[idx[0]] or not mask[ix2[0]]:
                continue
            tt2 = t[ix2]; w = (tt2 >= tau - win) & (tt2 < tau + win)
            if w.sum() < win:                           # needs at least ~half of the 2*win window recorded
                continue
            if np.unique(y[ix2][w]).size == 1:          # label constant in the window
                ctrl.append(d_of(ix2))
        ctrl = [c for c in ctrl if np.isfinite(c)]
        if np.isfinite(d) and ctrl:
            vals.append(d - np.mean(ctrl)); sids.append(sid)
    return (float(np.mean(vals)) if vals else np.nan), np.array(vals), sids


def tele_part(P, OFF):
    """Mean over repeats of logit(p) - offset (the telemetry part of an offset model's score)."""
    Pc = np.clip(P, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(Pc / (1 - Pc)) - OFF
    return np.nanmean(z, axis=0)


def stage_within(F, R, P):
    """Event-locked analysis (real labels) and the pair decomposition of the context strata."""
    out = {"event_locked": {}, "pair_share_same_session": {}}
    D = domains(F); sess = F.session_id.to_numpy(); ctx = ctx_strata(F); g = F.group_id.to_numpy()
    for lab in (1, 2):
        y = (F.severity.to_numpy() >= lab).astype(int)
        for fs, ln in (("CTXB+MOT", "lroff"), ("CTXB+SELFZ", "lroff"), ("CTXB+MOT", "xgboff")):
            z = np.load(P / f"sev{lab}__pooled__{fs}__{ln}.npz")
            stat, vals, sids = event_locked(F, y, tele_part(z["p"], z["off"]), np.ones(len(F), bool))
            # group bootstrap over onset sessions
            rng = np.random.default_rng(0); gs = np.array([g[F.session_id.to_numpy() == s][0] for s in sids])
            ug = np.unique(gs); bs = []
            for _ in range(10_000):
                pick = rng.choice(ug, len(ug)); v = np.concatenate([vals[gs == k] for k in pick])
                bs.append(v.mean())
            out["event_locked"][f"sev{lab}__pooled__{fs}__{ln}"] = {"did": stat, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                                     "n_onset_sessions": len(vals), "n_groups": int(len(ug))}
            print(f"event-locked sev{lab} {fs} {ln}: {stat:.3f} n={len(vals)}", flush=True)
        # share of context-stratum pairs that are same-session pairs, per protocol target
        for pr in PROTOCOLS:
            tgt = D[pr][1]; st = ctx[tgt]; ss = sess[tgt]; yy = y[tgt]; tot = same = 0
            d = pd.DataFrame({"st": st, "s": ss, "y": yy})
            for _, q in d.groupby("st"):
                pc = q[q.y == 1].s.value_counts(); nc = q[q.y == 0].s.value_counts()
                tot += pc.sum() * nc.sum(); same += sum(pc[k] * nc.get(k, 0) for k in pc.index)
            out["pair_share_same_session"][f"sev{lab}__{pr}"] = {"share": same / tot if tot else np.nan,
                                                                 "pairs": int(tot), "same_session_pairs": int(same),
                                                                 "sessions_both_classes": int(d.groupby("s").y.agg(lambda v: 0 < v.mean() < 1).sum())}
    json.dump(out, open(R / "rev3_within.json", "w"), indent=1)


# ----------------------------------------------------------------------------- stage 3: nulls
def transplant_labels_block(F, y, rng):
    """Addendum B6: group-block label transplant. Within each game x campaign cell the groups are
    permuted (a derangement over groups), so all sessions of a group receive sequences from one
    other group; recipient and donor sessions are matched by length rank (donor sessions re-used
    cyclically when the donor group has fewer). Aligned by elapsed time as in transplant_labels."""
    t = F.t_elapsed.to_numpy(); cell = (F.campaign + "|" + F.game).to_numpy()
    idx = F.groupby("session_id").indices; grp = F.groupby("session_id").group_id.first()
    out = np.empty_like(y)
    for c in np.unique(cell):
        sids = sorted(F.session_id[cell == c].unique())
        by_g = {}
        for s_ in sids:
            by_g.setdefault(grp[s_], []).append(s_)
        gs = sorted(by_g)
        for _ in range(10_000):
            perm = list(rng.permutation(gs))
            if all(a != b for a, b in zip(gs, perm)):
                break
        else:
            raise RuntimeError(f"no group derangement in cell {c}")
        for rg, dg in zip(gs, perm):
            rec = sorted(by_g[rg], key=lambda s_: (len(idx[s_]), s_))
            don = sorted(by_g[dg], key=lambda s_: (len(idx[s_]), s_))
            # match by length rank: spread the donors over the recipients in rank order
            for i, sid in enumerate(rec):
                d = don[min(len(don) - 1, i * len(don) // len(rec))] if len(rec) >= len(don) else don[i + (len(don) - len(rec))]
                a, b = idx[sid], idx[d]
                k = np.clip(np.searchsorted(t[b], t[a], side="right") - 1, 0, len(b) - 1)
                out[a] = y[b][k]
    return out


def _null_one(F, sim, learner, fsets, protocols, with_el, transplant="session", seed_base=700):
    rng = np.random.default_rng(seed_base + sim); D = domains(F)
    sess = F.session_id.to_numpy().astype(str); ctx = ctx_strata(F); rows = []
    tl = transplant_labels if transplant == "session" else transplant_labels_block
    for lab in (1, 2):
        ysim = tl(F, (F.severity.to_numpy() >= lab).astype(int), rng)
        keep = falls_mask(F, ysim)
        for pr in protocols:
            src, tgt = D[pr]
            Pb, _, Fb, Ob = transfer(F, "ctxb", [], ysim, src, tgt, inner=False, n_jobs=1)
            mb = point_metrics(ysim[tgt], Pb[:, tgt], Fb[:, tgt], ctx[tgt], sess[tgt])
            nb = within_nm(ysim[tgt], Pb[:, tgt], Fb[:, tgt], sess[tgt], keep[tgt])
            for fs in fsets:
                Pm, _, Fm, Om = transfer(F, learner, cols_for(F, fs), ysim, src, tgt, inner=False, n_jobs=1)
                mm = point_metrics(ysim[tgt], Pm[:, tgt], Fm[:, tgt], ctx[tgt], sess[tgt])
                nm = within_nm(ysim[tgt], Pm[:, tgt], Fm[:, tgt], sess[tgt], keep[tgt])
                row = {"sim": sim, "label": f"sev{lab}", "protocol": pr, "features": fs, "learner": learner,
                       "overall_auc": mm[0], "context_stratified_auc": mm[1], "within_session_auc": mm[2],
                       "base_overall_auc": mb[0], "base_context_stratified_auc": mb[1], "base_within_session_auc": mb[2],
                       "gain_ctx": mm[1] - mb[1], "gain_session": mm[2] - mb[2],
                       "within_session_nm_auc": nm, "base_within_session_nm_auc": nb, "gain_session_nm": nm - nb,
                       "transplant": transplant}
                if with_el and pr == "pooled":
                    row["event_locked"] = event_locked(F, ysim, tele_part(Pm, Om), np.ones(len(F), bool))[0]
                rows.append(row)
    return rows


def stage_null(F, R, P, learner, n_sim, fsets, fname, el_upto=200, transplant="session", seed_base=700, protocols=None):
    """Resumable: finished permutations (by seed) are kept. protocols (addendum 3): default NULL_PROTOCOLS."""
    protocols = NULL_PROTOCOLS if protocols is None else protocols
    f = R / fname; old = json.load(open(f)) if f.exists() else []
    per = 2 * len(protocols) * len(fsets)
    cnt = {}
    for r in old:
        cnt[r["sim"]] = cnt.get(r["sim"], 0) + 1
    done = {s for s, c in cnt.items() if c == per}; res = [r for r in old if r["sim"] in done]
    todo = [s for s in range(n_sim) if s not in done]; t0 = time.time()
    for i in range(0, len(todo), 4 * NJOBS):
        chunk = todo[i:i + 4 * NJOBS]
        out = Parallel(n_jobs=NJOBS)(delayed(_null_one)(F, s, learner, fsets, protocols, s < el_upto,
                                                        transplant, seed_base) for s in chunk)
        for rows in out:
            res += rows
        tmp = f.with_suffix(".tmp"); json.dump(res, open(tmp, "w")); tmp.replace(f)
        print(f"null {learner}: {len({r['sim'] for r in res})}/{n_sim} permutations ({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- stage 5: grouping
def merged_groups(S):
    """Connected components of inferred groups whose 7 demographic answers differ in <= 1 item."""
    from fa_data import DEMO  # noqa: F401
    fp = S.groupby("group_id").fingerprint.first(); keys = fp.index.to_numpy(); parts = [v.split("|") for v in fp]
    parent = {k: k for k in keys}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if sum(a != b for a, b in zip(parts[i], parts[j])) <= 1:
                parent[find(keys[i])] = find(keys[j])
    roots = {k: find(k) for k in keys}; codes = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {k: codes[roots[k]] for k in keys}


def stage_sens_group(F, R, P, S):
    out = {"runs": [], "n_groups": {}}
    mg = merged_groups(S)
    variants = {"sess_": F.session_id.astype("category").cat.codes.to_numpy(),
                "mrg_": F.group_id.map(mg).to_numpy()}
    for tag, gg in variants.items():
        Fg = F.copy(); Fg["group_id"] = gg
        out["n_groups"][tag] = {"all": int(len(np.unique(gg))),
                                "curated": int(len(np.unique(gg[F.curated.to_numpy().astype(bool)]))),
                                **{f"{c}|{gm}": int(len(np.unique(gg[(F.campaign == c).to_numpy() & (F.game == gm).to_numpy()])))
                                   for c in ("seated", "standing") for gm in ("Flight", "Race")},
                                "in_both_campaigns": int((Fg.groupby("group_id").campaign.nunique() > 1).sum()),
                                "played_both_games": int((Fg.groupby("group_id").game.nunique() > 1).sum())}
        print(tag, out["n_groups"][tag], flush=True)
        out["runs"] += run_grid(Fg, PROTOCOLS, FS_MAIN, ["lroff", "xgboff"], [1, 2], P, tag=tag)
        out["runs"] += fa_run.run_grid(Fg, ["within_seated", "pooled"], ["MOT+TIME", "CTX+TIME"], ["hgbreg"], [1, 2], P, tag=tag)
    out["merged_map"] = {int(k): int(v) for k, v in mg.items()}
    json.dump(out, open(R / "rev3_sens_group.json", "w"), indent=1)


# ----------------------------------------------------------------------------- stage 7: VRSQ zeros
def stage_vrsq0(F, R, P, S, bb=10_000):
    """Referee Q3: VRSQ files whose pre- and post-exposure items are all zero, and Table IX
    without them (Spearman's rho, participant-group bootstrap as in fa_run.stage_vrsq)."""
    S = S.set_index("session_id"); sev = F.groupby("session_id").severity
    S["ing_max"] = sev.max(); S["ing_mean"] = sev.mean()
    S["ing_frac_ge1"] = sev.apply(lambda v: (v >= 1).mean()); S["ing_frac_ge2"] = sev.apply(lambda v: (v >= 2).mean())
    V = S.dropna(subset=["vrsq_post_total"]).copy(); V["vrsq_delta_total"] = V.vrsq_post_total - V.vrsq_pre_total
    V["all_zero"] = (V.vrsq_post_raw_sum == 0) & (V.vrsq_pre_total == 0)

    def spear(a, b, d, seed=0):
        rng = np.random.default_rng(seed); r = d[a].rank().corr(d[b].rank())
        gs = d.group_id.unique(); by = {k: d.index[d.group_id == k] for k in gs}; v = []
        for _ in range(bb):
            dd = d.loc[np.concatenate([by[k] for k in rng.choice(gs, len(gs))])]; v.append(dd[a].rank().corr(dd[b].rank()))
        return {"rho": float(r), "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))], "n": int(len(d))}
    out = {"n_with_vrsq": int(len(V)), "all_zero": int(V.all_zero.sum()), "post_zero": int((V.vrsq_post_raw_sum == 0).sum()),
           "all_zero_by_cell": {f"{c}|{g}": {"n": int(len(d)), "all_zero": int(d.all_zero.sum())} for (c, g), d in V.groupby(["campaign", "game"])},
           "all_zero_with_ingame_max_ge2": int((V.all_zero & (V.ing_max >= 2)).sum()),
           "ingame_max_ge2": int((V.ing_max >= 2).sum()), "correlations_without_all_zero": {}}
    W = V[~V.all_zero]
    for a in ("ing_max", "ing_mean", "ing_frac_ge1", "ing_frac_ge2"):
        for b in ("vrsq_post_total", "vrsq_delta_total"):
            out["correlations_without_all_zero"][f"{a}~{b}"] = spear(a, b, W)
    json.dump(out, open(R / "rev3_vrsq0.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k != "correlations_without_all_zero"},
          {k: round(v["rho"], 3) for k, v in out["correlations_without_all_zero"].items()}, flush=True)


# ----------------------------------------------------------------------------- main
# ----------------------------------------------------------------------------- addendum (pre-submission review)
FAMILY_PROTOCOLS = NULL_PROTOCOLS


def stage_rotation(F, R, P):
    """Addendum B2a: are the rotation fields logged? Sessions whose three rotation fields never
    change, per cell; MOT features constant within every session of a campaign; single-feature
    campaign AUC of the rotation variability features."""
    from sklearn.metrics import roc_auc_score
    rot = ["CameraRotationX", "CameraRotationY", "CameraRotationZ"]
    g = F.groupby("session_id")
    const = (g[rot].nunique() <= 1).all(axis=1)
    cell = g.campaign.first() + "|" + g.game.first()
    out = {"constant_rotation_sessions": {c: {"constant": int(const[cell == c].sum()), "sessions": int((cell == c).sum())}
                                          for c in sorted(cell.unique())}}
    mot = tele_cols(F, "MOT"); nr = tele_cols(F, "MOTNR"); rotf = [c for c in mot if c not in nr]
    camp = F.campaign.to_numpy()
    out["rotation_features"] = rotf; out["non_rotation_features"] = nr
    for cp in ("seated", "standing"):
        m = camp == cp; G = F[m].groupby("session_id")
        out[f"{cp}_features_zero_everywhere"] = [c for c in mot if np.allclose(F.loc[m, c].to_numpy(float), 0)]
        out[f"{cp}_features_constant_within_every_session"] = [c for c in mot if (G[c].nunique() <= 1).all()]
    y = (camp == "standing").astype(int)
    out["single_feature_campaign_auc"] = {c: float(max(roc_auc_score(y, F[c]), 1 - roc_auc_score(y, F[c]))) for c in mot}
    json.dump(out, open(R / "rev3_rotation.json", "w"), indent=1)
    print(out["constant_rotation_sessions"], flush=True)


def stage_pairs(F, R, P):
    """Addendum A3: composition of the context-stratified pairs on the estimand's own strata
    (fold x campaign x game x 30-s bin, per repeat, averaged): share of pairs from different sessions,
    and among those the share whose sessions belong to the same inferred group."""
    D = domains(F); sess = F.session_id.to_numpy(); grp = F.group_id.to_numpy(); ctx = ctx_strata(F); out = {}
    for lab in (1, 2):
        y = (F.severity.to_numpy() >= lab).astype(int)
        for pr in PROTOCOLS:
            tgt = D[pr][1]; z = np.load(P / f"sev{lab}__{pr}__CTXB__ctxb.npz"); FO = z["fold"][:, tgt]
            res = []
            for r in range(FO.shape[0]):
                v = FO[r] >= 0
                d = pd.DataFrame({"st": np.char.add(np.char.add(FO[r][v].astype(str), "|"), ctx[tgt][v]),
                                  "s": sess[tgt][v], "g": grp[tgt][v], "y": y[tgt][v]})
                tot = same_s = same_g = 0
                for _, q in d.groupby("st"):
                    pos, neg = q[q.y == 1], q[q.y == 0]
                    if not len(pos) or not len(neg):
                        continue
                    tot += len(pos) * len(neg)
                    ps, ns = pos.s.value_counts(), neg.s.value_counts()
                    same_s += sum(ps[k] * ns.get(k, 0) for k in ps.index)
                    pg, ng = pos.g.value_counts(), neg.g.value_counts()
                    same_g += sum(pg[k] * ng.get(k, 0) for k in pg.index)
                if tot:
                    res.append((tot, same_s, same_g))
            if not res:
                continue
            a = np.array(res, float)
            out[f"sev{lab}__{pr}"] = {"pairs_mean": float(a[:, 0].mean()),
                                      "share_between_sessions": float(np.mean(1 - a[:, 1] / a[:, 0])),
                                      "share_same_group_among_between": float(np.mean((a[:, 2] - a[:, 1]) / (a[:, 0] - a[:, 1])))}
    json.dump(out, open(R / "rev3_pairs.json", "w"), indent=1)
    print({k: round(v["share_between_sessions"], 3) for k, v in out.items() if "pooled" in k}, flush=True)


def stage_main_nr(F, R, P):
    """Addendum B2c: LR+offset with the 14 non-rotation telemetry features, family protocols."""
    res = run_grid(F, FAMILY_PROTOCOLS, ["CTXB+MOTNR"], ["lroff"], [1, 2], P, inner=False)
    json.dump(res, open(R / "rev3_main_nr.json", "w"), indent=1)


def stage_pen(F, R, P):
    """Addendum B5: LR+offset with the session-scaled penalty, family protocols, and its positive
    control with the original planted-channel seed."""
    res = run_grid(F, FAMILY_PROTOCOLS, ["CTXB+MOT"], ["lroffp"], [1, 2], P, inner=False)
    for target_auc in POS_LEVELS:
        d = float(np.sqrt(2) * norm.ppf(target_auc)); tag = f"r3pos{int(round(100 * target_auc))}_"
        for lab in (1, 2):
            y = (F.severity.to_numpy() >= lab).astype(int)
            Fs = F.assign(spk=spike_feature(F, y, d, seed=900 + lab))
            res += run_grid(Fs, FAMILY_PROTOCOLS, ["CTXB+MOT+SPK"], ["lroffp"], [lab], P, tag=tag, inner=False)
    json.dump(res, open(R / "rev3_pen.json", "w"), indent=1)


def _seed_one(F, s, Pdir, models):
    """Addendum B1: one planted-channel seed, all levels, labels, family protocols and models.
    Returns point gains (context-stratified, within-session) over the study's CTXB and the planted
    channel's realised AUCs; predictions are kept in memory only (see stage_posctl_seeds)."""
    D = domains(F); sess = F.session_id.to_numpy().astype(str); ctx = ctx_strata(F); rows, keep = [], {}
    for target_auc in POS_LEVELS:
        d = float(np.sqrt(2) * norm.ppf(target_auc))
        for lab in (1, 2):
            y = (F.severity.to_numpy() >= lab).astype(int)
            spk = spike_feature(F, y, d, seed=5000 + 100 * s + lab); Fs = F.assign(spk=spk)
            for pr in FAMILY_PROTOCOLS:
                src, tgt = D[pr]
                zb = np.load(Pdir / f"sev{lab}__{pr}__CTXB__ctxb.npz")
                mb = point_metrics(y[tgt], zb["p"][:, tgt], zb["fold"][:, tgt], ctx[tgt], sess[tgt])
                real_ctx = float(strat_auc(y[tgt], spk[tgt], ctx[tgt]))
                real_ws = float(strat_auc(y[tgt], spk[tgt], sess[tgt]))
                for fs, ln in models:
                    Pm, _, Fm, _ = transfer(Fs, ln, cols_for(Fs, fs), y, src, tgt, inner=False, n_jobs=1)
                    mm = point_metrics(y[tgt], Pm[:, tgt], Fm[:, tgt], ctx[tgt], sess[tgt])
                    key = f"s{s}__{int(round(100 * target_auc))}__sev{lab}__{pr}__{fs}__{ln}"
                    rows.append({"seed": s, "level": target_auc, "label": f"sev{lab}", "protocol": pr, "features": fs,
                                 "learner": ln, "gain_ctx": mm[1] - mb[1], "gain_session": mm[2] - mb[2],
                                 "channel_ctx_auc": real_ctx, "channel_session_auc": real_ws, "key": key})
                    keep[key] = (Pm[:, tgt].astype(np.float32), Fm[:, tgt])
    return rows, keep


def _perm_p(null_sorted, obs):
    """One-sided permutation p, (1 + #{null >= obs}) / (1 + n), for a sorted finite null."""
    if not np.isfinite(obs):
        return float("nan")
    return float((1 + len(null_sorted) - np.searchsorted(null_sorted, obs, "left")) / (1 + len(null_sorted)))


def stage_posctl_seeds(F, R, P, n_seeds=20):
    """Addendum B1. For every seed and level, the 14 planted gains of a model are scored against the
    label-transplant null of the real-telemetry model (one-sided, (1+k)/(1+n)); Holm within the
    14; a planted signal is declared if Holm p < 0.05 and the group-bootstrap interval of its gain
    (B = 10,000, nested weights for the within-session AUC) excludes 0. The interval is computed only
    where Holm p < 0.05 (otherwise the signal is not declared whatever the interval)."""
    from fa_stats import Boot, holm
    from fa_rev3_stats import arrays, ci as _ci
    nulls = {}
    for fname, ln in (("rev3_null_lroff.json", "lroff"), ("rev3_null_xgboff.json", "xgboff")):
        df = pd.DataFrame(json.load(open(R / fname))); df = df[df.features == "CTXB+MOT"]
        for (lab, pr), d in df.groupby(["label", "protocol"]):
            for stat in ("ctx", "session"):
                v = d[f"gain_{stat}"].to_numpy(); nulls[(ln, lab, pr, stat)] = np.sort(v[np.isfinite(v)])
    models = [("CTXB+MOT+SPK", "lroff"), ("CTXB+MOT+SPK", "xgboff")]
    out_f = R / "rev3_posctl_seeds.json"
    old = json.load(open(out_f)) if out_f.exists() else {"rows": []}
    done = {r["seed"] for r in old["rows"]}; rows = old["rows"]
    D = domains(F); sess = F.session_id.to_numpy().astype(str); ctx = ctx_strata(F); g = F.group_id.to_numpy()
    todo = [s for s in range(n_seeds) if s not in done]; t0 = time.time()
    for i in range(0, len(todo), NJOBS):
        chunk = todo[i:i + NJOBS]
        res = Parallel(n_jobs=NJOBS)(delayed(_seed_one)(F, s, P, models) for s in chunk)
        for rs, keep in res:
            df = pd.DataFrame(rs)
            for stat in ("ctx", "session"):
                df[f"p_{stat}"] = [_perm_p(nulls[(r.learner, r.label, r.protocol, stat)], getattr(r, f"gain_{stat}"))
                                   for r in df.itertuples()]
                df[f"holm_{stat}"] = np.nan; df[f"ci_lo_{stat}"] = np.nan; df[f"ci_hi_{stat}"] = np.nan
                for _, d in df.groupby(["level", "learner"]):
                    adj = holm({k: p for k, p in zip(d.key, d[f"p_{stat}"])})
                    df.loc[d.index, f"holm_{stat}"] = [adj[k] for k in d.key]
            # bootstrap interval where Holm p < 0.05 (either estimand)
            for idx in df.index[(df.holm_ctx < 0.05) | (df.holm_session < 0.05)]:
                r = df.loc[idx]; lab = int(r.label[-1]); tgt = D[r.protocol][1]
                y = (F.severity.to_numpy() >= lab).astype(int)[tgt]
                zb = np.load(P / f"sev{lab}__{r.protocol}__CTXB__ctxb.npz"); Pm, Fm = keep[r.key]
                bt = Boot(g[tgt])
                am = arrays(bt, y, Pm.astype(float), Fm, ctx[tgt], sess[tgt])
                ab = arrays(bt, y, zb["p"][:, tgt], zb["fold"][:, tgt], ctx[tgt], sess[tgt])
                for stat in ("ctx", "session"):
                    lo, hi = _ci(am[stat] - ab[stat]); df.loc[idx, f"ci_lo_{stat}"] = lo; df.loc[idx, f"ci_hi_{stat}"] = hi
            rows += df.drop(columns=["key"]).to_dict("records")
        tmp = out_f.with_suffix(".tmp"); json.dump({"rows": rows, "n_null": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": len(v) for k, v in nulls.items()}},
                                                   open(tmp, "w")); tmp.replace(out_f)
        print(f"posctl seeds: {len({r['seed'] for r in rows})}/{n_seeds} ({time.time()-t0:.0f}s)", flush=True)


def cli(argv):
    RAWDIR, SELDIR, DATA, R = map(Path, argv[1:5]); P = R / "preds"; P.mkdir(parents=True, exist_ok=True)
    stages = argv[5:]
    raw, S = load_archive(RAWDIR)
    cur = set(curated_sessions(SELDIR, RAWDIR).values()); assert len(cur) == 35
    raw["curated"] = raw.session_id.isin(cur)
    F = add_context(build_features(raw)); F["curated"] = raw["curated"].to_numpy()
    fa_run.DATA = DATA
    for st in stages:
        t0 = time.time()
        if st == "decode": stage_decode(F, R, P)
        elif st == "main": stage_main(F, R, P)
        elif st == "control": stage_control(F, R, P)
        elif st == "hgb_control": stage_hgb_control(F, R, P)
        elif st == "posctl": stage_posctl(F, R, P)
        elif st == "within": stage_within(F, R, P)
        elif st == "null_lr": stage_null(F, R, P, "lroff", int(os.environ.get("FA_NSIM_LR", "1000")),
                                         ["CTXB+MOT", "CTXB+SELFZ"], "rev3_null_lroff.json")
        elif st == "null_xgb": stage_null(F, R, P, "xgboff", int(os.environ.get("FA_NSIM_XGB", "400")),
                                          ["CTXB+MOT"], "rev3_null_xgboff.json", el_upto=0)
        elif st == "sens_group": stage_sens_group(F, R, P, S)
        elif st == "vrsq0": stage_vrsq0(F, R, P, S)
        # addendum (docs/ANALYSIS_PLAN_REVISION3_ADDENDUM.md)
        elif st == "rotation": stage_rotation(F, R, P)
        elif st == "decode_nr": stage_decode(F, R, P, fsets=("MOTNR",), fname="rev3_decode_nr.json")
        elif st == "pairs": stage_pairs(F, R, P)
        elif st == "main_nr": stage_main_nr(F, R, P)
        elif st == "pen": stage_pen(F, R, P)
        elif st == "posctl_seeds": stage_posctl_seeds(F, R, P, int(os.environ.get("FA_NSEEDS", "20")))
        elif st == "null_nm": stage_null(F, R, P, "lroff", 200, ["CTXB+MOT"], "rev3_null_lroff_nm.json", el_upto=0)
        elif st == "null_nr": stage_null(F, R, P, "lroff", 200, ["CTXB+MOTNR"], "rev3_null_lroff_nr.json", el_upto=0)
        elif st == "null_pen": stage_null(F, R, P, "lroffp", 200, ["CTXB+MOT"], "rev3_null_lroffp.json", el_upto=0)
        elif st == "null_block": stage_null(F, R, P, "lroff", 200, ["CTXB+MOT"], "rev3_null_lroff_block.json", el_upto=0,
                                            transplant="block", seed_base=800)
        else:
            sys.exit(f"unknown stage {st}")
        print(f"== rev3 stage {st} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    # run through the imported module, so that worker processes unpickle module functions by name
    import fa_rev3
    fa_rev3.cli(sys.argv)
