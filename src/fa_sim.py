"""
fa_sim.py -- revision 3, Stage 6: simulation with known ground truth for the evaluation-design
findings (docs/ANALYSIS_PLAN_REVISION3.md). 30 participants x 2 sessions x 300 records (1 Hz);
each session's label switches 0 -> 1 once (earlier for a more susceptible session; never for
u < -0.5); ten telemetry features = participant fingerprint + AR(1) noise (rho 0.9).

Scenarios:  N  no link between telemetry and labels;
            S  feature 0 carries the session's susceptibility (between-session information only);
            M  feature 0 shifts by 0.5 SD after onset (momentary information).
20 replications per scenario (seeds 2000 + rep). Folds: participant-disjoint grouped 5-fold
(one repeat) or record-wise stratified 5-fold.

Episodic labels (addendum B4, docs/ANALYSIS_PLAN_REVISION3_ADDENDUM.md): with a single 0 -> 1 step,
exposure time orders the records of a session perfectly, so no momentary signal can raise the
within-session AUC. In scenarios Ne, Me and Me2 the label alternates after onset between
on- and off-episodes of U[20, 60] s each, drawn from a separate generator so that the step
scenarios are unchanged; feature 0 is shifted by 0 (Ne), 0.5 SD (Me) or 1.0 SD (Me2) while the
label is on. Seeds 3000 + rep.

Usage: python fa_sim.py <results dir>
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from fa_rev3 import fit_offset_lr, strat_auc, _std  # noqa: E402

N_PART, N_SESS, N_REC, N_FEAT, RHO, DELTA, REPS = 30, 2, 300, 10, 0.9, 0.5, 20
HGB = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, class_weight="balanced", random_state=0)


EPISODIC = {"Ne": 0.0, "Me": 0.5, "Me2": 1.0}


def _episodes(tau, rng, lo=20, hi=60):
    """0 before tau; then alternating on/off episodes of U[lo, hi] s, starting with 'on'."""
    y = np.zeros(N_REC, int)
    if not np.isfinite(tau):
        return y
    t, on = int(np.ceil(tau)), True
    while t < N_REC:
        L = int(rng.integers(lo, hi + 1))
        if on:
            y[t:t + L] = 1
        t += L; on = not on
    return y


def simulate(scenario, seed):
    rng = np.random.default_rng(seed); rows = []
    rng_ep = np.random.default_rng(10_000 + seed)                 # episodic labels only
    up = rng.standard_normal(N_PART); fp = rng.standard_normal((N_PART, N_FEAT))
    sid = 0
    for p in range(N_PART):
        for _ in range(N_SESS):
            u = 0.8 * up[p] + 0.6 * rng.standard_normal()
            t = np.arange(N_REC, dtype=float)
            tau = 30 + 240 * norm.cdf(-u) if u >= -0.5 else np.inf
            y = (t >= tau).astype(int) if scenario not in EPISODIC else _episodes(tau, rng_ep)
            e = np.empty((N_REC, N_FEAT)); z = rng.standard_normal((N_REC, N_FEAT)); e[0] = z[0]
            for i in range(1, N_REC):
                e[i] = RHO * e[i - 1] + np.sqrt(1 - RHO ** 2) * z[i]
            X = fp[p] + e
            if scenario == "S":
                X[:, 0] += u
            elif scenario == "M":
                X[:, 0] += DELTA * y
            elif scenario in EPISODIC:
                X[:, 0] += EPISODIC[scenario] * y
            d = pd.DataFrame(X, columns=[f"x{j}" for j in range(N_FEAT)]); d["t"] = t; d["y"] = y
            d["pid"] = p; d["sid"] = sid; rows.append(d); sid += 1
    return pd.concat(rows, ignore_index=True)


def one(scenario, rep):
    D = simulate(scenario, (3000 if scenario in EPISODIC else 2000) + rep); y = D.y.to_numpy(); g = D.pid.to_numpy(); s = D.sid.to_numpy()
    feats = [f"x{j}" for j in range(N_FEAT)]; X = D[feats + ["t"]].to_numpy(); Xf = D[feats].to_numpy(); t = D.t.to_numpy()
    tb = np.minimum(t // 30, 9).astype(int).astype(str); out = {"scenario": scenario, "rep": rep, "prevalence": float(y.mean())}
    # D1
    p = np.zeros(len(y)); fo = np.zeros(len(y), int)
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
        p[b] = clone(HGB).fit(X[a], y[a]).predict_proba(X[b])[:, 1]; fo[b] = k
    out["D1_recordwise_auc"] = strat_auc(y, p, fo.astype(str))
    folds = list(GroupKFold(5).split(X, y, g))

    def grouped(ytr, Xm=X):
        p = np.zeros(len(y)); fo = np.zeros(len(y), int)
        for k, (a, b) in enumerate(folds):
            p[b] = clone(HGB).fit(Xm[a], ytr[a]).predict_proba(Xm[b])[:, 1]; fo[b] = k
        return p, fo
    p, fo = grouped(y); fs = fo.astype(str)
    out["D1_grouped_auc"] = strat_auc(y, p, fs)
    out["hgb_within_session_auc"] = strat_auc(y, p, np.char.add(fs, s.astype(str)))
    # D2: prevalence score, leave one participant out, pooled across folds
    loo = np.zeros(len(y))
    for k in np.unique(g):
        loo[g == k] = y[g != k].mean()
    out["D2_prevalence_LOPO_pooled_auc"] = float(roc_auc_score(y, loo))
    # D3: label-shift control
    rng = np.random.default_rng(1); ys = y.copy()
    for k in np.unique(s):
        idx = np.where(s == k)[0]; n = len(idx); ys[idx] = np.roll(y[idx], int(rng.integers(n // 4, 3 * n // 4 + 1)))
    p2, _ = grouped(ys)
    out["D3_shifted_auc"] = strat_auc(y, p2, fs)
    out["D3_shifted_within_session_auc"] = strat_auc(y, p2, np.char.add(fs, s.astype(str)))
    # offset model with a time-only baseline: stratified (time bin) and within-session gains
    B = np.c_[t / 60, (t / 60) ** 2]; zb = np.zeros(len(y)); zf = np.zeros(len(y))
    for a, b in folds:
        Bs, Bt = _std(B[a], B[b]); b0, be = fit_offset_lr(Bs, y[a], np.zeros(len(a)))
        oa, ob = b0 + Bs @ be, b0 + Bt @ be; zb[b] = ob
        Xs, Xt = _std(Xf[a], Xf[b]); c0, ce = fit_offset_lr(Xs, y[a], oa); zf[b] = ob + c0 + Xt @ ce
    st = np.char.add(fs, tb); ss = np.char.add(fs, s.astype(str))
    out["lroff_ctx_gain"] = strat_auc(y, zf, st) - strat_auc(y, zb, st)
    out["lroff_session_gain"] = strat_auc(y, zf, ss) - strat_auc(y, zb, ss)
    out["base_within_session_auc"] = strat_auc(y, zb, ss)                 # time-only baseline
    out["x0_within_session_auc"] = strat_auc(y, D.x0.to_numpy(), s.astype(str))   # realised channel AUC
    out["x0_time_bin_auc"] = strat_auc(y, D.x0.to_numpy(), tb)
    return out


def sim_main():
    R = Path(sys.argv[1])
    res = Parallel(n_jobs=int(os.environ.get("FA_NJOBS", "4")))(delayed(one)(sc, r) for sc in ("N", "S", "M", "Ne", "Me", "Me2") for r in range(REPS))
    df = pd.DataFrame(res)
    summ = {sc: {c: {"mean": float(d[c].mean()), "q2.5": float(d[c].quantile(0.025)), "q97.5": float(d[c].quantile(0.975))}
                 for c in d.columns if c not in ("scenario", "rep")} for sc, d in df.groupby("scenario")}
    json.dump({"design": {"participants": N_PART, "sessions_per_participant": N_SESS, "records": N_REC,
                          "features": N_FEAT, "rho": RHO, "delta": DELTA, "replications": REPS,
                          "episodic": {"scenarios": EPISODIC, "episode_s": [20, 60], "seeds": "3000 + rep"}},
               "summary": summ, "replications": res}, open(R / "rev3_sim.json", "w"), indent=1)
    for sc, v in summ.items():
        print(sc, {k: round(x["mean"], 3) for k, x in v.items()})


if __name__ == "__main__":
    import fa_sim as _m
    _m.sim_main()
