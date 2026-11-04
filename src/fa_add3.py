"""
fa_add3.py -- addendum 3 (docs/addendum_3.md): post hoc analyses P1-P7, specified after all results
of the analysis plan and of addenda 1 and 2 were known. Every output is a new file
(results/add3_*.json, results/add3_gate/*.json, new prediction keys in results/preds); no existing
result file is written. Before every stage the leakage gate and the mutation test (Supplementary
Note S1) are run and their result is written to results/add3_gate/<stage>.json.

Usage:
  python fa_add3.py <RAW_DATABASE dir> <DATABASE_SELECTED_DATA dir> <dataset.csv> <results dir> stage [stage ...]
     stages: main_s null_p null_s null_block null_nr null_nm null_sess posctl6 posctl_sess diffuse vrsq vrlens_rw
  python fa_add3.py --stats <data dir> <results dir>        (writes results/add3_stats.json; nothing is refitted)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import fa_rev3  # noqa: E402
from fa_rev3 import (NJOBS, NULL_PROTOCOLS, FAMILY_PROTOCOLS, _perm_p, cols_for, ctx_strata, point_metrics,  # noqa: E402
                     stage_null, strat_auc, tele_cols, transfer)
from fa_run import domains, make_folds, spike_feature  # noqa: E402

P6 = ["within_seated", "within_standing", "pooled"]          # the post hoc 6-test family (P2, P4)
FAM = list(FAMILY_PROTOCOLS)                                  # the 7 protocols of the 14-test families
N_SEEDS = 20
B_SEED, D_SEED = 5000, 6000                                   # planted channel (addendum 1, B1); diffuse shift (P7)


# ----------------------------------------------------------------------------- gate
def gate(raw, F, R, name, session_groups=False):
    """Leakage gate and mutation test before a stage; the result goes to a new file. Raises if a check fails."""
    from fa_data import assert_no_leakage
    from fa_run import leak_mutation_test
    t0 = time.time(); g = F.group_id.to_numpy()
    folds = {pr: [(tr, te) for _, _, tr, te in make_folds(g, src, tgt)] for pr, (src, tgt) in domains(F).items()}
    checks = assert_no_leakage(raw, F, folds); checks["n_folds_checked"] = int(sum(len(v) for v in folds.values()))
    if session_groups:
        Fg = session_grouped(F); gs = Fg.group_id.to_numpy()
        sf = {f"sess:{pr}": [(tr, te) for _, _, tr, te in make_folds(gs, *domains(Fg)[pr])] for pr in P6}
        c2 = assert_no_leakage(raw, Fg, sf)
        checks["session_grouping_folds_disjoint"] = c2["train_test_group_and_session_disjoint_every_fold"]
        checks["n_session_grouping_folds_checked"] = int(sum(len(v) for v in sf.values()))
    checks["mutation_test_centred_window_caught"] = leak_mutation_test(raw)
    checks["stage"] = name; checks["utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()); checks["secs"] = round(time.time() - t0, 1)
    d = R / "add3_gate"; d.mkdir(exist_ok=True)
    json.dump(checks, open(d / f"{name}.json", "w"), indent=1)
    print(f"gate {name}: passed ({checks['secs']}s)", flush=True)


def session_grouped(F):
    """Every session its own group, exactly as fa_rev3.stage_sens_group."""
    Fg = F.copy(); Fg["group_id"] = F.session_id.astype("category").cat.codes.to_numpy()
    return Fg


# ----------------------------------------------------------------------------- P1: stronger penalty
def stage_main_s(F, R, P):
    res = fa_rev3.run_grid(F, FAM, ["CTXB+MOT"], ["lroffs"], [1, 2], P, inner=False)
    json.dump(res, open(R / "add3_main_lroffs.json", "w"), indent=1)


# ----------------------------------------------------------------------------- nulls (P1, P3, P4)
EXT = {  # stored 200-permutation null -> new 1,000-permutation file (same transplant, learner, seeds)
    "null_p": ("rev3_null_lroffp.json", "add3_null_lroffp.json", "lroffp", ["CTXB+MOT"], "session", 700),
    "null_block": ("rev3_null_lroff_block.json", "add3_null_lroff_block.json", "lroff", ["CTXB+MOT"], "block", 800),
    "null_nr": ("rev3_null_lroff_nr.json", "add3_null_lroff_nr.json", "lroff", ["CTXB+MOTNR"], "session", 700),
    "null_nm": ("rev3_null_lroff_nm.json", "add3_null_lroff_nm.json", "lroff", ["CTXB+MOT"], "session", 700),
}


def _same_rows(a, b, tol=1e-12):
    """Largest absolute difference over the numeric fields of matching rows (nan == nan)."""
    ka = {(r["sim"], r["label"], r["protocol"], r["features"]): r for r in a}
    worst = 0.0; n = 0
    for r in b:
        k = (r["sim"], r["label"], r["protocol"], r["features"]); o = ka[k]
        for f, v in r.items():
            if isinstance(v, float) or isinstance(o.get(f), float):
                x, y = float(v), float(o[f])
                if np.isnan(x) and np.isnan(y):
                    continue
                d = abs(x - y) if np.isfinite(x) and np.isfinite(y) else np.inf
                worst = max(worst, d); n += 1
    return worst, n


def stage_null_ext(F, R, P, name):
    src, dst, ln, fsets, tp, sb = EXT[name]
    if not (R / dst).exists():
        stored = [r for r in json.load(open(R / src))]
        assert len({r["sim"] for r in stored}) == 200
        chk = Parallel(n_jobs=2)(delayed(fa_rev3._null_one)(F, s, ln, fsets, NULL_PROTOCOLS, False, tp, sb) for s in (0, 199))
        rows = [r for rs in chk for r in rs]; worst, n = _same_rows(stored, rows)
        rep = {"source": src, "target": dst, "re_run_sims": [0, 199], "fields_compared": n, "max_abs_diff": worst,
               "identical": bool(worst <= 1e-12)}
        (R / "add3_gate").mkdir(exist_ok=True); json.dump(rep, open(R / "add3_gate" / f"repro_{name}.json", "w"), indent=1)
        print(f"{name}: re-run of permutations 0 and 199 vs stored: max |diff| = {worst} over {n} fields", flush=True)
        if worst > 1e-12:
            raise RuntimeError(f"{name}: stored permutations not reproduced (max |diff| {worst}); extension stopped")
        json.dump(stored, open(R / dst, "w"))
    stage_null(F, R, P, ln, 1000, fsets, dst, el_upto=0, transplant=tp, seed_base=sb)


def stage_null_s(F, R, P):
    stage_null(F, R, P, "lroffs", 1000, ["CTXB+MOT"], "add3_null_lroffs.json", el_upto=0)


def stage_null_sess(F, R, P):
    stage_null(session_grouped(F), R, P, "lroff", 1000, ["CTXB+MOT"], "add3_null_sess.json", el_upto=0, protocols=P6)


# ----------------------------------------------------------------------------- planted signals (P2, P4, P7)
def _qshift_prep(x, codes, u):
    """Per stratum: the sorted values and every record's mid-quantile (ties broken by u)."""
    prep = []
    for k in np.unique(codes):
        m = np.flatnonzero(codes == k); xs = x[m]; n = len(m)
        order = np.lexsort((u[m], xs)); q = np.empty(n); q[order] = (np.arange(n) + 0.5) / n
        prep.append((m, q, np.sort(xs)))
    return prep


def _qshift(x, pos, prep, d):
    """Amendment 1 of addendum 3: shift the positive records by d on the normal-score scale of their
    stratum and map back through the stratum's empirical quantile function, so every shifted value is a
    value observed in that stratum (d = 0 returns x exactly; monotone non-decreasing in d)."""
    out = x.copy()
    for m, q, sv in prep:
        pm = pos[m]
        if pm.any():
            q2 = norm.cdf(norm.ppf(q[pm]) + d); idx = np.minimum((q2 * len(sv)).astype(int), len(sv) - 1)
            out[m[pm]] = sv[idx]
    return out


def diffuse_shift(F, y, ctx, target, s, lab):
    """P7 (as amended before its run): five of the 14 non-rotation features; on the positive records
    each is shifted by the same amount d on the normal-score scale within its stratum and mapped back to
    observed values; d by bisection so that the within-stratum AUC of the equally weighted sum of the
    five shifted features, each divided by its pooled within-stratum SD, equals the target (all records,
    strata campaign x game x 30-s bin)."""
    pool = tele_cols(F, "MOTNR"); rng = np.random.default_rng(D_SEED + 100 * s + lab)
    feats = [str(c) for c in rng.choice(np.array(pool), 5, replace=False)]
    X = F[feats].to_numpy(float); codes = pd.factorize(ctx)[0]; cnt = np.bincount(codes)
    mu = np.column_stack([np.bincount(codes, weights=X[:, j]) / cnt for j in range(X.shape[1])])
    sig = np.sqrt(((X - mu[codes]) ** 2).mean(0)); assert (sig > 1e-12).all(), feats
    U = rng.random(X.shape)                                   # tie-breaking, drawn after the features
    preps = [_qshift_prep(X[:, j], codes, U[:, j]) for j in range(X.shape[1])]
    pos = y == 1

    def shifted(d):
        return np.column_stack([_qshift(X[:, j], pos, preps[j], d) for j in range(X.shape[1])])
    f = lambda d: strat_auc(y, (shifted(d) / sig).sum(1), ctx)
    a0 = f(0.0); flag = ""
    if a0 >= target:
        d = 0.0; flag = "auc_at_zero_not_below_target"
    elif f(10.0) < target:
        d = 10.0; flag = "target_not_reached_at_d_10"
    else:
        lo, hi = 0.0, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        d = hi
    Xs = shifted(d); Fs = F.copy(); Fs[feats] = Xs; S = (Xs / sig).sum(1)
    assert np.array_equal(shifted(0.0), X)
    return Fs, {"features_shifted": feats, "delta": float(d), "auc_S_at_0": float(a0), "auc_S": float(f(d)), "flag": flag,
                "share_values_changed": float((Xs != X).any(axis=1)[pos].mean())}, S


def _planted_one(F, s, Pdir, base_tag, models, levels, protocols, mode):
    """One realization: planted gains of every model over the stored baseline (point estimates), the
    realized AUC of the planted signal, and the test predictions (kept in memory for the intervals)."""
    D = domains(F); sess = F.session_id.to_numpy().astype(str); ctx = ctx_strata(F); rows, keep = [], {}
    for lev in levels:
        for lab in (1, 2):
            y = (F.severity.to_numpy() >= lab).astype(int)
            if mode == "spike":
                d = float(np.sqrt(2) * norm.ppf(lev)); chan = spike_feature(F, y, d, seed=B_SEED + 100 * s + lab)
                Fs = F.assign(spk=chan); info = {"d": d}
            else:
                Fs, info, chan = diffuse_shift(F, y, ctx, lev, s, lab)
            for pr in protocols:
                src, tgt = D[pr]
                zb = np.load(Pdir / f"{base_tag}sev{lab}__{pr}__CTXB__ctxb.npz")
                mb = point_metrics(y[tgt], zb["p"][:, tgt], zb["fold"][:, tgt], ctx[tgt], sess[tgt])
                ch_c = float(strat_auc(y[tgt], chan[tgt], ctx[tgt])); ch_w = float(strat_auc(y[tgt], chan[tgt], sess[tgt]))
                for fs, ln in models:
                    Pm, _, Fm, _ = transfer(Fs, ln, cols_for(Fs, fs), y, src, tgt, inner=False, n_jobs=1)
                    mm = point_metrics(y[tgt], Pm[:, tgt], Fm[:, tgt], ctx[tgt], sess[tgt])
                    key = f"s{s}__{int(round(100 * lev))}__sev{lab}__{pr}__{fs}__{ln}"
                    rows.append({"seed": s, "level": lev, "label": f"sev{lab}", "protocol": pr, "features": fs, "learner": ln,
                                 "gain_overall": mm[0] - mb[0], "gain_ctx": mm[1] - mb[1], "gain_session": mm[2] - mb[2],
                                 "channel_ctx_auc": ch_c, "channel_session_auc": ch_w, **info, "key": key})
                    keep[key] = (Pm[:, tgt].astype(np.float32), Fm[:, tgt])
    return rows, keep


def load_nulls(R, specs):
    """{(learner, label, protocol, stat): sorted finite null gains}; specs: [(file, learner, features)]."""
    nulls, n = {}, {}
    for fname, ln, fs in specs:
        df = pd.DataFrame(json.load(open(R / fname))); df = df[df.features == fs]
        n[f"{fname}|{fs}"] = int(df.sim.nunique())
        for (lab, pr), d in df.groupby(["label", "protocol"]):
            for stat in ("ctx", "session"):
                v = d[f"gain_{stat}"].to_numpy(float); nulls[(ln, lab, pr, stat)] = np.sort(v[np.isfinite(v)])
    return nulls, n


def score(df, keep, F, P, base_tag, nulls, families):
    """Permutation p against the real-telemetry null of the learner, Holm within each family (per
    realization, level, learner and estimand), and the group-bootstrap interval of the gain where a
    Holm-adjusted p < 0.05 in any family (as addendum 1, B1)."""
    from fa_stats import Boot, holm
    from fa_rev3_stats import arrays, ci as _ci
    D = domains(F); sess = F.session_id.to_numpy().astype(str); ctx = ctx_strata(F); g = F.group_id.to_numpy()
    for stat in ("ctx", "session"):
        ps = []
        for r in df.itertuples():
            nv = nulls.get((r.learner, r.label, r.protocol, stat))
            ps.append(_perm_p(nv, getattr(r, f"gain_{stat}")) if nv is not None and len(nv) else float("nan"))
        df[f"p_{stat}"] = ps
        for fam, prs in families.items():
            col = f"holm_{fam}_{stat}"; df[col] = np.nan
            for _, d in df[df.protocol.isin(prs)].groupby(["seed", "level", "learner"]):
                pv = {i: p for i, p in zip(d.index, d[f"p_{stat}"]) if np.isfinite(p)}
                adj = holm(pv)
                for i in pv:
                    df.loc[i, col] = adj[i]
        df[f"ci_lo_{stat}"] = np.nan; df[f"ci_hi_{stat}"] = np.nan
    hc = [c for c in df.columns if c.startswith("holm_")]
    need = (df[hc] < 0.05).any(axis=1); base_cache = {}
    for idx in df.index[need]:
        r = df.loc[idx]; lab = int(r.label[-1]); tgt = D[r.protocol][1]
        y = (F.severity.to_numpy() >= lab).astype(int)[tgt]; bt = Boot(g[tgt]); Pm, Fm = keep[r.key]
        am = arrays(bt, y, Pm.astype(float), Fm, ctx[tgt], sess[tgt])
        if (lab, r.protocol) not in base_cache:
            zb = np.load(P / f"{base_tag}sev{lab}__{r.protocol}__CTXB__ctxb.npz")
            base_cache[(lab, r.protocol)] = arrays(bt, y, zb["p"][:, tgt], zb["fold"][:, tgt], ctx[tgt], sess[tgt])
        ab = base_cache[(lab, r.protocol)]
        for stat in ("ctx", "session"):
            lo, hi = _ci(am[stat] - ab[stat]); df.loc[idx, f"ci_lo_{stat}"] = lo; df.loc[idx, f"ci_hi_{stat}"] = hi
    return df


def stage_planted(F, R, P, name, base_tag, models, levels, protocols, mode, null_specs, families, check_b1=False):
    """Resumable over realizations; scored per realization."""
    nulls, n_null = load_nulls(R, null_specs)
    out_f = R / f"add3_{name}.json"
    old = json.load(open(out_f)) if out_f.exists() else {"rows": []}
    rows = old["rows"]; done = {r["seed"] for r in rows}
    b1 = None
    if check_b1:
        b1 = pd.DataFrame(json.load(open(R / "rev3_posctl_seeds.json"))["rows"]).set_index(["seed", "level", "label", "protocol", "learner"])
    todo = [s for s in range(N_SEEDS) if s not in done]; t0 = time.time(); worst = old.get("b1_check_max_abs_diff", 0.0)
    for i in range(0, len(todo), NJOBS):
        chunk = todo[i:i + NJOBS]
        res = Parallel(n_jobs=NJOBS)(delayed(_planted_one)(F, s, P, base_tag, models, levels, protocols, mode) for s in chunk)
        for rs, keep in res:
            df = pd.DataFrame(rs)
            if b1 is not None:          # P2: the refitted realizations must reproduce the stored B1 gains
                for r in df.itertuples():
                    o = b1.loc[(r.seed, r.level, r.label, r.protocol, r.learner)]
                    dd = max(abs(o.gain_ctx - r.gain_ctx), abs(o.gain_session - r.gain_session))
                    worst = max(worst, dd)
                if worst > 1e-9:
                    raise RuntimeError(f"{name}: refitted planted gains differ from addendum-1 B1 by {worst}")
            df = score(df, keep, F, P, base_tag, nulls, families)
            rows += df.drop(columns=["key"]).to_dict("records")
        meta = {"rows": rows, "n_null": n_null, "families": families, "levels": list(levels), "protocols": protocols,
                "models": [list(m) for m in models], "mode": mode, "base_tag": base_tag}
        if b1 is not None:
            meta["b1_check_max_abs_diff"] = worst
        tmp = out_f.with_suffix(".tmp"); json.dump(meta, open(tmp, "w")); tmp.replace(out_f)
        print(f"{name}: {len({r['seed'] for r in rows})}/{N_SEEDS} realizations ({time.time()-t0:.0f}s)", flush=True)


def stage_posctl6(F, R, P):
    stage_planted(F, R, P, "posctl6", "", [("CTXB+MOT+SPK", "lroff"), ("CTXB+MOT+SPK", "xgboff")], (0.65, 0.70, 0.80), P6, "spike",
                  [("rev3_null_lroff.json", "lroff", "CTXB+MOT"), ("rev3_null_xgboff.json", "xgboff", "CTXB+MOT")],
                  {"f6": P6}, check_b1=True)


def stage_posctl_sess(F, R, P):
    stage_planted(session_grouped(F), R, P, "posctl_sess", "sess_", [("CTXB+MOT+SPK", "lroff")], (0.55, 0.60, 0.65, 0.70, 0.80), P6,
                  "spike", [("add3_null_sess.json", "lroff", "CTXB+MOT")], {"f6": P6})


def stage_diffuse(F, R, P):
    stage_planted(F, R, P, "posctl_diffuse", "", [("CTXB+MOT", "lroffp"), ("CTXB+MOT", "xgboff")], (0.70, 0.80), FAM, "diffuse",
                  [("add3_null_lroffp.json", "lroffp", "CTXB+MOT"), ("rev3_null_xgboff.json", "xgboff", "CTXB+MOT")],
                  {"f14": FAM, "f6": P6})


# ----------------------------------------------------------------------------- P5: VRSQ files
def stage_vrsq(F, R, S, bb=10_000):
    from scipy.stats import fisher_exact
    from fa_stats import holm
    S = S.set_index("session_id"); sev = F.groupby("session_id").severity
    S["ing_max"] = sev.max(); S["ing_mean"] = sev.mean()
    S["ing_frac_ge1"] = sev.apply(lambda v: (v >= 1).mean()); S["ing_frac_ge2"] = sev.apply(lambda v: (v >= 2).mean())
    V = S.dropna(subset=["vrsq_post_total"]).copy(); V["vrsq_delta_total"] = V.vrsq_post_total - V.vrsq_pre_total
    V["all_zero"] = ((V.vrsq_post_raw_sum == 0) & (V.vrsq_pre_total == 0)).astype(int)
    assert len(V) == 70 and V.all_zero.sum() == 32
    gs = V.group_id.unique(); by = {k: V.index[V.group_id == k] for k in gs}

    def boot_frames(seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(bb):
            yield V.loc[np.concatenate([by[k] for k in rng.choice(gs, len(gs))])]

    out = {"n_sessions": int(len(V)), "n_all_zero": int(V.all_zero.sum()), "B": bb, "groups": int(len(gs))}
    # (a) all-zero share by highest in-game rating; >= 2 against < 2
    out["by_ingame_max"] = {str(int(m)): {"n": int(len(d)), "all_zero": int(d.all_zero.sum())} for m, d in V.groupby("ing_max")}
    hi = V.ing_max >= 2
    tab = [[int(V.all_zero[hi].sum()), int((1 - V.all_zero[hi]).sum())], [int(V.all_zero[~hi].sum()), int((1 - V.all_zero[~hi]).sum())]]
    orr, pf = fisher_exact(tab, alternative="two-sided")
    diffs = []
    for d in boot_frames():
        h = d.ing_max >= 2
        diffs.append(d.all_zero[h].mean() - d.all_zero[~h].mean() if h.any() and (~h).any() else np.nan)
    diffs = np.array(diffs)
    out["ge2_vs_lt2"] = {"table_[allzero,not]_ge2_then_lt2": tab, "share_ge2": tab[0][0] / sum(tab[0]), "share_lt2": tab[1][0] / sum(tab[1]),
                         "difference": tab[0][0] / sum(tab[0]) - tab[1][0] / sum(tab[1]), "fisher_odds_ratio": float(orr),
                         "fisher_p_two_sided": float(pf), "diff_ci_group_bootstrap": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
                         "diff_boot_nan": int(np.isnan(diffs).sum())}

    def spear(a, b, d):
        """Spearman's rho, group-bootstrap 95 % interval (seed 0, as fa_run.stage_vrsq and stage_vrsq0)
        and two-sided group-bootstrap p, 2 min(share <= 0, share >= 0) (fa_stats.p_two_sided)."""
        r = d[a].rank().corr(d[b].rank()); gs_ = d.group_id.unique(); by_ = {k: d.index[d.group_id == k] for k in gs_}
        rng = np.random.default_rng(0); v = []
        for _ in range(bb):
            dd = d.loc[np.concatenate([by_[k] for k in rng.choice(gs_, len(gs_))])]; v.append(dd[a].rank().corr(dd[b].rank()))
        v = np.array(v); f = v[np.isfinite(v)]
        p = float(min(1.0, max(2 * min((f <= 0).mean(), (f >= 0).mean()), 1 / len(f))))
        return {"rho": float(r), "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))], "p_boot": p, "n": int(len(d))}
    SUM = ("ing_max", "ing_mean", "ing_frac_ge1", "ing_frac_ge2")
    sets = {"all_files": (V, ("vrsq_post_total", "vrsq_delta_total")),
            "without_all_zero": (V[V.all_zero == 0], ("vrsq_post_total", "vrsq_delta_total")),
            "allzero_indicator_all_files": (V, ("all_zero",))}
    for name, (d, bs) in sets.items():
        res = {f"{a}~{b}": spear(a, b, d) for a in SUM for b in bs}
        adj = holm({k: v["p_boot"] for k, v in res.items()})
        for k in res:
            res[k]["p_holm"] = adj[k]
        out[name] = {"m": len(res), "correlations": res}
    out["without_all_zero"]["note"] = "selects on a VRSQ total above 0 (only sessions whose file is not zero in every item)"
    # (e) listing and clustering
    Z = V[V.all_zero == 1].reset_index()[["session_id", "date", "group_id", "game", "campaign", "ing_max"]].sort_values(["date", "session_id"])
    out["all_zero_files"] = Z.astype({"group_id": int, "ing_max": int}).to_dict("records")
    out["by_date"] = {k: {"n": int(len(d)), "all_zero": int(d.all_zero.sum())} for k, d in V.groupby("date")}
    out["by_group"] = {str(int(k)): {"n": int(len(d)), "all_zero": int(d.all_zero.sum())} for k, d in V.groupby("group_id")}

    def homog(flag, key, min_n):
        n = 0
        for _, d in pd.DataFrame({"k": key, "f": flag}).groupby("k"):
            if len(d) >= min_n and d.f.nunique() == 1:
                n += 1
        return n
    rng = np.random.default_rng(0); flag = V.all_zero.to_numpy()
    for nm, key, mn in (("dates", V.date.to_numpy(), 1), ("groups_with_2plus_sessions", V.group_id.to_numpy(), 2)):
        obs = homog(flag, key, mn); sims = np.array([homog(rng.permutation(flag), key, mn) for _ in range(10_000)])
        out[f"clustering_{nm}"] = {"homogeneous_observed": int(obs), "units": int(sum(1 for _, d in pd.Series(key).groupby(key) if len(d) >= mn)),
                                   "homogeneous_null_mean": float(sims.mean()), "p_one_sided": float((1 + (sims >= obs).sum()) / (1 + len(sims))),
                                   "n_perm": 10_000}
    json.dump(out, open(R / "add3_vrsq.json", "w"), indent=1)
    print({k: out[k] for k in ("by_ingame_max", "ge2_vs_lt2")}, flush=True)


# ----------------------------------------------------------------------------- P6: VR-LENS record-wise without demographics
def stage_vrlens_rw(F, R, P, raw, CSV):
    import fa_vrlens as VL
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    D, rows = VL.curated_rows(raw, CSV); assert F.curated.to_numpy()[rows].all()
    feats = VL.ABLATIONS["sensVRLENSA_"]; X = D[feats].to_numpy(float); y4 = D.DiscomfortLevel.to_numpy().astype(int)
    ctx = ctx_strata(F)[rows]; t0 = time.time(); proba = np.zeros((len(D), 4)); fold = np.zeros(len(D), np.int8)
    for k, (a, b) in enumerate(StratifiedKFold(10, shuffle=True, random_state=0).split(X, y4)):
        proba[b] = VL.vrlens_reduced().fit(X[a], y4[a]).predict_proba(X[b]); fold[b] = k
        print(f"record-wise without demographics: fold {k + 1}/10 ({time.time() - t0:.0f}s)", flush=True)
    pred = proba.argmax(1)
    out = {"status": "post hoc (addendum 3, P6)", "features": feats, "n_records": int(len(D)),
           "folds": "StratifiedKFold(10, shuffle=True, random_state=0), as the original record-wise run",
           "accuracy_4class": float(accuracy_score(y4, pred)), "macro_f1_4class": float(f1_score(y4, pred, average="macro")),
           "majority_class_share": float(np.bincount(y4).max() / len(y4)), "secs": round(time.time() - t0)}
    st = np.char.add(np.char.add(fold.astype(str), "|"), ctx.astype(str))
    for lab in (1, 2):
        s = proba[:, lab:].sum(1); yy = (y4 >= lab).astype(int)
        out[f"auc_sev{lab}"] = float(roc_auc_score(yy, s)); out[f"ctx_auc_sev{lab}"] = float(strat_auc(yy, s, st))
        np.savez_compressed(P / f"a3rw_sensVRLENSA_sev{lab}__within_curated__VRLENS__stack.npz", p=s, idx=rows, fold=fold)
        # the same context-stratified AUC for the stored record-wise run with all ten features (no refit)
        z = np.load(P / f"rowwise_sev{lab}__within_curated__VRLENS__stack.npz")
        assert (z["idx"] == rows).all()
        st0 = np.char.add(np.char.add(z["fold"].astype(str), "|"), ctx.astype(str))
        out[f"stored_with_demographics_ctx_auc_sev{lab}"] = float(strat_auc(yy, z["p"], st0))
    json.dump(out, open(R / "add3_vrlens_rw_nodemo.json", "w"), indent=1)
    print(out, flush=True)


# ----------------------------------------------------------------------------- CLI
def cli(argv):
    if argv[1] == "--stats":
        import fa_add3_stats
        return fa_add3_stats.main(Path(argv[2]), Path(argv[3]))
    from fa_data import add_context, build_features, curated_sessions, load_archive
    import fa_run
    RAWDIR, SELDIR, CSV, R = map(Path, argv[1:5]); P = R / "preds"; stages = argv[5:]
    raw, S = load_archive(RAWDIR)
    cur = set(curated_sessions(SELDIR, RAWDIR).values()); assert len(cur) == 35
    raw["curated"] = raw.session_id.isin(cur)
    F = add_context(build_features(raw)); F["curated"] = raw["curated"].to_numpy()
    fa_run.DATA = Path("data")
    for st in stages:
        gate(raw, F, R, st, session_groups=st in ("null_sess", "posctl_sess"))
        t0 = time.time()
        if st == "main_s": stage_main_s(F, R, P)
        elif st in EXT: stage_null_ext(F, R, P, st)
        elif st == "null_s": stage_null_s(F, R, P)
        elif st == "null_sess": stage_null_sess(F, R, P)
        elif st == "posctl6": stage_posctl6(F, R, P)
        elif st == "posctl_sess": stage_posctl_sess(F, R, P)
        elif st == "diffuse": stage_diffuse(F, R, P)
        elif st == "vrsq": stage_vrsq(F, R, S)
        elif st == "vrlens_rw": stage_vrlens_rw(F, R, P, raw, CSV)
        else:
            sys.exit(f"unknown stage {st}")
        print(f"== add3 stage {st} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import fa_add3          # run through the imported module, so that workers unpickle module functions by name
    fa_add3.cli(sys.argv)
