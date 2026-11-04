"""
fa_rev3_stats.py -- statistics of revision 3 (docs/ANALYSIS_PLAN_REVISION3.md). Reads the saved
predictions and the null files; nothing is refitted. For every configuration: overall,
context-stratified and within-session ROC-AUC with participant-group bootstrap intervals
(B = 10,000, fitted models fixed; fa_stats.Boot). Telemetry gains (CTXB + telemetry minus CTXB)
with paired intervals; permutation p-values and Holm families from the label-transplant nulls;
the minimum detectable effect from the positive control.

Usage: python fa_rev3_stats.py <data dir> <results dir>
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from fa_stats import Boot, holm, load_frame  # noqa: E402
from fa_stats import ci as _ci, p_two_sided as _p2


def ci(v):
    """fa_stats.ci, or [nan, nan] when no bootstrap draw is finite (non-estimable configuration)."""
    return _ci(v) if np.isfinite(v[1:]).any() else [float("nan"), float("nan")]


def p_two_sided(d):
    return _p2(d) if np.isfinite(d[1:]).any() else float("nan")
from fa_run import CELL_PROTOCOLS, CTX_PROTOCOLS, domains  # noqa: E402

PROTOCOLS = CTX_PROTOCOLS + CELL_PROTOCOLS
NULL_PROTOCOLS = ["within_seated", "within_standing", "pooled"] + CELL_PROTOCOLS
POS = (55, 60, 65, 70, 80)
KEY = re.compile(r"^(?P<tag>[a-zA-Z0-9]*_)?sev(?P<lab>[12])__(?P<pr>[a-z_]+)__(?P<fs>[A-Z+]+)__(?P<mn>[a-z]+)$")
OLD_FS = ("TIME", "CTX+TIME", "MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME", "MOT")


def arrays(bt, y, P, FOLD, ctx, sess, keep=None):
    """Overall, context-stratified and within-session AUC arrays (row 0 observed, then bootstrap
    draws); with keep (addendum B3) also the within-session AUC on the 'falls' sessions only."""
    ov, cx, ws, nm = [], [], [], []
    for r in range(P.shape[0]):
        v = (FOLD[r] >= 0) & ~np.isnan(P[r]); f = FOLD[r].astype(str)
        ov.append(bt.auc(y, P[r], f, v))
        cx.append(bt.auc(y, P[r], np.char.add(np.char.add(f, "|"), ctx), v))
        fs = np.char.add(np.char.add(f, "|"), sess)
        ws.append(bt.auc(y, P[r], fs, v, nested=True))
        if keep is not None:
            nm.append(bt.auc(y, P[r], fs, v & keep, nested=True))
    with np.errstate(invalid="ignore"):
        out = {"overall": np.nanmean(ov, 0), "ctx": np.nanmean(cx, 0), "session": np.nanmean(ws, 0)}
        if keep is not None:
            out["session_nm"] = np.nanmean(nm, 0)
    return out


def summ(a):
    return {m: {"auc": float(v[0]), "ci": ci(v)} for m, v in a.items()}


def gain(a, b):
    out = {}
    for m in a:
        d = a[m] - b[m]
        out[m] = {"delta": float(d[0]), "ci": ci(d), "p_boot": p_two_sided(d)}
    return out


def job(DATA, P, keys, pr, grouping, gmap):
    """All configurations of one protocol target and one grouping (shared bootstrap draws)."""
    F, _ = load_frame(DATA); tgt = domains(F)[pr][1]; Ft = F[tgt]
    g = Ft.group_id.to_numpy() if grouping == "group" else (Ft.session_id.to_numpy() if grouping == "session"
                                                             else Ft.group_id.map(gmap).to_numpy())
    bt = Boot(g); ctx = Ft.s_ctx.to_numpy().astype(str); sess = Ft.session_id.to_numpy().astype(str)
    from fa_rev3 import falls_mask
    Ft = Ft.reset_index(drop=True); keeps = {lab: falls_mask(Ft, (Ft.severity.to_numpy() >= lab).astype(int)) for lab in (1, 2)}
    A, out = {}, {}
    for key in keys:
        d = KEY.match(key).groupdict(); y = (Ft.severity.to_numpy() >= int(d["lab"])).astype(int)
        z = np.load(P / f"{key}.npz")
        A[key] = arrays(bt, y, z["p"][:, tgt], z["fold"][:, tgt], ctx, sess, keep=keeps[int(d["lab"])])
        out[key] = {**summ(A[key]), "n_rows": int(tgt.sum()), "n_groups": int(len(bt.ug)), "prevalence": float(y.mean()),
                    "pos_groups": int(len(np.unique(g[y == 1]))), "n_clusters": int(len(bt.ug))}
    gains = {}
    for key in keys:
        d = KEY.match(key).groupdict(); tag = d["tag"] or ""
        base = f"{tag}sev{d['lab']}__{d['pr']}__CTXB__ctxb"
        if d["mn"] in ("lroff", "xgboff", "lroffp") and base in A:
            gains[key] = gain(A[key], A[base])
        if d["mn"] in ("hgb", "logreg", "hgbreg") and d["fs"] in ("MOT+TIME", "SELFZ+TIME", "MOT+CTX+TIME"):
            tk = f"{tag}sev{d['lab']}__{d['pr']}__{'TIME' if d['fs'] != 'MOT+CTX+TIME' else 'CTX+TIME'}__{d['mn']}"
            if tk in A:
                gains[key] = gain(A[key], A[tk])
    return out, gains


def null_tests(R, obs, fname, learner, fsets, n_expected):
    f = R / fname
    if not f.exists():
        return None
    df = pd.DataFrame(json.load(open(f))); n = df.sim.nunique()
    fam = {}
    stats = [("ctx", "gain_ctx"), ("session", "gain_session")] + ([("session_nm", "gain_session_nm")] if "gain_session_nm" in df else [])
    for fs in fsets:
        for stat, col in stats:
            tests, skipped = {}, {}
            for lab in (1, 2):
                for pr in NULL_PROTOCOLS:
                    key = f"sev{lab}__{pr}__{fs}__{learner}"
                    if key not in obs:
                        continue
                    if stat not in obs[key]:
                        continue
                    o = obs[key][stat]["delta"]
                    v = df[(df.label == f"sev{lab}") & (df.protocol == pr) & (df.features == fs)][col].to_numpy()
                    v = v[np.isfinite(v)]
                    if not len(v) or not np.isfinite(o):          # no finite null or observed value: not estimable
                        skipped[key] = {"observed_gain": o, "n_perm": int(len(v))}
                        continue
                    tests[key] = {"observed_gain": o, "p_perm": float((1 + (v >= o).sum()) / (1 + len(v))), "n_perm": int(len(v)),
                                  "null_mean": float(v.mean()), "null_q95": float(np.quantile(v, 0.95)),
                                  "null_q2.5": float(np.quantile(v, 0.025)), "null_q97.5": float(np.quantile(v, 0.975))}
            adj = holm({k: t["p_perm"] for k, t in tests.items()})
            for k in tests:
                tests[k]["p_holm"] = adj[k]
            fam[f"{learner}|{fs}|{stat}"] = {"tests": tests, "m": len(tests), "n_perm": int(n), "n_expected": n_expected,
                                             "not_estimable": skipped}
    # location of the models' own context-stratified and within-session AUC under the null
    loc = {}
    for (lab, pr, fs), d in df.groupby(["label", "protocol", "features"]):
        loc[f"{lab}__{pr}__{fs}__{learner}"] = {m: float(d[m].mean()) for m in
                                                ("overall_auc", "context_stratified_auc", "within_session_auc",
                                                 "base_overall_auc", "base_context_stratified_auc", "base_within_session_auc")}
    el = df[df.get("event_locked", pd.Series(np.nan, index=df.index)).notna()] if "event_locked" in df else df.iloc[:0]
    elnull = {}
    for (lab, fs), d in el.groupby(["label", "features"]):
        elnull[f"{lab}__{fs}"] = d.event_locked.to_numpy().tolist()
    return {"families": fam, "null_location": loc, "event_locked_null": elnull}


def power_summary(R, gains):
    """Addendum B1: power of the pre-specified rule over the planted-channel seeds, and the
    seed-based 'evidence against' check for the real gains."""
    f = R / "rev3_posctl_seeds.json"
    if not f.exists():
        return None
    df = pd.DataFrame(json.load(open(f))["rows"]); out = {"n_seeds": int(df.seed.nunique()), "cells": {}}
    for (ln, lev, lab, pr), d in df.groupby(["learner", "level", "label", "protocol"]):
        c = {}
        for stat in ("ctx", "session"):
            dec = (d[f"holm_{stat}"] < 0.05) & (d[f"ci_lo_{stat}"] > 0)
            g = np.sort(d[f"gain_{stat}"].to_numpy())
            real = gains.get(f"{lab}__{pr}__CTXB+MOT__{ln}", {}).get(stat, {}).get("delta", np.nan)
            c[stat] = {"power": float(dec.mean()), "power_single_protocol": float((d[f"p_{stat}"] <= 0.05 / 14).mean()),
                       "planted_gain_median": float(np.median(g)), "planted_gain_q05": float(np.quantile(g, 0.05)),
                       "planted_gain_2nd_smallest": float(g[1]) if len(g) > 1 else float("nan"),
                       "real_gain": float(real),
                       "evidence_against": bool(np.isfinite(real) and len(g) >= 20 and real < g[1])}
        c["channel_ctx_auc"] = float(d.channel_ctx_auc.mean()); c["channel_session_auc"] = float(d.channel_session_auc.mean())
        out["cells"][f"{ln}__{lev:.2f}__{lab}__{pr}"] = c
    return out


def main(DATA, R):
    t0 = time.time(); P = R / "preds"; F, S = load_frame(DATA)
    from fa_rev3 import merged_groups
    from fa_data import load_archive  # noqa: F401
    Sx = S.copy()
    gmap = merged_groups(Sx) if "fingerprint" in Sx else None
    files = sorted(f.stem for f in P.glob("*.npz") if KEY.match(f.stem))
    jobs = []
    for grouping, tags in (("group", ("", "r3control_", "control_") + tuple(f"r3pos{k}_" for k in POS)),
                           ("session", ("sess_",)), ("merged", ("mrg_",))):
        for pr in PROTOCOLS:
            ks = []
            for k in files:
                d = KEY.match(k).groupdict(); tag = d["tag"] or ""
                if d["pr"] != pr or tag not in tags:
                    continue
                if tag in ("", "control_") and not (d["mn"] in ("lroff", "xgboff", "ctxb", "lroffp") or d["fs"] in OLD_FS):
                    continue
                if tag in ("", "control_") and d["mn"] not in ("lroff", "xgboff", "ctxb", "hgb", "logreg", "hgbreg", "lroffp"):
                    continue
                ks.append(k)
            if ks:
                jobs.append((ks, pr, grouping))
    print(f"{len(jobs)} bootstrap jobs, {sum(len(j[0]) for j in jobs)} configurations", flush=True)
    res = Parallel(n_jobs=int(os.environ.get("FA_NJOBS", "4")), verbose=5)(
        delayed(job)(DATA, P, ks, pr, grouping, gmap) for ks, pr, grouping in jobs)
    configs, gains = {}, {}
    for (ks, pr, grouping), (c, gn) in zip(jobs, res):
        configs.update(c); gains.update(gn)
    out = {"B": 10_000, "configs": configs, "gains": gains}
    # nulls
    out["null_lroff"] = null_tests(R, gains, "rev3_null_lroff.json", "lroff", ["CTXB+MOT", "CTXB+SELFZ"], 1000)
    out["null_xgboff"] = null_tests(R, gains, "rev3_null_xgboff.json", "xgboff", ["CTXB+MOT"], 400)
    # addendum: secondary nulls (200 permutations each)
    out["null_lroff_nm"] = null_tests(R, gains, "rev3_null_lroff_nm.json", "lroff", ["CTXB+MOT"], 200)
    out["null_lroff_nr"] = null_tests(R, gains, "rev3_null_lroff_nr.json", "lroff", ["CTXB+MOTNR"], 200)
    out["null_lroffp"] = null_tests(R, gains, "rev3_null_lroffp.json", "lroffp", ["CTXB+MOT"], 200)
    out["null_lroff_block"] = null_tests(R, gains, "rev3_null_lroff_block.json", "lroff", ["CTXB+MOT"], 200)
    for k in ("null_lroff_nm", "null_lroff_nr", "null_lroffp", "null_lroff_block"):
        if out[k]:
            out[k].pop("event_locked_null", None)
    # event-locked p-values against the lroff null
    W = json.load(open(R / "rev3_within.json")) if (R / "rev3_within.json").exists() else None
    if W and out["null_lroff"]:
        for k, v in W["event_locked"].items():
            lab, _, fs, ln = k.split("__")
            nv = np.array(out["null_lroff"]["event_locked_null"].get(f"{lab}__{fs}", []), float); nv = nv[np.isfinite(nv)]
            if ln == "lroff" and len(nv):
                v["p_perm_one_sided"] = float((1 + (nv >= v["did"]).sum()) / (1 + len(nv))); v["n_perm"] = int(len(nv))
                v["null_mean"] = float(nv.mean()); v["null_q2.5"] = float(np.quantile(nv, 0.025)); v["null_q97.5"] = float(np.quantile(nv, 0.975))
        out["event_locked"] = W["event_locked"]
        for k in out["null_lroff"]["event_locked_null"]:
            out["null_lroff"]["event_locked_null"][k] = len(out["null_lroff"]["event_locked_null"][k])
    # minimum detectable effect (gain of CTXB + MOT + planted channel over CTXB; and channel alone)
    mde = {}
    for lab in (1, 2):
        for pr in NULL_PROTOCOLS:
            for fs, ln in (("CTXB+SPK", "lroff"), ("CTXB+MOT+SPK", "lroff"), ("CTXB+MOT+SPK", "xgboff"), ("CTXB+MOT+SPK", "lroffp")):
                for stat in ("ctx", "session"):
                    rows = {}
                    for k in POS:
                        key = f"r3pos{k}_sev{lab}__{pr}__{fs}__{ln}"
                        if key in gains:
                            rows[k / 100] = gains[key][stat]
                    det = [lv for lv, gg in sorted(rows.items()) if gg["ci"][0] > 0]
                    mde[f"sev{lab}__{pr}__{fs}__{ln}__{stat}"] = {"levels": {str(k): v for k, v in rows.items()},
                                                                  "mde": det[0] if det else None}
    out["mde"] = mde
    out["power"] = power_summary(R, gains)
    # referee Q2: does HGB-reg use the context columns? Share of outer-fold models whose test
    # predictions are identical with and without CTX, per protocol and label.
    use = {}
    for lab in (1, 2):
        for pr in CTX_PROTOCOLS:
            a, b = P / f"sev{lab}__{pr}__MOT+TIME__hgbreg.npz", P / f"sev{lab}__{pr}__MOT+CTX+TIME__hgbreg.npz"
            if not (a.exists() and b.exists()):
                continue
            za, zb = np.load(a), np.load(b); same = tot = 0; maxdiff = 0.0
            for r in range(za["p"].shape[0]):
                for f in np.unique(za["fold"][r][za["fold"][r] >= 0]):
                    m = za["fold"][r] == f; tot += 1; d = float(np.max(np.abs(za["p"][r][m] - zb["p"][r][m])))
                    same += d == 0.0; maxdiff = max(maxdiff, d)
            use[f"sev{lab}__{pr}"] = {"folds": tot, "identical_folds": same, "max_abs_diff": maxdiff}
    out["hgbreg_ctx_usage"] = use
    json.dump(out, open(R / "rev3_stats.json", "w"), indent=1)
    print(f"rev3_stats.json: {len(configs)} configurations, {len(gains)} gains ({time.time()-t0:.0f}s)")


def stats_main():
    main(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    import fa_rev3_stats as _m
    _m.stats_main()
