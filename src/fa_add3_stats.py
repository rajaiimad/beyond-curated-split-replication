"""
fa_add3_stats.py -- statistics of addendum 3 (docs/addendum_3.md; post hoc). Reads the saved
predictions, the stored statistics of revision 3 (results/rev3_stats.json, read only) and the null
and planted-signal files of fa_add3.py; nothing is refitted. Writes results/add3_stats.json only.

Decision rule (unchanged): Holm-adjusted one-sided permutation p < 0.05 within the family AND the
95 % group-bootstrap interval of the gain above 0. p = (1 + k)/(1 + n).

Usage: python fa_add3_stats.py <data dir> <results dir>
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from fa_stats import Boot, holm, load_frame  # noqa: E402
from fa_rev3_stats import arrays, gain  # noqa: E402
from fa_run import domains  # noqa: E402

FAM = ["within_seated", "within_standing", "pooled", "cell_seated_flight", "cell_seated_race",
       "cell_standing_flight", "cell_standing_race"]
P6 = ["within_seated", "within_standing", "pooled"]
T14 = [(lab, pr) for lab in (1, 2) for pr in FAM]
T6 = [(lab, pr) for lab in (1, 2) for pr in P6]
COL = {"overall": "gain_overall", "ctx": "gain_ctx", "session": "gain_session", "session_nm": "gain_session_nm"}


# ----------------------------------------------------------------------------- bootstrap gains (P1)
def _gain_job(DATA, P, pr, learners):
    from fa_rev3 import falls_mask
    F, _ = load_frame(DATA); tgt = domains(F)[pr][1]; Ft = F[tgt].reset_index(drop=True)
    bt = Boot(Ft.group_id.to_numpy()); ctx = Ft.s_ctx.to_numpy().astype(str); sess = Ft.session_id.to_numpy().astype(str)
    out = {}
    for lab in (1, 2):
        y = (Ft.severity.to_numpy() >= lab).astype(int); keep = falls_mask(Ft, y)
        A = {}
        for ln in ("ctxb",) + tuple(learners):
            key = f"sev{lab}__{pr}__CTXB__ctxb" if ln == "ctxb" else f"sev{lab}__{pr}__CTXB+MOT__{ln}"
            z = np.load(P / f"{key}.npz"); A[ln] = arrays(bt, y, z["p"][:, tgt], z["fold"][:, tgt], ctx, sess, keep=keep)
        for ln in learners:
            g = gain(A[ln], A["ctxb"])
            out[f"sev{lab}__{pr}__CTXB+MOT__{ln}"] = {**g, "auc": {m: float(v[0]) for m, v in A[ln].items()},
                                                     "base_auc": {m: float(v[0]) for m, v in A["ctxb"].items()},
                                                     "pos_groups": int(len(np.unique(Ft.group_id.to_numpy()[y == 1])))}
    return out


# ----------------------------------------------------------------------------- families
def null_frame(R, fname, fs):
    df = pd.DataFrame(json.load(open(R / fname))); df = df[df.features == fs].copy()
    df["gain_overall"] = df.overall_auc - df.base_overall_auc
    return df


def family(nd, stat, tests, obs, n_expected=None):
    """obs: {(lab, pr): {"delta": d, "ci": [lo, hi]}}. Holm within the estimable tests."""
    col = COL[stat]; res, skipped = {}, {}
    for lab, pr in tests:
        o = obs.get((lab, pr))
        v = nd[(nd.label == f"sev{lab}") & (nd.protocol == pr)][col].to_numpy(float); v = v[np.isfinite(v)]
        k = f"sev{lab}__{pr}"
        if o is None or not np.isfinite(o["delta"]) or not len(v):
            skipped[k] = {"observed_gain": None if o is None else o["delta"], "n_perm": int(len(v))}
            continue
        res[k] = {"observed_gain": o["delta"], "ci": o["ci"], "p_perm": float((1 + (v >= o["delta"]).sum()) / (1 + len(v))),
                  "n_perm": int(len(v)), "null_mean": float(v.mean()), "null_q2.5": float(np.quantile(v, 0.025)),
                  "null_q97.5": float(np.quantile(v, 0.975))}
    adj = holm({k: t["p_perm"] for k, t in res.items()})
    for k in res:
        lo = res[k]["ci"][0] if res[k]["ci"] is not None else np.nan
        res[k]["p_holm"] = adj[k]; res[k]["declared"] = bool(adj[k] < 0.05 and np.isfinite(lo) and lo > 0)
    n = int(nd.sim.nunique())
    return {"tests": res, "m": len(res), "n_perm_sims": n, "n_expected": n_expected, "not_estimable": skipped,
            "holm_floor": (len(res) / (n + 1)) if res else None,
            "n_declared": int(sum(t["declared"] for t in res.values())),
            "n_holm_lt_05": int(sum(t["p_holm"] < 0.05 for t in res.values())),
            "n_raw_lt_05": int(sum(t["p_perm"] < 0.05 for t in res.values())),
            "min_p_holm": float(min((t["p_holm"] for t in res.values()), default=np.nan)),
            "min_p_perm": float(min((t["p_perm"] for t in res.values()), default=np.nan))}


def obs_from(gains, fmt, stat, ne):
    """Observed gains of a family; where fewer than three groups have a positive record (ne), no
    interval is reported (as in the paper), so the test cannot meet the interval part of the rule."""
    out = {}
    for lab, pr in T14:
        g = gains.get(fmt.format(lab=lab, pr=pr), {}).get(stat)
        if g is not None:
            out[(lab, pr)] = {"delta": g["delta"], "ci": [float("nan"), float("nan")] if ne(lab, pr) else g["ci"]}
    return out


# ----------------------------------------------------------------------------- power
def power(R, fname, fams, ne):
    z = json.load(open(R / fname)); df = pd.DataFrame(z["rows"]); out = {"n_seeds": int(df.seed.nunique()), "n_null": z["n_null"],
                                                                         "levels": z["levels"], "cells": {}, "counts": {}}
    for (ln, lev, lab, pr), d in df.groupby(["learner", "level", "label", "protocol"]):
        c = {"n": int(len(d)), "channel_ctx_auc": float(d.channel_ctx_auc.mean()), "channel_session_auc": float(d.channel_session_auc.mean()),
             "channel_ctx_auc_min": float(d.channel_ctx_auc.min()), "channel_ctx_auc_max": float(d.channel_ctx_auc.max())}
        if "delta" in d:
            c.update(delta_mean=float(d.delta.mean()), delta_min=float(d.delta.min()), delta_max=float(d.delta.max()),
                     auc_S_at_0_min=float(d.auc_S_at_0.min()), auc_S_at_0_max=float(d.auc_S_at_0.max()),
                     flags=int((d.flag != "").sum()))
        for fam in fams:
            if f"holm_{fam}_ctx" not in d or d[f"holm_{fam}_ctx"].isna().all():
                continue
            for stat in ("ctx", "session"):
                h = d[f"holm_{fam}_{stat}"]; dec = (h < 0.05) & (d[f"ci_lo_{stat}"] > 0)
                c[f"{fam}|{stat}"] = {"power": float(dec.mean()), "power_perm_only": float((h < 0.05).mean()),
                                      "planted_gain_median": float(d[f"gain_{stat}"].median()),
                                      "planted_gain_sd": float(d[f"gain_{stat}"].std())}
        out["cells"][f"{ln}__{lev:.2f}__{lab}__{pr}"] = c
    for fam, prs in fams.items():
        for ln in sorted(df.learner.unique()):
            for lev in z["levels"]:
                for stat in ("ctx", "session"):
                    v = [out["cells"][f"{ln}__{lev:.2f}__sev{lab}__{pr}"][f"{fam}|{stat}"] for lab in (1, 2) for pr in prs
                         if f"{ln}__{lev:.2f}__sev{lab}__{pr}" in out["cells"] and not ne(lab, pr)]
                    out["counts"][f"{fam}|{ln}|{lev:.2f}|{stat}"] = {
                        "n_est": len(v), "n50": int(sum(x["power"] >= 0.5 for x in v)),
                        "n50_perm_only": int(sum(x["power_perm_only"] >= 0.5 for x in v)),
                        "n_any": int(sum(x["power"] > 0 for x in v))}
    return out


# ----------------------------------------------------------------------------- main
def main(DATA, R):
    t0 = time.time(); P = R / "preds"
    st3 = json.load(open(R / "rev3_stats.json")); G3 = st3["gains"]; C3 = st3["configs"]
    ne = lambda lab, pr, tag="": C3.get(f"{tag}sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3
    out = {"status": "post hoc (docs/addendum_3.md)", "B": 10_000}

    # P1: bootstrap gains of the two penalized learners (inferred groups)
    res = Parallel(n_jobs=int(os.environ.get("FA_NJOBS", "4")))(delayed(_gain_job)(DATA, P, pr, ("lroffp", "lroffs")) for pr in FAM)
    gains = {k: v for r in res for k, v in r.items()}
    out["gains_p1"] = gains
    # the lroffp gains were also computed in revision 3: they must agree
    dif = [abs(gains[k][s]["delta"] - G3[k][s]["delta"]) for k in gains if k.endswith("lroffp") and k in G3 for s in ("ctx", "session", "overall")]
    difci = [abs(a - b) for k in gains if k.endswith("lroffp") and k in G3 for s in ("ctx", "session")
             for a, b in zip(gains[k][s]["ci"], G3[k][s]["ci"]) if np.isfinite(a) and np.isfinite(b)]
    out["check_lroffp_vs_rev3_stats"] = {"max_abs_diff_gain": float(max(dif)), "max_abs_diff_ci": float(max(difci)), "n": len(dif)}

    fams = {}
    # P1 families (14 and post hoc 6), overall gain descriptive
    for ln, fname in (("lroffp", "add3_null_lroffp.json"), ("lroffs", "add3_null_lroffs.json")):
        nd = null_frame(R, fname, "CTXB+MOT")
        for stat in ("overall", "ctx", "session", "session_nm"):
            ob = obs_from(gains, "sev{lab}__{pr}__CTXB+MOT__" + ln, stat, ne)
            fams[f"P1|{ln}|CTXB+MOT|{stat}|14"] = family(nd, stat, T14, ob, 1000)
            if stat in ("ctx", "session"):
                fams[f"P1|{ln}|CTXB+MOT|{stat}|6"] = family(nd, stat, T6, ob, 1000)
    # P2: 6-test family of the stored primary/secondary nulls; the 14-test families recomputed as a check
    chk = []
    for ln, fs, fname, nexp in (("lroff", "CTXB+MOT", "rev3_null_lroff.json", 1000), ("lroff", "CTXB+SELFZ", "rev3_null_lroff.json", 1000),
                                ("xgboff", "CTXB+MOT", "rev3_null_xgboff.json", 400)):
        nd = null_frame(R, fname, fs)
        for stat in ("ctx", "session"):
            ob = obs_from(G3, "sev{lab}__{pr}__" + f"{fs}__{ln}", stat, ne)
            fams[f"P2|{ln}|{fs}|{stat}|6"] = family(nd, stat, T6, ob, nexp)
            f14 = family(nd, stat, T14, ob, nexp); old = st3[f"null_{ln}"]["families"][f"{ln}|{fs}|{stat}"]["tests"]
            chk += [abs(f14["tests"][f"sev{lab}__{pr}"]["p_holm"] - old[f"sev{lab}__{pr}__{fs}__{ln}"]["p_holm"])
                    for lab, pr in T14 if f"sev{lab}__{pr}" in f14["tests"]]
    out["check_14_families_vs_rev3_stats"] = {"max_abs_diff_p_holm": float(max(chk)), "n": len(chk)}
    # P3: secondary nulls with 1,000 permutations
    for name, ln, fs, fname in (("block", "lroff", "CTXB+MOT", "add3_null_lroff_block.json"), ("nr", "lroff", "CTXB+MOTNR", "add3_null_lroff_nr.json"),
                                ("nm", "lroff", "CTXB+MOT", "add3_null_lroff_nm.json")):
        nd = null_frame(R, fname, fs)
        for stat in ("ctx", "session", "session_nm"):
            if name == "nm" and stat != "session_nm":
                continue
            ob = obs_from(G3, "sev{lab}__{pr}__" + f"{fs}__{ln}", stat, ne)
            fams[f"P3|{name}|{ln}|{fs}|{stat}|14"] = family(nd, stat, T14, ob, 1000)
    # consistency: the falling-session null re-runs the main null's transplants and model
    a = null_frame(R, "rev3_null_lroff.json", "CTXB+MOT"); b = null_frame(R, "add3_null_lroff_nm.json", "CTXB+MOT")
    m = a.merge(b, on=["sim", "label", "protocol"], suffixes=("_a", "_b"))
    cons = {"rows_compared": int(len(m)), "sims": int(m.sim.nunique())}
    for c in ("gain_ctx", "gain_session", "overall_auc", "context_stratified_auc", "within_session_auc"):
        d = (m[c + "_a"] - m[c + "_b"]).abs(); both_nan = m[c + "_a"].isna() & m[c + "_b"].isna()
        cons[c] = {"max_abs_diff": float(d[~both_nan].max()), "nan_mismatch": int((m[c + "_a"].isna() != m[c + "_b"].isna()).sum())}
    out["check_nm_null_vs_main_null"] = cons
    # P4: session grouping
    nd = null_frame(R, "add3_null_sess.json", "CTXB+MOT")
    for stat in ("ctx", "session"):
        ob = obs_from(G3, "sess_sev{lab}__{pr}__CTXB+MOT__lroff", stat, lambda lab, pr: ne(lab, pr, "sess_"))
        fams[f"P4|sess|lroff|CTXB+MOT|{stat}|6"] = family(nd, stat, T6, ob, 1000)
    out["families"] = fams
    # permutations with a finite gain in the seated flight-game cell for severity >= 2 (two positive groups),
    # per 1,000-permutation null: they bound the smallest attainable unadjusted p of that test
    fin = {}
    for fname, fs in (("add3_null_lroffp.json", "CTXB+MOT"), ("add3_null_lroffs.json", "CTXB+MOT"), ("add3_null_lroff_block.json", "CTXB+MOT"),
                      ("add3_null_lroff_nr.json", "CTXB+MOTNR"), ("add3_null_lroff_nm.json", "CTXB+MOT")):
        d = null_frame(R, fname, fs); d = d[(d.label == "sev2") & (d.protocol == "cell_seated_flight")]
        fin[fname] = {"finite": int(np.isfinite(d.gain_ctx.to_numpy(float)).sum()), "n_perm": int(d.sim.nunique())}
    out["finite_seated_flight_s2"] = fin
    # power
    out["power_p2"] = power(R, "add3_posctl6.json", {"f6": P6}, ne)
    out["power_p4"] = power(R, "add3_posctl_sess.json", {"f6": P6}, lambda lab, pr: ne(lab, pr, "sess_"))
    out["power_p7"] = power(R, "add3_posctl_diffuse.json", {"f14": FAM, "f6": P6}, ne)
    z = json.load(open(R / "add3_posctl6.json")); out["power_p2"]["b1_check_max_abs_diff"] = z.get("b1_check_max_abs_diff")
    # the same 6-test power of LR+offset recomputed from the stored B1 p-values (inferred groups), for comparison
    out["vrsq"] = json.load(open(R / "add3_vrsq.json"))
    out["vrlens_rw_nodemo"] = json.load(open(R / "add3_vrlens_rw_nodemo.json"))
    gates = sorted((R / "add3_gate").glob("*.json"))
    out["gates"] = {f.stem: json.load(open(f)) for f in gates}
    json.dump(out, open(R / "add3_stats.json", "w"), indent=1)
    print(f"add3_stats.json written ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
