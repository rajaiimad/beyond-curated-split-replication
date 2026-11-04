"""
fa_stats.py -- the single statistics engine of the study. It reads the saved
out-of-group predictions (nothing is refitted) and computes, for every model:

  * ROC-AUC over all positive-negative pairs of target records that were scored by
    the SAME fold model ("within-fold AUC"), averaged over the R repeats;
  * context-stratified ROC-AUC: the same, but a positive record is only compared
    with negative records of the same fold AND the same posture x game x 30-s
    exposure bin:
        AUC_strat = sum_s n+_s n-_s AUC_s / sum_s n+_s n-_s .
    Game and posture are known before a session starts and exposure time at every
    moment, so a score that only encodes them gets AUC_strat = 0.5;

and the paired tests (difference of AUCs on identical target records with the same
bootstrap draws for both scores), with a Holm correction over the primary family.

Bootstrap: B = 10,000 resamples of participant groups (sessions for the
session-grouping sensitivity run); a group drawn k times enters with weight k (its
records with weight k; a cross-group pair with weight k_g k_h; a pair inside a stratum
nested in one unit, e.g. within one session, with weight k -- see Boot.auc(nested=True)).
Two-sided p = 2 * min(P*(delta <= 0), P*(delta >= 0)), floored at 1/B.

Usage: python fa_stats.py <data dir> <results dir>
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
B, SEED, STRATUM_S = 10_000, 0, 30


# ----------------------------------------------------------------------------- engine
class Boot:
    """Participant-group bootstrap over a fixed set of target records.

    Because a resampled group enters with one weight for all its records, the weighted
    AUC of bootstrap draw b is a ratio of two quadratic forms in the group weights c_b:
        AUC_b = c_b' A c_b / c_b' D c_b,
    where A[g, h] counts concordant (positive of group g, negative of group h) pairs in
    the same stratum (ties = 1/2) and D[g, h] counts all such pairs. A and D are
    computed once per score, so B = 10,000 draws cost almost nothing."""

    def __init__(self, groups, b=B, seed=SEED):
        self.ug, self.gi = np.unique(groups, return_inverse=True)
        G = len(self.ug); rng = np.random.default_rng(seed)
        self.C = np.vstack([np.ones((1, G)), rng.multinomial(G, np.full(G, 1 / G), size=b)]).astype(np.float64)

    def pair_mats(self, y, p, strata, valid):
        G = len(self.ug); A = np.zeros((G, G)); D = np.zeros((G, G))
        for s in np.unique(strata[valid]):
            m = valid & (strata == s)
            yy, pp, gg = y[m], p[m], self.gi[m]
            pos = yy == 1; neg = ~pos
            if not pos.any() or not neg.any():
                continue
            D += np.outer(np.bincount(gg[pos], minlength=G), np.bincount(gg[neg], minlength=G))
            ppos, gpos = pp[pos], gg[pos]
            for h in np.unique(gg[neg]):
                nv = np.sort(pp[neg & (gg == h)])
                lt = np.searchsorted(nv, ppos, "left"); le = np.searchsorted(nv, ppos, "right")
                A[:, h] += np.bincount(gpos, weights=lt + 0.5 * (le - lt), minlength=G)
        return A, D

    def auc(self, y, p, strata, valid=None, nested=False):
        """Stratified AUC for the observed sample (row 0) and every bootstrap draw.

        nested=True when every stratum lies inside one resampling unit (within-session or
        within-participant AUC): each copy of a unit drawn k times is a separate unit, and pairs exist
        only inside a copy, so the unit enters with weight k (num = sum_g k_g A_gg), not k^2."""
        valid = np.ones(len(y), bool) if valid is None else valid
        A, D = self.pair_mats(y, p, strata, valid)
        if nested:
            assert not (D - np.diag(np.diag(D))).any(), "nested=True but a stratum spans resampling units"
            num, den = self.C @ np.diag(A), self.C @ np.diag(D)
        else:
            num = np.einsum("bg,gh,bh->b", self.C, A, self.C); den = np.einsum("bg,gh,bh->b", self.C, D, self.C)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def ci(v):
    v = v[1:][~np.isnan(v[1:])]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def p_two_sided(d):
    d = d[1:][~np.isnan(d[1:])]
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d))))


def holm(pvals: dict) -> dict:
    k = sorted(pvals, key=pvals.get); m = len(k); adj, run = {}, 0.0
    for i, name in enumerate(k):
        run = max(run, min(1.0, pvals[name] * (m - i))); adj[name] = run
    return adj


def score_arrays(bt, y, P, FOLD, ctx):
    """Within-fold and context-stratified AUC arrays, averaged over repeats."""
    P = np.atleast_2d(P); FOLD = np.atleast_2d(FOLD)
    pooled, context = [], []
    for r in range(P.shape[0]):
        valid = (FOLD[r] >= 0) & ~np.isnan(P[r])
        fs = FOLD[r].astype(str)
        pooled.append(bt.auc(y, P[r], fs, valid))
        context.append(bt.auc(y, P[r], np.char.add(np.char.add(fs, "|"), ctx), valid))
    with np.errstate(invalid="ignore"):
        return {"pooled": np.nanmean(pooled, axis=0), "context": np.nanmean(context, axis=0)}


# ----------------------------------------------------------------------------- study
KEY = re.compile(r"^(?P<tag>(?:sens[A-Z0-9]+_|rowwise_|control_)?)sev(?P<lab>[12])__(?P<pr>[a-z_]+)__(?P<fs>[A-Z+]+)__(?P<mn>[a-z]+)$")
CELLS = ("cell_seated_flight", "cell_seated_race", "cell_standing_flight", "cell_standing_race")
TARGETS = {  # target record set -> protocols evaluated on exactly these records
    "seated": ["within_seated", "standing_to_seated", "pooled"],
    "standing": ["within_standing", "seated_to_standing", "pooled"],
    "flight": ["within_flight", "race_to_flight", "pooled"],
    "race": ["within_race", "flight_to_race", "pooled"],
    "all": ["pooled"],
    "curated": ["within_curated"],
    **{c: [c] for c in CELLS},
}
CROSS = {"seated": "standing_to_seated", "standing": "seated_to_standing",
         "flight": "race_to_flight", "race": "flight_to_race"}
WITHIN = {"seated": "within_seated", "standing": "within_standing", "flight": "within_flight",
          "race": "within_race", "all": "pooled", "curated": "within_curated", **{c: c for c in CELLS}}
PAIRS = {"telemetry_gain_over_time": ("MOT+TIME", "TIME"),
         "telemetry_gain_over_context": ("MOT+CTX+TIME", "CTX+TIME"),
         "context_gain_over_time": ("CTX+TIME", "TIME"),
         "telemetry_vs_context": ("MOT+TIME", "CTX+TIME"),
         "selfnorm_vs_raw": ("SELFZ+TIME", "MOT+TIME"),
         # revision 2, post hoc (added after the HGB-reg results had been seen; descriptive only):
         # how much of a self-normalised model's context-stratified AUC exposure time alone already gives
         "selfnorm_gain_over_time": ("SELFZ+TIME", "TIME")}


def load_frame(DATA):
    F = pd.read_csv(DATA / "features_w30.csv.gz", usecols=["session_id", "group_id", "campaign", "game",
                                                           "severity", "t_elapsed"])
    S = pd.read_csv(DATA / "sessions.csv")
    F["curated"] = F.session_id.map(S.set_index("session_id").curated_release).astype(bool).to_numpy()
    tbin = np.minimum((F.t_elapsed.to_numpy() // STRATUM_S).astype(int), 300 // STRATUM_S - 1)
    F["s_ctx"] = (F.campaign + "|" + F.game + "|" + pd.Series(tbin, index=F.index).astype(str)).to_numpy()
    return F, S


def target_mask(F, name):
    m = {"seated": F.campaign == "seated", "standing": F.campaign == "standing",
         "flight": F.game == "Flight", "race": F.game == "Race",
         "all": pd.Series(True, index=F.index), "curated": F.curated}
    m.update({f"cell_{p}_{g.lower()}": (F.campaign == p) & (F.game == g)
              for p in ("seated", "standing") for g in ("Flight", "Race")})
    return m[name].to_numpy()


def summary(a, n_rows, n_groups, prev):
    return {"auc": float(a["pooled"][0]), "ci": ci(a["pooled"]), "ctx_auc": float(a["context"][0]),
            "ctx_ci": ci(a["context"]), "n_rows": int(n_rows), "n_groups": int(n_groups), "prevalence": float(prev)}


def main(DATA, R):
    t0 = time.time(); P = R / "preds"
    F, S = load_frame(DATA)
    files = {m.group(0): m.groupdict() for f in sorted(P.glob("*.npz"))
             if (m := KEY.match(f.stem)) and m.group("mn") not in ("lroff", "lroffp", "xgboff", "ctxb")}   # revision 3: fa_rev3_stats.py
    out = {"B": B, "stratum_seconds": STRATUM_S, "auc_definition": "within-fold pairs, mean over repeats",
           "configs": {}, "tests": {}, "pooled_on_target": {}}
    arrays = {}

    for tname, prots in TARGETS.items():
        m = target_mask(F, tname); Ft = F[m]; ctx = Ft.s_ctx.to_numpy().astype(str)
        bt = Boot(Ft.group_id.to_numpy())
        for key, d in files.items():
            if d["tag"] or d["pr"] not in prots:
                continue
            z = np.load(P / f"{key}.npz")
            y = (Ft.severity.to_numpy() >= int(d["lab"])).astype(int)
            a = score_arrays(bt, y, z["p"][:, m], z["fold"][:, m], ctx); arrays[(tname, key)] = a
            if d["pr"] == "pooled" and tname in CROSS:
                out["pooled_on_target"][f"sev{d['lab']}__{tname}__{d['fs']}__{d['mn']}"] = summary(a, m.sum(), len(bt.ug), y.mean())
            if WITHIN.get(tname) == d["pr"] or CROSS.get(tname) == d["pr"]:
                out["configs"][key] = summary(a, m.sum(), len(bt.ug), y.mean())
        print(f"[{time.time()-t0:5.0f}s] target {tname}: {sum(1 for k in arrays if k[0] == tname)} scores", flush=True)

    def delta(name, tname, k1, k2):
        if (tname, k1) in arrays and (tname, k2) in arrays:
            out["tests"][name] = {}
            for sn in ("pooled", "context"):
                dd = arrays[(tname, k1)][sn] - arrays[(tname, k2)][sn]
                out["tests"][name][sn] = {"delta": float(dd[0]), "ci": ci(dd), "p_boot": p_two_sided(dd)}

    for lab in (1, 2):
        for mn in ("hgb", "logreg", "hgbreg"):
            k = lambda pr, fs: f"sev{lab}__{pr}__{fs}__{mn}"
            for tname, pr in list(WITHIN.items()) + list(CROSS.items()):
                for tn, (a, b) in PAIRS.items():
                    delta(f"sev{lab}_{mn}_{tn}__{pr}", tname, k(pr, a), k(pr, b))
            for tname, pr in CROSS.items():
                for fs in ("TIME", "CTX+TIME", "MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME"):
                    delta(f"sev{lab}_{mn}_shift_cost_{fs}__{tname}", tname, k(WITHIN[tname], fs), k(pr, fs))
                    delta(f"sev{lab}_{mn}_pooled_vs_cross_{fs}__{tname}", tname, k("pooled", fs), k(pr, fs))

    # --- primary family (declared in the manuscript, Sec. IV-F): severity >= 2, HGB
    primary = [f"sev2_hgb_shift_cost_MOT+TIME__{t}" for t in CROSS] + \
              [f"sev2_hgb_telemetry_gain_over_context__{pr}" for pr in
               ("within_seated", "within_standing", "within_flight", "within_race", "pooled",
                "seated_to_standing", "standing_to_seated", "flight_to_race", "race_to_flight")]
    missing = [n for n in primary if n not in out["tests"]]
    assert not missing, f"primary tests missing: {missing}"
    adj = holm({n: out["tests"][n]["pooled"]["p_boot"] for n in primary})
    for n in primary:
        out["tests"][n]["pooled"]["p_holm"] = adj[n]
    out["primary_family"] = {"tests": primary, "m": len(primary), "metric": "within-fold ROC-AUC difference",
                             "correction": "Holm"}
    # secondary: the same 13 comparisons with logistic regression, Holm-adjusted among themselves
    second = [n.replace("_hgb_", "_logreg_") for n in primary]
    adj2 = holm({n: out["tests"][n]["pooled"]["p_boot"] for n in second})
    for n in second:
        out["tests"][n]["pooled"]["p_holm"] = adj2[n]
    out["secondary_family_logreg"] = {"tests": second, "m": len(second), "correction": "Holm (within this family)"}
    # revision 2: the same 13 comparisons for every learner form an exploratory comparison family,
    # Holm-adjusted within each learner (HGB-reg is the learner reported in the main text)
    out["families"] = {}
    for mn in ("hgbreg", "hgb", "logreg"):
        fam = [n.replace("_hgb_", f"_{mn}_") for n in primary]
        if not all(n in out["tests"] for n in fam):
            print(f"comparison family for {mn}: incomplete, skipped", flush=True); continue
        adj3 = holm({n: out["tests"][n]["pooled"]["p_boot"] for n in fam})
        for n in fam:
            out["tests"][n]["pooled"]["p_holm"] = adj3[n]
        out["families"][mn] = {"tests": fam, "m": len(fam), "status": "exploratory (fixed after a first run)",
                               "correction": "Holm within the learner (descriptive)"}

    # --- tagged runs (sensitivity, control, record-wise): own record sets
    bad = set(S[S.flag_gap | S.flag_config | S.flag_short].session_id)
    from fa_run import domains
    for key, d in files.items():
        if not d["tag"]:
            continue
        z = np.load(P / f"{key}.npz")
        if d["tag"] == "rowwise_":
            Ft = F.iloc[z["idx"]]; Pm, Fo = z["p"][None, :], z["fold"][None, :]; grp = Ft.group_id.to_numpy()
        else:
            Fx = F[~F.session_id.isin(bad)].reset_index(drop=True) if d["tag"] == "sensCLEAN_" else F
            tgt = domains(Fx)[d["pr"]][1]; Ft = Fx[tgt]; Pm, Fo = z["p"][:, tgt], z["fold"][:, tgt]
            grp = Ft.session_id.to_numpy() if d["tag"] == "sensSESSGROUP_" else Ft.group_id.to_numpy()
        y = (Ft.severity.to_numpy() >= int(d["lab"])).astype(int)
        bt = Boot(grp)
        a = score_arrays(bt, y, Pm, Fo, Ft.s_ctx.to_numpy().astype(str))
        out["configs"][key] = summary(a, len(y), len(bt.ug), y.mean())
    json.dump(out, open(R / "stats.json", "w"), indent=1)
    print(f"stats.json: {len(out['configs'])} configs, {len(out['tests'])} tests ({time.time()-t0:.0f}s)")


def first_run(DATA, RL):
    """Primary family of the FIRST run (leave-one-group-out folds; scores of all folds
    pooled into one AUC, as analysed at the time). Reported only for transparency about
    the change of fold design (Sec. IV-F); written to <RL>/stats.json."""
    from fa_run import domains
    F, _ = load_frame(DATA); D = domains(F); P = RL / "preds"
    y = (F.severity.to_numpy() >= 2).astype(int); one = np.zeros(len(F)).astype(str); tests = {}

    def diff(m, k1, k2):
        bt = Boot(F.group_id.to_numpy()[m]); a = {}
        for k in (k1, k2):
            p = np.load(P / f"{k}.npz")["p"][m]
            a[k] = bt.auc(y[m], p, one[m], ~np.isnan(p))
        dd = a[k1] - a[k2]
        return {"pooled": {"delta": float(dd[0]), "ci": ci(dd), "p_boot": p_two_sided(dd)}}
    for t in CROSS:
        tests[f"sev2_hgb_shift_cost_MOT+TIME__{t}"] = diff(target_mask(F, t), f"sev2__{WITHIN[t]}__MOT+TIME__hgb",
                                                          f"sev2__{CROSS[t]}__MOT+TIME__hgb")
    for pr in ("within_seated", "within_standing", "within_flight", "within_race", "pooled",
               "seated_to_standing", "standing_to_seated", "flight_to_race", "race_to_flight"):
        tests[f"sev2_hgb_telemetry_gain_over_context__{pr}"] = diff(D[pr][1], f"sev2__{pr}__MOT+CTX+TIME__hgb",
                                                                   f"sev2__{pr}__CTX+TIME__hgb")
    adj = holm({n: t["pooled"]["p_boot"] for n, t in tests.items()})
    for n in tests:
        tests[n]["pooled"]["p_holm"] = adj[n]
    json.dump({"B": B, "design": "leave-one-group-out, scores pooled across folds", "tests": tests,
               "primary_family": {"tests": list(tests), "m": len(tests), "correction": "Holm"}},
              open(RL / "stats.json", "w"), indent=1)
    print(f"first run: {sum(t['pooled']['p_holm'] < 0.05 for t in tests.values())} of {len(tests)} primary tests "
          "significant after Holm")


def _selftest():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3); n = 600
    y = rng.integers(0, 2, n); p = np.round(rng.normal(y * 0.5, 1), 1); g = rng.integers(0, 12, n)
    st = rng.integers(0, 4, n).astype(str); one = np.zeros(n).astype(str)
    bt = Boot(g, b=200)
    a = bt.auc(y, p, one)
    assert abs(a[0] - roc_auc_score(y, p)) < 1e-12
    for r in (1, 57, 199):   # bootstrap row r == sklearn AUC on the replicated sample
        rep = np.concatenate([np.repeat(np.where(g == bt.ug[j])[0], int(bt.C[r, j])) for j in range(len(bt.ug))])
        assert abs(a[r] - roc_auc_score(y[rep], p[rep])) < 1e-12
    ref = sum(roc_auc_score(y[st == s], p[st == s]) * y[st == s].sum() * (1 - y[st == s]).sum() for s in np.unique(st)) / \
        sum(y[st == s].sum() * (1 - y[st == s]).sum() for s in np.unique(st))
    assert abs(bt.auc(y, p, st)[0] - ref) < 1e-12
    assert abs(bt.auc(y, st.astype(float), st)[0] - 0.5) < 1e-12      # stratum-only score -> 0.5
    # strata nested in units (within-session AUC): bootstrap row r == stratified AUC on the replicated
    # sample in which every copy of a unit is its own stratum (copies are distinct sessions)
    sub = np.char.add(g.astype(str), np.char.add("|", st))              # strata inside groups
    an = bt.auc(y, p, sub, nested=True)
    assert abs(an[0] - bt.auc(y, p, sub)[0]) < 1e-12
    for r in (1, 57, 199):
        idx, lab = [], []
        for j in range(len(bt.ug)):
            for c in range(int(bt.C[r, j])):
                w = np.where(g == bt.ug[j])[0]; idx.append(w); lab.append(np.char.add(sub[w], f"#{c}"))
        idx = np.concatenate(idx); lab = np.concatenate(lab); yr, pr_ = y[idx], p[idx]
        num = den = 0.0
        for s_ in np.unique(lab):
            m = lab == s_; npos, nneg = yr[m].sum(), (1 - yr[m]).sum()
            if npos and nneg:
                num += roc_auc_score(yr[m], pr_[m]) * npos * nneg; den += npos * nneg
        assert abs(an[r] - num / den) < 1e-12
    # a prevalence-only score fitted leave-one-group-out and pooled across folds is biased
    # below 0.5 (the reason for within-fold pairing)
    yy = rng.binomial(1, 0.4, 4000); gg = rng.integers(0, 20, 4000)
    loo = np.array([yy[gg != k].mean() for k in gg])
    assert roc_auc_score(yy, loo) < 0.5
    print("fa_stats self-test passed: weighted AUC == sklearn on replicated samples; stratified AUC formula; "
          "stratum-only score -> 0.5; nested strata == replicated sample with distinct copies; "
          "leave-one-group-out pooling bias reproduced "
          f"(AUC {roc_auc_score(yy, loo):.3f} for a pure prevalence score)")


if __name__ == "__main__":
    _selftest()
    if len(sys.argv) > 2:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
        RL = Path(sys.argv[2]).parent / "results_logo"          # archived first run, if present
        if (RL / "preds").is_dir() and any((RL / "preds").glob("*.npz")):   # predictions of the first run, if archived
            first_run(Path(sys.argv[1]), RL)
