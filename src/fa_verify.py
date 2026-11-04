"""
fa_verify.py -- consistency checks between the values printed in the paper and the result files.

1. every value the paper and its supplement print is defined in results/paper/values.json (addendum 3:
   values_add3.json), and
   no generated table contains a missing value. The printed keys are listed in
   results/paper/paper_keys.json (every \\V{key} of the submitted manuscript sources, with the
   source file that prints it);
2. every AUC that the paper prints -- overall (".auc") and context-stratified
   (".cauc") -- is recomputed from the saved predictions with scikit-learn, independently
   of fa_stats.py (within-fold pairs, mean over repeats; strata = fold x posture x game x
   30-s bin), and compared with the printed value (tolerance: rounding to 2 decimals);
4. re-checks the qualitative statements at full precision: all claim checks of fa_report
   (significant or not, interval includes 0.5 or not, counts written in words), the count of
   SELFZ+TIME intervals above 0.5, and lists printed intervals whose rounding hides the side
   of 0.5 on which a bound lies.
(Step 3, a list of the numbers typed by hand in the manuscript text, and a scan for placeholders
were run on the manuscript sources before submission; the sources are not part of this repository.)

Usage: python fa_verify.py <project dir>
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))


def wf_auc(y, p, strata):
    """AUC over positive-negative pairs within the same stratum, sklearn per stratum."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s; npos = int(y[m].sum()); nneg = int(m.sum()) - npos
        if npos and nneg:
            num += roc_auc_score(y[m], p[m]) * npos * nneg; den += npos * nneg
    return num / den if den else np.nan


def main(root):
    root = Path(root); P = root / "results" / "paper"
    vals = json.loads((P / "values.json").read_text())
    used = {k: set(v) for k, v in json.loads((P / "paper_keys.json").read_text()).items()}
    if (P / "paper_keys_add3.json").exists():          # addendum 3: keys printed since, in a new file
        for k, v in json.loads((P / "paper_keys_add3.json").read_text()).items():
            used.setdefault(k, set()).update(v)
    # addendum 3: its values live in values_add3.json (keys a3.*), written without changing values.json
    allvals = dict(vals)
    if (P / "values_add3.json").exists():
        allvals.update(json.loads((P / "values_add3.json").read_text()))
    # RQ3 resolution summary (post hoc, descriptive): values_rq3res.json (keys r3r.*) and the keys printed since
    # sample description (descriptive): values_sample.json (keys smp.*) and the keys printed since
    for vf, kf in (("values_rq3res.json", "paper_keys_rq3res.json"), ("values_sample.json", "paper_keys_sample.json")):
        if (P / vf).exists():
            allvals.update(json.loads((P / vf).read_text()))
            for k, v in json.loads((P / kf).read_text()).items():
                used.setdefault(k, set()).update(v)
    missing = sorted(k for k in used if k not in allvals)
    print(f"[1] \\V keys used: {len(used)}; missing: {len(missing)}")
    for k in missing:
        print("    MISSING", k, used[k])
    gen_missing = [f.name for f in P.glob("tab_*.tex")
                   if "MISSING" in f.read_text() or re.search(r"\bnan\b", f.read_text())]
    print(f"[1b] generated tables containing MISSING/nan: {gen_missing}")

    # [2] recompute every used AUC from the predictions
    from fa_run import domains
    from fa_report import PR, FS, MN, TAG
    F = pd.read_csv(root / "data" / "features_w30.csv.gz",
                    usecols=["session_id", "group_id", "campaign", "game", "severity", "t_elapsed"])
    S = pd.read_csv(root / "data" / "sessions.csv")
    F["curated"] = F.session_id.map(S.set_index("session_id").curated_release).astype(bool).to_numpy()
    tbin = np.minimum((F.t_elapsed.to_numpy() // 30).astype(int), 9).astype(str)
    F["ctx"] = (F.campaign + "|" + F.game + "|").to_numpy().astype(str)
    F["ctx"] = np.char.add(F["ctx"].to_numpy().astype(str), tbin)
    bad = set(S[S.flag_gap | S.flag_config | S.flag_short].session_id)
    inv = lambda d: {v: k for k, v in d.items()}
    PRi, FSi, MNi = inv(PR), inv(FS), inv(MN)
    TAGi = {v.rstrip("."): k for k, v in TAG.items() if v}
    tmask = {"seated": F.campaign == "seated", "standing": F.campaign == "standing",
             "flight": F.game == "Flight", "race": F.game == "Race"}
    checked = bad_n = 0
    for k in sorted(used):
        if not (k.endswith(".auc") or k.endswith(".cauc")) or k.startswith(("cur", "aud", "vrsq", "t.", "logo", "null", "ins",
                                                                            "r3", "y.", "sim.", "a3.")):   # revision 3: step [2b]; addendum 3: step [2d]
            continue
        parts = k.split(".")
        metric = parts[-1]; parts = parts[:-1]
        tag = ""
        if parts[0] in TAGi:
            tag = TAGi[parts[0]]; parts = parts[1:]
        on = None
        if parts[1] == "poolon":
            lab, _, on, fs, mn = parts; pr = "pool"
        else:
            lab, pr, fs, mn = parts
        key = f"{tag}sev{lab[1]}__{PRi[pr]}__{FSi[fs]}__{MNi[mn]}"
        z = np.load(root / "results" / "preds" / f"{key}.npz")
        Fx = F[~F.session_id.isin(bad)].reset_index(drop=True) if tag == "sensCLEAN_" else F
        y_all = (Fx.severity.to_numpy() >= int(lab[1])).astype(int)
        if tag == "rowwise_":
            idx = z["idx"]; Pm, Fo, yy, cx = z["p"][None, :], z["fold"][None, :], y_all[idx], Fx.ctx.to_numpy()[idx]
        else:
            tgt = domains(Fx)[PRi[pr]][1] if on is None else tmask[on].to_numpy()
            Pm, Fo, yy, cx = z["p"][:, tgt], z["fold"][:, tgt], y_all[tgt], Fx.ctx.to_numpy()[tgt]
        reps = []
        for r in range(Pm.shape[0]):
            v = (Fo[r] >= 0) & ~np.isnan(Pm[r])
            st = Fo[r][v].astype(str)
            if metric == "cauc":
                st = np.char.add(np.char.add(st, "|"), cx[v])
            reps.append(wf_auc(yy[v], Pm[r][v], st))
        a = np.nanmean(reps)
        printed = float(vals[k].replace("$-$", "-"))
        checked += 1
        if abs(a - printed) > 0.005 + 1e-9:
            bad_n += 1; print("    MISMATCH", k, vals[k], "recomputed", f"{a:.4f}")
    print(f"[2] AUC values recomputed independently with scikit-learn: {checked}; mismatches: {bad_n}")

    # [2b] revision 3: every printed AUC (overall, context-stratified, within-session) and telemetry
    # gain of the offset learners and the earlier learners, recomputed from the predictions
    from fa_report_rev3 import TAG3, FS3, LN
    LNi = inv(LN); NEW = ("lo", "xo", "cb", "lp")
    FS_new = {v: k for k, v in FS3.items() if k.startswith("CTXB")}; FS_old = {v: k for k, v in FS.items()}
    sess = F.session_id.to_numpy().astype(str)

    from functools import lru_cache
    # sessions whose label falls at least once (independent of fa_rev3.falls_mask)
    falls = {}
    for lab_ in (1, 2):
        yy = (F.severity.to_numpy() >= lab_).astype(int); fm = np.zeros(len(F), bool)
        for sid, d in F.assign(_y=yy).groupby("session_id"):
            d = d.sort_values("t_elapsed", kind="stable"); fm[F.index.get_indexer(d.index)] = bool((d._y.diff() < 0).any())
        falls[lab_] = fm

    @lru_cache(maxsize=None)
    def r3_auc(key, metric):
        z = np.load(root / "results" / "preds" / f"{key}.npz")
        pr = key.split("__")[1]; lab = int(key.split("__")[0][-1])
        tgt = domains(F)[pr][1]; y = (F.severity.to_numpy() >= lab).astype(int)[tgt]
        Pm, Fo = z["p"][:, tgt], z["fold"][:, tgt]; reps = []
        for r in range(Pm.shape[0]):
            v = (Fo[r] >= 0) & ~np.isnan(Pm[r]); s_ = Fo[r][v].astype(str)
            if metric == "c":
                s_ = np.char.add(np.char.add(s_, "|"), F.ctx.to_numpy()[tgt][v])
            elif metric == "w":
                s_ = np.char.add(np.char.add(s_, "|"), sess[tgt][v])
            elif metric == "n":                  # within session, sessions whose label also falls
                ft = falls[lab][tgt]; v = v & ft; s_ = np.char.add(np.char.add(Fo[r][v].astype(str), "|"), sess[tgt][v])
            reps.append(wf_auc(y[v], Pm[r][v], s_))
        return np.nanmean(reps)

    def r3_key(parts):
        tagv = ""
        if parts[0] not in ("s1", "s2"):
            tagv, parts = parts[0], parts[1:]
        lab, pr, fs, ln = parts
        fsn = FS_new[fs] if ln in NEW else FS_old[fs]
        tag = {"": "", "ctl": "r3control_" if ln in NEW else "control_", "sess": "sess_", "mrg": "mrg_"}.get(
            tagv, f"r3{tagv}_" if tagv.startswith("pos") else None)
        return f"{tag}sev{lab[1]}__{PRi[pr]}__{fsn}__{LNi[ln]}"
    n3 = bad3 = 0
    # every value of the revision-3 tables (main and control tags), not only those cited in the text
    def printed(k):
        """Values printed in the revision-3 tables: every offset-learner and baseline value, and the
        MOT / MOT+TIME values of default HGB and HGB-reg in the label-shift control table."""
        p = k.split(".")
        if p[-1] not in ("auc", "cauc", "wauc", "nauc", "d", "aucpt", "caucpt", "waucpt", "naucpt", "dpt") or len(p) < 6:
            return False
        tag = p[1] if p[1] not in ("s1", "s2") else ""
        q = p[2:] if tag else p[1:]
        if tag not in ("", "ctl") or len(q) < 4:
            return False
        lab, pr, fs, ln = q[:4]
        if ln in NEW:
            return True
        if ln in ("hgb", "hgbr", "lr") and (p[-1] == "wauc" or p[-2:] == ["w", "d"]):     # earlier learners within sessions (Table S26)
            return True
        return ln in ("hgb", "hgbr") and fs in ("MT", "M") and pr in ("ws", "wt", "pool", "cSF", "cSR", "cTF", "cTR") and p[0] == "r3"
    r3keys = sorted(set(used) | {k for k in vals if k.startswith(("r3.", "r3g.")) and printed(k)})
    for k in r3keys:
        if k not in vals or vals[k] in ("n/e", ""):
            continue
        parts = k.split(".")
        if parts[-1].endswith("pt"):                    # point estimate printed without an interval
            parts = parts[:-1] + [parts[-1][:-2]]
        try:
            if parts[0] == "r3" and parts[-1] in ("auc", "cauc", "wauc", "nauc"):
                key = r3_key(parts[1:-1]); m = {"auc": "o", "cauc": "c", "wauc": "w", "nauc": "n"}[parts[-1]]
                a = r3_auc(key, m)
            elif parts[0] == "r3g" and parts[-1] == "d":
                key = r3_key(parts[1:-2]); m = parts[-2]; d = key.split("__")
                base = (key.split("sev")[0] + f"sev{d[0][-1]}__{d[1]}__CTXB__ctxb") if d[3] in ("lroff", "xgboff", "lroffp") else \
                       f"{key.split('sev')[0]}sev{d[0][-1]}__{d[1]}__{'CTX+TIME' if d[2] == 'MOT+CTX+TIME' else 'TIME'}__{d[3]}"
                a = r3_auc(key, m) - r3_auc(base, m)
            else:
                continue
        except FileNotFoundError as e:
            print("    NOFILE", k, e); bad3 += 1; continue
        n3 += 1
        printed = float(vals[k].replace("$-$", "-").replace("+", ""))
        if abs(a - printed) > 0.005 + 1e-9:
            bad3 += 1; print("    MISMATCH", k, vals[k], "recomputed", f"{a:.4f}")
    print(f"[2b] revision-3 AUCs and gains recomputed independently: {n3}; mismatches: {bad3}")
    bad_n += bad3
    if (root / "results" / "claims_rev3.json").exists():
        c3 = json.loads((root / "results" / "claims_rev3.json").read_text())
        f3 = [k for k, v in c3.items() if v is False]
        print(f"[2c] revision-3 claim checks: {sum(v is True for v in c3.values())} true, {len(f3)} false {f3 if f3 else ''}")
        bad_n += len(f3)

    # [2d] addendum 3 (post hoc; docs/addendum_3.md): the values in results/paper/values_add3.json
    if (P / "values_add3.json").exists():
        bad_n += verify_add3(root, P, vals, r3_auc, PRi)

    # [2e] RQ3 resolution summary (post hoc, descriptive): the values in results/paper/values_rq3res.json
    if (P / "values_rq3res.json").exists():
        bad_n += verify_rq3res(root, P, vals)

    # [2f] the sample's demographic answers (descriptive): the values in results/paper/values_sample.json
    if (P / "values_sample.json").exists():
        bad_n += verify_sample(root, P, vals)

    # [4] qualitative statements at full precision (revision 2)
    st = json.loads((root / "results" / "stats.json").read_text())
    C = st["configs"]; q_bad = 0
    cl = json.loads((root / "results" / "claims.json").read_text())
    false = [k for k, v in cl.items() if v is False]
    print(f"[4a] qualitative claim checks (fa_report.claims, full precision): {sum(v is True for v in cl.values())} true, "
          f"{len(false)} false {false if false else ''}")
    q_bad += len(false)
    WITHIN6 = ["within_curated", "within_seated", "within_standing", "within_flight", "within_race", "pooled"]
    n_above = sum(C[f"sev1__{pr}__SELFZ+TIME__hgb"]["ctx_ci"][0] > 0.5 for pr in WITHIN6)
    ok4 = str(n_above) == str(vals.get("zs1.nciabove"))
    print(f"[4b] SELFZ+TIME (severity>=1, HGB) context-stratified intervals entirely above 0.5 among the six "
          f"same-setting protocols: {n_above} (text: {vals.get('zs1.nciabove')}) -> {'ok' if ok4 else 'MISMATCH'}")
    for pr in WITHIN6:
        lo, hi = C[f"sev1__{pr}__SELFZ+TIME__hgb"]["ctx_ci"]
        print(f"        {pr:16s} [{lo:.4f}, {hi:.4f}]")
    q_bad += (not ok4)
    # printed intervals whose two-decimal rounding hides which side of 0.5 a bound lies on
    amb = []
    for k in sorted(used):
        if not (k.endswith(".ci") or k.endswith(".cci")) or k not in vals:
            continue
        txt = vals[k].replace("$-$", "-")
        if "0.50" in re.findall(r"-?\d\.\d\d", txt):
            amb.append(k)
    print(f"[4c] printed intervals with a bound shown as 0.50 (check the wording near them): {amb}")
    missing += ["qualitative"] * q_bad

    ok = not missing and not bad_n and not gen_missing
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _holm(p):
    """Holm step-down adjustment (own implementation, independent of fa_stats.holm)."""
    k = sorted(p, key=lambda x: (p[x], x)); m = len(k); out, run = {}, 0.0
    for i, x in enumerate(k):
        run = max(run, min(1.0, (m - i) * p[x])); out[x] = run
    return out


def _num(s):
    s = s.replace("$-$", "-").replace("+", "")
    return 0.0 if s == "$<$0.001" else float(s)


def _same_p(printed, x):
    if printed == "$<$0.001":
        return x < 0.001
    return abs(_num(printed) - x) <= 0.0005 + 1e-9 or (printed.count(".") and len(printed.split(".")[1]) == 4 and abs(_num(printed) - x) <= 5e-5 + 1e-9)


def verify_add3(root, P, vals, r3_auc, PRi):
    """[2d] Addendum 3. (i) every printed gain of the two penalized learners and every observed gain
    entering a family, recomputed from the predictions; (ii) every permutation p and Holm-adjusted p
    of every family, recomputed from the null files; (iii) every printed power percentage, recomputed
    from the planted rows (p against the null files, Holm, stored interval); (iv) the claim checks; (v)
    the tables contain no missing value; (vi) no key of values.json is redefined; (vii) the named outputs
    add3_family6.json and add3_sess.json agree with add3_stats.json, and no Holm-adjusted p or decision
    of the overall gain (not a tested estimand) is printed."""
    R = root / "results"; va = json.loads((P / "values_add3.json").read_text()); A = json.load(open(R / "add3_stats.json"))
    bad = 0; clash = sorted(set(va) & set(vals))
    if clash:
        print("    values_add3 redefines existing keys:", clash[:5]); bad += len(clash)
    MET = {"overall": "o", "ctx": "c", "session": "w", "session_nm": "n"}
    LNA = {"lp": "lroffp", "ls": "lroffs"}
    n1 = 0
    for k, v in va.items():                                         # (i) printed P1 gains
        p = k.split(".")
        if p[1] == "g" and p[-1] in ("d", "dpt") and v != "n/e":
            key = f"sev{p[3][-1]}__{PRi[p[4]]}__CTXB+MOT__{LNA[p[2]]}"; base = f"sev{p[3][-1]}__{PRi[p[4]]}__CTXB__ctxb"
            a = r3_auc(key, p[5]) - r3_auc(base, p[5]); n1 += 1
            if abs(a - _num(v)) > 0.005 + 1e-9:
                bad += 1; print("    MISMATCH", k, v, f"{a:.4f}")
    FILES = {"P1": lambda r: (f"add3_null_{r[0]}.json", "", r[1], r[0]), "P2": lambda r: (f"rev3_null_{r[0]}.json", "", r[1], r[0]),
             "P3": lambda r: ({"block": "add3_null_lroff_block.json", "nr": "add3_null_lroff_nr.json", "nm": "add3_null_lroff_nm.json"}[r[0]], "", r[2], r[1]),
             "P4": lambda r: ("add3_null_sess.json", "sess_", r[2], r[1])}
    pre_of = {"P1": lambda r, st, sz: f"a3.n.{ {'lroffp': 'lp', 'lroffs': 'ls'}[r[0]]}.{MET[st]}.{sz}",
              "P2": lambda r, st, sz: f"a3.f6.{ {'lroff': 'lo', 'xgboff': 'xo'}[r[0]]}.{'M' if r[1] == 'CTXB+MOT' else 'Z'}.{MET[st]}",
              "P3": lambda r, st, sz: f"a3.sec.{r[0]}.{MET[st]}", "P4": lambda r, st, sz: f"a3.sess.{MET[st]}"}
    nulls = {}; n2 = n2o = 0
    for name, fam in A["families"].items():                          # (ii) families
        part, *rest = name.split("|"); st, sz = rest[-2], rest[-1]
        fname, tag, fs, ln = FILES[part](rest)
        if fname not in nulls:
            nulls[fname] = pd.DataFrame(json.load(open(R / fname)))
        nd = nulls[fname]; nd = nd[nd.features == fs]
        col = (nd.overall_auc - nd.base_overall_auc) if st == "overall" else nd[{"ctx": "gain_ctx", "session": "gain_session", "session_nm": "gain_session_nm"}[st]]
        ps = {}
        for t, rec in fam["tests"].items():
            lab, pr = t.split("__")
            o = r3_auc(f"{tag}{lab}__{pr}__{fs}__{ln}", MET[st]) - r3_auc(f"{tag}{lab}__{pr}__CTXB__ctxb", MET[st]); n2o += 1
            if abs(o - rec["observed_gain"]) > 1e-9:
                bad += 1; print("    MISMATCH observed gain", name, t, rec["observed_gain"], o)
            v = col[(nd.label == lab) & (nd.protocol == pr)].to_numpy(float); v = v[np.isfinite(v)]
            ps[t] = (1 + int((v >= o).sum())) / (1 + len(v))
        adj = _holm(ps); pre = pre_of[part](rest, st, sz)
        for t in ps:
            b = f"{pre}.{t.split('__')[0].replace('sev', 's')}.{PR_[t.split('__')[1]]}"; n2 += 1
            for suf, x in (("p", ps[t]), ("ph", adj[t])):
                if not _same_p(va[f"{b}.{suf}"], x):
                    bad += 1; print("    MISMATCH", f"{b}.{suf}", va[f"{b}.{suf}"], f"{x:.4f}")
            if va[f"{b}.dec"] != ("yes" if adj[t] < 0.05 and fam["tests"][t]["ci"][0] > 0 else "no"):
                bad += 1; print("    MISMATCH", f"{b}.dec")
        if int(va[f"{pre}.ndec"]) != sum(va[f"{pre}.{t.split('__')[0].replace('sev', 's')}.{PR_[t.split('__')[1]]}.dec"] == "yes" for t in ps):
            bad += 1; print("    MISMATCH", f"{pre}.ndec")
    n3 = 0                                                          # (iii) power
    for setn, fname, fams, specs in (("p2", "add3_posctl6.json", {"f6": None}, {"lroff": ("rev3_null_lroff.json", "CTXB+MOT"), "xgboff": ("rev3_null_xgboff.json", "CTXB+MOT")}),
                                     ("p4", "add3_posctl_sess.json", {"f6": None}, {"lroff": ("add3_null_sess.json", "CTXB+MOT")}),
                                     ("p7", "add3_posctl_diffuse.json", {"f14": None, "f6": None}, {"lroffp": ("add3_null_lroffp.json", "CTXB+MOT"), "xgboff": ("rev3_null_xgboff.json", "CTXB+MOT")})):
        z = json.load(open(R / fname)); df = pd.DataFrame(z["rows"])
        for ln, (nf, fs) in specs.items():
            if nf not in nulls:
                nulls[nf] = pd.DataFrame(json.load(open(R / nf)))
            nd = nulls[nf]; nd = nd[nd.features == fs]
            for stat, cn in (("ctx", "gain_ctx"), ("session", "gain_session")):
                for (lab, pr), d in nd.groupby(["label", "protocol"]):
                    v = np.sort(d[cn].to_numpy(float)); v = v[np.isfinite(v)]
                    m = (df.learner == ln) & (df.label == lab) & (df.protocol == pr)
                    df.loc[m, f"pp_{stat}"] = [(1 + int((v >= g).sum())) / (1 + len(v)) for g in df.loc[m, f"gain_{stat}"]]
        for fam in fams:
            prs = z["families"][fam]
            for stat, s in (("ctx", "c"), ("session", "w")):
                df[f"hh_{fam}_{stat}"] = np.nan
                for _, d in df[df.protocol.isin(prs)].groupby(["seed", "level", "learner"]):
                    adj = _holm({i: p for i, p in zip(d.index, d[f"pp_{stat}"])})
                    for i, x in adj.items():
                        df.loc[i, f"hh_{fam}_{stat}"] = x
                df[f"dd_{fam}_{stat}"] = (df[f"hh_{fam}_{stat}"] < 0.05) & (df[f"ci_lo_{stat}"] > 0)
                pre = {"f14": setn, "f6": setn if setn != "p7" else "p7f6"}[fam]
                for (ln, lev, lab, pr), d in df[df.protocol.isin(prs)].groupby(["learner", "level", "label", "protocol"]):
                    key = f"a3.pw.{pre}.{ {'lroff': 'lo', 'xgboff': 'xo', 'lroffp': 'lp'}[ln]}.{int(round(100 * lev))}.{s}.{lab.replace('sev', 's')}.{PR_[pr]}"
                    if key in va and va[key] != "n/e":
                        n3 += 1
                        if va[key] != f"{100 * d[f'dd_{fam}_{stat}'].mean():.0f}":
                            bad += 1; print("    MISMATCH", key, va[key], f"{100 * d[f'dd_{fam}_{stat}'].mean():.0f}")
    n4 = 0                                                          # VR-LENS record-wise AUCs (P6), from the saved predictions
    F_ = pd.read_csv(root / "data" / "features_w30.csv.gz", usecols=["campaign", "game", "t_elapsed", "severity"])
    tb = np.minimum((F_.t_elapsed.to_numpy() // 30).astype(int), 9).astype(str)
    ctx_ = np.char.add((F_.campaign + "|" + F_.game + "|").to_numpy().astype(str), tb)
    for tag, pre in (("a3rw_sensVRLENSA_", "a3.vrl.rwnd"), ("rowwise_", "a3.vrl.rw")):
        for lab in (1, 2):
            z = np.load(R / "preds" / f"{tag}sev{lab}__within_curated__VRLENS__stack.npz")
            y = (F_.severity.to_numpy()[z["idx"]] >= lab).astype(int)     # the curated records' ratings, as in step [2]
            checks = {f"{pre}.s{lab}.cauc": wf_auc(y, z["p"], np.char.add(np.char.add(z["fold"].astype(str), "|"), ctx_[z["idx"]]))}
            if tag.startswith("a3"):
                checks[f"{pre}.s{lab}.auc"] = roc_auc_score(y, z["p"])
            for k, a_ in checks.items():
                if k in va:
                    n4 += 1
                    if abs(a_ - _num(va[k])) > 0.005 + 1e-9:
                        bad += 1; print("    MISMATCH", k, va[k], f"{a_:.4f}")
    n5 = 0                                                          # named outputs of P2 and P4 (written after the audit)
    for fname, part, extra in (("add3_family6.json", ("P1", "P2"), None), ("add3_sess.json", ("P4",), "power_p4")):
        if not (R / fname).exists():
            bad += 1; print("    MISSING", fname); continue
        z = json.load(open(R / fname)); want = {k for k in A["families"] if k.split("|")[0] in part and (part != ("P1", "P2") or k.endswith("|6"))}
        if set(z["families"]) != want or any(json.dumps(z["families"][k]) != json.dumps(A["families"][k]) for k in want) \
                or (extra and json.dumps(z["power"]) != json.dumps(A[extra])):
            bad += 1; print("    MISMATCH", fname, "differs from add3_stats.json")
        n5 += len(want)
    pk = json.loads((P / "paper_keys_add3.json").read_text())       # the overall gain is not tested: no Holm p or decision printed
    ov = [k for k in pk if re.match(r"a3\.n\.l[ps]\.o\.", k) and k.split(".")[-1] in ("ph", "dec", "ndec")]
    if ov:
        bad += len(ov); print("    overall-gain Holm p or decision printed:", ov)
    cl = json.load(open(R / "claims_add3.json")); fl = [k for k, v in cl.items() if v is not True]       # (iv)
    bad += len(fl)
    tb = [f.name for f in P.glob("tab_a3_*.tex") if "MISSING" in f.read_text() or re.search(r"\bnan\b", f.read_text())
          or "(--)" in f.read_text()]                                                                       # (v)
    bad += len(tb)
    print(f"[2d] addendum 3: {len(va)} values; P1 gains recomputed {n1}; family tests recomputed {n2} (observed gains {n2o}); "
          f"power entries recomputed {n3}; VR-LENS AUCs recomputed {n4}; families in the named P2/P4 files {n5}; claims {sum(v is True for v in cl.values())} true, {len(fl)} false {fl if fl else ''}; "
          f"tables with missing values {tb}; mismatches {bad}")
    return bad


def verify_rq3res(root, P, vals):
    """[2e] RQ3 resolution summary (fa_report_rq3res.py; post hoc, descriptive). Recomputes, by a separate code
    path, every bar (smallest observed gain with permutation p < 0.05 and with p < 0.05/14) from the stored null,
    every median planted gain from the planted rows and the mean overall gain under the null; checks that no
    existing key is redefined and that the claim checks hold."""
    R = root / "results"; va = json.loads((P / "values_rq3res.json").read_text())
    old = dict(vals)
    if (P / "values_add3.json").exists():
        old.update(json.loads((P / "values_add3.json").read_text()))
    clash = sorted(set(va) & set(old)); bad = len(clash)
    if clash:
        print("    values_rq3res redefines existing keys:", clash[:5])
    nd = pd.DataFrame(json.load(open(R / "rev3_null_lroff.json")))
    nd = nd[(nd.features == "CTXB+MOT") & (nd.learner == "lroff")]
    pc = pd.DataFrame(json.load(open(R / "rev3_posctl_seeds.json"))["rows"])
    pc = pc[(pc.learner == "lroff") & (pc.features == "CTXB+MOT+SPK")]

    def bar(v, alpha):
        """Smallest null value x such that any observed gain above x has (1 + #null >= gain)/(1 + n) < alpha."""
        v = np.asarray(v, float); v = v[np.isfinite(v)]; n = len(v)
        for x in np.unique(v):
            if (1 + int((v > x).sum())) / (1 + n) < alpha:
                return float(x)
        return None

    def same(printed, x):
        return (printed == "–") if x is None else (printed != "–" and abs(_num(printed) - x) <= 0.005 + 1e-9)

    n = 0; agg = {"c": {"b05": [], "b14": [], "nm": []}, "w": {"b05": [], "b14": [], "nm": []}}; onm = []
    for (lab, pr), d in nd.groupby(["label", "protocol"]):
        b = f"r3r.{lab.replace('sev', 's')}.{PR_[pr]}"
        for e, col in (("c", "gain_ctx"), ("w", "gain_session")):
            v = d[col].to_numpy(float); v = v[np.isfinite(v)]
            x05, x14 = bar(v, 0.05), bar(v, 0.05 / 14)
            agg[e]["b05"].append(x05); agg[e]["nm"].append(v.mean())
            if x14 is not None:
                agg[e]["b14"].append(x14)
            checks = {f"{b}.{e}.b05": x05, f"{b}.{e}.b14": x14}
            for lev in (70, 80):
                g = pc[(pc.label == lab) & (pc.protocol == pr) & (np.round(pc.level * 100) == lev)][col].to_numpy(float)
                checks[f"{b}.{e}.pm{lev}"] = float(np.median(g[np.isfinite(g)]))
            for k, x in checks.items():
                n += 1
                if not same(va[k], x):
                    bad += 1; print("    MISMATCH", k, va[k], x)
            if va[f"{b}.{e}.nperm"] != str(len(v)):
                bad += 1; print("    MISMATCH", f"{b}.{e}.nperm")
        o = (d.overall_auc - d.base_overall_auc).to_numpy(float); o = o[np.isfinite(o)]; onm.append(o.mean()); n += 1
        if not same(va[f"r3r.o.nm.{lab.replace('sev', 's')}.{PR_[pr]}"], o.mean()):
            bad += 1; print("    MISMATCH", f"r3r.o.nm.{lab}.{pr}")
    for e in ("c", "w"):                                             # ranges printed in the text
        for s, xs in (("b05", agg[e]["b05"]), ("b14", agg[e]["b14"]), ("nm", agg[e]["nm"])):
            for f, x in (("min", min(xs)), ("max", max(xs))):
                n += 1
                if not same(va[f"r3r.{e}.{s}.{f}"], x):
                    bad += 1; print("    MISMATCH", f"r3r.{e}.{s}.{f}", va[f"r3r.{e}.{s}.{f}"], x)
    if va["r3r.o.nneg"] != str(sum(x < 0 for x in onm)) or va["r3r.o.n"] != str(len(onm)):
        bad += 1; print("    MISMATCH r3r.o.nneg / r3r.o.n")
    # the one boosting+offset entry where "interval excludes 0" (wording of the plan) and "above 0" (used) differ
    pa = pd.DataFrame(json.load(open(R / "rev3_posctl_seeds.json"))["rows"])
    x = pa[(pa.learner == "xgboff") & (pa.features == "CTXB+MOT+SPK") & (pa.label == "sev1")
           & (pa.protocol == "cell_seated_flight") & (np.round(pa.level * 100) == 80)]
    sig = x.holm_session < 0.05
    na, ne = int((sig & (x.ci_lo_session > 0)).sum()), int((sig & ((x.ci_lo_session > 0) | (x.ci_hi_session < 0))).sum())
    if (va["r3r.ir.xo.ndiff"], va["r3r.ir.xo.nreal"], va["r3r.ir.xo.pctabove"], va["r3r.ir.xo.pctexcl"]) != \
            (str(ne - na), str(len(x)), f"{100 * na / len(x):.0f}", f"{100 * ne / len(x):.0f}") or va["r3r.ir.xo.pctabove"] != vals["r3pw.xo.80.w.s1.cSF"]:
        bad += 1; print("    MISMATCH r3r.ir.xo.*")
    k05 = max(k for k in range(1001) if (1 + k) / 1001 < 0.05); k14 = max(k for k in range(1001) if (1 + k) / 1001 < 0.05 / 14)
    if (va["r3r.rank05"], va["r3r.rank14"]) != ("50th-largest", "third-largest") or (k05, k14) != (49, 2):
        bad += 1; print("    MISMATCH r3r.rank05 / r3r.rank14")
    if (va["r3r.k05"], va["r3r.k14"]) != (str(k05), str(k14)):
        bad += 1; print("    MISMATCH r3r.k05 / r3r.k14")
    cl = json.load(open(R / "claims_rq3res.json")); fl = [k for k, v in cl.items() if v is not True]
    bad += len(fl)
    print(f"[2e] RQ3 resolution summary: {len(va)} values; bars, planted medians and null means recomputed {n}; "
          f"claims {sum(v is True for v in cl.values())} true, {len(fl)} false {fl if fl else ''}; mismatches {bad}")
    return bad


def verify_sample(root, P, vals):
    """[2f] Sample description (fa_report_sample.py; descriptive). Recounts every answer from the raw session
    files (first record of each FILE.xml; the campaign from the posture field), not from data/sessions.csv:
    by session, by campaign and by distinct answer profile. Checks that no existing key is redefined and that
    the claim checks hold."""
    import os
    va = json.loads((P / "values_sample.json").read_text())
    old = dict(vals)
    for f in ("values_add3.json", "values_rq3res.json"):
        if (P / f).exists():
            old.update(json.loads((P / f).read_text()))
    clash = sorted(set(va) & set(old)); bad = len(clash)
    if clash:
        print("    values_sample redefines existing keys:", clash[:5])
    raw = Path(os.environ.get("UFFCS_RAW_DIR") or root / "data" / "raw" / "UFFCSData" / "RAW_DATABASE" / "RAW_DATABASE")
    files = sorted(raw.glob("*/FILE.xml")) if raw.is_dir() else []
    if not files:
        print(f"[2f] sample description: raw session files not found in {raw} (set UFFCS_RAW_DIR); not checked")
        return 1
    items = {"UserGenere": "gen", "UserAge": "age", "UserExperience": "exp", "UserFlicker": "flk",
             "UserGlassesUse": "gls", "UserVisionProblems": "vis", "UserEyeDominance": "eye"}
    first = re.compile(r"<Data\b.*?</Data>", re.S)
    sess = []
    for f in files:
        rec = first.search(f.read_text(encoding="utf-8-sig")).group(0)
        tag = lambda t: re.search(rf"<{t}>(.*?)</{t}>", rec, re.S).group(1).strip()
        sess.append(({it: tag(it) for it in items}, "seated" if "SENTADO" in tag("UserPosture").upper() else "standing"))
    prof = {tuple(a[it] for it in items): a for a, _ in sess}
    cols = {"s": [a for a, _ in sess], "st": [a for a, c in sess if c == "seated"],
            "sd": [a for a, c in sess if c == "standing"], "g": list(prof.values())}
    from fa_report_sample import ITEMS                           # answer labels and keys only
    n = 0
    for c, rows in cols.items():
        n += 1
        if va[f"smp.n.{c}"] != str(len(rows)):
            bad += 1; print("    MISMATCH", f"smp.n.{c}", va[f"smp.n.{c}"], len(rows))
        for it, ik in items.items():
            labels = {a for a, _, _ in ITEMS[it][2]}
            if {r[it] for r in rows} - labels:
                bad += 1; print("    UNLISTED ANSWER", it, {r[it] for r in rows} - labels)
            for a, ak, _ in ITEMS[it][2]:
                n += 1; x = sum(r[it] == a for r in rows)
                if va[f"smp.{ik}.{ak}.{c}"] != str(x):
                    bad += 1; print("    MISMATCH", f"smp.{ik}.{ak}.{c}", va[f"smp.{ik}.{ak}.{c}"], x)
    if str(len(prof)) != vals.get("aud.ngroups"):
        bad += 1; print("    MISMATCH answer profiles vs aud.ngroups", len(prof), vals.get("aud.ngroups"))
    # the largest answer profile spans both campaigns and both games (Note S4), recounted from the raw files
    key = lambda a: tuple(a[it] for it in items)
    size = {}
    for a, _ in sess:
        size[key(a)] = size.get(key(a), 0) + 1
    top = max(size, key=size.get)
    span = [(c, "Flight" if "_Flight_" in f.parent.name else "Race") for f, (a, c) in zip(files, sess) if key(a) == top]
    if str(size[top]) != vals.get("aud.biggroup") or len({c for c, _ in span}) != 2 or len({g for _, g in span}) != 2:
        bad += 1; print("    MISMATCH largest answer profile", size[top], span)
    if len(va) != n:
        bad += 1; print("    values_sample has keys not recounted:", len(va), n)
    cl = json.load(open(root / "results" / "claims_sample.json")); fl = [k for k, v in cl.items() if v is not True]
    bad += len(fl)
    print(f"[2f] sample description: {len(va)} values; recounted from {len(files)} raw session files {n}; "
          f"claims {sum(v is True for v in cl.values())} true, {len(fl)} false {fl if fl else ''}; mismatches {bad}")
    return bad


from fa_report import PR as PR_  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
