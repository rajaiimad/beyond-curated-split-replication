"""
fa_report_add3.py -- values and tables of addendum 3 (docs/addendum_3.md; post hoc analyses P1-P7).

Reads results/add3_stats.json (and results/rev3_stats.json, read only). Writes NEW files only:
  results/paper/values_add3.json and values_add3.tex   every number of the addendum, keys "a3.*"
                                                         (results/paper/values.json is not changed)
  results/paper/tab_a3_*.tex                           tables
  results/claims_add3.json                             qualitative statements checked at full precision
Value keys:
  a3.g.<ln>.<lab>.<pr>.<o|c|w|n>.{d,ci,dpt}             P1 gains (ln: lp = session-scaled C, ls = 0.1 x that)
  a3.<family>.{m,nperm,floor,ndec,nholm,nraw,minph,minp} and a3.<family>.<lab>.<pr>.{p,ph,nm,nr,dec}
      families: n.<ln>.<o|c|w|n>.<14|6> (P1), f6.<lo|xo>.<M|Z>.<c|w> (P2), sec.<block|nr|nm>.<c|w|n> (P3),
                sess.<c|w> (P4)
  a3.pw.<set>.<ln>.<lev>.<c|w>.<lab>.<pr>  power (% of realizations declared); a3.pwn.<set>.<ln>.<lev>.<c|w>.{n,n50,n50perm}
      sets: p2 (6-test family), p4 (session grouping), p7 / p7f6 (diffuse; 14- and 6-test families)
  a3.vrsq.*, a3.vrl.*, a3.chk.*
Usage: python fa_report_add3.py <project dir>
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fa_report import PR, PR_NAME, Values, a2, ci2, d2, pfmt  # noqa: E402
from fa_report_rev3 import WORDS  # noqa: E402

FAM = ["within_seated", "within_standing", "pooled", "cell_seated_flight", "cell_seated_race",
       "cell_standing_flight", "cell_standing_race"]
P6 = ["within_seated", "within_standing", "pooled"]
LNA = {"lroff": "lo", "xgboff": "xo", "lroffp": "lp", "lroffs": "ls"}
STA = {"overall": "o", "ctx": "c", "session": "w", "session_nm": "n"}
LNAME = {"lroff": "LR+offset", "xgboff": "Boosting+offset", "lroffp": "LR+offset, scaled penalty",
         "lroffs": "LR+offset, 10$\\times$ stronger penalty"}
LAB = {2: r"\sevtwo{}", 1: r"\sevone{}"}
DAG = r"$^\dagger$"


def lab_s(k):
    return k.replace("sev", "s")


# ----------------------------------------------------------------------------- values
def put_family(V, pre, fam):
    V.put(f"{pre}.m", fam["m"]); V.put(f"{pre}.nperm", fam["n_perm_sims"])
    V.put(f"{pre}.floor", pfmt(fam["holm_floor"]) if fam["holm_floor"] is not None else "n/e")
    V.put(f"{pre}.ndec", fam["n_declared"]); V.put(f"{pre}.ndecw", WORDS[fam["n_declared"]])
    V.put(f"{pre}.nholm", fam["n_holm_lt_05"]); V.put(f"{pre}.nraw", fam["n_raw_lt_05"]); V.put(f"{pre}.nraww", WORDS[fam["n_raw_lt_05"]])
    V.put(f"{pre}.minph", pfmt(fam["min_p_holm"])); V.put(f"{pre}.minp", pfmt(fam["min_p_perm"]))
    for k, t in fam["tests"].items():
        lab, pr = k.split("__"); b = f"{pre}.{lab_s(lab)}.{PR[pr]}"
        V.put(f"{b}.p", pfmt(t["p_perm"])); V.put(f"{b}.ph", pfmt(t["p_holm"])); V.put(f"{b}.nm", d2(t["null_mean"]))
        V.put(f"{b}.nr", ci2([t["null_q2.5"], t["null_q97.5"]])); V.put(f"{b}.dec", "yes" if t["declared"] else "no")
        V.put(f"{b}.np", t["n_perm"])
    for k, t in fam["not_estimable"].items():
        lab, pr = k.split("__"); V.put(f"{pre}.{lab_s(lab)}.{PR[pr]}.ph", "n/e")


def put_power(V, pre, pw, fam, learners, ne):
    for key, c in pw["cells"].items():
        ln, lev, lab, pr = key.split("__")
        if f"{fam}|ctx" not in c:
            continue
        for stat, s in (("ctx", "c"), ("session", "w")):
            V.put(f"a3.pw.{pre}.{LNA[ln]}.{int(round(100 * float(lev)))}.{s}.{lab_s(lab)}.{PR[pr]}",
                  "n/e" if ne(int(lab[-1]), pr) else f"{100 * c[f'{fam}|{stat}']['power']:.0f}")
    for key, c in pw["counts"].items():
        f, ln, lev, stat = key.split("|")
        if f != fam:
            continue
        b = f"a3.pwn.{pre}.{LNA[ln]}.{int(round(100 * float(lev)))}.{STA[stat]}"
        V.put(f"{b}.n", c["n_est"]); V.put(f"{b}.n50", c["n50"]); V.put(f"{b}.n50perm", c["n50_perm_only"]); V.put(f"{b}.nany", c["n_any"])


def values(V, A, C3, PW3):
    """PW3: power cells of the single planted channel (results/rev3_stats.json, read only)."""
    ne = lambda lab, pr: C3.get(f"sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3
    # P1 gains
    for key, g in A["gains_p1"].items():
        lab, pr, _, ln = key.split("__"); b = f"a3.g.{LNA[ln]}.{lab_s(lab)}.{PR[pr]}"; n_e = ne(int(lab[-1]), pr)
        for stat, s in STA.items():
            d = g[stat]["delta"]
            if not np.isfinite(d):
                V.put(f"{b}.{s}.d", "n/e"); V.put(f"{b}.{s}.ci", ""); continue
            V.put(f"{b}.{s}.d", "n/e" if n_e else d2(d)); V.put(f"{b}.{s}.ci", "" if n_e or not all(np.isfinite(g[stat]["ci"])) else ci2(g[stat]["ci"]))
            if n_e:
                V.put(f"{b}.{s}.dpt", d2(d))
    # families
    for name, fam in A["families"].items():
        part, *rest = name.split("|")
        if part == "P1":
            ln, fs, stat, size = rest; pre = f"a3.n.{LNA[ln]}.{STA[stat]}.{size}"
        elif part == "P2":
            ln, fs, stat, _ = rest; pre = f"a3.f6.{LNA[ln]}.{'M' if fs == 'CTXB+MOT' else 'Z'}.{STA[stat]}"
        elif part == "P3":
            nm, ln, fs, stat, _ = rest; pre = f"a3.sec.{nm}.{STA[stat]}"
        else:
            _, ln, fs, stat, _ = rest; pre = f"a3.sess.{STA[stat]}"
        put_family(V, pre, fam)
    # power
    put_power(V, "p2", A["power_p2"], "f6", ("lroff", "xgboff"), ne)
    put_power(V, "p4", A["power_p4"], "f6", ("lroff",), lambda lab, pr: False)
    put_power(V, "p7", A["power_p7"], "f14", ("lroffp", "xgboff"), ne)
    put_power(V, "p7f6", A["power_p7"], "f6", ("lroffp", "xgboff"), ne)
    for pre, pw in (("p2", A["power_p2"]), ("p4", A["power_p4"]), ("p7", A["power_p7"])):
        V.put(f"a3.pw.{pre}.nseeds", pw["n_seeds"])
    V.put("a3.pw.mcse", f"{100 * np.sqrt(0.25 / A['power_p2']['n_seeds']):.0f}")
    cells = A["power_p7"]["cells"]
    for lev in A["power_p7"]["levels"]:
        cs = [c for k, c in cells.items() if k.split("__")[1] == f"{lev:.2f}" and k.startswith("lroffp")]
        L = int(round(100 * lev))
        V.put(f"a3.p7.{L}.delta.min", a2(min(c["delta_min"] for c in cs))); V.put(f"a3.p7.{L}.delta.max", a2(max(c["delta_max"] for c in cs)))
        V.put(f"a3.p7.{L}.auc0.min", a2(min(c["auc_S_at_0_min"] for c in cs))); V.put(f"a3.p7.{L}.auc0.max", a2(max(c["auc_S_at_0_max"] for c in cs)))
        V.put(f"a3.p7.{L}.flags", sum(c["flags"] for c in cs))
        V.put(f"a3.p7.{L}.chan.min", a2(min(c["channel_ctx_auc_min"] for c in cs))); V.put(f"a3.p7.{L}.chan.max", a2(max(c["channel_ctx_auc_max"] for c in cs)))
    # P2: the families of 14 (Table VIII rule) on the same six combinations, for comparison
    for ln in ("lroff", "xgboff"):
        for lev in (0.65, 0.70, 0.80):
            for stat, s in (("ctx", "c"), ("session", "w")):
                v = [PW3[f"{ln}__{lev:.2f}__sev{lab}__{pr}"][stat]["power"] for lab in (1, 2) for pr in P6]
                V.put(f"a3.p2.c14.{LNA[ln]}.{int(round(100 * lev))}.{s}.n50", sum(x >= 0.5 for x in v))
    # P3/M3: permutations with a finite gain in the seated flight-game cell (severity >= 2), 1000-permutation nulls
    fin = [c["finite"] for c in A["finite_seated_flight_s2"].values()]
    V.put("a3.cSF.np.min", min(fin)); V.put("a3.cSF.np.max", max(fin))
    # smallest Holm-adjusted p over the four secondary analyses (P3 families and the scaled penalty, families of 14)
    sec = [f["min_p_holm"] for k, f in A["families"].items() if k.startswith("P3|")] + \
          [A["families"][f"P1|lroffp|CTXB+MOT|{s_}|14"]["min_p_holm"] for s_ in ("ctx", "session")]
    V.put("a3.sec.minph", pfmt(min(sec)))
    # P7: realized AUC of the planted score (mean over realizations, estimable label-protocol combinations),
    # diffuse shift against the single channel of Table VIII
    for lev in A["power_p7"]["levels"]:
        L = int(round(100 * lev))
        dc = [c for k, c in cells.items() if k.startswith(f"lroffp__{lev:.2f}__") and not ne(int(k.split("__")[2][-1]), k.split("__")[3])]
        V.put(f"a3.p7.{L}.cauc.min", a2(min(c["channel_ctx_auc"] for c in dc))); V.put(f"a3.p7.{L}.cauc.max", a2(max(c["channel_ctx_auc"] for c in dc)))
        V.put(f"a3.p7.{L}.wauc.min", a2(min(c["channel_session_auc"] for c in dc))); V.put(f"a3.p7.{L}.wauc.max", a2(max(c["channel_session_auc"] for c in dc)))
        V.put(f"a3.p7.{L}.wabove", sum(c["channel_session_auc"] > lev for c in dc)); V.put(f"a3.p7.{L}.nest", len(dc))
        bc = [c for k, c in PW3.items() if k.startswith(f"lroff__{lev:.2f}__") and not ne(int(k.split("__")[2][-1]), k.split("__")[3])]
        V.put(f"a3.b1.{L}.cauc.min", a2(min(c["channel_ctx_auc"] for c in bc))); V.put(f"a3.b1.{L}.cauc.max", a2(max(c["channel_ctx_auc"] for c in bc)))
        V.put(f"a3.b1.{L}.wauc.min", a2(min(c["channel_session_auc"] for c in bc))); V.put(f"a3.b1.{L}.wauc.max", a2(max(c["channel_session_auc"] for c in bc)))
    # P5: VRSQ
    q = A["vrsq"]; V.put("a3.vrsq.n", q["n_sessions"]); V.put("a3.vrsq.allzero", q["n_all_zero"]); V.put("a3.vrsq.groups", q["groups"])
    for m, c in q["by_ingame_max"].items():
        V.put(f"a3.vrsq.max{m}.n", c["n"]); V.put(f"a3.vrsq.max{m}.az", c["all_zero"])
    g2 = q["ge2_vs_lt2"]; t = g2["table_[allzero,not]_ge2_then_lt2"]
    V.put("a3.vrsq.ge2.az", t[0][0]); V.put("a3.vrsq.ge2.n", sum(t[0])); V.put("a3.vrsq.lt2.az", t[1][0]); V.put("a3.vrsq.lt2.n", sum(t[1]))
    V.put("a3.vrsq.ge2.pct", f"{100 * g2['share_ge2']:.0f}"); V.put("a3.vrsq.lt2.pct", f"{100 * g2['share_lt2']:.0f}")
    V.put("a3.vrsq.diff", f"{100 * g2['difference']:+.0f}".replace("-", "$-$"))
    V.put("a3.vrsq.diff.ci", "[" + ", ".join(f"{100 * x:.0f}".replace("-", "$-$") for x in g2["diff_ci_group_bootstrap"]) + "]")
    V.put("a3.vrsq.fisher.p", pfmt(g2["fisher_p_two_sided"])); V.put("a3.vrsq.fisher.or", a2(g2["fisher_odds_ratio"]))
    SUMK = {"ing_max": "max", "ing_mean": "mean", "ing_frac_ge1": "f1", "ing_frac_ge2": "f2"}
    for setn, sk in (("all_files", "all"), ("without_all_zero", "nz"), ("allzero_indicator_all_files", "az")):
        d = q[setn]; V.put(f"a3.vrsq.{sk}.m", d["m"])
        rh = [c["rho"] for c in d["correlations"].values()]
        V.put(f"a3.vrsq.{sk}.rho.min", a2(min(rh))); V.put(f"a3.vrsq.{sk}.rho.max", a2(max(rh)))
        V.put(f"a3.vrsq.{sk}.nholm", sum(c["p_holm"] < 0.05 for c in d["correlations"].values()))
        V.put(f"a3.vrsq.{sk}.nholmw", WORDS[sum(c["p_holm"] < 0.05 for c in d["correlations"].values())])
        V.put(f"a3.vrsq.{sk}.nraw", sum(c["p_boot"] < 0.05 for c in d["correlations"].values()))
        V.put(f"a3.vrsq.{sk}.minph", pfmt(min(c["p_holm"] for c in d["correlations"].values())))
        for k, c in d["correlations"].items():
            a, b = k.split("~"); bb = {"vrsq_post_total": "post", "vrsq_delta_total": "delta", "all_zero": "az"}[b]
            kk = f"a3.vrsq.{sk}.{SUMK[a]}.{bb}"
            V.put(f"{kk}.rho", a2(c["rho"])); V.put(f"{kk}.ci", ci2(c["ci"])); V.put(f"{kk}.p", pfmt(c["p_boot"])); V.put(f"{kk}.ph", pfmt(c["p_holm"]))
    V.put("a3.vrsq.ndates", len(q["by_date"]))
    for nm, kk in (("clustering_dates", "dates"), ("clustering_groups_with_2plus_sessions", "groups")):
        c = q[nm]; V.put(f"a3.vrsq.cl.{kk}.obs", c["homogeneous_observed"]); V.put(f"a3.vrsq.cl.{kk}.units", c["units"])
        V.put(f"a3.vrsq.cl.{kk}.null", f"{c['homogeneous_null_mean']:.1f}"); V.put(f"a3.vrsq.cl.{kk}.p", pfmt(c["p_one_sided"]))
    for dt, c in q["by_date"].items():
        V.put(f"a3.vrsq.date.{dt}", f"{c['all_zero']}/{c['n']}")
    # P6: VR-LENS
    r = A["vrlens_rw_nodemo"]
    V.put("a3.vrl.rwnd.acc", f"{100 * r['accuracy_4class']:.1f}"); V.put("a3.vrl.rwnd.f1", a2(r["macro_f1_4class"]))
    # three decimals: the two-decimal value lies at a rounding boundary (a repeated run gave 0.5350 against 0.5349;
    # docs/audit_addendum3.md), at three decimals both runs print the same value
    V.put("a3.vrl.rwnd.f1x", f"{r['macro_f1_4class']:.3f}")
    V.put("a3.vrl.rwnd.maj", f"{100 * r['majority_class_share']:.1f}"); V.put("a3.vrl.rwnd.secs", r["secs"])
    for lab in (1, 2):
        V.put(f"a3.vrl.rwnd.s{lab}.auc", a2(r[f"auc_sev{lab}"])); V.put(f"a3.vrl.rwnd.s{lab}.cauc", a2(r[f"ctx_auc_sev{lab}"]))
        V.put(f"a3.vrl.rw.s{lab}.cauc", a2(r[f"stored_with_demographics_ctx_auc_sev{lab}"]))
    # checks
    sci = lambda x: "0" if x == 0 else f"{x:.1e}"                      # largest differences of the reproducibility checks
    V.put("a3.chk.lroffp.maxdiff", sci(A['check_lroffp_vs_rev3_stats']['max_abs_diff_gain']))
    V.put("a3.chk.fam14.maxdiff", sci(A['check_14_families_vs_rev3_stats']['max_abs_diff_p_holm']))
    c = A["check_nm_null_vs_main_null"]
    V.put("a3.chk.nm.sims", c["sims"]); V.put("a3.chk.nm.maxdiff", sci(max(c[k]['max_abs_diff'] for k in ('gain_ctx', 'gain_session'))))
    V.put("a3.chk.b1.maxdiff", sci(A['power_p2']['b1_check_max_abs_diff']))


# ----------------------------------------------------------------------------- tables
def _gc(V, b, s, ph=None):
    d = V.v.get(f"{b}.{s}.d", "--")
    if d == "n/e" and f"{b}.{s}.dpt" in V.v:
        d = V.v[f"{b}.{s}.dpt"] + DAG
    out = f"{d} {V.v.get(f'{b}.{s}.ci', '')}".strip()
    if ph is not None:
        out += f" ({V.v.get(ph, '--')})"
    return out


def table_p1(V):
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Post hoc (specified after the main results; addendum 3, P1): telemetry gain of LR+offset with a penalty scaled to the number of training sessions ($C$ = training sessions / training records, \V{a3.n.lp.c.14.nperm} permutations) and with a ten times stronger penalty ($C$ = 0.1 $\times$ training sessions / training records, \V{a3.n.ls.c.14.nperm} permutations), CTXB+MOT, both labels, the seven protocols of the primary families. "
         r"Gain [95\,\% group-bootstrap interval] and, in parentheses, Holm-adjusted permutation $p$ within the family of 14 (between sessions: context-stratified AUC; within sessions: within-session AUC). The overall gain is not a tested estimand of the plan and is shown without a $p$. "
         r"$^\dagger$: fewer than three groups with a positive record, point estimate without interval.}",
         r"\label{tab:a3_p1}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Scaled penalty} & \multicolumn{3}{c}{10$\times$ stronger penalty} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Label & Protocol & Overall & Between & Within & Overall & Between & Within \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAM):
            cells = []
            for ln in ("lp", "ls"):
                b = f"a3.g.{ln}.s{lab}.{PR[pr]}"
                cells += [_gc(V, b, "o"), _gc(V, b, "c", f"a3.n.{ln}.c.14.s{lab}.{PR[pr]}.ph"), _gc(V, b, "w", f"a3.n.{ln}.w.14.s{lab}.{PR[pr]}.ph")]
            L.append(f"{LAB[lab] if i == 0 else ''} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_family6(V):
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Post hoc (specified after the main results; addendum 3, P2 and P4): the six same-setting tests (within seated, within standing, pooled; both labels) as a family of 6, Holm within each family; rule unchanged (Holm-adjusted $p<0.05$ and interval above 0). "
         r"Holm-adjusted $p$ within the family of 6 for LR+offset with the 28 telemetry features (MOT) and with self-normalized features (SELFZ; \V{a3.f6.lo.M.c.nperm} permutations), boosting+offset (MOT; \V{a3.f6.xo.M.c.nperm} permutations), and LR+offset with every session as its own group (P4: gain [95\,\% session-bootstrap interval] and Holm $p$; null with session grouping, \V{a3.sess.c.nperm} permutations). "
         r"Smallest attainable Holm-adjusted $p$: \V{a3.f6.lo.M.c.floor} (1000 permutations), \V{a3.f6.xo.M.c.floor} (400). The real-data $p$-values of these six tests were known before this family was defined (they are the same tests as in the families of 14).}",
         r"\label{tab:a3_family6}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{LR + offset, MOT} & \multicolumn{2}{c}{LR + offset, SELFZ} & \multicolumn{2}{c}{Boosting + offset} & \multicolumn{2}{c}{Session grouping, LR + offset} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Label & Protocol & B & W & B & W & B & W & B & W \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(P6):
            b = f"s{lab}.{PR[pr]}"
            cells = [V.v.get(f"a3.f6.{f}.{s}.{b}.ph", "--") for f in ("lo.M", "lo.Z", "xo.M") for s in ("c", "w")]
            cells += [f"{V.v.get(f'r3g.sess.{b}.M.lo.{s}.d', '--')} {V.v.get(f'r3g.sess.{b}.M.lo.{s}.ci', '')} ({V.v.get(f'a3.sess.{s}.{b}.ph', '--')})"
                      for s in ("c", "w")]
            L.append(f"{LAB[lab] if i == 0 else ''} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def _power_rows(V, pre, ln, prs, levels):
    rows = []
    for pr in prs:
        cells = []
        for lab in (2, 1):
            for s in ("c", "w"):
                for lev in levels:
                    cells.append(V.v.get(f"a3.pw.{pre}.{ln}.{int(round(100 * lev))}.{s}.s{lab}.{PR[pr]}", "--"))
        rows.append(f"{PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
    return rows


def _power_head(levels):
    n = 2 * len(levels)
    return [r"\toprule", f" & \\multicolumn{{{n}}}{{c}}{{\\sevtwo{{}}}} & \\multicolumn{{{n}}}{{c}}{{\\sevone{{}}}} \\\\",
            f"\\cmidrule(lr){{2-{1 + n}}}\\cmidrule(lr){{{2 + n}-{1 + 2 * n}}}",
            " & " + " & ".join([f"\\multicolumn{{{len(levels)}}}{{c}}{{B}} & \\multicolumn{{{len(levels)}}}{{c}}{{W}}"] * 2) + r" \\",
            "Protocol & " + " & ".join([" & ".join(f"{x:.2f}" for x in levels)] * 4) + r" \\", r"\midrule"]


def table_power6(V):
    lv = (0.65, 0.70, 0.80); n = 4 * len(lv)
    L = [r"\begin{table*}[ht]", r"\centering", r"\footnotesize",
         r"\caption{Post hoc (specified after the main results; addendum 3, P2 and P4): what the decision rule would have detected with the six same-setting tests as a family of 6. Format of Table~\mref{tab:mde}: percentage of \V{a3.pw.p2.nseeds} realizations of the planted channel of addendum 1 (added to the 28 telemetry features) in which a planted signal of the given within-stratum AUC is declared (Holm-adjusted permutation $p<0.05$ within the family of 6 against the label-transplant null of the learner, and a bootstrap interval of the gain above 0). "
         r"The realizations are those of Table~\mref{tab:mde}, refitted for these protocols (the refitted planted gains reproduce the stored ones; largest difference \V{a3.chk.b1.maxdiff}). Session grouping: every session its own group for the folds, the null (\V{a3.sess.c.nperm} permutations) and the bootstrap. "
         r"Conditional on the observed data; near 50\,\% the Monte Carlo standard error is about \V{a3.pw.mcse} points. B: between sessions; W: within sessions.}",
         r"\label{tab:a3_power6}", r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}l" + "c" * n + r"@{}}"] + _power_head(lv)
    for pre, ln, title in (("p2", "lo", "LR+offset (inferred groups)"), ("p2", "xo", "Boosting+offset (inferred groups)"),
                           ("p4", "lo", "LR+offset, every session its own group")):
        L.append(f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        L += _power_rows(V, pre, ln, P6, lv)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_diffuse(V):
    lv = (0.70, 0.80); n = 4 * len(lv)
    L = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
         r"\caption{Post hoc (specified after the main results; addendum 3, P7): a diffuse planted signal. In each of \V{a3.pw.p7.nseeds} realizations, five of the 14 non-rotation telemetry features are drawn at random and the positive records of each are shifted by the same amount on the normal-score scale of their context stratum (values mapped back to values observed in that stratum), calibrated so that the within-stratum AUC of the sum of the five shifted features is 0.70 or 0.80. "
         r"Percentage of realizations in which the signal is declared (Holm-adjusted permutation $p<0.05$ within the family of 14 against the learner's label-transplant null, and a bootstrap interval of the gain above 0); format of Table~\mref{tab:mde}. "
         r"LR+offset with the session-scaled penalty (null: \V{a3.n.lp.c.14.nperm} permutations) and boosting+offset (\V{a3.f6.xo.M.c.nperm} permutations). n/e: not estimable (fewer than three groups with a positive record).}",
         r"\label{tab:a3_diffuse}", r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}l" + "c" * n + r"@{}}"] + _power_head(lv)
    for ln, title in (("lp", "LR+offset, session-scaled penalty"), ("xo", "Boosting+offset")):
        L.append(f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        L += _power_rows(V, "p7", ln, FAM, lv)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def _pcell(V, gk, fk, s):
    d = V.v.get(f"{gk}.{s}.d", "--")
    if d == "n/e":
        if f"{gk}.{s}.dpt" in V.v:
            return f"{V.v[f'{gk}.{s}.dpt']}{DAG} ({V.v.get(f'{fk}.ph', '--')})"
        return "n/e"
    return f"{d} {V.v.get(f'{gk}.{s}.ci', '')} ({V.v.get(f'{fk}.ph', '--')})"


def table_secondary(V):
    """Supplementary Table S20 with the secondary nulls extended to 1000 permutations (P3)."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Secondary analyses of the telemetry gain with 1000-permutation nulls (post hoc, specified after the main results; addendum 3, P3; replaces the 200-permutation version). Main learner unless stated; gain [95\,\% group-bootstrap interval] and, in parentheses, Holm-adjusted permutation $p$ within the family of 14 (13 for falling sessions, because for \sevone{} no session of the seated flight-game cell has a falling rating). "
         r"Main: Holm $p$ of families A and B (\V{r3n.lo.M.c.nperm} permutations). Group-block null: the same gains against a null that transplants whole groups (\V{a3.sec.block.c.nperm} permutations). Falling sessions: within-session gain on the sessions whose rating also falls (\V{a3.sec.nm.n.nperm} permutations). Scaled penalty: LR+offset with $C$ = training sessions / training records (\V{a3.n.lp.c.14.nperm} permutations). Without rotation: the 14 telemetry features not derived from the rotation fields (\V{a3.sec.nr.c.nperm} permutations). "
         r"Smallest attainable Holm-adjusted $p$: \V{a3.sec.block.c.floor} (\V{a3.sec.nm.n.floor} for falling sessions). n/e: not estimable; $^\dagger$: fewer than three groups with a positive record, no interval.}",
         r"\label{tab:a3_secondary}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Main ($p_\mathrm{Holm}$)} & \multicolumn{2}{c}{Group-block null ($p_\mathrm{Holm}$)} & Falling sessions & \multicolumn{2}{c}{Scaled penalty} & Without rotation \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}",
         r"Label & Protocol & B & W & B & W & W & B & W & B / W \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAM):
            b = f"s{lab}.{PR[pr]}"
            main = [V.v.get(f"r3n.{b}.M.lo.{s}.ph", "--") for s in ("c", "w")]
            blk = [V.v.get(f"a3.sec.block.{s}.{b}.ph", "--") for s in ("c", "w")]
            nm = _pcell(V, f"r3g.{b}.M.lo", f"a3.sec.nm.n.{b}", "n")
            pen = [_pcell(V, f"a3.g.lp.{b}", f"a3.n.lp.{s}.14.{b}", s) for s in ("c", "w")]
            nrd = lambda s: (V.v[f"r3g.{b}.MNR.lo.{s}.dpt"] + DAG) if V.v.get(f"r3g.{b}.MNR.lo.{s}.d") == "n/e" and f"r3g.{b}.MNR.lo.{s}.dpt" in V.v \
                else V.v.get(f"r3g.{b}.MNR.lo.{s}.d", "--")
            nr = f"{nrd('c')} ({V.v.get(f'a3.sec.nr.c.{b}.ph', '--')}) / {nrd('w')} ({V.v.get(f'a3.sec.nr.w.{b}.ph', '--')})"
            L.append(f"{LAB[lab] if i == 0 else ''} & {PR_NAME[pr]} & " + " & ".join(main + blk + [nm] + pen + [nr]) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_falling(V):
    """Table S22 with the 1000-permutation falling-session null (P3); otherwise as fa_report_rev3.table_falling."""
    from fa_report_rev3 import _a
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Within-session AUC on the sessions whose rating also falls, where exposure time cannot order the records (addendum B3), for the context baseline and both offset learners in the protocols of the primary families: AUC of the baseline, and gain of each learner over it, with 95\,\% group-bootstrap intervals (nested weights); in parentheses, the Holm-adjusted permutation $p$ of LR+offset against the null of this estimand, extended post hoc from 200 to \V{a3.sec.nm.n.nperm} permutations (smallest attainable Holm-adjusted $p$ \V{a3.sec.nm.n.floor}; boosting+offset was not tested). n/e: no session with a falling rating and both classes. $^\dagger$: fewer than three groups with a positive record, so no interval is reported.}",
         r"\label{tab:supp_falling}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Label & Protocol & Baseline AUC & LR + offset: gain ($p_\mathrm{Holm}$) & Boosting + offset: gain \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAM):
            b = f"s{lab}.{PR[pr]}"
            base = f"{_a(V, f'r3.{b}.B.cb', 'nauc')} {V.v.get(f'r3.{b}.B.cb.nci', '')}".strip()
            lo = _pcell(V, f"r3g.{b}.M.lo", f"a3.sec.nm.n.{b}", "n")
            xo = _gc(V, f"r3g.{b}.M.xo", "n")
            L.append(f"{LAB[lab] if i == 0 else ''} & {PR_NAME[pr]} & {base} & {lo} & {xo} \\\\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_nr(V):
    """Table S23 with the 1000-permutation null of the analysis without rotation (P3)."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{LR+offset with the 14 telemetry features not derived from the rotation fields (context baseline plus the 14 non-rotation features, CTXB+MOTNR; addendum B2c): gain over the context baseline with 95\,\% group-bootstrap intervals and, in parentheses, the Holm-adjusted permutation $p$ within the family of 14 against a null extended post hoc from 200 to \V{a3.sec.nr.c.nperm} permutations (smallest attainable Holm-adjusted $p$ \V{a3.sec.nr.c.floor}). For comparison, the gains with all 28 features (Table~\mref{tab:gain}, Table~S14). $^\dagger$: fewer than three groups with a positive record, so no interval is reported.}",
         r"\label{tab:supp_nr}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Without rotation (14 features)} & All 28 features \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-6}",
         r"Label & Protocol & Overall & Between sessions & Within sessions & Between / within \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAM):
            b = f"s{lab}.{PR[pr]}"; g = f"r3g.{b}.MNR.lo"
            full = f"{_gc(V, f'r3g.{b}.M.lo', 'c').split(' [')[0]} / {_gc(V, f'r3g.{b}.M.lo', 'w').split(' [')[0]}"
            L.append(f"{LAB[lab] if i == 0 else ''} & {PR_NAME[pr]} & {_gc(V, g, 'o')} & {_gc(V, g, 'c', f'a3.sec.nr.c.{b}.ph')} & "
                     f"{_gc(V, g, 'w', f'a3.sec.nr.w.{b}.ph')} & {full} \\\\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_nullranges(V):
    """Table S25 with the 1000-permutation group-block null (P3)."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Gain of LR+offset (MOT) under the two label-transplant nulls: mean and 2.5--97.5 percentiles over the permutations of the session-level null (\V{r3n.lo.M.c.nperm} permutations) and of the group-block null (extended post hoc from 200 to \V{a3.sec.block.c.nperm} permutations), next to the observed gain, between sessions (B) and within sessions (W). For the seated flight-game cell (\sevtwo{}), only the permutations with a finite gain enter (\V{r3n.cSF.np} and \V{a3.cSF.np.max}). $^\dagger$: fewer than three groups with a positive record, so no interval is reported.}",
         r"\label{tab:supp_nullranges}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Observed gain} & \multicolumn{2}{c}{Session-level null} & \multicolumn{2}{c}{Group-block null} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Label & Protocol & B & W & B & W & B & W \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAM):
            b = f"s{lab}.{PR[pr]}"
            cells = [_gc(V, f"r3g.{b}.M.lo", "c").split(" [")[0], _gc(V, f"r3g.{b}.M.lo", "w").split(" [")[0]]
            cells += [f"{V.v.get(f'r3n.{b}.M.lo.{s}.nm', '--')} {V.v.get(f'r3n.{b}.M.lo.{s}.nr', '')}".strip() for s in ("c", "w")]
            cells += [f"{V.v.get(f'a3.sec.block.{s}.{b}.nm', '--')} {V.v.get(f'a3.sec.block.{s}.{b}.nr', '')}".strip() for s in ("c", "w")]
            L.append(f"{LAB[lab] if i == 0 else ''} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_vrsq(V, q):
    SUMN = {"max": "Highest rating", "mean": "Mean rating", "f1": r"Share $\geq$1", "f2": r"Share $\geq$2"}
    L = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
         r"\caption{Post hoc (specified after the main results; addendum 3, P5): VRSQ files that are zero in every item. "
         r"Top: all-zero files by the highest in-game rating of the session (\V{a3.vrsq.n} sessions with a VRSQ file); rating $\geq$2 against $<$2: difference in shares \V{a3.vrsq.diff} percentage points, 95\,\% group-bootstrap interval \V{a3.vrsq.diff.ci}, Fisher's exact test $p=\V{a3.vrsq.fisher.p}$. "
         r"Bottom: Spearman's $\rho$ [95\,\% group-bootstrap interval] (Holm-adjusted two-sided bootstrap $p$ within each column set; exploratory). Without all-zero files the correlations select on a VRSQ total above 0; the all-zero indicator column does not condition on it.}",
         r"\label{tab:a3_vrsq}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Highest in-game rating & 0 & 1 & 2 & 3 & \\",
         "All-zero / sessions & " + " & ".join(f"{V.v.get(f'a3.vrsq.max{m}.az', '0')}/{V.v.get(f'a3.vrsq.max{m}.n', '0')}" for m in range(4)) + r" & \\",
         r"\midrule",
         r" & \multicolumn{2}{c}{All files} & \multicolumn{2}{c}{Without all-zero files} & All-zero \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"In-game summary & Post & Change & Post & Change & indicator \\", r"\midrule"]
    for s in ("max", "mean", "f1", "f2"):
        cells = [f"{V.v[f'a3.vrsq.{st}.{s}.{b}.rho']} {V.v[f'a3.vrsq.{st}.{s}.{b}.ci']} ({V.v[f'a3.vrsq.{st}.{s}.{b}.ph']})"
                 for st, b in (("all", "post"), ("all", "delta"), ("nz", "post"), ("nz", "delta"), ("az", "az"))]
        L.append(f"{SUMN[s]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_allzero(V, q):
    L = [r"\begin{table}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Post hoc (addendum 3, P5): the \V{a3.vrsq.allzero} questionnaire files that are zero in every item, by recording date. Group: inferred participant group (answer profile). The archive does not document how these files arose; no origin is inferred here. "
         r"Dates on which all or none of the sessions have an all-zero file: \V{a3.vrsq.cl.dates.obs} of \V{a3.vrsq.cl.dates.units} (random assignment: \V{a3.vrsq.cl.dates.null} on average, one-sided $p=\V{a3.vrsq.cl.dates.p}$); groups with two or more sessions whose sessions all agree: \V{a3.vrsq.cl.groups.obs} of \V{a3.vrsq.cl.groups.units} (random assignment: \V{a3.vrsq.cl.groups.null}, $p=\V{a3.vrsq.cl.groups.p}$).}",
         r"\label{tab:a3_allzero}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llllcc@{}}", r"\toprule",
         r"Date & Start & Campaign & Game & Group & Highest rating \\", r"\midrule"]
    for r in q["all_zero_files"]:
        start = ":".join(f"{int(x):02d}" for x in r["session_id"].rsplit("_", 3)[-3:])     # session start time, zero-padded
        L.append(f"{r['date']} & {start} & {r['campaign']} & {r['game']} & {r['group_id']} & {r['ing_max']} \\\\")
    L += [r"\midrule", r"\multicolumn{6}{@{}l}{All-zero / sessions per date: " +
          "; ".join(f"{d} {c['all_zero']}/{c['n']}" for d, c in q["by_date"].items()) + r"} \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_vrlens(V):
    return "\n".join([
        r"\begin{table}[ht]", r"\centering", r"\footnotesize",
        r"\caption{Post hoc (addendum 3, P6): the re-implemented VR-LENS detector under the original record-wise 10-fold protocol with and without the \V{vrl.ndemo} demographic answers, and under group-disjoint evaluation. Four-class accuracy; AUC and context-stratified AUC (fold $\times$ context stratum) for \sevtwo{}. The record-wise run without demographic answers is new; the other rows are the stored runs.}",
        r"\label{tab:a3_vrlens}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"Evaluation & Acc (\%) & AUC & Ctx-strat. \\", r"\midrule",
        f"Record-wise, all ten inputs & {V.v['vrl.rw.acc']} & {V.v['rw.s2.wc.VRL.stack.auc']} & {V.v['a3.vrl.rw.s2.cauc']} \\\\",
        f"Record-wise, without demographic answers & {V.v['a3.vrl.rwnd.acc']} & {V.v['a3.vrl.rwnd.s2.auc']} & {V.v['a3.vrl.rwnd.s2.cauc']} \\\\",
        f"Group-disjoint, all ten inputs & {V.v['vrl.grp.acc']} & {V.v['vrl.s2.wc.VRL.stack.auc']} & {V.v['vrl.s2.wc.VRL.stack.cauc']} \\\\",
        f"Majority class & {V.v['a3.vrl.rwnd.maj']} & -- & -- \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])


# ----------------------------------------------------------------------------- claims
def claims(A):
    """Qualitative statements of docs/results_addendum3.md, checked at full precision."""
    F = A["families"]; out = {}
    out["p1_no_test_declared_either_penalty"] = all(F[f"P1|{ln}|CTXB+MOT|{s}|{m}"]["n_declared"] == 0
                                                    for ln in ("lroffp", "lroffs") for s in ("ctx", "session") for m in ("14", "6"))
    out["p2_no_real_test_declared_in_6_family"] = all(F[k]["n_declared"] == 0 for k in F if k.startswith("P2|"))
    out["p3_no_secondary_test_declared"] = all(F[k]["n_declared"] == 0 for k in F if k.startswith("P3|"))
    out["p4_no_test_declared_session_grouping"] = all(F[k]["n_declared"] == 0 for k in F if k.startswith("P4|"))
    out["check_lroffp_gains_identical_to_rev3"] = A["check_lroffp_vs_rev3_stats"]["max_abs_diff_gain"] < 1e-12
    out["check_14_families_identical_to_rev3"] = A["check_14_families_vs_rev3_stats"]["max_abs_diff_p_holm"] < 1e-12
    c = A["check_nm_null_vs_main_null"]
    out["check_falling_null_reproduces_main_null"] = c["sims"] == 1000 and all(c[k]["max_abs_diff"] < 1e-12 and c[k]["nan_mismatch"] == 0
                                                                                for k in ("gain_ctx", "gain_session"))
    out["check_refitted_planted_gains_reproduce_b1"] = A["power_p2"]["b1_check_max_abs_diff"] < 1e-9
    out["gates_all_passed"] = all(g.get("mutation_test_centred_window_caught", True) for g in A["gates"].values()) and \
        all(g.get("identical", True) for g in A["gates"].values())
    out["p7_no_flagged_calibration"] = all(c["flags"] == 0 for c in A["power_p7"]["cells"].values())
    q = A["vrsq"]
    out["p5_fisher_not_below_005"] = q["ge2_vs_lt2"]["fisher_p_two_sided"] >= 0.05
    out["p5_diff_ci_includes_0"] = q["ge2_vs_lt2"]["diff_ci_group_bootstrap"][0] < 0 < q["ge2_vs_lt2"]["diff_ci_group_bootstrap"][1]
    return out


def claims_text(A, V, PW3, R):
    """Statements added to the manuscript from this addendum, checked at full precision."""
    v = V.v; out = {}; n = lambda k: int(v[k])
    out["p2_family6_raises_between_power_at_080"] = n("a3.pwn.p2.lo.80.c.n50") > n("a3.p2.c14.lo.80.c.n50")
    out["p2_family6_same_within_power_at_080"] = n("a3.pwn.p2.lo.80.w.n50") == n("a3.p2.c14.lo.80.w.n50")
    out["p2_family6_070_at_most_two_between_none_within"] = n("a3.pwn.p2.lo.70.c.n50") <= 2 and n("a3.pwn.p2.lo.70.w.n50") == 0
    out["p4_power_within_one_of_inferred_groups"] = all(abs(n(f"a3.pwn.p4.lo.{L}.{s}.n50") - n(f"a3.pwn.p2.lo.{L}.{s}.n50")) <= 1
                                                         for L in (65, 70, 80) for s in ("c", "w"))
    out["p7_diffuse_declared_more_often_than_single_channel_at_080"] = (
        n("a3.pwn.p7.lp.80.c.n50") > n("r3pwn.lo.80.c.n50d") and n("a3.pwn.p7.lp.80.w.n50") > n("r3pwn.lo.80.w.n50d")
        and n("a3.pwn.p7.xo.80.c.n50") > n("r3pwn.xo.80.c.n50d") and n("a3.pwn.p7.xo.80.w.n50") > n("r3pwn.xo.80.w.n50d"))
    out["p7_within_session_auc_above_target_in_most_combinations"] = all(n(f"a3.p7.{L}.wabove") > n(f"a3.p7.{L}.nest") / 2 for L in (70, 80))
    out["p7_single_channel_within_session_auc_close_to_target"] = all(
        abs(c["channel_session_auc"] - float(k.split("__")[1])) < 0.035 for k, c in PW3.items()
        if k.startswith("lroff__0.70__") or k.startswith("lroff__0.80__") if not k.endswith("sev2__cell_seated_flight"))
    q = A["vrsq"]
    out["p5_indicator_intervals_all_include_0"] = all(c["ci"][0] < 0 < c["ci"][1] for c in q["allzero_indicator_all_files"]["correlations"].values())
    out["p5_no_holm_below_005_with_all_files"] = all(c["p_holm"] >= 0.05 for c in q["all_files"]["correlations"].values())
    out["p5_four_holm_below_005_without_all_zero"] = sum(c["p_holm"] < 0.05 for c in q["without_all_zero"]["correlations"].values()) == 4
    out["p5_dates_cluster_groups_do_not"] = q["clustering_dates"]["p_one_sided"] < 0.05 <= q["clustering_groups_with_2plus_sessions"]["p_one_sided"]
    r = A["vrlens_rw_nodemo"]; vl = json.load(open(R / "vrlens.json")); ab = json.load(open(R / "rev3_vrlens_ablation.json"))
    out["p6_recordwise_without_demographics_between_grouped_and_recordwise"] = (
        ab["sensVRLENSA_"]["accuracy_4class_mean_over_repeats"] < r["accuracy_4class"] < vl["recordwise_10fold"]["accuracy_4class"])
    out["p6_recordwise_ctx_auc_lower_without_demographics"] = r["ctx_auc_sev2"] < r["stored_with_demographics_ctx_auc_sev2"]
    sec = [f for k, f in A["families"].items() if k.startswith("P3|")] + [A["families"][f"P1|lroffp|CTXB+MOT|{s_}|14"] for s_ in ("ctx", "session")]
    out["p3_every_secondary_family_could_pass"] = all(f["holm_floor"] < 0.05 for f in sec)
    out["p3_smallest_secondary_holm_is_scaled_penalty_within"] = min(f["min_p_holm"] for f in sec) == A["families"]["P1|lroffp|CTXB+MOT|session|14"]["min_p_holm"]
    return out


def write_defs(vals, path):
    """Value definitions only: \\V itself is defined by values.tex, which must be loaded first."""
    lines = ["% generated by src/fa_report_add3.py -- do not edit by hand; load after values.tex, which defines \\V", "\\makeatletter"]
    lines += [f"\\expandafter\\def\\csname fa@{k}\\endcsname{{{v}}}" for k, v in sorted(vals.items())]
    lines.append("\\makeatother")
    Path(path).write_text("\n".join(lines) + "\n")
    Path(path).with_suffix(".json").write_text(json.dumps(dict(sorted(vals.items())), indent=1, ensure_ascii=False))


# ----------------------------------------------------------------------------- main
def main(root):
    root = Path(root); R = root / "results"; G = R / "paper"
    A = json.load(open(R / "add3_stats.json")); S3 = json.load(open(R / "rev3_stats.json")); C3, PW3 = S3["configs"], S3["power"]["cells"]
    old = json.loads((G / "values.json").read_text())
    V = Values(); V.v.update(old); n_old = len(old)           # existing values are read, never rewritten
    values(V, A, C3, PW3)
    new = {k: v for k, v in V.v.items() if k not in old}
    assert all(V.v[k] == old[k] for k in old) and all(k.startswith("a3.") for k in new)
    write_defs(new, G / "values_add3.tex")
    for name, fn in (("tab_a3_p1.tex", lambda: table_p1(V)), ("tab_a3_family6.tex", lambda: table_family6(V)),
                     ("tab_a3_power6.tex", lambda: table_power6(V)), ("tab_a3_diffuse.tex", lambda: table_diffuse(V)),
                     ("tab_a3_secondary.tex", lambda: table_secondary(V)), ("tab_a3_vrsq.tex", lambda: table_vrsq(V, A["vrsq"])),
                     ("tab_a3_allzero.tex", lambda: table_allzero(V, A["vrsq"])), ("tab_a3_vrlens.tex", lambda: table_vrlens(V)),
                     ("tab_a3_falling.tex", lambda: table_falling(V)), ("tab_a3_nr.tex", lambda: table_nr(V)),
                     ("tab_a3_nullranges.tex", lambda: table_nullranges(V))):
        (G / name).write_text(fn() + "\n")
    cl = claims(A); cl.update(claims_text(A, V, PW3, R)); json.dump(cl, open(R / "claims_add3.json", "w"), indent=1)
    print(f"values_add3.json: {len(new)} new values (values.json unchanged, {n_old} values); "
          f"claims: {sum(cl.values())} true, {sum(not v for v in cl.values())} false {[k for k, v in cl.items() if not v]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
