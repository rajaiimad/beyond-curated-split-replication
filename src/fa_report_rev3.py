"""
fa_report_rev3.py -- numbers, tables and the figure of revision 3 (docs/ANALYSIS_PLAN_REVISION3.md).
Called by fa_report.py (main), which writes all values into results/paper/values.json.

Reads results/rev3_*.json; writes results/paper/r3_*.tex and figures/fig_gain_null.pdf.
Value keys (\\V{...}):
  r3.<tag><lab>.<protocol>.<fs>.<learner>.{auc,ci,cauc,cci,wauc,wci,rec,fpr}
  r3g.<tag><lab>.<protocol>.<fs>.<learner>.{o,c,w}.{d,ci,p}     telemetry gain over the context baseline
  r3n.<lab>.<protocol>.<fs>.<learner>.{c,w}.{p,ph,nm}            label-transplant null: p, Holm p, null mean
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fa_report import (PR, PR_NAME, CELL_P, WITHIN_P, CROSS_P, FS, MN, a2, d2, ci2, pct, pfmt, n_,
                       INK, INK2, MUTED, GREY_L, BLUE, ORANGE, COL2)

FS3 = {"CTXB": "B", "CTXB+MOT": "M", "CTXB+SELFZ": "Z", "CTXB+SPK": "S", "CTXB+MOT+SPK": "MS", "CTXB+MOTNR": "MNR", **FS}
LN = {"lroff": "lo", "xgboff": "xo", "ctxb": "cb", "lroffp": "lp", **MN}
TAG3 = {"": "", "r3control_": "ctl.", "control_": "ctl.", "sess_": "sess.", "mrg_": "mrg.",
        **{f"r3pos{k}_": f"pos{k}." for k in (55, 60, 65, 70, 80)}}
NULL_P = ["within_seated", "within_standing", "pooled"] + CELL_P
METRIC = {"overall": ("auc", "ci", "o"), "ctx": ("cauc", "cci", "c"), "session": ("wauc", "wci", "w"),
          "session_nm": ("nauc", "nci", "n")}       # within-session AUC on sessions whose rating also falls (addendum B3)
SL = {"ctx": "c", "session": "w", "session_nm": "n"}
NULLS = {"null_lroff": "r3n", "null_xgboff": "r3n", "null_lroffp": "r3n", "null_lroff_nr": "r3n",
         "null_lroff_block": "r3nb", "null_lroff_nm": "r3nm"}   # value prefix of each null's families
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
         9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen"}


def split_key(key):
    tag = next(t for t in sorted(TAG3, key=len, reverse=True) if key.startswith(t))
    lab, pr, fs, ln = key[len(tag):].split("__")
    return tag, lab.replace("sev", "s"), pr, fs, ln


def vkey(prefix, key):
    tag, lab, pr, fs, ln = split_key(key)
    return f"{prefix}.{TAG3[tag]}{lab}.{PR[pr]}.{FS3[fs]}.{LN[ln]}"


def values(V, R):
    st = json.load(open(R / "rev3_stats.json"))
    ops = {}
    for f in ("rev3_main.json", "rev3_control.json", "rev3_sens_group.json"):
        if (R / f).exists():
            d = json.load(open(R / f)); ops.update({r["key"]: r for r in (d["runs"] if isinstance(d, dict) else d)})
    NE = {key for key, c in st["configs"].items() if c["pos_groups"] < 3}      # n/e: < 3 groups with a positive record
    for key, c in st["configs"].items():
        k = vkey("r3", key)
        for m, (a, b, _) in METRIC.items():
            if m not in c:
                continue
            ok = key not in NE and c[m]["auc"] == c[m]["auc"]
            V.put(f"{k}.{a}", a2(c[m]["auc"]) if ok else "n/e")
            if key in NE and c[m]["auc"] == c[m]["auc"]:
                V.put(f"{k}.{a}pt", a2(c[m]["auc"]))          # point estimate only (no interval)
            V.put(f"{k}.{b}", ci2(c[m]["ci"]) if ok and all(x == x for x in c[m]["ci"]) else "")
        V.put(f"{k}.prev", pct(c["prevalence"])); V.put(f"{k}.ngroups", c["n_groups"]); V.put(f"{k}.posgroups", c["pos_groups"])
        o = ops.get(key)
        if o is not None:
            if o["roc_auc"] == o["roc_auc"]:
                assert abs(o["roc_auc"] - c["overall"]["auc"]) < 1e-9, key          # two engines, same predictions
            if "op_fpr" in o and key not in NE:
                V.put(f"{k}.fpr", pct(o["op_fpr"])); V.put(f"{k}.rec", pct(o["op_recall"]) if o["op_recall"] == o["op_recall"] else "n/e")
    for key, g in st["gains"].items():
        k = vkey("r3g", key)
        for m, (_, _, s) in METRIC.items():
            if m not in g:
                continue
            ok = key not in NE and g[m]["delta"] == g[m]["delta"]
            V.put(f"{k}.{s}.d", d2(g[m]["delta"]) if ok else "n/e")
            V.put(f"{k}.{s}.ci", ci2(g[m]["ci"]) if ok and all(x == x for x in g[m]["ci"]) else "")
            V.put(f"{k}.{s}.p", pfmt(g[m]["p_boot"]) if ok else "")
            if key in NE and g[m]["delta"] == g[m]["delta"]:
                V.put(f"{k}.{s}.dpt", d2(g[m]["delta"]))      # point estimate only (no interval)
    for nk, pre in NULLS.items():
        nl = st.get(nk)
        if not nl:
            continue
        for fam, d in nl["families"].items():
            ln, fs, stat = fam.split("|"); s = SL[stat]
            V.put(f"{pre}.{LN[ln]}.{FS3[fs]}.{s}.nperm", d["n_perm"]); V.put(f"{pre}.{LN[ln]}.{FS3[fs]}.{s}.m", d["m"])
            V.put(f"{pre}.{LN[ln]}.{FS3[fs]}.{s}.nsig", WORDS[sum(t["p_holm"] < 0.05 for t in d["tests"].values())])
            V.put(f"{pre}.{LN[ln]}.{FS3[fs]}.{s}.nsigraw", WORDS[sum(t["p_perm"] < 0.05 for t in d["tests"].values())])
            V.put(f"{pre}.{LN[ln]}.{FS3[fs]}.{s}.holmfloor", pfmt(d["m"] / (d["n_perm"] + 1)))
            V.put(f"{pre}.{LN[ln]}.{FS3[fs]}.{s}.minp", pfmt(min(t["p_holm"] for t in d["tests"].values())))
            for key, t in d["tests"].items():
                k = vkey(pre, key)
                if key in NE:
                    V.put(f"{k}.{s}.ne", "1")
                V.put(f"{k}.{s}.p", pfmt(t["p_perm"])); V.put(f"{k}.{s}.ph", pfmt(t["p_holm"]))
                V.put(f"{k}.{s}.nm", d2(t["null_mean"])); V.put(f"{k}.{s}.nr", ci2([t["null_q2.5"], t["null_q97.5"]]))
                V.put(f"{k}.{s}.np", t["n_perm"])
    V.put("r3.ne.list", ", ".join(sorted(NE)).replace("_", r"\_") or "none")
    return st


# ---------------------------------------------------------------- tables
DAG = r"$^\dagger$"


def _d(V, base, s):
    """Gain, or its point estimate marked with a dagger where no interval exists."""
    d = V.v.get(f"{base}.{s}.d", "--")
    return V.v[f"{base}.{s}.dpt"] + DAG if d == "n/e" and f"{base}.{s}.dpt" in V.v else d


def _a(V, k, a="auc"):
    """AUC, or its point estimate marked with a dagger where no interval exists."""
    v = V.v.get(f"{k}.{a}", "--")
    return V.v[f"{k}.{a}pt"] + DAG if v == "n/e" and f"{k}.{a}pt" in V.v else v


def _g(V, base, s, with_p=None):
    d = _d(V, base, s); c = V.v.get(f"{base}.{s}.ci", "")
    out = f"{d} {c}".strip()
    if with_p is not None and d != "n/e":
        p = V.v.get(f"{with_p}.{s}.ph")
        if p is not None:
            out += f" ({p})"
    return out


def table_gain(V, lab="s2", ln="lo", fs="M", label="tab:gain", star=True, caption=None):
    env = "table*" if star else "table"
    cap = caption or (r"Telemetry beyond context (" + ("\\sevtwo{}" if lab == "s2" else "\\sevone{}") + r"). The context baseline knows only game, campaign and exposure time; the telemetry model is the same baseline plus the 28 telemetry features, which can only add to it (logistic regression with the baseline as a fixed offset). "
                      r"Gain = telemetry model minus baseline on identical records, with 95\,\% group-bootstrap intervals and, in parentheses, the Holm-adjusted permutation $p$ against the label-transplant null where it was run. "
                      r"Between sessions: context-stratified AUC (positive and negative records of the same game, campaign and 30-s bin, mostly from different sessions, \V{r3pair.btw.min}--\V{r3pair.btw.max}\,\%). Within sessions: only records of the same session are compared. Prev.: share of target records with the label. Recall and false-positive rate of the telemetry model at a threshold set for a 10\,\% false-alarm budget on training groups. $^\dagger$: fewer than three groups with a positive record, so no interval is reported; the point estimate is shown, and the cell's tests stay in the families. ``--'': no threshold could be evaluated.")
    L = [f"\\begin{{{env}}}[t]", r"\centering", r"\footnotesize", f"\\caption{{{cap}}}", f"\\label{{{label}}}",
         r"\setlength{\tabcolsep}{1.5pt}", r"\begin{tabular}{@{}lrcccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Overall ROC-AUC} & \multicolumn{3}{c}{Telemetry gain, $\Delta$AUC [95\,\% CI] ($p_{\mathrm{Holm}}$)} & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
         r"Protocol & Prev. (\%) & Baseline & Telemetry & Overall & Between sessions & Within sessions & Rec. / FPR (\%) \\",
         r"\midrule", r"\multicolumn{8}{l}{\emph{Same setting, held-out groups}} \\"]
    for pr in WITHIN_P + CELL_P + CROSS_P:
        if pr == CELL_P[0]:
            L.append(r"\multicolumn{8}{l}{\emph{One game--campaign cell (context constant)}} \\")
        if pr == CROSS_P[0]:
            L.append(r"\multicolumn{8}{l}{\emph{Shifted setting (campaign or game), held-out groups}} \\")
        b = f"{lab}.{PR[pr]}"; base = f"r3.{b}.B.cb"; tel = f"r3.{b}.{fs}.{ln}"; g = f"r3g.{b}.{fs}.{ln}"
        pk = f"r3n.{b}.{fs}.{ln}" if pr in NULL_P else None
        rec = f"{V.v.get(tel + '.rec', '--')} / {V.v.get(tel + '.fpr', '--')}"
        L.append(f"{PR_NAME[pr]} & {V.v[base + '.prev']} & {_a(V, base)} & {_a(V, tel)} & "
                 f"{_g(V, g, 'o')} & {_g(V, g, 'c', pk)} & {_g(V, g, 'w', pk)} & {rec} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(L)


def table_decode(V, R):
    d = json.load(open(R / "rev3_decode.json"))["runs"]
    nr = R / "rev3_decode_nr.json"
    if nr.exists():
        d.update(json.load(open(nr))["runs"])
    rows = [("game_race", "Game (racing vs flight)"), ("campaign_standing", "Campaign (standing vs seated)"),
            ("cell_seated|Flight", "Cell: seated, flight"), ("cell_seated|Race", "Cell: seated, racing"),
            ("cell_standing|Flight", "Cell: standing, flight"), ("cell_standing|Race", "Cell: standing, racing")]
    fsets = [f for f in ("MOT", "SELFZ", "MOTNR") if f"{f}__logreg__game_race" in d]
    for fs in fsets:
        for ln in ("logreg", "hgb"):
            for t, _ in rows:
                r = d[f"{fs}__{ln}__{t}"]; k = f"r3d.{fs}.{MN[ln]}.{t.replace('|', '').replace('_', '')}"
                V.put(k + ".auc", a2(r["auc"])); V.put(k + ".ci", ci2(r["ci"])); V.put(k + ".sauc", a2(r["session_auc"]))
    head = {"MOT": "MOT (28)", "SELFZ": "SELFZ (28)", "MOTNR": "MOT without rotation (14)"}
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{What the telemetry encodes. ROC-AUC (95\,\% group-bootstrap interval) with which the 28 telemetry features (MOT), their self-normalized versions (SELFZ) and the 14 MOT features not derived from the camera-rotation fields (speed, acceleration, region of interest) recognize the game, the campaign and each game--campaign cell of records from held-out groups (pooled protocol folds; cells one-vs-rest). The rotation fields are constant within every standing-campaign session (Section~\ref{sec:parsing}); without the features derived from them, the campaign is recognized only weakly. LR: logistic regression; HGB: histogram gradient boosting (scikit-learn, 150 iterations).}",
         r"\label{tab:decode}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l" + "cc" * len(fsets) + r"@{}}", r"\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{head[f]}}}" for f in fsets) + r" \\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(fsets))),
         "Target & " + " & ".join(["LR & HGB"] * len(fsets)) + r" \\", r"\midrule"]
    for t, name in rows:
        cells = []
        for fs in fsets:
            for ln in ("logreg", "hgb"):
                k = f"r3d.{fs}.{MN[ln]}.{t.replace('|', '').replace('_', '')}"
                cells.append(f"{V.v[k + '.auc']} {{\\scriptsize {V.v[k + '.ci']}}}")
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_mde(V, st):
    """Minimum detectable effect: smallest planted within-stratum AUC whose recovered gain has a
    95 % interval above 0, for the channel alone and among the telemetry features."""
    mde = st["mde"]
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{What the design could detect (positive control). Smallest planted within-stratum AUC (0.55, 0.60, 0.65, 0.70 or 0.80) whose recovered gain over the context baseline has a 95\,\% interval above 0, between sessions (B) and within sessions (W); ``--'' = not recovered even at 0.80. Alone: the planted channel is the only addition to the baseline; among MOT: it is added to the 28 telemetry features. Main learner (logistic regression with the baseline as offset).}",
         r"\label{tab:mde}", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{\sevtwo{}} & \multicolumn{4}{c}{\sevone{}} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r" & \multicolumn{2}{c}{Alone} & \multicolumn{2}{c}{Among MOT} & \multicolumn{2}{c}{Alone} & \multicolumn{2}{c}{Among MOT} \\",
         r"Protocol & B & W & B & W & B & W & B & W \\", r"\midrule"]
    for pr in NULL_P:
        cells = []
        for lab in (2, 1):
            for fs in ("CTXB+SPK", "CTXB+MOT+SPK"):
                ne = st["configs"].get(f"r3pos70_sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 3) < 3
                for stat in ("ctx", "session"):
                    v = mde.get(f"sev{lab}__{pr}__{fs}__lroff__{stat}", {}).get("mde")
                    txt = "n/e" if ne else (f"{v:.2f}" if v else "--")
                    cells.append(txt)
                    V.put(f"r3m.s{lab}.{PR[pr]}.{FS3[fs]}.{'c' if stat == 'ctx' else 'w'}", "none" if txt == "--" else txt)
        L.append(f"{PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_control(V):
    """Label-shift control: overall and within-session AUC with true and shifted training labels."""
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Label-shift control (\sevtwo{}). Overall ROC-AUC on held-out groups with correct training labels and with each training session's labels circularly displaced (test labels unchanged). Inside a single cell, game and campaign are constant, so a telemetry-only model that stays predictive under the shift there must use differences between sessions. Default HGB on MOT alone is the telemetry-only model; the other columns also contain exposure time (MOT+TIME for default HGB and HGB-reg; context baseline plus MOT for the offset learners), so their drop under the shift can also reflect the lost time course. n/e: not estimable (fewer than three groups with a positive record).}",
         r"\label{tab:control}", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lcccccccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Default HGB, MOT} & \multicolumn{2}{c}{Default HGB, MOT+TIME} & \multicolumn{2}{c}{HGB-reg, MOT+TIME} & \multicolumn{2}{c}{LR + offset} & \multicolumn{2}{c}{Boosting + offset} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
         r"Protocol & correct & shifted & correct & shifted & correct & shifted & correct & shifted & correct & shifted \\", r"\midrule"]
    for pr in NULL_P:
        b = PR[pr]; cells = []
        for real, shift in ((f"r3.s2.{b}.M.hgb", f"r3.ctl.s2.{b}.M.hgb"), (f"r3.s2.{b}.MT.hgb", f"r3.ctl.s2.{b}.MT.hgb"),
                            (f"r3.s2.{b}.MT.hgbr", f"r3.ctl.s2.{b}.MT.hgbr"),
                            (f"r3.s2.{b}.M.lo", f"r3.ctl.s2.{b}.M.lo"), (f"r3.s2.{b}.M.xo", f"r3.ctl.s2.{b}.M.xo")):
            cells += [V.v.get(real + ".auc", "--"), V.v.get(shift + ".auc", "--")]
        L.append(f"{PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


LEVELS = (0.55, 0.60, 0.65, 0.70, 0.80)
SHORT = {"within_seated": "within seated", "within_standing": "within standing", "pooled": "pooled",
         "cell_seated_flight": "seated flight-game cell", "cell_seated_race": "seated racing-game cell",
         "cell_standing_flight": "standing flight-game cell", "cell_standing_race": "standing racing-game cell"}


def power_values(V, st):
    """Addendum B1: power of the pre-specified decision rule over the planted-channel seeds."""
    pw = st.get("power")
    if not pw:
        return None
    C = st["configs"]; ne = lambda lab, pr: C.get(f"sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3
    V.put("r3pw.nseeds", pw["n_seeds"])
    cells = pw["cells"]
    for key, c in cells.items():
        ln, lev, lab, pr = key.split("__")
        for stat, s_ in (("ctx", "c"), ("session", "w")):
            V.put(f"r3pw.{LN[ln]}.{int(round(100 * float(lev)))}.{s_}.{lab.replace('sev', 's')}.{PR[pr]}",
                  "n/e" if ne(int(lab[-1]), pr) else f"{100 * c[stat]['power']:.0f}")
    for ln in ("lroff", "xgboff"):
        for stat, s_ in (("ctx", "c"), ("session", "w")):
            for lev in LEVELS:
                v = [cells[f"{ln}__{lev:.2f}__sev{lab}__{pr}"][stat] for lab in (1, 2) for pr in FAMILY_P
                     if f"{ln}__{lev:.2f}__sev{lab}__{pr}" in cells and not ne(lab, pr)]
                if not v:
                    continue
                b = f"r3pwn.{LN[ln]}.{int(round(100 * lev))}.{s_}"
                V.put(b + ".n", len(v)); V.put(b + ".n50", WORDS.get(sum(x["power"] >= 0.5 for x in v), sum(x["power"] >= 0.5 for x in v)))
                V.put(b + ".n50d", sum(x["power"] >= 0.5 for x in v))
                V.put(b + ".n80", WORDS.get(sum(x["power"] >= 0.8 for x in v), sum(x["power"] >= 0.8 for x in v)))
                V.put(b + ".nany", WORDS.get(sum(x["power"] > 0 for x in v), sum(x["power"] > 0 for x in v)))
                V.put(b + ".med", f"{100 * float(np.median([x['power'] for x in v])):.0f}")
                V.put(b + ".max", f"{100 * max(x['power'] for x in v):.0f}")
                V.put(b + ".single", f"{100 * float(np.mean([x['power_single_protocol'] for x in v])):.0f}")
            for lev in LEVELS:
                ch = [cells[k]["channel_ctx_auc" if stat == "ctx" else "channel_session_auc"] for k in cells
                      if k.startswith(f"{ln}__{lev:.2f}__")]
                if ch:
                    V.put(f"r3chan.{int(round(100 * lev))}.{s_}.min", a2(min(ch))); V.put(f"r3chan.{int(round(100 * lev))}.{s_}.max", a2(max(ch)))
    # the planted channel is added to the observed data, so power is conditional on it: the spread of
    # the planted gain over seeds, the Monte Carlo error of a 20-seed proportion, and the counts the
    # permutation part of the rule alone would give (the interval condition is conservative within sessions)
    from pathlib import Path
    sd = pd.DataFrame(json.load(open(Path(V.R) / "rev3_posctl_seeds.json"))["rows"])
    sd = sd[[not ne(int(l[-1]), p) for l, p in zip(sd.label, sd.protocol)]]
    n_s = pw["n_seeds"]; V.put("r3pw.mcse", f"{100 * np.sqrt(0.25 / n_s):.0f}")
    sp = sd[(sd.learner == "lroff") & (sd.level == 0.8)].groupby(["label", "protocol"])
    gs = pd.concat([sp.gain_ctx.std(), sp.gain_session.std()])
    V.put("r3pw.gsd.min", f"{gs.min():.2f}"); V.put("r3pw.gsd.max", f"{gs.max():.2f}")
    for ln in ("lroff", "xgboff"):
        for stat, s_ in (("ctx", "c"), ("session", "w")):
            for lev in LEVELS:
                x = sd[(sd.learner == ln) & (sd.level == lev)].groupby(["label", "protocol"])[f"holm_{stat}"]
                n50p = int(((x.apply(lambda z: (z < 0.05).mean())) >= 0.5).sum())
                V.put(f"r3pwn.{LN[ln]}.{int(round(100 * lev))}.{s_}.n50perm", n50p)
    p80 = [cells[f"lroff__0.80__sev{lab}__{pr}"][stat]["power"] for lab in (1, 2) for pr in FAMILY_P
           for stat in ("ctx", "session") if f"lroff__0.80__sev{lab}__{pr}" in cells and not ne(lab, pr)]
    V.put("r3pw.near50", sum(0.3 <= x <= 0.7 for x in p80)); V.put("r3pw.n80", len(p80))
    # seed-based evidence against (LR+offset): real gain below 19 of the 20 planted gains of that size
    lab_s = {2: r"\sevtwo{}", 1: r"\sevone{}"}
    ag = []
    for lab in (2, 1):
        for pr in FAMILY_P:
            if ne(lab, pr):
                continue
            for stat in ("ctx", "session"):
                hit = [lev for lev in LEVELS if cells.get(f"lroff__{lev:.2f}__sev{lab}__{pr}", {}).get(stat, {}).get("evidence_against")]
                if hit:
                    ag.append((lab, pr, stat, hit[0]))
    V.put("r3ag.n", WORDS.get(len(ag), len(ag)))
    parts = []
    for l in (2, 1):
        it = [f"{SHORT[p]} ({x:.2f}{'' if st_ == 'ctx' else ', within sessions'})" for ll, p, st_, x in ag if ll == l]
        if it:
            parts.append(f"{lab_s[l]}: " + (", ".join(it[:-1]) + " and " + it[-1] if len(it) > 1 else it[0]))
    V.put("r3ag.list", "; ".join(parts) or "none")
    V.put("r3ag.nb", WORDS.get(sum(a[2] == "ctx" for a in ag), sum(a[2] == "ctx" for a in ag)))
    V.put("r3ag.nw", WORDS.get(sum(a[2] == "session" for a in ag), sum(a[2] == "session" for a in ag)))
    if ag:
        V.put("r3ag.min", f"{min(x for *_, x in ag):.2f}")
    return ag


def table_power(V, st, ln="lroff", label="tab:mde", star=False, levels=(0.70, 0.80), wlevels=(0.70, 0.80), caption=None):
    """Power of the pre-specified rule: share of planted-channel seeds in which a planted signal
    added to the 28 telemetry features is declared (Holm-adjusted permutation p < 0.05 and bootstrap
    interval of the gain above 0)."""
    pw = st["power"]["cells"]; C = st["configs"]
    ne = lambda lab, pr: C.get(f"sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3
    n = len(levels) + len(wlevels)
    cap = caption or (r"What the decision rule would have detected (positive control as specified in the revision addendum: \V{r3pw.nseeds} realizations of a planted channel added to the 28 telemetry features; the channel alone was fitted in one realization only and has no power estimate, Supplementary Table~S19). "
                      r"Percentage of realizations in which a planted signal of the given within-stratum AUC, added to the 28 telemetry features, is declared by the rule applied to the real data: "
                      r"Holm-adjusted permutation $p<0.05$ against the label-transplant null and a bootstrap interval of the gain above 0. "
                      r"Entries are conditional on the observed data (only the channel's noise varies between realizations; near 50\,\% the Monte Carlo standard error is about \V{r3pw.mcse} points) and assume that the signal is present in every test of the family. "
                      r"B: between sessions (context-stratified AUC); W: within sessions. "
                      r"Main learner (logistic regression with the context baseline as offset). Smaller planted sizes and both learners: Supplementary Table~S18. n/e: not estimable (fewer than three groups with a positive record).")
    env = "table*" if star else "table"
    L = [f"\\begin{{{env}}}[t]", r"\centering", r"\footnotesize", f"\\caption{{{cap}}}", f"\\label{{{label}}}",
         r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}l" + "c" * (2 * n) + r"@{}}", r"\toprule",
         f" & \\multicolumn{{{n}}}{{c}}{{\\sevtwo{{}}}} & \\multicolumn{{{n}}}{{c}}{{\\sevone{{}}}} \\\\",
         f"\\cmidrule(lr){{2-{1 + n}}}\\cmidrule(lr){{{2 + n}-{1 + 2 * n}}}",
         " & " + " & ".join([f"\\multicolumn{{{len(levels)}}}{{c}}{{B}} & \\multicolumn{{{len(wlevels)}}}{{c}}{{W}}"] * 2) + r" \\",
         "Protocol & " + " & ".join([" & ".join(f"{x:.2f}" for x in levels + wlevels)] * 2) + r" \\", r"\midrule"]
    for pr in FAMILY_P:
        cells = []
        for lab in (2, 1):
            for stat, levs in (("ctx", levels), ("session", wlevels)):
                for lev in levs:
                    c = pw.get(f"{ln}__{lev:.2f}__sev{lab}__{pr}")
                    if ne(lab, pr) or c is None:
                        cells.append("n/e"); continue
                    cells.append(f"{100 * c[stat]['power']:.0f}")
        L.append(f"{PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(L)


def table_power_full(V, st):
    """Supplement: power for both offset learners, every level, both estimands and labels."""
    pw = st["power"]["cells"]; C = st["configs"]
    ne = lambda lab, pr: C.get(f"sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3
    L = [r"\begin{table}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Power of the decision rule in full, for the positive control as specified in the revision addendum (\V{r3pw.nseeds} realizations of a planted channel added to the 28 telemetry features; the channel alone was fitted in one realization only, Table~S19): percentage of realizations in which the planted signal is declared (Holm-adjusted permutation $p<0.05$ within the family of 14 and bootstrap interval of the gain above 0), for both offset learners, every planted within-stratum AUC, between sessions (B) and within sessions (W). The channel is added to the observed telemetry and labels and only its noise changes between realizations, so each entry is conditional on the observed data; with \V{r3pw.nseeds} realizations, an entry near 50\,\% has a Monte Carlo standard error of about \V{r3pw.mcse} percentage points. Second-last row: share of realizations reaching the single-protocol criterion of a signal confined to one protocol (unadjusted permutation $p\le 0.05/14$), mean over the estimable combinations. Last row: mean realized AUC of the planted channel on the target records. n/e: fewer than three groups with a positive record, so no interval is reported and the rule is not applied.}",
         r"\label{tab:supp_power}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}ll" + "c" * 20 + r"@{}}", r"\toprule",
         r" & & \multicolumn{10}{c}{LR + offset} & \multicolumn{10}{c}{Boosting + offset} \\", r"\cmidrule(lr){3-12}\cmidrule(lr){13-22}",
         r" & & \multicolumn{5}{c}{B} & \multicolumn{5}{c}{W} & \multicolumn{5}{c}{B} & \multicolumn{5}{c}{W} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}\cmidrule(lr){18-22}",
         "Label & Protocol & " + " & ".join([" & ".join(f"{x:.2f}" for x in LEVELS)] * 4) + r" \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAMILY_P):
            cells = []
            for ln in ("lroff", "xgboff"):
                for stat in ("ctx", "session"):
                    for lev in LEVELS:
                        c = pw.get(f"{ln}__{lev:.2f}__sev{lab}__{pr}")
                        cells.append("n/e" if ne(lab, pr) or c is None else f"{100 * c[stat]['power']:.0f}")
            L.append(f"{(chr(92) + 'sevtwo{}') if lab == 2 and i == 0 else ((chr(92) + 'sevone{}') if i == 0 else '')} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    sg = []
    for ln in ("lroff", "xgboff"):
        for stat in ("ctx", "session"):
            for lev in LEVELS:
                v = [pw[f"{ln}__{lev:.2f}__sev{lab}__{pr}"][stat]["power_single_protocol"] for lab in (1, 2) for pr in FAMILY_P
                     if f"{ln}__{lev:.2f}__sev{lab}__{pr}" in pw and not ne(lab, pr)]
                sg.append(f"{100 * float(np.mean(v)):.0f}" if v else "--")
    L.append(r"\multicolumn{2}{@{}l}{Single-protocol criterion} & " + " & ".join(sg) + r" \\")
    ch = []
    for ln in ("lroff", "xgboff"):
        for stat in ("ctx", "session"):
            for lev in LEVELS:
                v = [pw[k]["channel_ctx_auc" if stat == "ctx" else "channel_session_auc"] for k in pw if k.startswith(f"{ln}__{lev:.2f}__")]
                ch.append(a2(float(np.mean(v))) if v else "--")
    L.append(r"\multicolumn{2}{@{}l}{Realized channel AUC} & " + " & ".join(ch) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_recovery(V, st):
    """Supplement: the single-seed recovery table (bootstrap-only rule) for the three offset learners."""
    mde = st["mde"]; C = st["configs"]
    L = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
         r"\caption{Recovery of one realization of the planted channel under the bootstrap-only rule: smallest planted within-stratum AUC whose gain over the context baseline has a 95\,\% group-bootstrap interval above 0, between sessions (B) and within sessions (W); ``--'' = not recovered even at 0.80. This rule ignores refitting variability and is looser than the rule applied to the real data (Table~\mref{tab:mde}). Alone: the channel is the only addition to the baseline; among MOT: it is added to the 28 telemetry features. Scaled penalty: $C$ = training sessions / training records (Note~S2). The channel alone and the scaled-penalty model were fitted only in this realization; the replication over \V{r3pw.nseeds} realizations (Table~S18) covers LR+offset and boosting+offset among MOT.}",
         r"\label{tab:supp_recovery}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}ll" + "c" * 8 + r"@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{LR + offset, alone} & \multicolumn{2}{c}{LR + offset, among MOT} & \multicolumn{2}{c}{Boosting + offset, among MOT} & \multicolumn{2}{c}{LR + offset (scaled penalty), among MOT} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Label & Protocol & B & W & B & W & B & W & B & W \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAMILY_P):
            ne = C.get(f"r3pos70_sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 3) < 3
            cells = []
            for fs, ln in (("CTXB+SPK", "lroff"), ("CTXB+MOT+SPK", "lroff"), ("CTXB+MOT+SPK", "xgboff"), ("CTXB+MOT+SPK", "lroffp")):
                for stat in ("ctx", "session"):
                    v = mde.get(f"sev{lab}__{pr}__{fs}__{ln}__{stat}", {}).get("mde")
                    cells.append("n/e" if ne else (f"{v:.2f}" if v else "--"))
            L.append(f"{(chr(92) + 'sevtwo{}') if lab == 2 and i == 0 else ((chr(92) + 'sevone{}') if i == 0 else '')} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def _pcell(V, gkey, nkey, s):
    """'gain [CI] (Holm p)' for a secondary analysis."""
    d = _d(V, gkey, s)
    if d == "n/e":
        return "n/e"
    p = V.v.get(f"{nkey}.{s}.ph")
    return f"{d} {V.v.get(gkey + '.' + s + '.ci', '')}".strip() + (f" ({p})" if p else "")


def table_secondary(V, st):
    """Supplement: the main family tests next to the secondary nulls, estimand, learner and features."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Secondary analyses of the telemetry gain (main learner unless stated; gain [95\,\% group-bootstrap interval] and, in parentheses, Holm-adjusted permutation $p$ within the family of 14; 13 for falling sessions, because for \sevone{} no session of the seated flight-game cell has a falling rating). "
         r"Main: Holm $p$ of families A (between sessions) and B (within sessions) against the session-level label-transplant null (\V{r3n.lo.M.c.nperm} permutations). "
         r"Group-block null: the same gains against a null that transplants whole groups (\V{r3nb.lo.M.c.nperm} permutations). "
         r"Falling sessions: within-session gain on the sessions whose rating also falls, where exposure time does not order the records (\V{r3nm.lo.M.n.nperm} permutations). "
         r"Scaled penalty: LR+offset with $C$ = training sessions / training records (\V{r3n.lp.M.c.nperm} permutations). "
         r"Without rotation: the 14 telemetry features not derived from the rotation fields (\V{r3n.lo.MNR.c.nperm} permutations). With \V{r3nb.lo.M.c.nperm} permutations the smallest attainable Holm-adjusted $p$ is \V{r3nb.lo.M.c.holmfloor} (\V{r3nm.lo.M.n.holmfloor} for falling sessions), so no test of the secondary analyses could pass the decision rule; they are descriptive. n/e: not estimable (fewer than three groups with a positive record, or no falling session); $^\dagger$: fewer than three groups with a positive record, so no interval is reported (the seated flight-game cell for \sevtwo{} stays in its families as planned). Intervals of the gains without rotation: Table~S23.}",
         r"\label{tab:supp_secondary}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Main ($p_\mathrm{Holm}$)} & \multicolumn{2}{c}{Group-block null ($p_\mathrm{Holm}$)} & Falling sessions & \multicolumn{2}{c}{Scaled penalty} & Without rotation \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}",
         r"Label & Protocol & B & W & B & W & W & B & W & B / W \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAMILY_P):
            b = f"s{lab}.{PR[pr]}"
            ne = V.v.get(f"r3g.{b}.M.lo.c.d") == "n/e"
            main = [V.v.get(f"r3n.{b}.M.lo.{s}.ph", "--") for s in ("c", "w")]
            blk = [V.v.get(f"r3nb.{b}.M.lo.{s}.ph", "--") for s in ("c", "w")]
            nm = _pcell(V, f"r3g.{b}.M.lo", f"r3nm.{b}.M.lo", "n")
            pen = [_pcell(V, f"r3g.{b}.M.lp", f"r3n.{b}.M.lp", s) for s in ("c", "w")]
            nr = f"{_d(V, f'r3g.{b}.MNR.lo', 'c')} ({V.v.get(f'r3n.{b}.MNR.lo.c.ph', '--')}) / {_d(V, f'r3g.{b}.MNR.lo', 'w')} ({V.v.get(f'r3n.{b}.MNR.lo.w.ph', '--')})"
            cells = main + blk + [nm] + pen + [nr]
            L.append(f"{(chr(92) + 'sevtwo{}') if lab == 2 and i == 0 else ((chr(92) + 'sevone{}') if i == 0 else '')} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_sim_full(V):
    """Supplement: the simulation in full (means and 2.5-97.5 percentiles over replications)."""
    rows = [("Record-wise AUC (HGB)", "D1_recordwise_auc"), ("Group-disjoint AUC (HGB)", "D1_grouped_auc"),
            ("Within-session AUC (HGB)", "hgb_within_session_auc"), ("Prevalence score, pooled leave-one-out", "D2_prevalence_LOPO_pooled_auc"),
            ("HGB, shifted training labels", "D3_shifted_auc"), ("Time-only baseline, within-session AUC", "base_within_session_auc"),
            ("Offset model: gain between sessions", "lroff_ctx_gain"), ("Offset model: gain within sessions", "lroff_session_gain"),
            ("First feature alone: within-session AUC", "x0_within_session_auc")]
    sc = ["N", "S", "M", "Ne", "Me", "Me2"]
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{The simulation in full: mean and 2.5--97.5 percentiles over \V{sim.reps} replications. Step labels (N, S, M): each session's label switches on once; episodic labels (Ne, Me, Me2): after onset the label alternates between on- and off-episodes of 20--60\,s, so exposure time no longer orders the records of a session. N, Ne: no information in the telemetry; S: between-session information; M, Me, Me2: the first feature shifted by 0.5, 0.5 and 1.0 standard deviations while the label is on.}",
         r"\label{tab:supp_sim}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}l" + "c" * len(sc) + r"@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Step labels} & \multicolumn{3}{c}{Episodic labels} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         "Quantity & " + " & ".join(sc) + r" \\", r"\midrule"]
    for name, k in rows:
        L.append(f"{name} & " + " & ".join(f"{V.v.get(f'sim.{x}.{k}', '--')} {{\\tiny {V.v.get(f'sim.{x}.{k}.r', '')}}}" for x in sc) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def second_values(V, R):
    y = json.load(open(R / "rev3_yalcin.json")); m = y["main"]; w = y["without_long_recordings"]
    V.put("y.n", m["n_participants"]); V.put("y.nrec", n_(m["n_records"])); V.put("y.prev", pct(m["prevalence"]))
    V.put("y.npos", m["participants_with_positives"]); V.put("y.commit", y["commit"][:7])
    V.put("y.nfolders", len(y["participants"]))
    out = [k for k, v in y["participants"].items() if "presses_outside_span" in v and v["presses_outside_span"] > 0]
    V.put("y.nout", sum(y["participants"][k]["presses_outside_span"] for k in out))
    V.put("y.nout.ep", sum(y["participants"][k].get("press_episodes_outside_span", 0) for k in out))
    V.put("y.nout.np", len(out))
    V.put("y.nlong", len(w["excluded"]))
    # bins without an eye-tracking sample (features carried forward), and the logged sampling rate
    P_ = {k: v for k, v in y["participants"].items() if "empty_bins" in v}
    if P_:
        longs = [str(k) for k in w["excluded"]]; emp = sum(v["empty_bins"] for v in P_.values())
        emp_long = sum(P_[k]["empty_bins"] for k in longs); other = [k for k in P_ if k not in longs and P_[k]["empty_bins"] > 0]
        V.put("y.nempty", n_(emp)); V.put("y.nempty.pct", pct(emp / m["n_records"]))
        V.put("y.long.nempty", n_(emp_long)); V.put("y.long.gap", f"{max(P_[k]['first_gap_s'] for k in longs):.0f}")
        V.put("y.nempty.other", n_(emp - emp_long)); V.put("y.nempty.other.np", WORDS.get(len(other), len(other)))
        V.put("y.nl.nempty.pct", pct((emp - emp_long) / w["n_records"]))
        V.put("y.rate", f"{np.median([v['samples_per_s'] for v in P_.values()]):.0f}")
    for tag, d in (("", m), ("nl.", w)):
        for k in ("D1_recordwise_hgb", "D1_recordwise_logreg"):
            V.put(f"y.{tag}{k}", a2(d[k]["auc"]))
        for k in ("D1_grouped_hgb", "D1_grouped_logreg", "D1_grouped_TIME", "D3_true_hgb", "D3_shifted_hgb", "D3_true_logreg", "D3_shifted_logreg"):
            V.put(f"y.{tag}{k}", a2(d[k]["auc"])); V.put(f"y.{tag}{k}.ci", ci2(d[k]["ci"]))
            V.put(f"y.{tag}{k}.w", a2(d[k]["within_session_auc"])); V.put(f"y.{tag}{k}.wci", ci2(d[k]["within_session_ci"]))
        V.put(f"y.{tag}id.acc", pct(d["D1_participant_identification"]["accuracy"], 0))
        V.put(f"y.{tag}id.maj", pct(d["D1_participant_identification"]["majority_share"], 0))
        V.put(f"y.{tag}id.k", d["D1_participant_identification"]["n_classes"])
        for k, v in d["D2"].items():
            V.put(f"y.{tag}{k}", a2(v))
        if "offset" in d:
            o = d["offset"]
            for nm_ in ("baseline", "offset"):
                V.put(f"y.{tag}off.{nm_}", a2(o[nm_]["auc"])); V.put(f"y.{tag}off.{nm_}.ci", ci2(o[nm_]["ci"]))
                V.put(f"y.{tag}off.{nm_}.w", a2(o[nm_]["within_participant_auc"])); V.put(f"y.{tag}off.{nm_}.wci", ci2(o[nm_]["within_participant_ci"]))
            for gk_, g_ in (("o", "gain_overall"), ("w", "gain_within_participant")):
                V.put(f"y.{tag}off.g{gk_}", d2(o[g_]["delta"])); V.put(f"y.{tag}off.g{gk_}.ci", ci2(o[g_]["ci"]))
    s = json.load(open(R / "rev3_sim.json"))
    V.put("sim.reps", s["design"]["replications"]); V.put("sim.npart", s["design"]["participants"])
    for sc, d in s["summary"].items():
        for k, v in d.items():
            V.put(f"sim.{sc}.{k}", a2(v["mean"]) if not k.endswith("gain") else d2(v["mean"]))
            V.put(f"sim.{sc}.{k}.r", ci2([v["q2.5"], v["q97.5"]]))
    return y, s


def table_second(V):
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{Three evaluation-design findings (D1--D3) on a second, independent archive (Yalcin et al. \cite{yalcin2024}: \V{y.n} participants, eye tracking, button presses during a roller-coaster ride) and in simulation with known ground truth (means over \V{sim.reps} replications per scenario; percentiles and episodic-label scenarios in Supplementary Table~S21). Default HGB unless stated. N: no telemetry--label link; S: between-session information only; M: momentary information (0.5\,SD shift of one feature while the label is on). Empty cells: not computed for that setting.}",
         r"\label{tab:second}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.5cm}cccc@{}}", r"\toprule",
         r" & Second archive & \multicolumn{3}{c}{Simulation} \\", r"\cmidrule(lr){3-5}",
         r"Finding / quantity & & N & S & M \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\emph{D1: record-wise splits identify sessions}} \\",
         r"Record-wise AUC & \V{y.D1_recordwise_hgb} & \V{sim.N.D1_recordwise_auc} & \V{sim.S.D1_recordwise_auc} & \V{sim.M.D1_recordwise_auc} \\",
         r"Group-disjoint AUC & \V{y.D1_grouped_hgb} & \V{sim.N.D1_grouped_auc} & \V{sim.S.D1_grouped_auc} & \V{sim.M.D1_grouped_auc} \\",
         r"Participant identified (\%) & \V{y.id.acc} & & & \\",
         r"\multicolumn{5}{@{}l}{\emph{D2: prevalence scores fall below 0.5 under grouped cross-validation}} \\",
         r"Prevalence score, pooled leave-one-out & \V{y.prevalence_score_LOPO_pooled_auc} & \V{sim.N.D2_prevalence_LOPO_pooled_auc} & \V{sim.S.D2_prevalence_LOPO_pooled_auc} & \V{sim.M.D2_prevalence_LOPO_pooled_auc} \\",
         r"\multicolumn{5}{@{}l}{\emph{D3: a model survives displaced training labels (between-session learning)}} \\",
         r"AUC, correct labels & \V{y.D3_true_hgb} & \V{sim.N.D1_grouped_auc} & \V{sim.S.D1_grouped_auc} & \V{sim.M.D1_grouped_auc} \\",
         r"AUC, shifted labels & \V{y.D3_shifted_hgb} & \V{sim.N.D3_shifted_auc} & \V{sim.S.D3_shifted_auc} & \V{sim.M.D3_shifted_auc} \\",
         r"\multicolumn{5}{@{}l}{\emph{Offset model: gain over a time-only baseline}} \\",
         r"Overall & \V{y.off.go} & & & \\",
         r"Between sessions & & \V{sim.N.lroff_ctx_gain} & \V{sim.S.lroff_ctx_gain} & \V{sim.M.lroff_ctx_gain} \\",
         r"Within sessions (participants) & \V{y.off.gw} & \V{sim.N.lroff_session_gain} & \V{sim.S.lroff_session_gain} & \V{sim.M.lroff_session_gain} \\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_groups(V, R):
    g = json.load(open(R / "rev3_sens_group.json"))
    for tag, d in g["n_groups"].items():
        for k, v in d.items():
            V.put(f"grp.{tag.rstrip('_')}.{k.replace('|', '')}", v)
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Sensitivity to the grouping. Telemetry gain over the context baseline between sessions (B) and within sessions (W), with 95\,\% bootstrap intervals over the units of each grouping: inferred groups (main, " + V.v["aud.ngroups"] + r"), every session its own group (lenient, " + V.v["grp.sess.all"] + r" units) and groups merged when their answers differ in at most one item (strict, " + V.v["grp.mrg.all"] + r" groups). n/e: not estimable (a protocol without scored records); $^\dagger$: fewer than three units with a positive record, so no interval is reported (point estimate shown). The merged grouping leaves only \V{grp.mrg.all} units, so its intervals are descriptive. Last rows: overall AUC of HGB-reg on MOT+TIME and CTX+TIME.}",
         r"\label{tab:supp_groups}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}ll" + "c" * 6 + r"@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Inferred} & \multicolumn{2}{c}{Session} & \multicolumn{2}{c}{Merged} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Label & Protocol & B & W & B & W & B & W \\", r"\midrule"]
    for ln, lname in (("lo", "LR + offset"), ("xo", "Boosting + offset")):
        L.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{lname}}}}} \\\\")
        for lab in (2, 1):
            for i, pr in enumerate(FAMILY_P):
                cells = []
                for tag in ("", "sess.", "mrg."):
                    b = f"r3g.{tag}s{lab}.{PR[pr]}.M.{ln}"
                    cells += [_g(V, b, "c"), _g(V, b, "w")]
                L.append(f"{(chr(92) + 'sevtwo{}') if lab == 2 and i == 0 else ((chr(92) + 'sevone{}') if i == 0 else '')} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L.append(r"\multicolumn{8}{@{}l}{\emph{HGB-reg, overall AUC (MOT+TIME / CTX+TIME), \sevtwo{}}} \\")
    for pr in ("within_seated", "pooled"):
        row = [f"{V.v.get(f'r3.{tag}s2.{PR[pr]}.MT.hgbr.auc', '--')} / {V.v.get(f'r3.{tag}s2.{PR[pr]}.CT.hgbr.auc', '--')}" for tag in ("", "sess.", "mrg.")]
        L.append(f" & {PR_NAME[pr]} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{x}}}" for x in row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def fig_gain_null(st, R, out):
    """Observed telemetry gains of the main learner against its label-transplant null."""
    import matplotlib.pyplot as plt
    nl = pd.DataFrame(json.load(open(R / "rev3_null_lroff.json"))); nl = nl[nl.features == "CTXB+MOT"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6), sharey=True)
    for ax, (col, stat, title) in zip(axes, (("gain_ctx", "ctx", "(a) Between sessions (context-stratified)"),
                                             ("gain_session", "session", "(b) Within sessions"))):
        ylab = []
        for i, (lab, pr) in enumerate([(l, p) for l in (2, 1) for p in NULL_P]):
            v = nl[(nl.label == f"sev{lab}") & (nl.protocol == pr)][col].dropna().to_numpy()
            yy = i + np.random.default_rng(i).uniform(-0.25, 0.25, len(v))
            ax.scatter(v, yy, s=1.2, color=GREY_L, alpha=0.35, lw=0, rasterized=True)
            g = st["gains"].get(f"sev{lab}__{pr}__CTXB+MOT__lroff", {}).get(stat)
            if g and st["configs"].get(f"sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3:
                ax.plot(g["delta"], i, marker="o", ms=3.4, mfc="white", mec=BLUE if lab == 2 else ORANGE, mew=1.0, ls="none")
                g = None                                   # fewer than three positive groups: point estimate only
            if g:
                ax.errorbar(g["delta"], i, xerr=[[g["delta"] - g["ci"][0]], [g["ci"][1] - g["delta"]]], fmt="o",
                            ms=3.2, color=BLUE if lab == 2 else ORANGE, lw=1.0, capsize=0)
            ylab.append(f"{'≥2' if lab == 2 else '≥1'}  {PR_NAME[pr].replace(chr(92) + 'ra{}', '→')}")
        ax.axvline(0, color=MUTED, lw=0.6); ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel("Telemetry gain, ΔAUC")
        ax.set_yticks(range(len(ylab))); ax.set_yticklabels(ylab)
    axes[0].invert_yaxis()                             # shared y-axis: invert once, primary label on top
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def range_values(V, st):
    """Ranges over the shifted (X) and same-setting (W) protocols, as in fa_report.extra_values."""
    ops = {}
    for f in ("rev3_main.json",):
        ops.update({r["key"]: r for r in json.load(open(Path(V.R) / f))})
    for lab in (1, 2):
        for sc, prs in (("X", CROSS_P), ("W", WITHIN_P), ("C", CELL_P)):
            for fs, ln in (("CTXB", "ctxb"), ("CTXB+MOT", "lroff"), ("CTXB+MOT", "xgboff"), ("CTXB+SELFZ", "lroff")):
                cs = [st["configs"].get(f"sev{lab}__{pr}__{fs}__{ln}") for pr in prs]
                cs = [c for c in cs if c]
                if not cs:
                    continue
                k = f"r3rng.s{lab}.{sc}.{FS3[fs]}.{LN[ln]}"
                ov = [c["overall"]["auc"] for c in cs if c["overall"]["auc"] == c["overall"]["auc"]]
                cx = [c["ctx"]["auc"] for c in cs if c["ctx"]["auc"] == c["ctx"]["auc"]]
                V.put(k + ".min", a2(min(ov))); V.put(k + ".max", a2(max(ov)))
                V.put(k + ".cmin", a2(min(cx))); V.put(k + ".cmax", a2(max(cx)))
                o = [ops[f"sev{lab}__{pr}__{fs}__{ln}"] for pr in prs if f"sev{lab}__{pr}__{fs}__{ln}" in ops]
                o = [x for x in o if "op_fpr" in x]
                if o:
                    V.put(k + ".recmin", pct(min(x["op_recall"] for x in o))); V.put(k + ".recmax", pct(max(x["op_recall"] for x in o)))
                    V.put(k + ".fprmin", pct(min(x["op_fpr"] for x in o))); V.put(k + ".fprmax", pct(max(x["op_fpr"] for x in o)))


def misc_values(V, R, st, F):
    # referee Q2: identical HGB-reg predictions with and without the context columns
    u = st.get("hgbreg_ctx_usage", {})
    if u:
        V.put("r3u.nident", sum(x["identical_folds"] for x in u.values())); V.put("r3u.nfolds", sum(x["folds"] for x in u.values()))
        V.put("r3u.nprot.allsame", sum(x["identical_folds"] == x["folds"] for x in u.values()))
        V.put("r3u.ncomb", len(u))
        diffs = [abs(st["configs"][f"{k}__MOT+TIME__hgbreg"]["overall"]["auc"] - st["configs"][f"{k}__MOT+CTX+TIME__hgbreg"]["overall"]["auc"])
                 for k in u if f"{k}__MOT+TIME__hgbreg" in st["configs"]]
        V.put("r3u.maxdiff", a2(max(diffs)))
    # pair decomposition and event-locked analysis
    W = json.load(open(R / "rev3_within.json"))
    for k, v in W["pair_share_same_session"].items():
        lab, pr = k.split("__")
        b = f"r3w.{lab.replace('sev', 's')}.{PR[pr]}"
        V.put(b + ".share", pct(1 - v["share"])); V.put(b + ".sameshare", pct(v["share"])); V.put(b + ".nboth", v["sessions_both_classes"])
    for k, v in st.get("event_locked", W["event_locked"]).items():
        lab, pr, fs, ln = k.split("__")
        b = f"r3el.{lab.replace('sev', 's')}.{FS3[fs]}.{LN[ln]}"
        V.put(b + ".d", d2(v["did"])); V.put(b + ".ci", ci2(v["ci"])); V.put(b + ".n", v["n_onset_sessions"]); V.put(b + ".ng", v["n_groups"])
        if "p_perm_one_sided" in v:
            V.put(b + ".p", pfmt(v["p_perm_one_sided"])); V.put(b + ".nr", ci2([v["null_q2.5"], v["null_q97.5"]]))
    # VRSQ files that are all zero (referee Q3)
    q = json.load(open(R / "rev3_vrsq0.json"))
    V.put("r3v.n", q["n_with_vrsq"]); V.put("r3v.allzero", q["all_zero"]); V.put("r3v.post0", q["post_zero"])
    V.put("r3v.allzero.ge2", q["all_zero_with_ingame_max_ge2"]); V.put("r3v.ge2", q["ingame_max_ge2"])
    for cell, d in q["all_zero_by_cell"].items():
        V.put(f"r3v.cell.{cell.replace('|', '')}", f"{d['all_zero']}/{d['n']}")
    rho = {k: v for k, v in q["correlations_without_all_zero"].items()}
    for k, v in rho.items():
        a, b = k.split("~")
        V.put(f"r3v.{a}.{'post' if b == 'vrsq_post_total' else 'delta'}.rho", a2(v["rho"]))
        V.put(f"r3v.{a}.{'post' if b == 'vrsq_post_total' else 'delta'}.ci", ci2(v["ci"]))
        V.put(f"r3v.{a}.{'post' if b == 'vrsq_post_total' else 'delta'}.rho3", f"{v['rho']:.3f}")
    V.put("r3v.nw", next(iter(rho.values()))["n"])
    # smallest number of permutations n for which a family of m tests can reach a Holm-adjusted
    # p below 0.05 at all (m / (n + 1) < 0.05, with p = (1 + k) / (1 + n))
    for m_ in (13, 14):
        n_need = next(n for n in range(1, 10_000) if m_ / (n + 1) < 0.05)
        V.put(f"r3sec.nperm.need{m_}", n_need)
    V.put("r3v.rho.min", a2(min(v["rho"] for v in rho.values()))); V.put("r3v.rho.max", a2(max(v["rho"] for v in rho.values())))
    # reproducibility of the revision-2 result files (fa_repro.py)
    rp = json.load(open(R / "reproducibility.json"))
    V.put("repro.python", rp["python"]); V.put("repro.nvalues", n_(rp["total_values"]))
    V.put("repro.nfiles", len(rp["files"])); V.put("repro.ndiff", n_(rp["total_differ"]))
    bad = [k for k, v in rp["files"].items() if v["differ"]]
    V.put("repro.note", "" if not bad else " (all in " + ", ".join(r"\texttt{" + b.replace("_", r"\_") + "}" for b in bad) + ")")
    # how far the grouping could be tightened: merge profiles that differ in one answer AND share a date
    S = pd.read_csv(Path(V.R).parent / "data" / "sessions.csv")
    fp = S.groupby("group_id").fingerprint.first(); dates = S.groupby("group_id").date.apply(set)
    keys = list(fp.index); parent = {k: k for k in keys}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = fp[keys[i]].split("|"), fp[keys[j]].split("|")
            if sum(x != y for x, y in zip(a, b)) <= 1 and dates[keys[i]] & dates[keys[j]]:
                parent[find(keys[i])] = find(keys[j])
    V.put("grp.mrgdate.all", len({find(k) for k in keys}))
    # majority-class share of the curated release (four-class accuracy reference for VR-LENS)
    cur = F[F.curated.astype(bool)]
    V.put("vrl.majority", f"{100 * cur.severity.value_counts(normalize=True).max():.1f}")


def addendum_values(V, R, st, F):
    """Values of docs/ANALYSIS_PLAN_REVISION3_ADDENDUM.md that are not part of a table."""
    C, g = st["configs"], st["gains"]
    ne = lambda lab, pr: C.get(f"sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3
    # B2a: frozen rotation fields in the standing campaign
    rt = json.load(open(R / "rev3_rotation.json"))
    cs = rt["constant_rotation_sessions"]
    V.put("r3rot.const.standing", sum(v["constant"] for k, v in cs.items() if k.startswith("standing")))
    V.put("r3rot.n.standing", sum(v["sessions"] for k, v in cs.items() if k.startswith("standing")))
    V.put("r3rot.const.seated", sum(v["constant"] for k, v in cs.items() if k.startswith("seated")))
    V.put("r3rot.n.seated", sum(v["sessions"] for k, v in cs.items() if k.startswith("seated")))
    V.put("r3rot.nrot", len(rt["rotation_features"])); V.put("r3rot.nnr", len(rt["non_rotation_features"]))
    # standing sessions: one constant rotation vector per session, and how many different ones
    rc = ["session_id", "campaign", "CameraRotationX", "CameraRotationY", "CameraRotationZ"]
    Fr = pd.read_csv(R.parent / "data" / "features_w30.csv.gz", usecols=rc)
    rs = Fr[Fr.campaign == "standing"].groupby("session_id")[rc[2:]]
    assert (rs.nunique() == 1).all().all()
    V.put("r3rot.distinct.standing", int(rs.first().round(6).drop_duplicates().shape[0]))
    V.put("r3rot.nzero", len(rt["standing_features_zero_everywhere"]))
    V.put("r3rot.nconst", len(rt["standing_features_constant_within_every_session"]))
    sf = rt["single_feature_campaign_auc"]
    V.put("r3rot.single.max", a2(max(sf[c] for c in rt["rotation_features"])))
    V.put("r3rot.single.nrmax", a2(max(sf[c] for c in rt["non_rotation_features"])))
    # A3: pair composition on the estimand's strata
    pr_ = json.load(open(R / "rev3_pairs.json"))
    for k, v in pr_.items():
        lab, pr = k.split("__"); b = f"r3pair.{lab.replace('sev', 's')}.{PR[pr]}"
        V.put(b + ".btw", pct(v["share_between_sessions"])); V.put(b + ".sg", pct(v["share_same_group_among_between"], 0))
    sg = [pr_[f"sev{lab}__{pr}"]["share_same_group_among_between"] for lab in (1, 2) for pr in CELL_P
          if f"sev{lab}__{pr}" in pr_ and not ne(lab, pr)]
    V.put("r3pair.cells.sg.min", pct(min(sg), 0)); V.put("r3pair.cells.sg.max", pct(max(sg), 0))
    bt = [v["share_between_sessions"] for k, v in pr_.items() if not ne(*[int(k[3]), k.split("__")[1]])]
    V.put("r3pair.btw.min", pct(min(bt), 0)); V.put("r3pair.btw.max", pct(max(bt), 0))
    # B3: sessions whose label is a single upward step (exposure time ranks them perfectly)
    t = F.t_elapsed.to_numpy()
    for lab in (1, 2):
        y = (F.severity.to_numpy() >= lab).astype(int); nboth = nstep = 0; pa = ps = 0
        for _, idx in F.groupby("session_id").indices.items():
            o = idx[np.argsort(t[idx], kind="stable")]; v = y[o]
            if v.min() == v.max():
                continue
            nboth += 1; npos = v.sum(); pa += npos * (len(v) - npos)
            if (np.diff(v) >= 0).all():
                nstep += 1; ps += npos * (len(v) - npos)
        V.put(f"r3sat.s{lab}.nboth", nboth); V.put(f"r3sat.s{lab}.nstep", nstep); V.put(f"r3sat.s{lab}.pairs", pct(ps / pa, 0))
        V.put(f"r3sat.s{lab}.nfall", nboth - nstep)
    # restricted within-session gains (sessions whose rating also falls): range over the family protocols
    for lab in (1, 2):
        v = [g[k]["session_nm"]["delta"] for pr in FAMILY_P if (k := f"sev{lab}__{pr}__CTXB+MOT__lroff") in g
             and "session_nm" in g[k] and not ne(lab, pr) and g[k]["session_nm"]["delta"] == g[k]["session_nm"]["delta"]]
        if v:
            V.put(f"r3grng.s{lab}.F.M.lo.n.min", d2(min(v))); V.put(f"r3grng.s{lab}.F.M.lo.n.max", d2(max(v)))
    # null spread against bootstrap width (family A, estimable tests)
    famA = st["null_lroff"]["families"]["lroff|CTXB+MOT|ctx"]["tests"]
    rat = [(t_["null_q97.5"] - t_["null_q2.5"]) / (g[k]["ctx"]["ci"][1] - g[k]["ctx"]["ci"][0]) for k, t_ in famA.items()
           if C[k]["pos_groups"] >= 3]
    V.put("r3nw.med", f"{np.median(rat):.1f}"); V.put("r3nw.min", f"{min(rat):.1f}"); V.put("r3nw.max", f"{max(rat):.1f}")
    # null location of the overall gain (LR+offset, MOT)
    og = [v["overall_auc"] - v["base_overall_auc"] for k, v in st["null_lroff"]["null_location"].items() if k.endswith("CTXB+MOT__lroff")]
    z0 = lambda x: "0.00" if abs(x) < 0.005 else d2(x)
    V.put("r3nl.og.min", z0(min(og))); V.put("r3nl.og.max", z0(max(og)))
    # the standing racing-game overall gain (sev2) against the null of the same model (descriptive:
    # the overall gain is not in the pre-specified families)
    nlf = pd.DataFrame(json.load(open(R / "rev3_null_lroff.json")))
    for lab, pr in ((2, "cell_standing_race"),):
        d = nlf[(nlf.label == f"sev{lab}") & (nlf.protocol == pr) & (nlf.features == "CTXB+MOT")]
        v = (d.overall_auc - d.base_overall_auc).dropna().to_numpy(); obs = g[f"sev{lab}__{pr}__CTXB+MOT__lroff"]["overall"]["delta"]
        V.put(f"r3nl.s{lab}.{PR[pr]}.og.p", pfmt((1 + (v >= obs).sum()) / (1 + len(v))))
        V.put(f"r3nl.s{lab}.{PR[pr]}.og.nr", ci2(np.quantile(v, [0.025, 0.975])))
    # the context baseline uses continuous exposure time, so it still ranks records inside a 30-s bin
    cb = [c["ctx"]["auc"] for k, c in C.items() if re.fullmatch(r"sev[12]__\w+__CTXB__ctxb", k) and c["pos_groups"] >= 3
          and c["ctx"]["auc"] == c["ctx"]["auc"]]
    V.put("r3cb.cauc.min", a2(min(cb))); V.put("r3cb.cauc.max", a2(max(cb)))
    # VR-LENS ablation: four-class accuracy on held-out groups without the inputs that identify people and games
    ab = json.load(open(R / "rev3_vrlens_ablation.json"))
    for t in ("A", "B", "C"):
        V.put(f"vrl.abl.{t}.acc", f"{100 * ab[f'sensVRLENS{t}_']['accuracy_4class_mean_over_repeats']:.1f}")
    # addendum 2 (post hoc): the group-disjoint run with group-aware internal stacking folds
    gcv = R / "rev3_vrlens_groupcv.json"
    if gcv.exists():
        V.put("vrl.gcv.acc", f"{100 * json.load(open(gcv))['accuracy_4class_mean_over_repeats']:.1f}")
        V.put("vrl.grp.acc1", f"{100 * json.load(open(R / 'vrlens.json'))['grouped_5x5']['accuracy_4class_mean_over_repeats']:.1f}")
    # recovery of the single-seed planted channel (bootstrap-only rule, Supplementary Table S19): the channel
    # alone against the same channel among the 28 telemetry features, between sessions
    mde = st.get("mde", {})
    est = [(lab, pr) for lab in (2, 1) for pr in FAMILY_P
           if C.get(f"r3pos70_sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 3) >= 3]
    le70 = lambda fs: sum(1 for lab, pr in est if (mde.get(f"sev{lab}__{pr}__{fs}__lroff__ctx", {}).get("mde") or 9) <= 0.70 + 1e-9)
    V.put("r3recov.c.n", len(est)); V.put("r3recov.alone.c.le70", le70("CTXB+SPK")); V.put("r3recov.mot.c.le70", le70("CTXB+MOT+SPK"))
    # one cell recurs: severity >= 1, standing racing game, within sessions (every null in which it was tested)
    rec = [("null_lroff", "lroff|CTXB+MOT|session", "CTXB+MOT__lroff"), ("null_lroff", "lroff|CTXB+SELFZ|session", "CTXB+SELFZ__lroff"),
           ("null_xgboff", "xgboff|CTXB+MOT|session", "CTXB+MOT__xgboff"),
           ("null_lroff_nr", "lroff|CTXB+MOTNR|session", "CTXB+MOTNR__lroff"), ("null_lroff_nm", "lroff|CTXB+MOT|session_nm", "CTXB+MOT__lroff"),
           ("null_lroffp", "lroffp|CTXB+MOT|session", "CTXB+MOT__lroffp"), ("null_lroff_block", "lroff|CTXB+MOT|session", "CTXB+MOT__lroff")]
    tt = [st[nk]["families"][fam]["tests"][f"sev1__cell_standing_race__{m}"] for nk, fam, m in rec if nk in st]
    V.put("r3rec.n", WORDS.get(len(tt), len(tt)))
    V.put("r3rec.d.min", d2(min(t_["observed_gain"] for t_ in tt))); V.put("r3rec.d.max", d2(max(t_["observed_gain"] for t_ in tt)))
    V.put("r3rec.p.min", pfmt(min(t_["p_perm"] for t_ in tt))); V.put("r3rec.p.max", pfmt(max(t_["p_perm"] for t_ in tt)))
    V.put("r3rec.ph.min", pfmt(min(t_["p_holm"] for t_ in tt)))
    # the 200-permutation secondary nulls: smallest Holm-adjusted p over all their families
    # ... over the families that Supplementary Table S20 shows
    s20 = {"null_lroff_block": ("ctx", "session"), "null_lroff_nm": ("session_nm",), "null_lroffp": ("ctx", "session"), "null_lroff_nr": ("ctx", "session")}
    V.put("r3sec.ph.min", pfmt(min(t_["p_holm"] for nk, ests in s20.items() for fam, f in st[nk]["families"].items()
                                   if fam.split("|")[2] in ests for t_ in f["tests"].values() if t_["p_holm"] == t_["p_holm"])))
    V.put("r3rec.nsess", int(F[(F.campaign == "standing") & (F.game == "Race")].session_id.nunique()))
    # finite permutations of the n/e cell
    tA = famA.get("sev2__cell_seated_flight__CTXB+MOT__lroff")
    if tA:
        V.put("r3n.cSF.np", tA["n_perm"]); V.put("r3n.cSF.pmin", pfmt(1 / (tA["n_perm"] + 1)))
    k_sf = "sev2__cell_seated_flight__CTXB+MOT__xgboff"
    if "null_xgboff" in st:
        V.put("r3n.cSF.np.xo", st["null_xgboff"]["families"]["xgboff|CTXB+MOT|ctx"]["tests"][k_sf]["n_perm"])
    n2 = [t_["n_perm"] for nk in ("null_lroff_block", "null_lroffp", "null_lroff_nr", "null_lroff_nm") if nk in st
          for d in st[nk]["families"].values() for k, t_ in d["tests"].items() if k.startswith("sev2__cell_seated_flight__")]
    if n2:
        V.put("r3n.cSF.np2.min", min(n2)); V.put("r3n.cSF.np2.max", max(n2))
    # VRSQ: cell medians without the all-zero files
    S = pd.read_csv(R / "session_level.csv"); Vq = S[S.vrsq_post_total.notna()].copy()
    Vq["all_zero"] = (Vq.vrsq_post_raw_sum == 0) & (Vq.vrsq_pre_total == 0)
    for (c, gm), d in Vq.groupby(["campaign", "game"]):
        w = d[~d.all_zero]; k = f"r3v.med.{c}{gm}"
        V.put(k, f"{w.vrsq_post_total.median():.1f}"); V.put(k + ".n", len(w)); V.put(k + ".all", f"{d.vrsq_post_total.median():.1f}")
        V.put(k + ".sev2", pct(w.ing_frac_ge2.mean()))
    # secondary nulls: how many tests reach Holm p < 0.05 with an interval excluding 0
    for nk, pre in (("null_lroff_block", "r3nb"), ("null_lroffp", "r3n"), ("null_lroff_nr", "r3n"), ("null_lroff_nm", "r3nm")):
        nl = st.get(nk)
        if not nl:
            continue
        for fam, d in nl["families"].items():
            ln, fs, stat = fam.split("|")
            n_ = sum(t_["p_holm"] < 0.05 and g.get(k, {}).get(stat, {}).get("ci", [0])[0] > 0 for k, t_ in d["tests"].items())
            V.put(f"{pre}.{LN[ln]}.{FS3[fs]}.{SL[stat]}.ndecl", WORDS.get(n_, n_))


def run(V, F, R, GEN, FIG):
    """Writes every revision-3 value, table and figure. A result file that does not exist yet is
    reported as TODO-RUN and its outputs are skipped (fa_verify then reports the missing values)."""
    V.R = str(R)
    st = values(V, R)
    range_values(V, st); second_values(V, R); family_values(V, st); control_values(V, st); power_values(V, st)
    try:
        grouping_values(V, st)
    except KeyError as e:
        print('TODO-RUN (grouping values):', repr(e))
    try:
        misc_values(V, R, st, F)
    except FileNotFoundError as e:
        print("TODO-RUN (misc values):", e)
    try:
        addendum_values(V, R, st, F)
    except (FileNotFoundError, KeyError) as e:
        print("TODO-RUN (addendum values):", repr(e))
    jobs = [("r3_tab_decode.tex", lambda: table_decode(V, R)), ("r3_tab_gain.tex", lambda: table_gain(V)),
            ("r3_supp_gain_s1.tex", lambda: table_gain(V, lab="s1", label="tab:supp_gain_s1", star=False, caption=
                r"As Table~\mref{tab:gain} for the secondary label \sevone{}.")),
            ("r3_supp_second.tex", lambda: table_second_full(V)),
            ("r3_supp_gain_z.tex", lambda: table_gain(V, fs="Z", label="tab:supp_gain_z", star=False, caption=
                r"As Table~\mref{tab:gain} for the self-normalized features (SELFZ) instead of MOT (\sevtwo{})."))
            , ("r3_supp_gain_xgb.tex", lambda: table_gain(V, ln="xo", label="tab:supp_gain_xgb", star=False, caption=
                r"As Table~\mref{tab:gain} for gradient boosting with the context baseline as offset instead of logistic regression (\sevtwo{}); Holm-adjusted permutation $p$ against its own null.")),
            ("r3_tab_mde.tex", lambda: table_power(V, st)), ("r3_tab_control.tex", lambda: table_control(V)),
            ("r3_tab_second.tex", lambda: table_second(V)), ("r3_supp_groups.tex", lambda: table_groups(V, R)),
            ("r3_supp_power.tex", lambda: table_power_full(V, st)), ("r3_supp_recovery.tex", lambda: table_recovery(V, st)),
            ("r3_supp_secondary.tex", lambda: table_secondary(V, st)), ("r3_supp_sim.tex", lambda: table_sim_full(V)),
            ("r3_supp_falling.tex", lambda: table_falling(V)), ("r3_supp_nr.tex", lambda: table_nr(V)),
            ("r3_supp_control.tex", lambda: table_control_full(V)), ("r3_supp_nullranges.tex", lambda: table_nullranges(V)),
            ("r3_supp_oldwithin.tex", lambda: table_old_within(V)), ("r3_supp_decode_session.tex", lambda: table_decode_session(V))]
    for name, fn in jobs:
        try:
            (GEN / name).write_text(fn())
        except (FileNotFoundError, KeyError) as e:
            print(f"TODO-RUN ({name}):", repr(e))
    fig_gain_null(st, R, FIG / "fig_gain_null.pdf")
    claims(R, st)
    return st


# ---------------------------------------------------------------- supplementary tables from existing results
def _lab_cell(lab, i):
    return (chr(92) + "sevtwo{}" if lab == 2 else chr(92) + "sevone{}") if i == 0 else ""


def _ci(V, k):
    return V.v.get(k, "")


def table_falling(V):
    """S22: within-session AUC on the sessions whose rating also falls, for all three models (addendum B3)."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Within-session AUC on the sessions whose rating also falls, where exposure time cannot order the records (addendum B3), for the context baseline and both offset learners in the protocols of the primary families: AUC of the baseline, and gain of each learner over it, with 95\,\% group-bootstrap intervals (nested weights); in parentheses, the Holm-adjusted permutation $p$ of LR+offset against the null of this estimand (\V{r3nm.lo.M.n.nperm} permutations, so the smallest attainable Holm-adjusted $p$ is \V{r3nm.lo.M.n.holmfloor}; boosting+offset was not tested). n/e: no session with a falling rating and both classes. $^\dagger$: fewer than three groups with a positive record, so no interval is reported.}",
         r"\label{tab:supp_falling}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Label & Protocol & Baseline AUC & LR + offset: gain ($p_\mathrm{Holm}$) & Boosting + offset: gain \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAMILY_P):
            b = f"s{lab}.{PR[pr]}"
            base = f"{_a(V, f'r3.{b}.B.cb', 'nauc')} {_ci(V, f'r3.{b}.B.cb.nci')}".strip()
            lo = _pcell(V, f"r3g.{b}.M.lo", f"r3nm.{b}.M.lo", "n"); xo = _pcell(V, f"r3g.{b}.M.xo", "-", "n")
            L.append(f"{_lab_cell(lab, i)} & {PR_NAME[pr]} & {base} & {lo} & {xo} \\\\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_nr(V):
    """S23: LR+offset without the rotation-derived features (addendum B2c), with intervals."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{LR+offset with the 14 telemetry features not derived from the rotation fields (context baseline plus the 14 non-rotation features, CTXB+MOTNR; addendum B2c): gain over the context baseline with 95\,\% group-bootstrap intervals and, in parentheses, the Holm-adjusted permutation $p$ within the family of 14 (\V{r3n.lo.MNR.c.nperm} permutations, so the smallest attainable Holm-adjusted $p$ is \V{r3n.lo.MNR.c.holmfloor}). For comparison, the gains with all 28 features (Table~\mref{tab:gain}, Table~S14). $^\dagger$: fewer than three groups with a positive record, so no interval is reported.}",
         r"\label{tab:supp_nr}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Without rotation (14 features)} & All 28 features \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-6}",
         r"Label & Protocol & Overall & Between sessions & Within sessions & Between / within \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAMILY_P):
            b = f"s{lab}.{PR[pr]}"; g = f"r3g.{b}.MNR.lo"; pk = f"r3n.{b}.MNR.lo"
            full = f"{_d(V, f'r3g.{b}.M.lo', 'c')} / {_d(V, f'r3g.{b}.M.lo', 'w')}"
            L.append(f"{_lab_cell(lab, i)} & {PR_NAME[pr]} & {_g(V, g, 'o')} & {_g(V, g, 'c', pk)} & {_g(V, g, 'w', pk)} & {full} \\\\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_control_full(V):
    """S24: label-shift control for both labels, including the context baseline alone (plan stage 3)."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Label-shift control in full: overall ROC-AUC on held-out groups with correct (c) and circularly displaced (s) training labels, for both labels, every learner of the control and the context baseline alone (which knows only game, campaign and exposure time). Supplementary Note~S6 discusses it; inside a single cell, game and campaign are constant, so a telemetry-only model that stays predictive under the shift there must use differences between sessions. n/e: not estimable (fewer than three groups with a positive record); ``--'': not run.}",
         r"\label{tab:supp_control}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}ll" + "c" * 12 + r"@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Context baseline} & \multicolumn{2}{c}{Default HGB, MOT} & \multicolumn{2}{c}{Default HGB, MOT+TIME} & \multicolumn{2}{c}{HGB-reg, MOT+TIME} & \multicolumn{2}{c}{LR + offset} & \multicolumn{2}{c}{Boosting + offset} \\",
         "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(6)),
         "Label & Protocol & " + " & ".join(["c & s"] * 6) + r" \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(NULL_P):
            b = f"s{lab}.{PR[pr]}"; cells = []
            for fl in ("B.cb", "M.hgb", "MT.hgb", "MT.hgbr", "M.lo", "M.xo"):
                cells += [V.v.get(f"r3.{b}.{fl}.auc", "--"), V.v.get(f"r3.ctl.{b}.{fl}.auc", "--")]
            L.append(f"{_lab_cell(lab, i)} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_nullranges(V):
    """S25: where the two label-transplant nulls put the gain (mean and 95 % range), next to the observed gain."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Gain of LR+offset (MOT) under the two label-transplant nulls: mean and 2.5--97.5 percentiles over the permutations of the session-level null (\V{r3n.lo.M.c.nperm} permutations) and of the group-block null (\V{r3nb.lo.M.c.nperm} permutations), next to the observed gain, between sessions (B) and within sessions (W). For the seated flight-game cell (\sevtwo{}), only the permutations with a finite gain enter (\V{r3n.cSF.np} and \V{r3n.cSF.np2.max}). $^\dagger$: fewer than three groups with a positive record, so no interval is reported.}",
         r"\label{tab:supp_nullranges}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Observed gain} & \multicolumn{2}{c}{Session-level null} & \multicolumn{2}{c}{Group-block null} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Label & Protocol & B & W & B & W & B & W \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(FAMILY_P):
            b = f"s{lab}.{PR[pr]}"; cells = [_d(V, f"r3g.{b}.M.lo", "c"), _d(V, f"r3g.{b}.M.lo", "w")]
            for pre in ("r3n", "r3nb"):
                for s in ("c", "w"):
                    cells.append(f"{V.v.get(f'{pre}.{b}.M.lo.{s}.nm', '--')} {V.v.get(f'{pre}.{b}.M.lo.{s}.nr', '')}".strip())
            L.append(f"{_lab_cell(lab, i)} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_old_within(V):
    """S26: within-session AUC of the earlier learners (plan stage 4)."""
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Earlier learners within sessions (plan stage 4): within-session AUC of exposure time alone (logistic regression) and the within-session gain of MOT+TIME over TIME and of MOT+CTX+TIME over CTX+TIME for the same learner, with 95\,\% group-bootstrap intervals (nested weights), in the protocols the earlier analysis ran. No null was run for these gains. n/e: not estimable (fewer than three groups with a positive record).}",
         r"\label{tab:supp_oldwithin}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r" & & TIME & \multicolumn{3}{c}{MOT+TIME minus TIME} & \multicolumn{2}{c}{MOT+CTX+TIME minus CTX+TIME} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
         r"Label & Protocol & LR & HGB-reg & Default HGB & LR & HGB-reg & LR \\", r"\midrule"]
    for lab in (2, 1):
        for i, pr in enumerate(WITHIN_P + CROSS_P):
            b = f"s{lab}.{PR[pr]}"
            t = f"{V.v.get(f'r3.{b}.T.lr.wauc', '--')} {V.v.get(f'r3.{b}.T.lr.wci', '')}".strip()
            cells = [t] + [_g(V, f"r3g.{b}.{fs}.{ln}", "w") for fs, ln in (("MT", "hgbr"), ("MT", "hgb"), ("MT", "lr"), ("MCT", "hgbr"), ("MCT", "lr"))]
            L.append(f"{_lab_cell(lab, i)} & {PR_NAME[pr]} & " + " & ".join(cells) + r" \\")
        if lab == 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_decode_session(V):
    """S27: decoding at the session level (plan stage 1)."""
    rows = [("gamerace", "Game (racing vs flight)"), ("campaignstanding", "Campaign (standing vs seated)"),
            ("cellseatedFlight", "Cell: seated, flight"), ("cellseatedRace", "Cell: seated, racing"),
            ("cellstandingFlight", "Cell: standing, flight"), ("cellstandingRace", "Cell: standing, racing")]
    L = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
         r"\caption{What the telemetry encodes, at the session level (plan stage 1): ROC-AUC over held-out sessions, each scored by the mean of its records' scores, compared within the models of a fold and averaged over repeats (no interval). Record-level values: Table~\mref{tab:decode}.}",
         r"\label{tab:supp_decode_session}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{MOT (28)} & \multicolumn{2}{c}{SELFZ (28)} & \multicolumn{2}{c}{Without rotation (14)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"Target & LR & HGB & LR & HGB & LR & HGB \\", r"\midrule"]
    for t, name in rows:
        L.append(f"{name} & " + " & ".join(V.v.get(f"r3d.{fs}.{ln}.{t}.sauc", "--") for fs in ("MOT", "SELFZ", "MOTNR") for ln in ("lr", "hgb")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


FAMILY_P = NULL_P


def family_values(V, st):
    """Ranges of the gains, null location, minimum-detectable-effect ranges and the
    pre-specified 'evidence against' rule (docs/ANALYSIS_PLAN_REVISION3.md, Stage 3)."""
    g = st["gains"]; C = st["configs"]
    ne = {k for k, c in C.items() if c["pos_groups"] < 3}
    for lab in (1, 2):
        for sc, prs in (("W", WITHIN_P), ("F", FAMILY_P), ("X", CROSS_P), ("C", CELL_P)):
            for fs, ln in (("CTXB+MOT", "lroff"), ("CTXB+MOT", "xgboff"), ("CTXB+SELFZ", "lroff"),
                           ("CTXB+MOTNR", "lroff"), ("CTXB+MOT", "lroffp")):
                for m, s in (("overall", "o"), ("ctx", "c"), ("session", "w"), ("session_nm", "n")):
                    v = [g[k][m]["delta"] for pr in prs if (k := f"sev{lab}__{pr}__{fs}__{ln}") in g and k not in ne
                         and m in g[k] and g[k][m]["delta"] == g[k][m]["delta"]]
                    if v:
                        b = f"r3grng.s{lab}.{sc}.{FS3[fs]}.{LN[ln]}.{s}"
                        V.put(b + ".min", d2(min(v))); V.put(b + ".max", d2(max(v)))
                        V.put(b + ".nneg", sum(x < 0 for x in v)); V.put(b + ".n", len(v))
                        V.put(b + ".nciabove", sum(g[f"sev{lab}__{pr}__{fs}__{ln}"][m]["ci"][0] > 0 for pr in prs
                                                   if f"sev{lab}__{pr}__{fs}__{ln}" in g and f"sev{lab}__{pr}__{fs}__{ln}" not in ne
                                                   and m in g[f"sev{lab}__{pr}__{fs}__{ln}"]))
    nl = st.get("null_lroff")
    if nl:
        cx = [v["context_stratified_auc"] for k, v in nl["null_location"].items() if k.endswith("CTXB+MOT__lroff")]
        V.put("r3nl.cx.min", a2(min(cx))); V.put("r3nl.cx.max", a2(max(cx)))
        bw = [v["base_within_session_auc"] for k, v in nl["null_location"].items() if k.endswith("CTXB+MOT__lroff")]
        V.put("r3nl.bw.min", a2(min(bw))); V.put("r3nl.bw.max", a2(max(bw)))
    # minimum detectable effect among MOT (ranges, counts)
    mde = st["mde"]
    for stat, s in (("ctx", "c"), ("session", "w")):
        rec, none = [], 0
        for lab in (1, 2):
            vals = []
            for pr in FAMILY_P:
                if f"r3pos70_sev{lab}__{pr}__CTXB__ctxb" in ne:
                    continue
                v = mde.get(f"sev{lab}__{pr}__CTXB+MOT+SPK__lroff__{stat}", {}).get("mde")
                if v:
                    vals.append(v); rec.append(v)
                else:
                    none += 1
            if vals:
                V.put(f"r3mrng.s{lab}.MS.{s}.min", f"{min(vals):.2f}"); V.put(f"r3mrng.s{lab}.MS.{s}.max", f"{max(vals):.2f}")
        V.put(f"r3mrng.MS.{s}.nnone", WORDS.get(none, none)); V.put(f"r3mrng.MS.{s}.nrec", WORDS.get(len(rec), len(rec)))
    # evidence against a signal of size X: smallest planted X recovered (gain interval > 0) whose
    # recovered lower bound exceeds the upper bound of the real gain
    against = []
    for lab in (2, 1):
        for pr in FAMILY_P:
            real = g.get(f"sev{lab}__{pr}__CTXB+MOT__lroff")
            if real is None or f"sev{lab}__{pr}__CTXB+MOT__lroff" in ne:
                continue
            for stat in ("ctx", "session"):
                lv = mde.get(f"sev{lab}__{pr}__CTXB+MOT+SPK__lroff__{stat}", {}).get("levels", {})
                hit = [float(x) for x, p in sorted(lv.items(), key=lambda t: float(t[0]))
                       if p["ci"][0] > 0 and real[stat]["ci"][1] < p["ci"][0]]
                if hit:
                    against.append((lab, pr, stat, hit[0]))
    V.put("r3against.n", WORDS.get(len(against), len(against)))
    lab_s = {2: r"\sevtwo{}", 1: r"\sevone{}"}
    short = {"within_seated": "within seated", "within_standing": "within standing", "pooled": "pooled",
             "cell_seated_flight": "seated flight-game cell", "cell_seated_race": "seated racing-game cell",
             "cell_standing_flight": "standing flight-game cell", "cell_standing_race": "standing racing-game cell"}
    allb = all(st_ == "ctx" for _, _, st_, _ in against)
    parts = []
    for l in (2, 1):
        it = [f"{short[p]} ({x:.2f}{'' if allb or st_ == 'ctx' else ', within sessions'})" for ll, p, st_, x in against if ll == l]
        if it:
            parts.append(f"{lab_s[l]}: " + (", ".join(it[:-1]) + " and " + it[-1] if len(it) > 1 else it[0]))
    V.put("r3against.list", "; ".join(parts) or "none")
    V.put("r3against.where", "all between sessions" if allb and against else "")
    if against:
        V.put("r3against.min", f"{min(x for *_, x in against):.2f}"); V.put("r3against.max", f"{max(x for *_, x in against):.2f}")
    allc = [v for lab in (1, 2) for pr in FAMILY_P if f"r3pos70_sev{lab}__{pr}__CTXB__ctxb" not in ne
            and (v := mde.get(f"sev{lab}__{pr}__CTXB+MOT+SPK__lroff__ctx", {}).get("mde"))]
    V.put("r3mrng.MS.c.allmin", f"{min(allc):.2f}"); V.put("r3mrng.MS.c.allmax", f"{max(allc):.2f}")
    return against


def table_second_full(V):
    rows = [("D1: record-wise AUC (HGB / LR)", "D1_recordwise_hgb", "D1_recordwise_logreg", None),
            ("D1: participant-disjoint AUC (HGB)", "D1_grouped_hgb", None, "ci"),
            ("D1: participant-disjoint AUC (LR)", "D1_grouped_logreg", None, "ci"),
            ("Exposure time alone (LR)", "D1_grouped_TIME", None, "ci"),
            ("D3: HGB, correct labels", "D3_true_hgb", None, "ci"), ("D3: HGB, shifted labels", "D3_shifted_hgb", None, "ci"),
            ("D3: LR, correct labels", "D3_true_logreg", None, "ci"), ("D3: LR, shifted labels", "D3_shifted_logreg", None, "ci")]
    L = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
         r"\caption{The second archive (Yalcin et al.\ \cite{yalcin2024}) in full: participant-disjoint grouped 5-fold, 5 repeats; within-participant AUC compares records of the same participant only. With all " + V.v["y.n"] + r" recordings and without the one that lasts more than twice the median ride.}",
         r"\label{tab:supp_second}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{All recordings} & \multicolumn{2}{c}{Without the long recording} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Quantity & AUC & Within participant & AUC & Within participant \\", r"\midrule"]
    for name, k, k2, ci in rows:
        cells = []
        for t in ("", "nl."):
            if k2:
                cells += [f"{V.v['y.' + t + k]} / {V.v['y.' + t + k2]}", ""]
            else:
                cells += [f"{V.v['y.' + t + k]} {V.v['y.' + t + k + '.ci']}", f"{V.v['y.' + t + k + '.w']} {V.v['y.' + t + k + '.wci']}"]
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L.append(r"D1: participant identified (\%; most frequent) & " + f"{V.v['y.id.acc']} ({V.v['y.id.maj']}) & & {V.v['y.nl.id.acc']} ({V.v['y.nl.id.maj']}) & " + r"\\")
    L.append(r"D2: prevalence score, pooled leave-one-out & " + f"{V.v['y.prevalence_score_LOPO_pooled_auc']} & & {V.v['y.nl.prevalence_score_LOPO_pooled_auc']} & " + r"\\")
    L.append(r"D2: HGB, pooled leave-one-out / within fold & " + f"{V.v['y.hgb_LOPO_pooled_auc']} / {V.v['y.hgb_grouped5_within_fold_auc']} & & {V.v['y.nl.hgb_LOPO_pooled_auc']} / {V.v['y.nl.hgb_grouped5_within_fold_auc']} & " + r"\\")
    if "y.off.baseline" in V.v:
        for nm_, lab_ in (("baseline", "Offset design: time-only baseline ($t$, $t^2$)"), ("offset", "Offset design: baseline + 18 eye-tracking features")):
            L.append(f"{lab_} & " + " & ".join(f"{V.v[f'y.{t}off.{nm_}']} {V.v[f'y.{t}off.{nm_}.ci']} & {V.v[f'y.{t}off.{nm_}.w']} {V.v[f'y.{t}off.{nm_}.wci']}" for t in ("", "nl.")) + r" \\")
        L.append(r"Offset design: gain & " + " & ".join(f"{V.v[f'y.{t}off.go']} {V.v[f'y.{t}off.go.ci']} & {V.v[f'y.{t}off.gw']} {V.v[f'y.{t}off.gw.ci']}" for t in ("", "nl.")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def power_claims(st):
    """Statements about the power of the pre-specified rule (addendum B1) and the secondary analyses."""
    out = {}; pw = st.get("power")
    if not pw:
        return {"power_results_present": False}
    C, g = st["configs"], st["gains"]; cells = pw["cells"]
    ne = lambda lab, pr: C.get(f"sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3
    est = [(lab, pr) for lab in (1, 2) for pr in FAMILY_P if not ne(lab, pr)]
    P = lambda ln, lev, stat: [cells[f"{ln}__{lev:.2f}__sev{lab}__{pr}"][stat]["power"] for lab, pr in est]
    # "only ... in a minority": 0.80 between sessions declared in at least half the realizations in fewer than half the combinations
    out["power_080_between_minority"] = sum(x >= 0.5 for x in P("lroff", 0.80, "ctx")) < len(est) / 2
    out["power_080_within_minority"] = sum(x >= 0.5 for x in P("lroff", 0.80, "session")) < len(est) / 2
    out["power_070_almost_never"] = all(sum(x >= 0.5 for x in P("lroff", 0.70, st_)) <= 1 for st_ in ("ctx", "session"))
    out["boosting_similar_within_power"] = abs(sum(x >= 0.5 for x in P("xgboff", 0.80, "session")) - sum(x >= 0.5 for x in P("lroff", 0.80, "session"))) <= 2
    out["power_decreases_with_size"] = (np.mean(P("lroff", 0.65, "ctx")) <= np.mean(P("lroff", 0.70, "ctx")) <= np.mean(P("lroff", 0.80, "ctx")))
    out["boosting_lower_power_than_lr"] = np.mean(P("xgboff", 0.80, "ctx")) < np.mean(P("lroff", 0.80, "ctx"))
    # the bootstrap-only rule of the original plan looked far more sensitive
    mde = st["mde"]
    rec = sum((v := mde.get(f"sev{lab}__{pr}__CTXB+MOT+SPK__lroff__ctx", {}).get("mde")) is not None for lab, pr in est)
    out["bootstrap_only_rule_looked_more_sensitive"] = rec > sum(x >= 0.5 for x in P("lroff", 0.80, "ctx")) + 2
    # evidence against: some, and only between sessions in few protocols
    ag = [(lab, pr, stat) for lab, pr in est for stat in ("ctx", "session") for lev in LEVELS
          if cells[f"lroff__{lev:.2f}__sev{lab}__{pr}"][stat]["evidence_against"]]
    # the seed-based 'evidence against' rule is uninformative: it flags planted sizes no design of this size could exclude
    out["evidence_against_rule_flags_small_sizes"] = any(cells[f"lroff__0.60__sev{lab}__{pr}"][stat]["evidence_against"]
                                                         for lab, pr in est for stat in ("ctx", "session"))
    # secondary analyses: nothing declared (Holm p < 0.05 and interval above 0)
    for nk in ("null_lroff_block", "null_lroffp", "null_lroff_nr", "null_lroff_nm"):
        nl = st.get(nk)
        if not nl:
            out[f"{nk}_present"] = False; continue
        decl = [k for fam, d in nl["families"].items() for k, t in d["tests"].items()
                if t["p_holm"] < 0.05 and g.get(k, {}).get(fam.split("|")[2], {}).get("ci", [0])[0] > 0]
        out[f"{nk}_nothing_declared"] = not decl
    # the standing racing-game overall gain is the only same-setting or single-cell gain whose interval excludes 0
    pos = [(pr, e) for pr in dict.fromkeys(WITHIN_P + NULL_P) if not ne(2, pr) for e in ("overall", "ctx", "session", "session_nm")
           if e in g[f"sev2__{pr}__CTXB+MOT__lroff"] and g[f"sev2__{pr}__CTXB+MOT__lroff"][e]["ci"][0] > 0]
    out["only_cTR_overall_gain_ci_above_0"] = pos == [("cell_standing_race", "overall")]
    # severity >= 1: the within-session gain in the same cell has an interval above 0 but no Holm p < 0.05
    t1 = st["null_lroff"]["families"]["lroff|CTXB+MOT|session"]["tests"]["sev1__cell_standing_race__CTXB+MOT__lroff"]
    out["sev1_cTR_within_ci_above_0_not_declared"] = (g["sev1__cell_standing_race__CTXB+MOT__lroff"]["session"]["ci"][0] > 0
                                                      and t1["p_holm"] >= 0.05)
    # event-locked: other learners do not reproduce the main learner's change
    el = st["event_locked"]
    out["event_locked_not_reproduced"] = all(el[k]["did"] < 0.05 for k in ("sev2__pooled__CTXB+SELFZ__lroff", "sev2__pooled__CTXB+MOT__xgboff"))
    return out


def claims(R, st):
    """Full-precision checks of the qualitative statements of the revision-3 text. Each entry is
    True or False; fa_verify fails if any is False."""
    C, g = st["configs"], st["gains"]; out = {}
    ne = {k for k, c in C.items() if c["pos_groups"] < 3}
    gk = lambda lab, pr, fs="CTXB+MOT", ln="lroff": f"sev{lab}__{pr}__{fs}__{ln}"
    # telemetry beyond context
    out["overall_gain_negative_all_same_setting_s2"] = all(g[gk(2, pr)]["overall"]["delta"] < 0 for pr in WITHIN_P)
    famA = st["null_lroff"]["families"]["lroff|CTXB+MOT|ctx"]["tests"]; famB = st["null_lroff"]["families"]["lroff|CTXB+MOT|session"]["tests"]
    out["familyA_no_raw_p_below_0.05"] = all(t["p_perm"] >= 0.05 for t in famA.values())
    out["familyA_B_none_after_holm"] = all(t["p_holm"] >= 0.05 for t in list(famA.values()) + list(famB.values()))
    out["familyB_raw_set"] = {k for k, t in famB.items() if t["p_perm"] < 0.05} == {
        gk(1, "pooled"), gk(1, "cell_standing_race"), gk(2, "cell_seated_flight")}
    out["familyB_seated_flight_s2_is_ne"] = gk(2, "cell_seated_flight") in ne or C[f"sev2__cell_seated_flight__CTXB__ctxb"]["pos_groups"] < 3
    for name in ("null_xgboff", "null_lroff"):
        for fam, d in st[name]["families"].items():
            out[f"{fam}_none_after_holm"] = all(t["p_holm"] >= 0.05 for t in d["tests"].values())
    loc = st["null_lroff"]["families"]
    nA = [t["null_mean"] for t in famA.values()]; nB = [t["null_mean"] for t in famB.values()]
    out["null_between_gain_centres_on_zero"] = all(abs(x) < 0.04 for x in nA)
    out["null_within_gain_negative"] = all(x < 0 for x in nB)
    # null spread vs bootstrap width (median ratio between 0.5 and 2)
    rat = [(t["null_q97.5"] - t["null_q2.5"]) / (g[k]["ctx"]["ci"][1] - g[k]["ctx"]["ci"][0]) for k, t in famA.items() if k not in ne]
    out["null_spread_wider_than_bootstrap"] = float(np.median(rat)) >= 1.2
    # second archive: the flagged recording is long only because of its first gap, and most bins
    # without a sample lie in it
    y_ = json.load(open(R / "rev3_yalcin.json")); P_ = {k: v for k, v in y_["participants"].items() if "empty_bins" in v}
    if P_:
        longs = [str(k) for k in y_["without_long_recordings"]["excluded"]]
        med = float(np.median([v["duration_s"] for v in P_.values()]))
        out["yalcin_long_recording_long_only_because_of_first_gap"] = all(
            P_[k]["duration_s"] > 2 * med and P_[k]["duration_s"] - P_[k]["first_gap_s"] <= 2 * med for k in longs)
        emp = sum(v["empty_bins"] for v in P_.values())
        out["yalcin_most_empty_bins_in_long_recording"] = sum(P_[k]["empty_bins"] for k in longs) > emp / 2
    # event-locked: bootstrap interval excludes zero but inside the null range (severity >= 2)
    el = st["event_locked"]["sev2__pooled__CTXB+MOT__lroff"]
    out["event_locked_s2_ci_excludes_zero_but_p_ge_0.05"] = el["ci"][0] > 0 and el["p_perm_one_sided"] >= 0.05 and el["did"] < el["null_q97.5"]
    # shift: telemetry moves the overall AUC only towards chance
    xs = [C[gk(2, pr)]["overall"]["auc"] for pr in CROSS_P]; xb = [C[f"sev2__{pr}__CTXB__ctxb"]["overall"]["auc"] for pr in CROSS_P]
    out["shift_lroff_towards_chance"] = all(a >= b for a, b in zip(xs, xb)) and max(xs) < 0.6
    # flight -> racing: telemetry corrects a reversed baseline time course to about chance, not above it
    f2r, f2rb = C[gk(2, "flight_to_race")], C["sev2__flight_to_race__CTXB__ctxb"]
    out["f2r_baseline_within_session_below_chance"] = f2rb["session"]["auc"] < 0.5
    out["f2r_lroff_overall_and_within_ci_include_chance"] = (f2r["overall"]["ci"][0] <= 0.5 <= f2r["overall"]["ci"][1]
                                                           and f2r["session"]["ci"][0] <= 0.5 <= f2r["session"]["ci"][1])
    out["f2r_is_largest_shift_gain"] = max(CROSS_P, key=lambda pr: g[gk(2, pr)]["overall"]["delta"]) == "flight_to_race"
    out["shift_ctx_gain_ci_includes_zero_all"] = all(g[gk(2, pr)]["ctx"]["ci"][0] <= 0 for pr in CROSS_P)
    # power of the pre-specified rule (addendum B1): see power_claims()
    out.update(power_claims(st))
    # conditional power: within sessions the interval condition removes power the permutation part has
    sd = pd.DataFrame(json.load(open(R / "rev3_posctl_seeds.json"))["rows"])
    sd = sd[[C.get(f"{l}__{p}__CTXB__ctxb", {}).get("pos_groups", 0) >= 3 for l, p in zip(sd.label, sd.protocol)]]
    x = sd[(sd.learner == "lroff") & (sd.level == 0.8)].groupby(["label", "protocol"])
    full_w = (x.apply(lambda z: ((z.holm_session < 0.05) & (z.ci_lo_session > 0)).mean()) >= 0.5).sum()
    perm_w = (x.apply(lambda z: (z.holm_session < 0.05).mean()) >= 0.5).sum()
    out["within_interval_condition_lowers_power"] = bool(perm_w > full_w)
    # the recurring cell (sev1, standing racing game, within sessions): positive in every null, never declared
    rec = [("null_lroff", "lroff|CTXB+MOT|session", "CTXB+MOT__lroff"), ("null_lroff", "lroff|CTXB+SELFZ|session", "CTXB+SELFZ__lroff"),
           ("null_xgboff", "xgboff|CTXB+MOT|session", "CTXB+MOT__xgboff"),
           ("null_lroff_nr", "lroff|CTXB+MOTNR|session", "CTXB+MOTNR__lroff"), ("null_lroff_nm", "lroff|CTXB+MOT|session_nm", "CTXB+MOT__lroff"),
           ("null_lroffp", "lroffp|CTXB+MOT|session", "CTXB+MOT__lroffp"), ("null_lroff_block", "lroff|CTXB+MOT|session", "CTXB+MOT__lroff")]
    tt = [st[nk]["families"][fam]["tests"][f"sev1__cell_standing_race__{m}"] for nk, fam, m in rec]
    out["recurring_cell_positive_never_declared"] = bool(all(t_["observed_gain"] > 0 and t_["p_holm"] >= 0.05 for t_ in tt))
    xo_rec = st["null_xgboff"]["families"]["xgboff|CTXB+MOT|session"]["tests"]["sev1__cell_standing_race__CTXB+MOT__xgboff"]["p_holm"]
    out["recurring_cell_smallest_holm_is_boosting"] = bool(all(xo_rec < t_["p_holm"] for t_ in tt if t_["p_holm"] != xo_rec))
    # the 200-permutation secondary nulls cannot reach a Holm-adjusted p below 0.05, and none came near that floor
    sec_f = [f for nk in ("null_lroff_block", "null_lroff_nm", "null_lroff_nr", "null_lroffp") for f in st[nk]["families"].values()]
    floor = [f["m"] / (f["n_perm"] + 1) for f in sec_f]
    out["secondary_nulls_cannot_declare"] = bool(min(floor) > 0.05)
    out["secondary_nulls_min_holm_far_above_floor"] = bool(min(t_["p_holm"] for f in sec_f for t_ in f["tests"].values()
                                                               if t_["p_holm"] == t_["p_holm"]) > 2 * max(floor))
    # every gain of a secondary analysis with an interval above 0 outside the shifted protocols lies in the
    # standing racing-game cell (sev1) or comes from boosting+offset on falling sessions, which has no null
    sec = []
    for k, v in g.items():
        p_ = k.split("__")
        if p_[0] not in ("sev1", "sev2") or p_[1] in CROSS_P or C.get(f"{p_[0]}__{p_[1]}__CTXB__ctxb", {}).get("pos_groups", 0) < 3:
            continue
        for e in ("overall", "ctx", "session", "session_nm"):
            if e in v and v[e]["ci"][0] > 0 and (p_[2], p_[3]) in (("CTXB+MOTNR", "lroff"), ("CTXB+MOT", "xgboff"), ("CTXB+MOT", "lroffp")):
                sec.append((p_[0], p_[1], p_[2], p_[3], e))
    out["secondary_ci_above_0_only_cTR_s1_or_boosting_falling"] = all(
        (s_[0] == "sev1" and s_[1] == "cell_standing_race") or (s_[3] == "xgboff" and s_[4] == "session_nm") for s_ in sec)
    # control, severity >= 1: no learner-cell combination above 0.55 keeps its AUC within 0.05 under the shift
    kk = 0
    for fs, ln in (("MOT", "hgb"), ("MOT+TIME", "hgb"), ("MOT+TIME", "hgbreg"), ("CTXB+MOT", "lroff"), ("CTXB+MOT", "xgboff")):
        tg = "r3control_" if ln in ("lroff", "xgboff") else "control_"
        for pr in CELL_P:
            k = f"sev1__{pr}__{fs}__{ln}"
            if k in C and f"{tg}{k}" in C and C[k]["pos_groups"] >= 3 and C[k]["overall"]["auc"] > 0.55:
                kk += C[f"{tg}{k}"]["overall"]["auc"] > C[k]["overall"]["auc"] - 0.05
    out["control_sev1_none_kept"] = kk == 0
    # earlier learners within sessions: no same-setting gain over exposure time with an interval above 0
    out["old_within_no_same_setting_gain_above_0"] = not any(
        g[k]["session"]["ci"][0] > 0 for k in g if k.split("__")[0] in ("sev1", "sev2") and k.split("__")[1] in WITHIN_P
        and k.split("__")[3] in ("hgb", "hgbreg", "logreg") and "session" in g[k])
    out["planted_gain_varies_little_over_seeds"] = bool(max(x.gain_ctx.std().max(), x.gain_session.std().max()) < 0.1)
    # the standing racing-game overall gain (sev2) lies inside the null of the same model
    nlf = pd.DataFrame(json.load(open(R / "rev3_null_lroff.json")))
    d = nlf[(nlf.label == "sev2") & (nlf.protocol == "cell_standing_race") & (nlf.features == "CTXB+MOT")]
    v = (d.overall_auc - d.base_overall_auc).dropna().to_numpy(); obs = g["sev2__cell_standing_race__CTXB+MOT__lroff"]["overall"]["delta"]
    out["cTR_overall_gain_inside_its_null"] = bool((1 + (v >= obs).sum()) / (1 + len(v)) > 0.05)
    # the two-group seated flight-game cell (sev2): nominal within-session p, not significant after Holm
    tsf = st["null_lroff"]["families"]["lroff|CTXB+MOT|session"]["tests"]["sev2__cell_seated_flight__CTXB+MOT__lroff"]
    out["sev2_cSF_within_nominal_only"] = bool(tsf["p_perm"] < 0.05 <= tsf["p_holm"])
    # context baseline: its own context-stratified AUC is not exactly 0.5 (continuous exposure time)
    cb = [c["ctx"]["auc"] for k, c in C.items() if re.fullmatch(r"sev[12]__\w+__CTXB__ctxb", k) and c["pos_groups"] >= 3]
    out["ctx_baseline_cauc_between_045_and_065"] = bool(0.45 <= min(cb) and max(cb) <= 0.65 and max(cb) > 0.55)
    # VR-LENS: below the majority share with all inputs, above it without the demographic answers
    maj = 1 - json.load(open(R / "stats.json"))["configs"]["rowwise_sev1__within_curated__VRLENS__stack"]["prevalence"]   # rating 0
    ab = json.load(open(R / "rev3_vrlens_ablation.json"))
    gr = json.load(open(R / "vrlens.json"))
    acc_full = gr["grouped_5x5"]["accuracy_4class_mean_over_repeats"]
    out["vrlens_full_below_majority"] = bool(acc_full < maj)
    out["vrlens_ablations_above_majority_by_little"] = bool(all(maj < ab[f"sensVRLENS{t}_"]["accuracy_4class_mean_over_repeats"] < maj + 0.02 for t in "ABC"))
    # addendum 2 (post hoc): group-aware internal stacking folds do not lift the detector above the majority share
    gcv = R / "rev3_vrlens_groupcv.json"
    if gcv.exists():
        a_g = json.load(open(gcv))["accuracy_4class_mean_over_repeats"]
        out["vrlens_groupaware_below_majority"] = bool(a_g < maj)
        out["vrlens_groupaware_within_2pts_of_recordwise_inner"] = bool(abs(a_g - acc_full) < 0.02)
        cf = json.load(open(R / "stats.json"))["configs"]
        out["vrlens_groupaware_auc_lower_than_recordwise_inner"] = bool(
            cf["sensVRLENSG_sev2__within_curated__VRLENS__stack"]["auc"] < cf["sensVRLENS_sev2__within_curated__VRLENS__stack"]["auc"])
    # the planted channel alone is recovered at smaller sizes than the same channel among the 28 features
    mde = st.get("mde", {})
    est = [(lab, pr) for lab in (2, 1) for pr in FAMILY_P if C.get(f"r3pos70_sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 3) >= 3]
    le70 = lambda fs: sum(1 for lab, pr in est if (mde.get(f"sev{lab}__{pr}__{fs}__lroff__ctx", {}).get("mde") or 9) <= 0.70 + 1e-9)
    out["recovery_alone_le70_more_often_than_among_mot"] = bool(le70("CTXB+SPK") > le70("CTXB+MOT+SPK"))
    # decoding
    d = json.load(open(R / "rev3_decode.json"))["runs"]
    out["selfz_hgb_still_identifies_campaign"] = d["SELFZ__hgb__campaign_standing"]["auc"] >= 0.99
    out["selfz_lr_identifies_campaign_less_well"] = d["SELFZ__logreg__campaign_standing"]["auc"] < d["MOT__logreg__campaign_standing"]["auc"] - 0.1
    dn = json.load(open(R / "rev3_decode_nr.json"))["runs"]
    out["without_rotation_campaign_weak"] = all(dn[f"MOTNR__{l}__campaign_standing"]["auc"] < 0.75 for l in ("logreg", "hgb"))
    out["without_rotation_game_still_recognised"] = all(dn[f"MOTNR__{l}__game_race"]["auc"] >= 0.9 for l in ("logreg", "hgb"))
    rt = json.load(open(R / "rev3_rotation.json"))["constant_rotation_sessions"]
    out["rotation_constant_in_every_standing_session_and_no_seated"] = all(
        (v["constant"] == v["sessions"]) if k.startswith("standing") else (v["constant"] == 0) for k, v in rt.items())
    rtf = json.load(open(R / "rev3_rotation.json"))
    out["standing_constant_features_are_the_rotation_features"] = (
        sorted(rtf["standing_features_constant_within_every_session"]) == sorted(rtf["rotation_features"])
        and set(rtf["standing_features_zero_everywhere"]) < set(rtf["rotation_features"]))
    # cells: 8 groups in the racing-game cells, 16 in the flight-game cells (5 test folds)
    C_ = st["configs"]
    out["cell_groups_8_racing_16_flight"] = all(
        C_[f"sev2__cell_{c}_{gm}__CTXB__ctxb"]["n_groups"] == (8 if gm == "race" else 16)
        for c in ("seated", "standing") for gm in ("race", "flight"))
    # falling-session family: 13 tests, the missing one is the seated flight-game cell for sev1
    fnm = st["null_lroff_nm"]["families"]["lroff|CTXB+MOT|session_nm"]
    out["falling_family_13_without_sev1_cSF"] = (len(fnm["tests"]) == 13
                                                 and list(fnm["not_estimable"]) == ["sev1__cell_seated_flight__CTXB+MOT__lroff"])
    pairs = json.load(open(R / "rev3_pairs.json"))
    out["pairs_almost_all_between_sessions_pooled"] = all(pairs[f"sev{l}__pooled"]["share_between_sessions"] > 0.95 for l in (1, 2))
    # labels
    q = json.load(open(R / "rev3_vrsq0.json"))
    out["allzero_in_every_cell"] = all(v["all_zero"] > 0 for v in q["all_zero_by_cell"].values())
    allc = json.load(open(R / "vrsq.json"))["correlations"]
    out["weak_all_sessions"] = all(abs(c["rho"]) < 0.4 for c in allc.values())          # moderate: rho >= 0.4
    wz_ = {k: v["rho"] for k, v in q["correlations_without_all_zero"].items()}
    up = {k for k in wz_ if wz_[k] > allc[k]["rho"]}
    out["without_zeros_rises_for_three_of_four_summaries_not_frac_ge1"] = up == {
        f"{a}~{b}" for a in ("ing_max", "ing_mean", "ing_frac_ge2") for b in ("vrsq_post_total", "vrsq_delta_total")}
    wz = {k: v["rho"] for k, v in q["correlations_without_all_zero"].items()}
    out["moderate_for_two_of_four_without_zeros"] = (
        {k.split("~")[0] for k, v in wz.items() if v >= 0.4} == {"ing_frac_ge2", "ing_max"}
        and all(wz[f"{a}~{b}"] >= 0.4 for a in ("ing_frac_ge2", "ing_max") for b in ("vrsq_post_total", "vrsq_delta_total")))
    Sq = pd.read_csv(R / "session_level.csv"); Sq = Sq[Sq.vrsq_post_total.notna()]
    az = (Sq.vrsq_post_raw_sum == 0) & (Sq.vrsq_pre_total == 0)
    med = Sq[~az].groupby(["campaign", "game"]).vrsq_post_total.median(); med_all = Sq.groupby(["campaign", "game"]).vrsq_post_total.median()
    out["vrsq_cells_line_up_without_zero_files"] = (med.idxmax() == ("seated", "Race") and med.idxmin() == ("seated", "Flight")
                                                    and med_all[("seated", "Race")] < med_all[("standing", "Race")])
    shr = Sq[~az].groupby(["campaign", "game"]).ing_frac_ge2.mean()       # share rated >= 2, same sessions
    out["vrsq_extreme_cells_match_ratings"] = shr.idxmax() == med.idxmax() and shr.idxmin() == med.idxmin()
    nr = Sq[~az].groupby(["campaign", "game"]).size()
    out["vrsq_racing_cells_same_n"] = nr[("seated", "Race")] == nr[("standing", "Race")]
    out["mean_rises_less_ge1_not_at_all"] = all(
        allc[f"ing_mean~{b}"]["rho"] < wz[f"ing_mean~{b}"] < 0.4 and wz[f"ing_frac_ge1~{b}"] <= allc[f"ing_frac_ge1~{b}"]["rho"]
        for b in ("vrsq_post_total", "vrsq_delta_total"))
    # second archive and simulation
    y = json.load(open(R / "rev3_yalcin.json"))["main"]
    out["yalcin_recordwise_higher"] = y["D1_recordwise_hgb"]["auc"] > y["D1_grouped_hgb"]["auc"]
    out["yalcin_prevalence_below_half"] = y["D2"]["prevalence_score_LOPO_pooled_auc"] < 0.5
    out["yalcin_D3_does_not_hold"] = y["D3_shifted_hgb"]["ci"][1] < y["D3_true_hgb"]["ci"][0]
    yy = json.load(open(R / "rev3_yalcin.json"))
    out["yalcin_offset_gain_ci_above_0_both"] = all(yy[sec]["offset"][g_]["ci"][0] > 0 for sec in ("main", "without_long_recordings")
                                                    for g_ in ("gain_overall", "gain_within_participant"))
    s = json.load(open(R / "rev3_sim.json"))["summary"]; m = lambda sc, k: s[sc][k]["mean"]
    out["sim_recordwise_high_even_in_N"] = m("N", "D1_recordwise_auc") > m("N", "D1_grouped_auc") + 0.1
    out["sim_prevalence_below_half_every_scenario"] = all(m(sc, "D2_prevalence_LOPO_pooled_auc") < 0.5 for sc in "NSM")
    out["sim_shift_lowers_every_scenario"] = all(m(sc, "D3_shifted_auc") < m(sc, "D1_grouped_auc") for sc in "NSM")
    out["sim_S_softens_drop"] = m("S", "D3_shifted_auc") > m("N", "D3_shifted_auc")
    out["sim_ctx_gain_only_S"] = m("S", "lroff_ctx_gain") > 0.1 and abs(m("N", "lroff_ctx_gain")) < 0.05 and abs(m("M", "lroff_ctx_gain")) < 0.05
    out["sim_M_no_within_gain"] = abs(m("M", "lroff_session_gain") - m("N", "lroff_session_gain")) < 0.02
    try:
        out["sim_step_time_baseline_saturates_within"] = all(m(sc, "base_within_session_auc") > 0.99 for sc in "NSM")
        out["sim_episodic_time_baseline_not_saturated"] = all(m(sc, "base_within_session_auc") < 0.9 for sc in ("Ne", "Me", "Me2"))
        out["sim_Me2_within_gain_positive"] = s["Me2"]["lroff_session_gain"]["q2.5"] > 0
        out["sim_Ne_within_gain_not_positive"] = m("Ne", "lroff_session_gain") < 0.01
        out["sim_Me_within_gain_weak"] = 0 < m("Me", "lroff_session_gain") < 0.05 and s["Me"]["lroff_session_gain"]["q2.5"] <= 0
        out["sim_Me2_ctx_gain_positive"] = s["Me2"]["lroff_ctx_gain"]["q2.5"] > 0
        out["sim_Ne_ctx_gain_near_0"] = abs(m("Ne", "lroff_ctx_gain")) < 0.05
    except KeyError:
        out["sim_episodic_results_present"] = False
    # label-shift control in the single cells: the learner-cell combinations above chance (> 0.55) with
    # correct labels that keep that AUC (within 0.05) under the shift lie in cells of 16 and 8 groups
    keep5 = []
    for fs, ln, tg in (("MOT", "hgb", "control_"), ("MOT+TIME", "hgb", "control_"), ("MOT+TIME", "hgbreg", "control_"),
                       ("CTXB+MOT", "lroff", "r3control_"), ("CTXB+MOT", "xgboff", "r3control_")):
        for pr in CELL_P:
            k = f"sev2__{pr}__{fs}__{ln}"
            if k in C and f"{tg}{k}" in C and C[f"sev2__{pr}__CTXB__ctxb"]["pos_groups"] >= 3:
                if C[k]["overall"]["auc"] > 0.55 and C[f"{tg}{k}"]["overall"]["auc"] > C[k]["overall"]["auc"] - 0.05:
                    keep5.append((pr, C[k]["n_groups"]))
    out["control_keep_cells_have_16_and_8_groups"] = keep5 == [("cell_standing_flight", 16), ("cell_standing_race", 8)]
    hm = [(C[f"sev2__{pr}__MOT__hgb"]["overall"]["auc"], C[f"control_sev2__{pr}__MOT__hgb"]["overall"]["auc"]) for pr in CELL_P
          if C.get(f"sev2__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) >= 3]
    out["control_cells_default_hgb_at_chance"] = all(0.4 < a < 0.6 and 0.4 < b < 0.6 for a, b in hm)
    # VR-LENS: the above-chance context-stratified AUC (severity >= 2) does not survive removing the
    # demographic answers; for severity >= 1 it does, narrowly, until player position is removed too
    s2 = json.load(open(R / "stats.json"))["configs"]; v = lambda t, l: s2[f"{t}sev{l}__within_curated__VRLENS__stack"]["ctx_ci"]
    out["vrlens_s2_lead_gone_without_demographics"] = v("sensVRLENS_", 2)[0] > 0.5 and all(v(t, 2)[0] <= 0.5 <= v(t, 2)[1] for t in ("sensVRLENSA_", "sensVRLENSB_", "sensVRLENSC_"))
    out["vrlens_s1_narrow_then_gone"] = v("sensVRLENSA_", 1)[0] > 0.5 and v("sensVRLENSB_", 1)[0] <= 0.5
    # without the demographic answers: accuracy on held-out groups rises, context-stratified AUC falls
    cs_ = json.load(open(R / "stats.json"))["configs"]
    out["vrlens_s1_ctx_auc_not_lower_without_demographics"] = bool(
        cs_["sensVRLENSA_sev1__within_curated__VRLENS__stack"]["ctx_auc"] >= cs_["sensVRLENS_sev1__within_curated__VRLENS__stack"]["ctx_auc"])
    out["shift_ctx_gain_ci_includes_zero_both_offset_learners_both_labels"] = all(
        g[f"sev{l}__{pr}__CTXB+MOT__{ln}"]["ctx"]["ci"][0] <= 0 for l in (1, 2) for pr in CROSS_P for ln in ("lroff", "xgboff"))
    out["seated_flight_s2_two_positive_groups"] = C["sev2__cell_seated_flight__CTXB__ctxb"]["pos_groups"] == 2
    out["vrlens_without_demographics_accuracy_up_ctx_auc_down"] = bool(
        ab["sensVRLENSA_"]["accuracy_4class_mean_over_repeats"] > acc_full
        and cs_["sensVRLENSA_sev2__within_curated__VRLENS__stack"]["ctx_auc"] < cs_["sensVRLENS_sev2__within_curated__VRLENS__stack"]["ctx_auc"])
    out = {k: bool(v) for k, v in out.items()}
    json.dump(out, open(R / "claims_rev3.json", "w"), indent=1)
    bad = [k for k, v in out.items() if v is not True]
    print("REV3 CLAIM CHECKS:", "all passed" if not bad else f"FAILED {bad}")
    return out


def control_values(V, st):
    """Ranges of the label-shift control inside the single cells (estimable cells only)."""
    C = st["configs"]
    ne = lambda lab, pr: C.get(f"sev{lab}__{pr}__CTXB__ctxb", {}).get("pos_groups", 0) < 3
    for lab in (1, 2):
        for fs, ln, tagr in (("MOT", "hgb", "control_"), ("MOT+TIME", "hgb", "control_"), ("MOT+TIME", "hgbreg", "control_"),
                             ("CTXB+MOT", "lroff", "r3control_"), ("CTXB+MOT", "xgboff", "r3control_")):
            t = [C[f"sev{lab}__{pr}__{fs}__{ln}"]["overall"]["auc"] for pr in CELL_P
                 if not ne(lab, pr) and f"sev{lab}__{pr}__{fs}__{ln}" in C]
            s = [C[f"{tagr}sev{lab}__{pr}__{fs}__{ln}"]["overall"]["auc"] for pr in CELL_P
                 if not ne(lab, pr) and f"{tagr}sev{lab}__{pr}__{fs}__{ln}" in C]
            b = f"r3ctlrng.s{lab}.{FS3[fs]}.{LN[ln]}"
            if t:
                V.put(b + ".tmin", a2(min(t))); V.put(b + ".tmax", a2(max(t)))
            if s:
                V.put(b + ".smin", a2(min(s))); V.put(b + ".smax", a2(max(s)))
    # learner-cell combinations above chance (> 0.55) with correct labels that keep their AUC (within 0.05)
    # under the shift, over every learner of the label-shift table (severity >= 2)
    names = {("MOT", "hgb"): "default HGB on MOT", ("MOT+TIME", "hgb"): "default HGB on MOT+TIME",
             ("MOT+TIME", "hgbreg"): "HGB-reg", ("CTXB+MOT", "lroff"): "LR+offset", ("CTXB+MOT", "xgboff"): "boosting+offset"}
    keep, above = [], 0
    for (fs, ln), nm in names.items():
        tg = "r3control_" if ln in ("lroff", "xgboff") else "control_"
        for pr in CELL_P:
            k = f"sev2__{pr}__{fs}__{ln}"
            if ne(2, pr) or k not in C or f"{tg}{k}" not in C:
                continue
            a_, b_ = C[k]["overall"]["auc"], C[f"{tg}{k}"]["overall"]["auc"]
            if a_ > 0.55:
                above += 1
                if b_ > a_ - 0.05:
                    keep.append(f"{nm} in the {SHORT[pr]} ({a2(a_)} with correct, {a2(b_)} with displaced labels)")
    # the same count for severity >= 1 (Table S24)
    a1 = k1 = 0
    for (fs, ln), nm in names.items():
        tg = "r3control_" if ln in ("lroff", "xgboff") else "control_"
        for pr in CELL_P:
            k = f"sev1__{pr}__{fs}__{ln}"
            if ne(1, pr) or k not in C or f"{tg}{k}" not in C or C[k]["overall"]["auc"] <= 0.55:
                continue
            a1 += 1; k1 += C[f"{tg}{k}"]["overall"]["auc"] > C[k]["overall"]["auc"] - 0.05
    V.put("r3ctl.s1.nabove", WORDS.get(a1, a1)); V.put("r3ctl.s1.nkeep", WORDS.get(k1, k1))
    V.put("r3ctl.nabove", WORDS.get(above, above)); V.put("r3ctl.nkeep", WORDS.get(len(keep), len(keep)))
    V.put("r3ctl.keep", " and ".join([", ".join(keep[:-1]), keep[-1]]) if len(keep) > 1 else (keep[0] if keep else "none"))


def grouping_values(V, st):
    """Ranges of the main learner's gains under the session grouping and the number of estimable
    label-protocol combinations under the merged grouping (seven primary protocols)."""
    g, C = st["gains"], st["configs"]
    for m, s in (("ctx", "c"), ("session", "w")):
        v = [g[k][m]["delta"] for lab in (1, 2) for pr in FAMILY_P
             if (k := f"sess_sev{lab}__{pr}__CTXB+MOT__lroff") in g and g[k][m]["delta"] == g[k][m]["delta"]
             and C[f"sess_sev{lab}__{pr}__CTXB__ctxb"]["pos_groups"] >= 3]
        if v:
            V.put(f"r3grng.sess.{s}.min", d2(min(v))); V.put(f"r3grng.sess.{s}.max", d2(max(v)))
            V.put(f"r3grng.sess.{s}.nciabove", WORDS.get(sum(g[f"sess_sev{lab}__{pr}__CTXB+MOT__lroff"][m]["ci"][0] > 0 for lab in (1, 2)
                                                             for pr in FAMILY_P if f"sess_sev{lab}__{pr}__CTXB+MOT__lroff" in g
                                                             and C[f"sess_sev{lab}__{pr}__CTXB__ctxb"]["pos_groups"] >= 3), 0))
        v2 = [g[k][m]["delta"] for pr in FAMILY_P
              if (k := f"sess_sev2__{pr}__CTXB+MOT__lroff") in g and g[k][m]["delta"] == g[k][m]["delta"]
              and C[f"sess_sev2__{pr}__CTXB__ctxb"]["pos_groups"] >= 3]
        if v2:
            V.put(f"r3grng.sess.s2.{s}.min", d2(min(v2))); V.put(f"r3grng.sess.s2.{s}.max", d2(max(v2)))
    est = [(lab, pr) for lab in (1, 2) for pr in FAMILY_P
           if f"mrg_sev{lab}__{pr}__CTXB__ctxb" in C and C[f"mrg_sev{lab}__{pr}__CTXB__ctxb"]["pos_groups"] >= 3
           and C[f"mrg_sev{lab}__{pr}__CTXB+MOT__lroff"]["ctx"]["auc"] == C[f"mrg_sev{lab}__{pr}__CTXB+MOT__lroff"]["ctx"]["auc"]]
    V.put("r3mrg.nest", WORDS.get(len(est), len(est)))
