"""
fa_report.py -- every number, table and figure in the paper comes from here.

Reads   data/sessions.csv, data/features_w30.csv.gz, results/*.json, and (for the
        audit of the curated release) the raw and curated archive folders.
Writes  results/paper/values.json  (one entry per reported number, by key; committed)
        results/paper/values.tex   (the same values as LaTeX macros, \\V{key})
        results/paper/*.tex        (LaTeX tables)
        figures/fig_*.pdf          (vector figures)

Usage: python fa_report.py <project dir> <RAW_DATABASE dir> <DATABASE_SELECTED_DATA dir> <curated dataset.csv>
"""
from __future__ import annotations
import json, sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------- style (validated palette, light mode)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY_L, GREY_D = "#aeada8", "#52514e"                 # baselines stay gray
ORD = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]    # ordinal ramp, steps 250/400/550/700
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False,
})
COL1, COL2 = 3.5, 7.16
CELLS = [("seated", "Flight"), ("seated", "Race"), ("standing", "Flight"), ("standing", "Race")]
CELL_KEY = {("seated", "Flight"): "seatF", ("seated", "Race"): "seatR",
            ("standing", "Flight"): "standF", ("standing", "Race"): "standR"}
POST_COL = {"seated": BLUE, "standing": ORANGE}
GAME_LS = {"Flight": "-", "Race": "--"}
GAME_MK = {"Flight": "o", "Race": "^"}

PR = {"within_curated": "wc", "within_seated": "ws", "within_standing": "wt", "within_flight": "wf",
      "within_race": "wr", "pooled": "pool", "seated_to_standing": "s2t", "standing_to_seated": "t2s",
      "flight_to_race": "f2r", "race_to_flight": "r2f", "cell_seated_flight": "cSF", "cell_seated_race": "cSR",
      "cell_standing_flight": "cTF", "cell_standing_race": "cTR"}
PR_NAME = {"within_curated": "Curated release (seated)", "within_seated": "Within seated",
           "within_standing": "Within standing", "within_flight": "Within flight game",
           "within_race": "Within racing game", "pooled": "All cells (pooled)",
           "seated_to_standing": "Seated \\ra{} standing", "standing_to_seated": "Standing \\ra{} seated",
           "flight_to_race": "Flight \\ra{} racing", "race_to_flight": "Racing \\ra{} flight",
           "cell_seated_flight": "Seated, flight game", "cell_seated_race": "Seated, racing game",
           "cell_standing_flight": "Standing, flight game", "cell_standing_race": "Standing, racing game"}
CELL_P = ["cell_seated_flight", "cell_seated_race", "cell_standing_flight", "cell_standing_race"]
WITHIN_P = ["within_curated", "within_seated", "within_standing", "within_flight", "within_race", "pooled"]
CROSS_P = ["seated_to_standing", "standing_to_seated", "flight_to_race", "race_to_flight"]
FS = {"TIME": "T", "CTX": "C", "CTX+TIME": "CT", "MOT": "M", "MOT+TIME": "MT", "MOT+CTX+TIME": "MCT", "MOTQ+TIME": "MQT",
      "SELFZ+TIME": "ZT", "MOT+FPS+TIME": "MFT", "MOT+TIME+DEMO": "MTD", "MOT+SPK+TIME": "MST",
      "SPK+TIME": "ST", "SELFZ+SPK+TIME": "ZST", "VRLENS": "VRL"}
MN = {"hgb": "hgb", "logreg": "lr", "mlp": "mlp", "hgbreg": "hgbr", "stack": "stack"}
TAG = {"": "", "sensFPS_": "fps.", "sensMLP_": "mlp.", "sensDEMO_": "demo.", "sensCLEAN_": "clean.",
       "sensSESSGROUP_": "sess.", "sensW10_": "w10.", "sensW60_": "w60.", "sensCURATED_": "cur.",
       "rowwise_": "rw.", "control_": "ctl.", "sensQUAT_": "quat.", "sensSPK60_": "spk60.", "sensSPK70_": "spk70.",
       "sensVRLENS_": "vrl.", "sensVRLENSA_": "vrlA.", "sensVRLENSB_": "vrlB.", "sensVRLENSC_": "vrlC.", "sensVRLENSG_": "vrlG."}


# ---------------------------------------------------------------- formatting
def a2(x):
    """AUC / correlation / difference, two decimals, LaTeX minus."""
    s = f"{x:.2f}"
    return s.replace("-", "$-$")


def d2(x):
    s = f"{x:+.2f}"
    return s.replace("-", "$-$")


def ci2(c):
    return f"[{a2(c[0])}, {a2(c[1])}]"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def pfmt(p):
    """Three decimals; four where three would hide which side of 0.05 the value is on."""
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.4f}" if f"{p:.3f}" == "0.050" else f"{p:.3f}"


def n_(x):
    return f"{int(x):,}"


class Values:
    def __init__(self):
        self.v = {}

    def put(self, key, val):
        val = str(val)
        assert "\\" not in key and "{" not in key and "#" not in key and "%" not in key, key
        assert key not in self.v or self.v[key] == val, (key, self.v.get(key), val)
        self.v[key] = val

    def write(self, path):
        lines = ["% generated by code/fa_report.py -- do not edit by hand", "\\makeatletter"]
        for k, v in sorted(self.v.items()):
            lines.append(f"\\expandafter\\def\\csname fa@{k}\\endcsname{{{v}}}")
        lines += ["\\newcommand{\\V}[1]{\\ifcsname fa@#1\\endcsname\\csname fa@#1\\endcsname"
                  "\\else\\errmessage{fa_report: no value for key #1}\\fi}", "\\makeatother"]
        Path(path).write_text("\n".join(lines) + "\n")
        Path(path).with_suffix(".json").write_text(json.dumps(self.v, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- audit
def audit(V, F, S, RAW, SEL, CSV):
    from fa_data import curated_sessions
    V.put("aud.nsess", len(S)); V.put("aud.nrec", n_(len(F))); V.put("aud.ngroups", S.group_id.nunique())
    V.put("aud.nsess.seated", (S.campaign == "seated").sum()); V.put("aud.nsess.standing", (S.campaign == "standing").sum())
    V.put("aud.nvrsq", S.vrsq_post_total.notna().sum())
    gs = S.groupby("group_id")
    V.put("aud.groups.bothpost", int((gs.campaign.nunique() > 1).sum()))
    V.put("aud.groups.bothgame", int((gs.game.nunique() > 1).sum()))
    big = gs.size().idxmax(); V.put("aud.biggroup", int(gs.size().max()))
    V.put("aud.biggroup.dates", S[S.group_id == big].date.nunique())
    for f in ("gap", "config", "short"):
        V.put(f"aud.flag.{f}", int(S[f"flag_{f}"].sum()))
    V.put("aud.flag.any", int((S.flag_gap | S.flag_config | S.flag_short).sum()))
    dt = F.groupby("session_id").t_elapsed.diff().dropna()
    V.put("aud.dt", f"{dt.median():.2f}")
    V.put("aud.prev1", pct((F.severity >= 1).mean())); V.put("aud.prev2", pct((F.severity >= 2).mean()))
    ch = F.groupby("session_id").severity.apply(lambda x: np.diff(x.to_numpy()))
    V.put("aud.rchg.med", f"{np.median([np.count_nonzero(c) for c in ch]):.0f}")
    V.put("aud.rchg.max", int(max(np.count_nonzero(c) for c in ch)))
    V.put("aud.rchg.none", int(sum(np.count_nonzero(c) == 0 for c in ch)))
    V.put("aud.rdec.nsess", int(sum((c < 0).any() for c in ch)))
    V.put("aud.rdec.share", pct(sum((c < 0).sum() for c in ch) / sum((c != 0).sum() for c in ch), 0))
    V.put("aud.rjump.n", int(sum((np.abs(c) > 1).sum() for c in ch)))
    early = F[F.t_elapsed < 60]; late = F[F.t_elapsed >= 120]
    V.put("aud.prev2.first", pct((early.severity >= 2).mean())); V.put("aud.prev2.late", pct((late.severity >= 2).mean()))
    rows = []
    for post, game in CELLS:
        k = CELL_KEY[(post, game)]; d = F[(F.campaign == post) & (F.game == game)]
        s = S[(S.campaign == post) & (S.game == game)]
        vals = dict(nsess=len(s), ngroups=s.group_id.nunique(), nrec=n_(len(d)),
                    dur=f"{s.duration_s.median():.0f}", fps=f"{d.GameFps.mean():.0f}",
                    sev1=pct((d.severity >= 1).mean()), sev2=pct((d.severity >= 2).mean()),
                    vrsqmed=f"{s.vrsq_post_total.median():.1f}", vrsqn=int(s.vrsq_post_total.notna().sum()),
                    cur=int(s.curated_release.sum()))
        for kk, vv in vals.items():
            V.put(f"aud.{k}.{kk}", vv)
        rows.append((post, game, vals))
    fps = {k: F[(F.campaign == p) & (F.game == g)].GameFps.mean() for (p, g), k in CELL_KEY.items()}
    others = [v for k, v in fps.items() if k != "standR"]
    V.put("aud.fps.others", f"{min(others):.0f}--{max(others):.0f}")
    V.put("aud.standR.fpsmean", f"{fps['standR']:.0f}")
    # curated release
    cur_ids = set(S[S.curated_release].session_id)
    V.put("aud.cur.nsess", len(cur_ids)); V.put("aud.cur.nrec", n_(F.session_id.isin(cur_ids).sum()))
    V.put("aud.cur.ngroups", S[S.curated_release].group_id.nunique())
    for camp, d in S.groupby("campaign"):
        dt = pd.to_datetime(d.date)
        lo, hi = dt.min(), dt.max()
        txt = (f"{lo.day}--{hi.day} {lo.strftime('%B')} {lo.year}" if (lo.month, lo.year) == (hi.month, hi.year)
               else f"{lo.day} {lo.strftime('%B')} -- {hi.day} {hi.strftime('%B')} {hi.year}")
        V.put(f"aud.dates.{camp}", txt)
    notcur = S[(S.campaign == "seated") & ~S.curated_release]
    V.put("aud.gap.max", f"{notcur.max_gap_s.max():.0f}")
    m = curated_sessions(SEL, RAW)

    def rec(p):
        return [{c.tag: (c.text or "").strip().replace(",", ".") for c in e} for e in ET.parse(p).getroot().findall("Data")]
    ident = sum(rec(Path(SEL) / str(c) / "FILE.xml") == rec(Path(RAW) / s / "FILE.xml") for c, s in m.items())
    V.put("aud.cur.identical", ident)
    D = pd.read_csv(CSV)
    V.put("aud.cur.labels", "/".join(n_(x) for x in D.DiscomfortLevel.value_counts().sort_index()))
    assert D.DiscomfortLevel.value_counts().sort_index().tolist() == \
        F[F.session_id.isin(cur_ids)].severity.value_counts().sort_index().tolist()
    y2 = (D.DiscomfortLevel >= 2).astype(int)
    V.put("cur.raceshare", pct(y2[D.StaticFrame == 1].sum() / y2.sum()))
    V.put("cur.prev2", pct(y2.mean()))
    for c in ("StaticFrame", "TimeStamp", "CameraFieldOfView", "PlayerSpeed", "GameFps"):
        a = roc_auc_score(y2, D[c]); V.put(f"cur.col.{c}", a2(max(a, 1 - a)))
    V.put("cur.col.StaticFrame.raw", f"{roc_auc_score(y2, D.StaticFrame):.3f}")
    # table: archive
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\caption{The raw UFFCSData archive by game--campaign cell. Seated and standing are the two recording campaigns, which differ in more than posture (Section~\ref{sec:why}). Groups are inferred participant groups; a group can appear in more than one cell. Rates are shares of records; VRSQ is the post-exposure total (0--100). The curated release contains the seated sessions marked in the last column.}",
         r"\label{tab:archive}", r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
         r"Campaign & Game & Sessions & Groups & Records & Length (s)$^{a}$ & FPS$^{b}$ & \sevone{} (\%) & \sevtwo{} (\%) & VRSQ post$^{c}$ & Curated \\", r"\midrule"]
    for post, game, v in rows:
        L.append(f"{post.capitalize()} & {'Racing' if game == 'Race' else 'Flight'} & {v['nsess']} & {v['ngroups']} & {v['nrec']} & {v['dur']} & {v['fps']} & {v['sev1']} & {v['sev2']} & {v['vrsqmed']} ({v['vrsqn']}) & {v['cur']} \\\\")
    L += [r"\midrule", f"All & & {len(S)} & {S.group_id.nunique()} & {n_(len(F))} & {S.duration_s.median():.0f} & {F.GameFps.mean():.0f} & {pct((F.severity >= 1).mean())} & {pct((F.severity >= 2).mean())} & {S.vrsq_post_total.median():.1f} ({S.vrsq_post_total.notna().sum()}) & {len(cur_ids)} \\\\",
          r"\bottomrule", r"\multicolumn{11}{l}{$^{a}$Median session length. $^{b}$Mean rendering frame rate. $^{c}$Median post-exposure VRSQ total (0--100), with the number of sessions that have a VRSQ file.}",
          r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_fields(raw_fields):
    """Session-level condition fields (computed in main from the raw archive)."""
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\caption{Session-level condition fields of the raw archive. None varies within a session; a game alias takes one value per game.}",
         r"\label{tab:fields}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.9cm}>{\raggedright\arraybackslash}p{3.2cm}>{\raggedright\arraybackslash}p{2.1cm}@{}}", r"\toprule",
         r"Field & Values (flight / racing game) & Carries \\", r"\midrule"]
    for f, (vals, carries) in raw_fields.items():
        L.append(f"{{\\scriptsize\\texttt{{{f}}}}} & {vals} & {carries} \\\\")
    L += [r"\bottomrule", r"\multicolumn{3}{@{}l}{$^{a}$One standing flight-game session: 60.}", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ---------------------------------------------------------------- results values
def config_values(V, R):
    runs = []
    for f in ("main.json", "mitigation.json", "context.json", "cell.json", "sensitivity.json", "control.json", "posctl.json",
              "hgbreg.json", "posctl_reg.json"):
        if not (R / f).exists() and f in ("hgbreg.json", "posctl_reg.json"):
            continue                                   # revision-2 runs, present once run
        d = json.load(open(R / f))
        runs += d["runs"] if isinstance(d, dict) else d
    ops = {r["key"]: r for r in runs}
    st = json.load(open(R / "stats.json"))
    for key, c in st["configs"].items():
        tag = next(t for t in sorted(TAG, key=len, reverse=True) if key.startswith(t))
        lab, pr, fs, mn = key[len(tag):].split("__")
        k = f"{TAG[tag]}{lab.replace('sev', 's')}.{PR[pr]}.{FS[fs]}.{MN[mn]}"
        if tag == "rowwise_":        # near 1: three decimals, so that 0.999 is not printed as 1.00
            V.put(k + ".auc", f"{c['auc']:.3f}"); V.put(k + ".ci", f"[{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]")
        else:
            V.put(k + ".auc", a2(c["auc"])); V.put(k + ".ci", ci2(c["ci"]))
        V.put(k + ".prev", pct(c["prevalence"])); V.put(k + ".npos", n_(round(c["prevalence"] * c["n_rows"])))
        V.put(k + ".ngroups", c["n_groups"])
        if "ctx_auc" in c:
            if tag == "rowwise_":
                V.put(k + ".cauc", f"{c['ctx_auc']:.3f}"); V.put(k + ".cci", f"[{c['ctx_ci'][0]:.3f}, {c['ctx_ci'][1]:.3f}]")
            else:
                V.put(k + ".cauc", a2(c["ctx_auc"])); V.put(k + ".cci", ci2(c["ctx_ci"]))
        o = ops.get(key)
        if o is not None:
            if o["roc_auc"] == o["roc_auc"]:                    # nan when a repeat had no estimable fold
                assert abs(o["roc_auc"] - c["auc"]) < 1e-9, key  # same predictions, two engines
            if "op_fpr" in o:
                V.put(k + ".fpr", pct(o["op_fpr"])); V.put(k + ".rec", pct(o["op_recall"]))
                V.put(k + ".prec", pct(o["op_precision"])); V.put(k + ".mcc", a2(o["op_mcc"]))
            if o.get("per_group_auc_median") == o.get("per_group_auc_median"):
                V.put(k + ".gmed", a2(o["per_group_auc_median"]))
                V.put(k + ".giqr", ci2(o["per_group_auc_iqr"])); V.put(k + ".gn", o["n_groups_both_classes"])
            V.put(k + ".prauc", a2(o["pr_auc"]))
    return st, ops


RULE = {}   # outcome of the pre-specified reporting rule, filled by extra_values, checked by claims

TEST_CODE = {"telemetry_gain_over_time": "gainT", "telemetry_gain_over_context": "gainC",
             "context_gain_over_time": "ctxT", "telemetry_vs_context": "telVsCtx", "selfnorm_vs_raw": "selfz",
             "selfnorm_gain_over_time": "selfzT"}


def test_key(name):
    lab, mn, rest = name.split("_", 2)
    base, where = rest.split("__")
    if base.startswith("shift_cost_"):
        code = "shift" + FS[base[len("shift_cost_"):]]
    elif base.startswith("pooled_vs_cross_"):
        code = "poolX" + FS[base[len("pooled_vs_cross_"):]]
    else:
        code = TEST_CODE[base]
    where = PR.get(where, where)
    return f"t.{lab.replace('sev', 's')}.{MN[mn]}.{code}.{where}"


def test_values(V, st):
    for name, t in st["tests"].items():
        k = test_key(name)
        for sn, pre in (("pooled", ""), ("context", "c")):
            r = t[sn]
            V.put(f"{k}.{pre}d", d2(r["delta"])); V.put(f"{k}.{pre}ci", ci2(r["ci"])); V.put(f"{k}.{pre}p", pfmt(r["p_boot"]))
            if "p_holm" in r:
                V.put(f"{k}.holm", pfmt(r["p_holm"]))
    for mn, fam in st.get("families", {}).items():
        V.put(f"fam.{MN[mn]}.m", fam["m"])
        V.put(f"fam.{MN[mn]}.nsig", sum(st["tests"][n]["pooled"]["p_holm"] < 0.05 for n in fam["tests"]))
    V.put("fam.m", st["primary_family"]["m"])
    V.put("fam.nsig", sum(st["tests"][n]["pooled"]["p_holm"] < 0.05 for n in st["primary_family"]["tests"]))
    V.put("fam.lr.nsig", sum(st["tests"][n]["pooled"]["p_holm"] < 0.05 for n in st["secondary_family_logreg"]["tests"]))


# ---------------------------------------------------------------- tables
def cell(V, k, what="auc"):
    return f"{V.v[k + '.auc']} {V.v[k + '.ci']}" if what == "auc" else V.v[k + "." + what]


def table_protocols(F):
    from fa_run import domains
    F = F.assign(curated=F.curated.astype(bool)); D = domains(F)
    # research questions each protocol serves (Section I): RQ1 the audit (context baseline, exposure time),
    # RQ2 evaluation design (curated release, change of setting); every protocol enters Table gain (RQ3)
    desc = {"within_curated": ("Curated release", "Curated release", "2, 3"),
            "within_seated": ("Seated", "Seated", "1, 3"), "within_standing": ("Standing", "Standing", "1, 3"),
            "within_flight": ("Flight game", "Flight game", "1, 3"), "within_race": ("Racing game", "Racing game", "1, 3"),
            "pooled": ("All cells", "All cells", "1, 3"),
            "seated_to_standing": ("Seated", "Standing", "2, 3"), "standing_to_seated": ("Standing", "Seated", "2, 3"),
            "flight_to_race": ("Flight game", "Racing game", "2, 3"), "race_to_flight": ("Racing game", "Flight game", "2, 3"),
            **{c: (PR_NAME[c], PR_NAME[c], "3") for c in CELL_P}}
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{Evaluation protocols. The inferred participant groups of the target are split into five folds (five repeats); for each fold, training uses the source records of all groups outside the fold. Seated and standing denote the two recording campaigns.}",
         r"\label{tab:protocols}", r"\setlength{\tabcolsep}{1.3pt}", r"\begin{tabular}{@{}llrr@{\hspace{6pt}}l@{}}", r"\toprule",
         r"Train on (source) & Test on (target) & Groups$^{a}$ & Records$^{b}$ & RQ$^{c}$ \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\emph{Same setting}} \\"]
    g = F.group_id.to_numpy()
    for pr in WITHIN_P + CROSS_P + CELL_P:
        src, tgt = D[pr]; s, t, rq = desc[pr]
        if pr == "seated_to_standing":
            L.append(r"\multicolumn{5}{@{}l}{\emph{Shifted setting}} \\")
        if pr == CELL_P[0]:
            L.append(r"\multicolumn{5}{@{}l}{\emph{One game--campaign cell}} \\")
        L.append(f"{s} & {t} & {len(np.unique(g[tgt]))} & {n_(tgt.sum())} & {rq} \\\\")
    L += [r"\bottomrule", r"\multicolumn{5}{@{}l}{$^{a}$Inferred groups with target records. $^{b}$Target records.}", r"\\", r"\multicolumn{5}{@{}l}{$^{c}$Research questions (Section~\ref{sec:intro}) the protocol serves.}",
          r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_main(V, lab="s2", mn="hgb", label="tab:main", caption=None, star=True):
    env = "table*" if star else "table"
    cap = caption or (r"Group-disjoint results for \sevtwo{} (HGB). Overall ROC-AUC for the three context baselines and, with 95\,\% group-bootstrap intervals, "
                      r"for the two telemetry models; context-stratified AUC of MOT+TIME (0.5 = no knowledge beyond game, campaign and 30-s exposure bin); and recall with the "
                      r"false-positive rate reached on the target, at a threshold set for a 10\,\% false-alarm budget on training groups. "
                      r"Group-disjoint: no inferred participant group is on both sides of a split. Seated and standing denote the two recording campaigns, which differ in more than posture.")
    L = [f"\\begin{{{env}}}[t]", r"\centering", r"\footnotesize", f"\\caption{{{cap}}}", f"\\label{{{label}}}",
         r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{lrccccccc}", r"\toprule",
         r" & & \multicolumn{5}{c}{Overall ROC-AUC} & Context-strat. AUC & Recall / FPR (\%) \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-8}\cmidrule(lr){9-9}",
         r"Protocol & Prev. (\%) & TIME & CTX & CTX+TIME & MOT+TIME & MOT+CTX+TIME & MOT+TIME & MOT+TIME \\", r"\midrule",
         r"\multicolumn{9}{l}{\emph{Held-out groups, same setting}} \\"]
    for pr in WITHIN_P + CROSS_P:
        if pr == "seated_to_standing":
            L.append(r"\multicolumn{9}{l}{\emph{Held-out groups and a new setting (campaign or game)}} \\")
        b = f"{lab}.{PR[pr]}"
        ctx = V.v.get(f"{b}.C.{mn}.auc", "--")
        rec = f"{V.v.get(f'{b}.MT.{mn}.rec', '--')} / {V.v.get(f'{b}.MT.{mn}.fpr', '--')}"
        L.append(f"{PR_NAME[pr]} & {V.v[f'{b}.MT.{mn}.prev']} & {V.v[f'{b}.T.{mn}.auc']} & {ctx} & {V.v[f'{b}.CT.{mn}.auc']} & "
                 f"{cell(V, f'{b}.MT.{mn}')} & {cell(V, f'{b}.MCT.{mn}')} & {V.v[f'{b}.MT.{mn}.cauc']} {V.v[f'{b}.MT.{mn}.cci']} & {rec} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(L)


def table_tests(V, st):
    """Exploratory comparison family (13 comparisons, severity >= 2) for each learner; Holm within learner."""
    fams = st.get("families", {})
    main = "hgbreg" if "hgbreg" in fams else "hgb"
    others = [m for m in ("logreg", "hgb") if m != main and m in fams] or ["logreg"]
    nm = {"hgbreg": "HGB-reg", "hgb": "default HGB", "logreg": "LR"}
    ncol = 4 + 2 * len(others)
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Exploratory comparison family (\sevtwo{}): differences in overall ROC-AUC on identical target records, with group-bootstrap 95\,\% CI, "
         r"two-sided bootstrap $p$ and Holm-adjusted $p$ (within each learner). The family was fixed after a first run had been seen (Supplementary Table~S4); "
         r"the Holm adjustment is a descriptive guard, not confirmatory testing; $p$ has four decimals where three would round to 0.050. " + nm[main] + r" was the main learner of the earlier analysis.}",
         r"\label{tab:tests}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l" + "c" * (ncol - 1) + r"@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{" + nm[main] + "}" + "".join(r" & \multicolumn{2}{c}{" + nm[o] + "}" for o in others) + r" \\ "
         + r"\cmidrule(lr){2-4}" + "".join(rf"\cmidrule(lr){{{5 + 2 * i}-{6 + 2 * i}}}" for i in range(len(others))),
         r"Comparison & $\Delta$AUC [95\,\% CI] & $p$ & $p_\mathrm{Holm}$" + r" & $\Delta$AUC & $p_\mathrm{Holm}$" * len(others) + r" \\", r"\midrule",
         r"\multicolumn{" + str(ncol) + r"}{l}{\emph{Shift cost: same setting minus cross setting (MOT+TIME)}} \\"]
    tn = {"standing": "on standing (seated \\ra{} standing)", "seated": "on seated (standing \\ra{} seated)",
          "race": "on racing (flight \\ra{} racing)", "flight": "on flight (racing \\ra{} flight)"}

    def row(name, n):
        k = test_key(n.replace("_hgb_", f"_{main}_"))
        out = f"{name} & {V.v[k + '.d']} {V.v[k + '.ci']} & {V.v[k + '.p']} & {V.v[k + '.holm']}"
        for o in others:
            ko = test_key(n.replace("_hgb_", f"_{o}_"))
            out += f" & {V.v[ko + '.d']} & {V.v[ko + '.holm']}"
        return out + r" \\"
    for t in ("standing", "seated", "race", "flight"):
        L.append(row(tn[t], f"sev2_hgb_shift_cost_MOT+TIME__{t}"))
    L.append(r"\multicolumn{" + str(ncol) + r"}{l}{\emph{Telemetry gain over context: MOT+CTX+TIME minus CTX+TIME}} \\")
    for pr in ["within_seated", "within_standing", "within_flight", "within_race", "pooled"] + CROSS_P:
        L.append(row(PR_NAME[pr], f"sev2_hgb_telemetry_gain_over_context__{pr}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_curated(V, mn="hgb"):
    """Curated release under different evaluations; with the re-implemented VR-LENS detector (Stage 4 of
    revision 2) when its results exist. Group-disjoint rows use the main learner."""
    vr = "rw.s2.wc.VRL.stack.auc" in V.v
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{Results for the curated 35-session release under different evaluation protocols. Single-column results are reported as released, using the better of the two directions. "
         r"Record-wise splits allow records of the same participants and sessions to appear in both training and test sets; their point estimates are given to three decimals, with intervals in Supplementary Table~S1. "
         r"They may reflect label discrimination, identification of sessions seen in training and recognition of context, and are not evidence of generalization to unseen participants. Group-disjoint results: grouped 5-fold cross-validation, 5 repeats. Ctx-strat.: context-stratified AUC; ``--'': undefined for the timestamp, which is itself a stratification variable"
         + (r"; VR-LENS is the re-implemented published detector (Section~\ref{sec:vrlens})" if vr else "") + ".}",
         r"\label{tab:curated}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{\sevtwo{}} & \sevone{} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-4}",
         r"Evaluation & AUC [95\,\% CI] & Ctx-strat. & AUC [95\,\% CI] \\", r"\midrule",
         r"\multicolumn{4}{@{}l}{\emph{No model: one released column}} \\",
         f"\\texttt{{StaticFrame}} (game alias) & {V.v['cur.col.StaticFrame']} & 0.50 & {V.v['cur1.col.StaticFrame']} \\\\",
         f"\\texttt{{TimeStamp}} & {V.v['cur.col.TimeStamp']} & -- & {V.v['cur1.col.TimeStamp']} \\\\",
         r"\multicolumn{4}{@{}l}{\emph{Record-wise stratified 5-fold, default HGB}} \\"]
    for fs, nm in (("MT", "MOT+TIME"), ("CT", "CTX+TIME")):
        k2, k1 = f"rw.s2.wc.{fs}.hgb", f"rw.s1.wc.{fs}.hgb"
        L.append(f"{nm} & {V.v[k2 + '.auc']} & {V.v[k2 + '.cauc']} & {V.v[k1 + '.auc']} \\\\")
    if vr:
        L += [r"\multicolumn{4}{@{}l}{\emph{Record-wise 10-fold (the original protocol of VR-LENS)}} \\",
              f"VR-LENS & {V.v['rw.s2.wc.VRL.stack.auc']} & {V.v['rw.s2.wc.VRL.stack.cauc']} & {V.v['rw.s1.wc.VRL.stack.auc']} \\\\"]
    L.append(r"\multicolumn{4}{@{}l}{\emph{Group-disjoint, " + {"hgbr": "HGB-reg", "hgb": "default HGB"}[mn] + (r" and VR-LENS" if vr else "") + r"}} \\")
    for fs, nm in (("T", "TIME"), ("CT", "CTX+TIME"), ("MT", "MOT+TIME"), ("MCT", "MOT+CTX+TIME")):
        k2, k1 = f"s2.wc.{fs}.{mn}", f"s1.wc.{fs}.{mn}"
        L.append(f"{nm} & {cell(V, k2)} & {V.v[k2 + '.cauc']} & {cell(V, k1)} \\\\")
    if vr:
        L.append(f"VR-LENS & {cell(V, 'vrl.s2.wc.VRL.stack')} & {V.v['vrl.s2.wc.VRL.stack.cauc']} & {cell(V, 'vrl.s1.wc.VRL.stack')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_vrlens(V, R, mn):
    """Stage 4 (revision 2): the re-implemented VR-LENS detector (reduced model) on the curated
    release, under its original record-wise 10-fold split and under the study's group-disjoint
    protocol, next to the context baseline on the same records. Written only if results/vrlens.json
    exists (fa_vrlens.py needs xgboost)."""
    vr = json.load(open(R / "vrlens.json"))
    rw, gr = vr["recordwise_10fold"], vr["grouped_5x5"]
    V.put("vrl.rw.acc", pct(rw["accuracy_4class"], 0)); V.put("vrl.rw.f1", a2(rw["macro_f1_4class"]))
    V.put("vrl.grp.acc", pct(gr["accuracy_4class_mean_over_repeats"], 0))
    V.put("vrl.nfeat", len(vr["features"])); V.put("vrl.nrec", n_(vr["n_records"]))
    V.put("vrl.v.sklearn", vr["versions"]["sklearn"]); V.put("vrl.v.xgboost", vr["versions"]["xgboost"])
    V.put("vrl.ndemo", sum(f.startswith("User") for f in vr["features"]))
    att1 = R / "vrlens_attempt1.json"      # first run, with unshuffled internal folds (kept for transparency)
    if att1.exists():
        V.put("vrl.att1.acc", pct(json.load(open(att1))["recordwise_10fold"]["accuracy_4class"], 0))
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{A published curated-release detector under three evaluations. VR-LENS (reduced model) \cite{kundu2023vrlens}, re-implemented from its paper "
         r"(not the authors' code; assumptions in Section~\ref{sec:vrlens}): " + V.v["vrl.nfeat"] + r" released columns, among them the game alias \texttt{StaticFrame} and " + V.v["vrl.ndemo"] + r" demographic answers, "
         r"and a stacked ensemble trained on the four rating classes. Acc: four-class accuracy. Record-wise rows: point estimates with three decimals (intervals in Supplementary Table~S1). "
         r"The context baseline uses the same records.}",
         r"\label{tab:vrlens}", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{\sevtwo{}} & \sevone{} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-5}",
         r"Evaluation & Acc (\%) & AUC [95\,\% CI] & Ctx-strat. & AUC \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\emph{(i) Original protocol: record-wise 10-fold}} \\",
         f"VR-LENS & {V.v['vrl.rw.acc']} & {V.v['rw.s2.wc.VRL.stack.auc']} & {V.v['rw.s2.wc.VRL.stack.cauc']} & {V.v['rw.s1.wc.VRL.stack.auc']} \\\\",
         f"CTX+TIME (record-wise 5-fold) & -- & {V.v['rw.s2.wc.CT.hgb.auc']} & {V.v['rw.s2.wc.CT.hgb.cauc']} & {V.v['rw.s1.wc.CT.hgb.auc']} \\\\",
         r"\multicolumn{5}{@{}l}{\emph{(ii)--(iv) Group-disjoint: grouped 5-fold, 5 repeats}} \\",
         f"VR-LENS & {V.v['vrl.grp.acc']} & {cell(V, 'vrl.s2.wc.VRL.stack')} & {V.v['vrl.s2.wc.VRL.stack.cauc']} {V.v['vrl.s2.wc.VRL.stack.cci']} & {V.v['vrl.s1.wc.VRL.stack.auc']} \\\\",
         f"CTX+TIME ({V.v['main.learner']}) & -- & {cell(V, f's2.wc.CT.{mn}')} & {V.v[f's2.wc.CT.{mn}.cauc']} {V.v[f's2.wc.CT.{mn}.cci']} & {V.v[f's1.wc.CT.{mn}.auc']} \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_mitig(V, lab="s2"):
    tn = {"standing": ("wt", "s2t", "Standing"), "seated": ("ws", "t2s", "Seated"),
          "race": ("wr", "f2r", "Racing game"), "flight": ("wf", "r2f", "Flight game")}
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{Remedies under shift (\sevtwo{}, HGB, overall ROC-AUC on the target records). Same setting: trained on the target setting (other groups). Cross: trained on the other setting only. All cells: the pooled protocol, trained on every cell (other groups). MT = MOT+TIME, Z = SELFZ+TIME.}",
         r"\label{tab:mitig}", r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Same setting} & \multicolumn{3}{c}{Cross} & All cells \\ \cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-7}",
         r"Target & MT & Z & TIME & MT & Z & MT \\", r"\midrule"]
    for t, (w, x, nm) in tn.items():
        L.append(f"{nm} & {V.v[f'{lab}.{w}.MT.hgb.auc']} & {V.v[f'{lab}.{w}.ZT.hgb.auc']} & {V.v[f'{lab}.{x}.T.hgb.auc']} & "
                 f"{V.v[f'{lab}.{x}.MT.hgb.auc']} & {V.v[f'{lab}.{x}.ZT.hgb.auc']} & {V.v[f'{lab}.poolon.{t}.MT.hgb.auc']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


VRSQ_ROWS = [("ing_max", "In-game maximum rating"), ("ing_mean", "In-game mean rating"),
             ("ing_frac_ge1", "Share of records \\sevone{}"), ("ing_frac_ge2", "Share of records \\sevtwo{}")]


def vrsq_values(V, R):
    q = json.load(open(R / "vrsq.json"))
    for a, _ in VRSQ_ROWS:
        for b, bk in (("vrsq_post_total", "post"), ("vrsq_delta_total", "delta")):
            c = q["correlations"][f"{a}~{b}"]
            V.put(f"vrsq.{a}.{bk}.rho", a2(c["rho"])); V.put(f"vrsq.{a}.{bk}.ci", ci2(c["ci"]))
    V.put("vrsq.n", q["n_sessions_with_vrsq"])
    for camp, d in q["by_campaign"].items():
        for a, c in d.items():
            V.put(f"vrsq.{camp}.{a}.rho", a2(c["rho"])); V.put(f"vrsq.{camp}.{a}.ci", ci2(c["ci"])); V.put(f"vrsq.{camp}.n", c["n"])
    V.put("vrsq.disc.ge2post0", q["discordant"]["ingame_ge2_but_vrsq_post_0"])
    V.put("vrsq.disc.in0postHigh", q["discordant"]["ingame_0_but_vrsq_post_ge_median_pos"])
    ct = q["crosstab"]
    for lv, z, p in zip(ct["rows_ingame_max"], ct["vrsq_post_zero"], ct["vrsq_post_positive"]):
        V.put(f"vrsq.ct.max{int(lv)}.zero", z); V.put(f"vrsq.ct.max{int(lv)}.pos", p)
    V.put("vrsq.ge2.n", sum(z + p for lv, z, p in zip(ct["rows_ingame_max"], ct["vrsq_post_zero"], ct["vrsq_post_positive"]) if lv >= 2))
    V.put("vrsq.post0.n", sum(ct["vrsq_post_zero"]))
    for camp, d in q["onset"].items():
        V.put(f"onset.{camp}.ge1", f"{d['ge1_median_s']:.0f}"); V.put(f"onset.{camp}.ge1n", d["ge1_n"])
        V.put(f"onset.{camp}.ge2", f"{d['ge2_median_s']:.0f}"); V.put(f"onset.{camp}.ge2n", d["ge2_n"])
        V.put(f"onset.{camp}.n", d["n_sessions"])
    for camp, d in q["vrsq_summary"].items():
        V.put(f"vrsq.{camp}.premean", f"{d['pre_mean']:.1f}"); V.put(f"vrsq.{camp}.postmean", f"{d['post_mean']:.1f}")
        V.put(f"vrsq.{camp}.postmed", f"{d['post_median']:.1f}"); V.put(f"vrsq.{camp}.post0", pct(d["post_zero_frac"], 0))
    return q


def table_vrsq(V):
    """Spearman correlations of the rating summaries with the VRSQ, for all files and, as a
    sensitivity analysis, without the files that are zero in every item (values of fa_report_rev3)."""
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         f"\\caption{{Spearman correlation of session-level summaries of the in-game ratings with the VRSQ, with participant-group bootstrap 95\\,\\% CI, for all {V.v['vrsq.n']} sessions with a questionnaire file and, as a sensitivity analysis, without the {V.v['r3v.allzero']} files that are zero in every item before and after play ({V.v['r3v.nw']} sessions). The origin of the all-zero files is not documented, so neither treatment measures agreement reliably.}}",
         r"\label{tab:vrsq}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccc}", r"\toprule",
         f" & \\multicolumn{{2}}{{c}}{{All files ($n={V.v['vrsq.n']}$)}} & \\multicolumn{{2}}{{c}}{{Without all-zero files ($n={V.v['r3v.nw']}$)}} \\\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Session summary & VRSQ post & VRSQ post $-$ pre & VRSQ post & VRSQ post $-$ pre \\", r"\midrule"]
    for a, nm in VRSQ_ROWS:
        L.append(f"{nm} & {V.v[f'vrsq.{a}.post.rho']} {V.v[f'vrsq.{a}.post.ci']} & {V.v[f'vrsq.{a}.delta.rho']} {V.v[f'vrsq.{a}.delta.ci']}"
                 f" & {V.v[f'r3v.{a}.post.rho']} {V.v[f'r3v.{a}.post.ci']} & {V.v[f'r3v.{a}.delta.rho']} {V.v[f'r3v.{a}.delta.ci']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


SENS_ROWS = [("", "MT", "hgb", "Reference (default HGB)"), ("fps.", "MFT", "hgb", "+ frame rate"),
             ("mlp.", "MT", "mlp", "MLP instead of HGB"), ("demo.", "MTD", "hgb", "+ demographic answers"),
             ("clean.", "MT", "hgb", "Flagged sessions removed"), ("sess.", "MT", "hgb", "Session as split unit"),
             ("w10.", "MT", "hgb", "10-s windows"), ("w60.", "MT", "hgb", "60-s windows"),
             ("quat.", "MQT", "hgb", "Quaternion-aware rotation")]


def sens_ranges(V, st):
    """Range of the AUC over the sensitivity variants (main analysis excluded), per label x protocol."""
    for lab in ("s2", "s1"):
        for p in ("ws", "pool", "s2t", "t2s"):
            v = [float(V.v[f"{tag}{lab}.{p}.{fs}.{mn}.auc"].replace("$-$", "-")) for tag, fs, mn, _ in SENS_ROWS[1:]]
            V.put(f"sensrange.{lab}.{p}.min", a2(min(v))); V.put(f"sensrange.{lab}.{p}.max", a2(max(v)))
            vn = [float(V.v[f"{tag}{lab}.{p}.{fs}.{mn}.auc"].replace("$-$", "-")) for tag, fs, mn, _ in SENS_ROWS[1:] if tag != "fps."]
            V.put(f"sensrange.{lab}.{p}.maxnofps", a2(max(vn)))
            c = [float(V.v[f"{tag}{lab}.{p}.{fs}.{mn}.cauc"].replace("$-$", "-")) for tag, fs, mn, _ in SENS_ROWS[1:]]
            V.put(f"sensrange.{lab}.{p}.cmin", a2(min(c))); V.put(f"sensrange.{lab}.{p}.cmax", a2(max(c)))
    diffs = [abs(st["configs"][f"sensSESSGROUP_sev{l}__{p}__MOT+TIME__hgb"]["auc"] - st["configs"][f"sev{l}__{p}__MOT+TIME__hgb"]["auc"])
             for l in (1, 2) for p in ("within_seated", "pooled", "seated_to_standing", "standing_to_seated")]
    V.put("sess.maxdiff", a2(max(diffs)))
    # largest change of any sensitivity variant from the main analysis (overall AUC, same protocol and label)
    allv = []
    for tag, fs, mn, _ in SENS_ROWS[1:]:
        for l in ("s1", "s2"):
            for p in ("ws", "pool", "s2t", "t2s"):
                allv.append(abs(float(V.v[f"{tag}{l}.{p}.{fs}.{mn}.auc"].replace("$-$", "-")) -
                                float(V.v[f"{l}.{p}.MT.hgb.auc"].replace("$-$", "-"))))
    V.put("sens.maxdiff", a2(max(allv)))


def table_posctl(V):
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Positive control of the earlier analysis (added there post hoc, one realization). A synthetic channel with a planted within-stratum AUC of 0.60 or 0.70 is added to three feature sets and fitted with the folds of the study in three protocols (within seated, within standing, pooled). Each cell gives the context-stratified AUC recovered in the three protocols (minimum--maximum) and, in parentheses, the number of the three whose group-bootstrap interval lies above 0.5. The channel's own context-stratified AUC on the target records is \V{spk60.chanctx.min}--\V{spk60.chanctx.max} (planted 0.60) and \V{spk70.chanctx.min}--\V{spk70.chanctx.max} (planted 0.70).}",
         r"\label{tab:posctl}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{\sevtwo{}} & \multicolumn{2}{c}{\sevone{}} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Channel added to & Model & Planted 0.60 & Planted 0.70 & Planted 0.60 & Planted 0.70 \\", r"\midrule"]
    names = {"ST": "Exposure time only (TIME)", "MST": "Raw telemetry (MOT+TIME)", "ZST": "Self-normalized (SELFZ+TIME)"}
    lrn = [m for m in ("hgbr", "hgb", "lr") if f"spk70.ST.{m}.s2.cmin" in V.v]
    lab_of = {"hgbr": "HGB-reg", "hgb": "Default HGB", "lr": "LR"}
    for fs in ("ST", "MST", "ZST"):
        for mn in lrn:
            row = [names[fs] if mn == lrn[0] else "", lab_of[mn]]
            for lab in ("s2", "s1"):
                for t in ("spk60.", "spk70."):
                    k = f"{t}{fs}.{mn}.{lab}"
                    row.append(f"{V.v[k + '.cmin']}--{V.v[k + '.cmax']} ({V.v[k + '.nci']}/3)")
            L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_sens(V):
    prs = [("ws", "Within seated"), ("pool", "All cells"), ("s2t", "Seated\\ra{}stand."), ("t2s", "Stand.\\ra{}seated")]
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Sensitivity analyses: ROC-AUC [95\,\% group-bootstrap CI]. The first row is the reference: default HGB on MOT+TIME (the earlier sensitivity learner, Table~S6), 30-s windows, inferred participant groups as the unit of splitting; each other row changes one element. Splitting by session is less strict than splitting by inferred participant group.}",
         r"\label{tab:sens}", r"\setlength{\tabcolsep}{2.6pt}", r"\begin{tabular}{@{}l" + "c" * 8 + "@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{\sevtwo{}} & \multicolumn{4}{c}{\sevone{}} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         "Variant & " + " & ".join(n for _, n in prs) * 1 + " & " + " & ".join(n for _, n in prs) + r" \\", r"\midrule"]
    for tag, fs, mn, nm in SENS_ROWS:
        cells = []
        for lab in ("s2", "s1"):
            for p, _ in prs:
                k = f"{tag}{lab}.{p}.{fs}.{mn}"
                cells.append(f"\\makecell{{{V.v.get(k + '.auc', 'MISSING')}\\\\[-1pt]{{\\scriptsize {V.v.get(k + '.ci', 'MISSING')}}}}}")
        L.append(nm + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


NONEST = set()   # (label, cell protocol) pairs with fewer than three groups having a positive record


def table_cells(V, F):
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Training and testing inside one game--campaign cell (disjoint in inferred participant groups). Game and campaign are constant within a cell, so the context-stratified AUC compares records of the same 30-s exposure bin only (0.5 = no knowledge beyond the 30-s exposure bin). n/e: not estimable, because fewer than three groups have any positive record.}",
         r"\label{tab:cells}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r" & & & \multicolumn{3}{c}{Overall AUC, LR / HGB} & \multicolumn{2}{c}{Time-stratified AUC, HGB [95\,\% CI]} \\ \cmidrule(lr){4-6}\cmidrule(lr){7-8}",
         r"Cell & Label & Groups$^{\dagger}$ & TIME & MOT+TIME & SELFZ+TIME & MOT+TIME & SELFZ+TIME \\", r"\midrule"]
    for pr in CELL_P:
        post, game = pr.split("_")[1], pr.split("_")[2].capitalize()
        d = F[(F.campaign == post) & (F.game == game)]
        for lab in ("s2", "s1"):
            thr = int(lab[1])
            gpos, gall = d[d.severity >= thr].group_id.nunique(), d.group_id.nunique()
            V.put(f"{lab}.{PR[pr]}.gpos", gpos); V.put(f"{lab}.{PR[pr]}.gall", gall)
            b = f"{lab}.{PR[pr]}"; lname = r"$\geq$" + str(thr)
            if gpos < 3:
                NONEST.add((f"sev{thr}", pr))
                L.append(f"{PR_NAME[pr]} & {lname} & {gpos}/{gall} & n/e & n/e & n/e & n/e & n/e \\\\")
                continue
            L.append(f"{PR_NAME[pr] if lab == 's2' else ''} & {lname} & {gpos}/{gall} & "
                     f"{V.v[b + '.T.lr.auc']} / {V.v[b + '.T.hgb.auc']} & {V.v[b + '.MT.lr.auc']} / {V.v[b + '.MT.hgb.auc']} & "
                     f"{V.v[b + '.ZT.lr.auc']} / {V.v[b + '.ZT.hgb.auc']} & "
                     f"{V.v[b + '.MT.hgb.cauc']} {V.v[b + '.MT.hgb.cci']} & {V.v[b + '.ZT.hgb.cauc']} {V.v[b + '.ZT.hgb.cci']} \\\\")
    L += [r"\bottomrule", r"\multicolumn{8}{@{}l}{$^{\dagger}$Groups with at least one positive record / all groups in the cell.}", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


# ---------------------------------------------------------------- figures
def fig_archive(F, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.15), gridspec_kw={"width_ratios": [1.25, 1]})
    F = F.assign(bin=np.minimum(F.t_elapsed // 30, 9).astype(int))
    for post, game in CELLS:
        d = F[(F.campaign == post) & (F.game == game)]
        s = d.groupby("bin").severity.apply(lambda x: (x >= 2).mean()); n = d.groupby("bin").size()
        s = s[n >= 30]; x = s.index * 30 + 15
        a.plot(x, s.values * 100, color=POST_COL[post], ls=GAME_LS[game], marker=GAME_MK[game], ms=3.2,
               mfc="white", mew=1.0, label=f"{post.capitalize()}, {'racing' if game == 'Race' else 'flight'}")
    a.set_xlabel("Exposure time (s), 30-s bins"); a.set_ylabel("Records with severity ≥ 2 (%)")
    a.set_xlim(0, 305); a.set_ylim(0, 100); a.set_xticks(range(0, 301, 60))
    a.set_title("(a) High-severity share over exposure", loc="left", color=INK)
    a.legend(loc="upper left", ncol=2, handlelength=2.6, columnspacing=1.0)
    labels = []
    for i, (post, game) in enumerate(CELLS):
        d = F[(F.campaign == post) & (F.game == game)]
        share = d.severity.value_counts(normalize=True).reindex(range(4), fill_value=0).values * 100
        left = 0
        for lv in range(4):
            b.barh(i, share[lv], left=left, color=ORD[lv], height=0.62, edgecolor="white", linewidth=1.0,
                   label=f"Rating {lv}" if i == 0 else None)
            if share[lv] >= 7:
                b.text(left + share[lv] / 2, i, f"{share[lv]:.0f}", ha="center", va="center", fontsize=6.3,
                       color="white" if lv >= 1 else INK)
            left += share[lv]
        labels.append(f"{post.capitalize()}, {'racing' if game == 'Race' else 'flight'}\n{len(d):,} rec., {d.session_id.nunique()} sess.")
    b.set_yticks(range(4)); b.set_yticklabels(labels, color=INK2); b.invert_yaxis()
    b.set_xlim(0, 100); b.set_xlabel("Share of records (%)"); b.grid(axis="y", visible=False)
    b.set_title("(b) Rating composition per cell", loc="left", color=INK)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4, handlelength=1.0, columnspacing=0.9)
    fig.tight_layout(w_pad=1.5); fig.savefig(out); plt.close(fig)


def fig_auc(st, out, lab="sev2", mn="hgb"):
    series = [("TIME", "TIME", GREY_L, "s"), ("CTX+TIME", "CTX+TIME", GREY_D, "D"),
              ("MOT+TIME", "MOT+TIME", BLUE, "o"), ("MOT+CTX+TIME", "MOT+CTX+TIME", ORANGE, "^")]
    prs = WITHIN_P + CROSS_P
    names = [PR_NAME[p].replace("\\ra{}", "→") for p in prs]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0), sharey=True)
    for ax, (metric, cim, title) in zip(axes, (("auc", "ci", "(a) Overall ROC-AUC"),
                                                ("ctx_auc", "ctx_ci", "(b) Context-stratified ROC-AUC"))):
        for j, (fs, nm, col, mk) in enumerate(series):
            off = (j - 1.5) * 0.17
            for i, p in enumerate(prs):
                c = st["configs"][f"{lab}__{p}__{fs}__{mn}"]
                y = i + off
                ax.plot(c[cim], [y, y], color=col, lw=1.0, solid_capstyle="round")
                ax.plot(c[metric], y, marker=mk, color=col, ms=4.2, mec="white", mew=0.6, ls="none",
                        label=nm if i == 0 else None)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.axhspan(len(WITHIN_P) - 0.5, len(prs) - 0.5, color="#f3f2ee", zorder=0, lw=0)
        ax.set_xlim(0.1, 1.0); ax.set_title(title, loc="left", color=INK); ax.set_xlabel("ROC-AUC (chance = 0.5)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(prs))); axes[0].set_yticklabels(names, color=INK2); axes[0].invert_yaxis()
    axes[1].text(0.12, len(WITHIN_P) - 0.35, "shifted target", fontsize=6.5, color=MUTED, va="top")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.16), ncol=4, handlelength=1.2, columnspacing=1.4)
    fig.tight_layout(w_pad=0.8); fig.savefig(out); plt.close(fig)


def fig_vrsq(R, out):
    S = pd.read_csv(R / "session_level.csv").dropna(subset=["vrsq_post_total"])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    S["all_zero"] = (S.vrsq_post_raw_sum == 0) & (S.vrsq_pre_total == 0)       # every item 0 before and after
    for post, mk in (("seated", "o"), ("standing", "^")):
        d = S[(S.campaign == post) & ~S.all_zero]
        x = d.ing_max + rng.uniform(-0.18, 0.18, len(d)) + (-0.08 if post == "seated" else 0.08)
        ax.scatter(x, d.vrsq_post_total, s=16, marker=mk, facecolor="white", edgecolor=POST_COL[post], lw=1.0,
                   label=post.capitalize(), zorder=3)
    z = S[S.all_zero]                                  # stacked at 0: shown as counts per rating
    for lv, n in z.groupby("ing_max").size().items():
        ax.text(lv, -9, f"{n} all-zero", ha="center", va="top", fontsize=6.5, color=MUTED)
    med = S.groupby("ing_max").vrsq_post_total.median()
    for lv, m in med.items():
        ax.plot([lv - 0.3, lv + 0.3], [m, m], color=INK, lw=1.4, zorder=4)
    ax.plot([], [], color=INK, lw=1.4, label="Median (both campaigns)")
    ax.set_xticks(range(4)); ax.set_xlabel("Highest in-game rating in the session")
    ax.set_ylabel("Post-exposure VRSQ total (0–100)"); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-20, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="x", visible=False); ax.legend(loc="upper left", ncol=3, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_null(st, R, out):
    """Context-stratified AUC: label-transplant null (gray), observed models, planted signals."""
    nl = pd.DataFrame(json.load(open(R / "null.json")))
    prs = [("within_seated", "Within seated"), ("within_standing", "Within standing"), ("pooled", "All cells")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35), sharey=True)
    rng = np.random.default_rng(0)
    for ax, lab, title in zip(axes, ("sev2", "sev1"), ("(a) Severity ≥ 2", "(b) Severity ≥ 1")):
        ticks, names = [], []
        for i, (pr, nm) in enumerate(prs):
            for j, (fs, col, mk, fsn) in enumerate((("MOT+TIME", BLUE, "o", "MOT+TIME"), ("SELFZ+TIME", ORANGE, "^", "SELFZ+TIME"))):
                y = i * 2.6 + j
                d = nl[(nl.label == lab) & (nl.protocol == pr) & (nl.features == fs)].context_stratified_auc
                ax.scatter(d, y + rng.uniform(-0.2, 0.2, len(d)), s=4, color=GREY_L, lw=0, zorder=2, alpha=0.6,
                           label="Null (label transplant)" if (i, j) == (0, 0) else None)
                c = st["configs"][f"{lab}__{pr}__{fs}__hgb"]
                ax.plot(c["ctx_ci"], [y, y], color=col, lw=1.0, zorder=3)
                ax.plot(c["ctx_auc"], y, marker=mk, color=col, ms=4.6, mec="white", mew=0.6, ls="none", zorder=4,
                        label=f"Observed, {fsn}" if i == 0 else None)
                spk_fs = "MOT+SPK+TIME" if fs == "MOT+TIME" else "SELFZ+SPK+TIME"
                for tg, fill, nm2 in (("sensSPK60_", "white", "With planted 0.60"), ("sensSPK70_", INK, "With planted 0.70")):
                    cc = st["configs"][f"{tg}{lab}__{pr}__{spk_fs}__hgb"]
                    ax.plot(cc["ctx_auc"], y - 0.42, marker="D", ms=3.8, mfc=fill, mec=INK, mew=0.8, ls="none", zorder=4,
                            label=nm2 if (i, j) == (0, 0) else None)
                ticks.append(y); names.append(f"{nm}, {fsn}")
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0.2, 0.85); ax.set_title(title, loc="left", color=INK); ax.set_xlabel("Context-stratified ROC-AUC")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(names, color=INK2)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.55, -0.1), ncol=5, handlelength=1.0, columnspacing=1.1)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.06, 1, 1)); fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------- main
def field_table(RAW):
    from fa_data import load_archive
    raw, _ = load_archive(Path(RAW))
    out = {}
    for f in ("StaticFrame", "CameraAutoMovement", "CameraFieldOfView", "DegreeOfControl", "HapticFeedback",
              "DofSimulation", "Locomotion", "UserSymptoms"):
        assert (raw.groupby("session_id")[f].nunique() == 1).all(), f
        v = raw.groupby(["campaign", "game"])[f].agg(lambda s: "/".join(sorted(map(str, s.unique()))))
        if f == "CameraFieldOfView":
            sf = sorted(float(x) for x in v["standing", "Flight"].split("/"))
            assert len(sf) == 2 and sf[0] == 60.0
            vals = (f"{float(v['seated','Flight']):.1f} / {float(v['seated','Race']):.1f} seated; "
                    f"{sf[1]:.1f} / {float(v['standing','Race']):.1f} standing$^{{a}}$")
            carries = "game alias (seated)"
        else:
            fl, rc = {v["seated", "Flight"], v["standing", "Flight"]}, {v["seated", "Race"], v["standing", "Race"]}
            assert len(fl) == 1 and len(rc) == 1, f
            fl, rc = fl.pop(), rc.pop()
            vals = f"{fl} / {rc}" if fl != rc else f"{fl} (both)"
            carries = "game alias" if fl.lower() != rc.lower() else ("game alias (letter case only)" if fl != rc else "nothing (constant)")
        out[f] = (vals.replace("NENHUM", "``none''"), carries)
    return out


def main(root, RAW, SEL, CSV):
    root = Path(root); DATA, R = root / "data", root / "results"
    FIG, GEN = root / "figures", root / "results" / "paper"
    FIG.mkdir(exist_ok=True, parents=True); GEN.mkdir(exist_ok=True, parents=True)
    F = pd.read_csv(DATA / "features_w30.csv.gz", usecols=["session_id", "group_id", "campaign", "game",
                                                           "severity", "t_elapsed", "GameFps"])
    S = pd.read_csv(DATA / "sessions.csv")
    F["curated"] = F.session_id.map(S.set_index("session_id").curated_release).astype(bool)
    V = Values()
    (GEN / "tab_archive.tex").write_text(audit(V, F, S, RAW, SEL, CSV))
    D = pd.read_csv(CSV); y1 = (D.DiscomfortLevel >= 1).astype(int)
    for c in ("StaticFrame", "TimeStamp"):
        a = roc_auc_score(y1, D[c]); V.put(f"cur1.col.{c}", a2(max(a, 1 - a)))
    (GEN / "tab_fields.tex").write_text(table_fields(field_table(RAW)))
    (GEN / "tab_protocols.tex").write_text(table_protocols(F))
    st, ops = config_values(V, R)
    for k, c in st["pooled_on_target"].items():
        lab, t, fs, mn = k.split("__")
        V.put(f"{lab.replace('sev', 's')}.poolon.{t}.{FS[fs]}.{MN[mn]}.auc", a2(c["auc"]))
        V.put(f"{lab.replace('sev', 's')}.poolon.{t}.{FS[fs]}.{MN[mn]}.ci", ci2(c["ci"]))
        V.put(f"{lab.replace('sev', 's')}.poolon.{t}.{FS[fs]}.{MN[mn]}.cauc", a2(c["ctx_auc"]))
        V.put(f"{lab.replace('sev', 's')}.poolon.{t}.{FS[fs]}.{MN[mn]}.cci", ci2(c["ctx_ci"]))
    for lab in ("1", "2"):
        cz = [st["pooled_on_target"][f"sev{lab}__{t}__MOT+TIME__hgb"]["ctx_auc"] for t in ("seated", "standing", "race", "flight")]
        V.put(f"poolon.s{lab}.MT.cmin", a2(min(cz))); V.put(f"poolon.s{lab}.MT.cmax", a2(max(cz)))
    test_values(V, st)
    w = [c["ci"][1] - c["ci"][0] for k, c in st["configs"].items()
         if k.startswith("sev2__") and k.split("__")[1] in WITHIN_P + CROSS_P and k.split("__")[2] in ("MOT+TIME", "MOT+CTX+TIME")
         and k.endswith("__hgb")]
    V.put("ciw.min", a2(min(w))); V.put("ciw.max", a2(max(w)))
    for mn in ("hgbreg", "logreg"):
        w = [c["ci"][1] - c["ci"][0] for k, c in st["configs"].items()
             if k.startswith("sev2__") and k.split("__")[1] in WITHIN_P + CROSS_P and k.split("__")[2] in ("MOT+TIME", "MOT+CTX+TIME")
             and k.endswith(f"__{mn}")]
        if w:
            V.put(f"ciw.{MN[mn]}.min", a2(min(w))); V.put(f"ciw.{MN[mn]}.max", a2(max(w)))
    # which feature set has the highest AUC (full precision, every feature set run), per label x learner,
    # over the ten protocols of Table main
    for lab in ("1", "2"):
        for mn, mk in (("hgb", "hgb"), ("logreg", "lr"), ("hgbreg", "hgbr")):
            if not any(k.endswith("__hgbreg") for k in st["configs"]) and mn == "hgbreg":
                continue
            best = []
            for pr in WITHIN_P + CROSS_P:
                cand = {k.split("__")[2]: c["auc"] for k, c in st["configs"].items()
                        if len(k.split("__")) == 4 and k.split("__")[0] == f"sev{lab}" and k.split("__")[1] == pr
                        and k.split("__")[3] == mn}
                best.append(max(cand, key=cand.get))
            V.put(f"best.s{lab}.{mk}.T", sum(b == "TIME" for b in best))
            V.put(f"best.s{lab}.{mk}.cross.T", sum(b == "TIME" for b, pr in zip(best, WITHIN_P + CROSS_P) if pr in CROSS_P))
            V.put(f"best.s{lab}.{mk}.within.CT", sum(b == "CTX+TIME" for b, pr in zip(best, WITHIN_P + CROSS_P) if pr in WITHIN_P))
    # every context-stratified interval of a telemetry model in a same-setting or single-cell protocol
    above = below = n_tot = 0; hits = []
    for k, c in st["configs"].items():
        m = k.split("__")
        if len(m) != 4 or m[0] not in ("sev1", "sev2") or m[1] not in WITHIN_P + CELL_P \
                or m[2] not in ("MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME") or c["ctx_auc"] != c["ctx_auc"]:
            continue
        if m[1] in CELL_P:
            post, game = m[1].split("_")[1], m[1].split("_")[2].capitalize()
            if F[(F.campaign == post) & (F.game == game) & (F.severity >= int(m[0][-1]))].group_id.nunique() < 3:
                continue                                               # not estimable (Table cells)
        n_tot += 1
        if c["ctx_ci"][0] > 0.5:
            above += 1; hits.append(k)
        elif c["ctx_ci"][1] < 0.5:
            below += 1
    V.put("ctxsum.n", n_tot); V.put("ctxsum.above", above); V.put("ctxsum.below", below)
    V.put("ctxsum.above.Z", sum("SELFZ" in h for h in hits)); V.put("ctxsum.above.raw", sum("SELFZ" not in h for h in hits))
    fw = [o["op_fpr"] for k, o in ops.items() if len(k.split("__")) == 4 and k.split("__")[0] in ("sev1", "sev2")
          and k.split("__")[1] in WITHIN_P
          and k.split("__")[2] == "MOT+TIME" and k.split("__")[3] == "hgb" and "op_fpr" in o]
    V.put("fpr.w.min", pct(min(fw))); V.put("fpr.w.max", pct(max(fw)))
    print("context-stratified intervals above 0.5:", hits)
    vrsq_values(V, R)
    base = (r"Group-disjoint results (grouped 5-fold, 5 repeats). Overall ROC-AUC for the three context baselines and, with 95\,\% "
            r"group-bootstrap intervals, for the two telemetry models; context-stratified AUC of MOT+TIME (0.5 = no knowledge beyond game, "
            r"campaign and 30-s exposure bin); recall and false-positive rate on the target at a threshold set for a 10\,\% false-alarm budget "
            r"on training groups. Group-disjoint: no inferred participant group is on both sides of a split. Seated and standing denote the two recording campaigns.")
    main_mn = "hgbr" if "s2.ws.MT.hgbr.auc" in V.v and "s2.r2f.MCT.hgbr.auc" in V.v else "hgb"
    main_nm = {"hgbr": "HGB-reg, the regularized, group-validated learner", "hgb": "default HGB"}[main_mn]
    V.put("main.learner", {"hgbr": "HGB-reg", "hgb": "HGB"}[main_mn])
    (GEN / "tab_main.tex").write_text(table_main(V, mn=main_mn, caption=r"Main results for \sevtwo{} (" + main_nm + "). " + base))
    (GEN / "supp_main_s1.tex").write_text(table_main(V, lab="s1", mn=main_mn, label="tab:supp_s1", star=False, caption=
        r"Secondary label \sevone{}, " + main_nm + ". " + base))
    (GEN / "supp_main_lr.tex").write_text(table_main(V, mn="lr", label="tab:supp_lr", star=False, caption=
        r"Primary label \sevtwo{}, logistic regression (no CTX-only model was fitted with it). " + base))
    (GEN / "supp_main_hgb.tex").write_text(table_main(V, mn="hgb", label="tab:supp_hgb", star=False, caption=
        r"Primary label \sevtwo{}, default HGB (sensitivity learner). " + base))
    (GEN / "tab_tests.tex").write_text(table_tests(V, st))
    if (R / "vrlens.json").exists():
        table_vrlens(V, R, main_mn)          # values only; the rows are part of Table curated
    else:
        print("results/vrlens.json missing: VR-LENS rows not written (TODO-RUN: fa_vrlens.py)")
    (GEN / "tab_curated.tex").write_text(table_curated(V, main_mn))
    (GEN / "tab_mitig.tex").write_text(table_mitig(V))
    (GEN / "tab_sens.tex").write_text(table_sens(V)); sens_ranges(V, st)
    (GEN / "tab_cells.tex").write_text(table_cells(V, F))
    lc = json.load(open(R / "leakage_checks.json")); V.put("chk.nfolds", lc["n_folds_checked"])
    lg = json.load(open(R / "logo_ctx.json"))
    for k, c in lg.items():
        lab, pr, fs, mn = k.split("__")
        V.put(f"logo.{lab.replace('sev', 's')}.{PR[pr]}.{FS[fs]}.{MN[mn]}.auc", a2(c["logo_pooled_auc"]))
    V.put("chk.all", "passed" if all(v is True or isinstance(v, list) or isinstance(v, int) for v in lc.values()) else "FAILED")
    extra_values(V, F, S, R, root)
    demo_sizes = vrlens_profiles(V, RAW, CSV, S)
    (GEN / "tab_posctl.tex").write_text(table_posctl(V))
    (GEN / "supp_table.tex").write_text(supp_table(st, ops))
    (GEN / "supp_firstrun.tex").write_text(supp_firstrun(st, json.load(open(root / "results_logo" / "stats.json"))))
    (GEN / "supp_strata.tex").write_text(supp_strata(V, F, R / "preds"))
    # revision 3: the revision-2 main table, comparison family and positive control move to the Supplement
    (GEN / "supp_main_hgbr.tex").write_text(table_main(V, mn=main_mn, label="tab:supp_hgbr", star=False, caption=
        r"Earlier analysis: main table for \sevtwo{}, " + main_nm + ". " + base))
    to_supp = lambda t, lab: t.replace("table*", "table").replace(f"\\label{{tab:{lab}}}", f"\\label{{tab:supp_{lab}}}")
    (GEN / "supp_tests.tex").write_text(to_supp((GEN / "tab_tests.tex").read_text(), "tests").replace(
        r"\caption{Exploratory comparison family", r"\caption{Earlier analysis: exploratory comparison family", 1).replace(
        r"Supplementary Table~S4", r"Table~S4"))
    (GEN / "supp_posctl.tex").write_text(to_supp((GEN / "tab_posctl.tex").read_text(), "posctl").replace(
        r"\caption{", r"\caption{Earlier analysis: ", 1))
    import fa_report_rev3                       # revision 3: offset learners, controls, second archive
    fa_report_rev3.run(V, F, R, GEN, FIG)
    (GEN / "tab_vrsq.tex").write_text(table_vrsq(V))      # needs the values of the VRSQ zero check
    V.write(GEN / "values.tex")
    fig_archive(F, FIG / "fig_archive.pdf"); fig_vrsq(R, FIG / "fig_vrsq.pdf")
    fig_auc(st, FIG / "fig_auc.pdf", mn="hgbreg" if "sev2__race_to_flight__MOT+CTX+TIME__hgbreg" in st["configs"] else "hgb")
    fig_null(st, R, FIG / "fig_null.pdf")
    y2c = (D.DiscomfortLevel >= 2).astype(int); a_sf = roc_auc_score(y2c, D.StaticFrame)
    claims(st, R, staticframe_auc=max(a_sf, 1 - a_sf), demo_shared_sizes=demo_sizes)
    print(f"values: {len(V.v)}; tables and figures written")


def vrlens_profiles(V, RAW, CSV, S):
    """The VR-LENS detector uses five of the seven demographic answers that define the inferred groups.
    Groups of the curated release whose five answers coincide (they differ only in the other two)."""
    from fa_data import load_archive
    from fa_vrlens import FEATURES, curated_rows
    raw, _ = load_archive(Path(RAW))
    Dc, rows = curated_rows(raw, CSV)
    sid = raw.session_id.to_numpy()[rows]
    grp = pd.Series(sid).map(S.set_index("session_id").group_id).to_numpy()
    demo = [c for c in FEATURES if c.startswith("User")]
    prof = Dc.assign(_g=grp).groupby("_g")[demo].first()
    size = prof.groupby(demo).size(); shared = size[size > 1]
    gs = set(prof.reset_index().merge(shared.rename("n").reset_index(), on=demo)["_g"])
    V.put("vrl.demo.n", len(demo)); V.put("vrl.cur.ngroups", len(prof))
    V.put("vrl.demo.nprof", len(shared)); V.put("vrl.demo.ngroups", len(gs))
    V.put("vrl.demo.nsess", len(set(sid[np.isin(grp, list(gs))])))
    return [int(x) for x in shared.values]


def extra_values(V, F, S, R, root):
    from fa_run import domains
    st_all = json.load(open(R / "stats.json"))
    D = domains(F.assign(curated=F.curated.astype(bool)))
    cell = (F.campaign + "|" + F.game).to_numpy()
    for lab in (1, 2):
        y = (F.severity.to_numpy() >= lab).astype(int)
        for pr in ("within_seated", "pooled", "within_curated"):
            t = D[pr][1]; prev = pd.Series(y[t]).groupby(cell[t]).transform("mean").to_numpy()
            V.put(f"ins.s{lab}.{PR[pr]}.C.auc", a2(roc_auc_score(y[t], prev)))   # in-sample cell prevalence as score
    # revision 2: ranges over protocol groups, per label, feature set and learner (W: same setting,
    # X: cross setting, A: all ten protocols of Table main)
    ops_all = {}
    for f in ("main.json", "context.json", "hgbreg.json"):
        if (R / f).exists():
            d = json.load(open(R / f)); ops_all.update({r["key"]: r for r in (d["runs"] if isinstance(d, dict) else d)})
    for lab in (1, 2):
        for fs in ("TIME", "CTX+TIME", "MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME"):
            for mn in ("hgbreg", "hgb", "logreg"):
                for sc, prs in (("W", WITHIN_P), ("X", CROSS_P), ("A", WITHIN_P + CROSS_P)):
                    cs = [st_all["configs"].get(f"sev{lab}__{pr}__{fs}__{mn}") for pr in prs]
                    if not all(cs):
                        continue
                    k = f"rng.s{lab}.{sc}.{FS[fs]}.{MN[mn]}"
                    V.put(k + ".min", a2(min(c["auc"] for c in cs))); V.put(k + ".max", a2(max(c["auc"] for c in cs)))
                    V.put(k + ".cmin", a2(min(c["ctx_auc"] for c in cs))); V.put(k + ".cmax", a2(max(c["ctx_auc"] for c in cs)))
                    os_ = [ops_all.get(f"sev{lab}__{pr}__{fs}__{mn}", {}) for pr in prs]
                    if all("op_recall" in o for o in os_):
                        V.put(k + ".recmin", pct(min(o["op_recall"] for o in os_))); V.put(k + ".recmax", pct(max(o["op_recall"] for o in os_)))
                        V.put(k + ".fprmin", pct(min(o["op_fpr"] for o in os_))); V.put(k + ".fprmax", pct(max(o["op_fpr"] for o in os_)))
    TT = st_all["tests"]
    for mn in ("hgbreg", "hgb", "logreg"):
        g = [TT.get(f"sev2_{mn}_telemetry_gain_over_context__{p}", {}).get("pooled", {}) for p in WITHIN_P[1:]]
        if all("p_holm" in x for x in g):
            V.put(f"w.gainC.neg.{MN[mn]}", sum(x["delta"] < 0 and x["p_holm"] < 0.05 for x in g))
            V.put(f"w.gainC.nneg.{MN[mn]}", sum(x["delta"] < 0 for x in g))
        sc = [TT.get(f"sev2_{mn}_shift_cost_MOT+TIME__{t}") for t in ("seated", "standing", "race", "flight")]
        if all(sc):
            V.put(f"rng.shift.{MN[mn]}.min", a2(min(x["pooled"]["delta"] for x in sc)))
            V.put(f"rng.shift.{MN[mn]}.max", a2(max(x["pooled"]["delta"] for x in sc)))
    tc = [st_all["configs"][f"sev{l}__{pr}__TIME__hgb"]["ctx_auc"] for l in (1, 2) for pr in WITHIN_P + CROSS_P]
    V.put("tctx.min", a2(min(tc))); V.put("tctx.max", a2(max(tc)))
    xc = [st_all["configs"][f"sev2__{pr}__MOT+TIME__hgb"]["ctx_auc"] for pr in CROSS_P]
    V.put("xctx.min", a2(min(xc))); V.put("xctx.max", a2(max(xc)))
    for l in (1, 2):
        yy = (F.severity >= l).astype(int); gs = yy.groupby(F.session_id)
        V.put(f"ctl.s{l}.poschance", pct(float((gs.sum() ** 2 / gs.size()).sum() / yy.sum()), 0))
    es = json.load(open(R / "es_check.json")); act = [r for v in es.values() for r in v if r["early_stopping"]]
    V.put("es.nfold", sum(len(v) for v in es.values())); V.put("es.nactive", len(act))
    V.put("es.nstop", sum(r["n_iter"] < 150 for r in act)); V.put("es.minit", min(r["n_iter"] for r in act))
    ctl = json.load(open(R / "control.json"))["label_agreement_after_shift"]
    for lab, d in ctl.items():
        V.put(f"ctl.{lab.replace('sev', 's')}.agree", pct(d["record_agreement"], 0))
        V.put(f"ctl.{lab.replace('sev', 's')}.pospos", pct(d["positives_still_positive"], 0))
    nl = pd.DataFrame(json.load(open(R / "null.json")))
    V.put("null.nsim", nl.sim.nunique())
    st = json.load(open(R / "stats.json"))["configs"]
    outside = []
    for (lab, pr, fs), d in nl.groupby(["label", "protocol", "features"]):
        k = f"null.{lab.replace('sev', 's')}.{PR[pr]}.{FS[fs]}"
        cx, ov = d.context_stratified_auc, d.overall_auc
        V.put(k + ".ctx", a2(cx.mean())); V.put(k + ".ctxlo", a2(cx.quantile(0.025))); V.put(k + ".ctxhi", a2(cx.quantile(0.975)))
        V.put(k + ".ov", a2(ov.mean())); V.put(k + ".ovlo", a2(ov.quantile(0.025))); V.put(k + ".ovhi", a2(ov.quantile(0.975)))
        obs = st[f"{lab}__{pr}__{fs}__hgb"]["ctx_auc"]
        V.put(k + ".obs", a2(obs))
        if fs != "CTX" and not (cx.quantile(0.025) <= obs <= cx.quantile(0.975)):
            outside.append(f"{lab}/{pr}/{fs}: observed {obs:.3f}, null 95% range [{cx.quantile(0.025):.3f}, {cx.quantile(0.975):.3f}]")
    V.put("null.noutside", len(outside)); V.put("null.nchecked", int((nl.features != "CTX").sum() / nl.sim.nunique()))
    labn = {"sev1": "\\sevone{}", "sev2": "\\sevtwo{}"}
    prn = {"within_seated": "within the seated campaign", "within_standing": "within the standing campaign", "pooled": "pooled"}
    items = []
    for (lab, pr, fs), d in nl.groupby(["label", "protocol", "features"]):
        cx = d.context_stratified_auc; obs = st[f"{lab}__{pr}__{fs}__hgb"]["ctx_auc"]
        if fs != "CTX" and not (cx.quantile(0.025) <= obs <= cx.quantile(0.975)):
            items.append(f"{fs} for {labn[lab]} {prn[pr]} ({a2(obs)}; null range {a2(cx.quantile(0.025))}--{a2(cx.quantile(0.975))})")
    V.put("null.outside.list", "; ".join(items) if items else "none")
    for lab in ("1", "2"):
        z = [st[f"sev{lab}__{pr}__SELFZ+TIME__hgb"]["ctx_auc"] for pr in WITHIN_P]
        V.put(f"zs{lab}.cmin", a2(min(z))); V.put(f"zs{lab}.cmax", a2(max(z)))
        m = [st[f"sev{lab}__{pr}__MOT+TIME__hgb"]["ctx_auc"] for pr in WITHIN_P]
        V.put(f"ms{lab}.cmin", a2(min(m))); V.put(f"ms{lab}.cmax", a2(max(m)))
        m = [st[f"sev{lab}__{pr}__MOT+TIME__logreg"]["ctx_auc"] for pr in WITHIN_P]
        V.put(f"ms{lab}lr.cmin", a2(min(m))); V.put(f"ms{lab}lr.cmax", a2(max(m)))
        V.put(f"zs{lab}.nciabove", sum(st[f"sev{lab}__{pr}__SELFZ+TIME__hgb"]["ctx_ci"][0] > 0.5 for pr in WITHIN_P))
        z = [st[f"sev{lab}__{pr}__SELFZ+TIME__logreg"]["ctx_auc"] for pr in WITHIN_P]
        V.put(f"zs{lab}lr.cmin", a2(min(z))); V.put(f"zs{lab}lr.cmax", a2(max(z)))
    half = [(d.context_stratified_auc.quantile(0.975) - d.context_stratified_auc.quantile(0.025)) / 2
            for (_, _, fs), d in nl.groupby(["label", "protocol", "features"]) if fs != "CTX"]
    V.put("null.maxhalf", a2(max(half))); V.put("null.minhalf", a2(min(half)))
    for (lab, pr, fs), d in nl.groupby(["label", "protocol", "features"]):
        if fs == "CTX":
            continue
        obs = st[f"{lab}__{pr}__{fs}__hgb"]["ctx_auc"]; cx = d.context_stratified_auc.to_numpy()
        V.put(f"null.{lab.replace('sev', 's')}.{PR[pr]}.{FS[fs]}.pn", f"{(1 + (cx >= obs).sum()) / (1 + len(cx)):.3f}")
    V.put("null.pmin", f"{1 / (1 + nl.sim.nunique()):.3f}")
    pns = {}
    for (lab, pr, fs), d in nl.groupby(["label", "protocol", "features"]):
        if fs != "CTX":
            obs = st[f"{lab}__{pr}__{fs}__hgb"]["ctx_auc"]; cx = d.context_stratified_auc.to_numpy()
            pns[(lab, pr, fs)] = (1 + (cx >= obs).sum()) / (1 + len(cx))
    V.put("null.np05", sum(v < 0.05 for v in pns.values()))
    pmin = min(pns.values()); V.put("null.pnmin", f"{pmin:.3f}")
    from fa_stats import holm
    hadj = holm({"/".join(k): v for k, v in pns.items()})
    V.put("null.holmmin", f"{min(hadj.values()):.2f}"); V.put("null.nholm05", sum(v < 0.05 for v in hadj.values()))
    words = lambda n: {0: "none does", 1: "one does"}.get(n, f"{n} do")
    has = lambda n: {0: "none has", 1: "one has"}.get(n, f"{n} have")
    V.put("null.np05.words", has(sum(v < 0.05 for v in pns.values())))
    V.put("null.nholm05.words", words(sum(v < 0.05 for v in hadj.values())))
    V.put("null.holmfloor", f"{len(pns) / (1 + nl.sim.nunique()):.3f}")   # smallest attainable Holm-adjusted permutation p
    stc = json.load(open(R / "stats.json"))["configs"]
    nw = lambda df, mn: sum((d.context_stratified_auc.quantile(0.975) - d.context_stratified_auc.quantile(0.025)) >
                            (stc[f"{lab}__{pr}__{fs}__{mn}"]["ctx_ci"][1] - stc[f"{lab}__{pr}__{fs}__{mn}"]["ctx_ci"][0])
                            for (lab, pr, fs), d in df.groupby(["label", "protocol", "features"]) if fs != "CTX")
    V.put("null.nwider", nw(nl, "hgb")); V.put("nulllr.nwider", nw(pd.DataFrame(json.load(open(R / "null_lr.json"))), "logreg"))
    nlr = pd.DataFrame(json.load(open(R / "null_lr.json"))); V.put("nulllr.nsim", nlr.sim.nunique())
    pl = {}
    for (lab, pr, fs), d in nlr.groupby(["label", "protocol", "features"]):
        if fs != "CTX":
            obs = st[f"{lab}__{pr}__{fs}__logreg"]["ctx_auc"]; cx = d.context_stratified_auc.to_numpy()
            pl[(lab, pr, fs)] = (1 + (cx >= obs).sum()) / (1 + len(cx))
            V.put(f"nulllr.{lab.replace('sev', 's')}.{PR[pr]}.{FS[fs]}.pn", f"{pl[(lab, pr, fs)]:.3f}")
            V.put(f"nulllr.{lab.replace('sev', 's')}.{PR[pr]}.{FS[fs]}.ctxlo", a2(d.context_stratified_auc.quantile(0.025)))
            V.put(f"nulllr.{lab.replace('sev', 's')}.{PR[pr]}.{FS[fs]}.ctxhi", a2(d.context_stratified_auc.quantile(0.975)))
    hl = holm({"/".join(k): v for k, v in pl.items()})
    V.put("nulllr.np05", sum(v < 0.05 for v in pl.values())); V.put("nulllr.pnmin", f"{min(pl.values()):.3f}")
    V.put("nulllr.holmmin", f"{min(hl.values()):.2f}"); V.put("nulllr.nholm05", sum(v < 0.05 for v in hl.values()))
    V.put("nulllr.nholm05.words", words(sum(v < 0.05 for v in hl.values())))
    V.put("nulllr.np05.words", has(sum(v < 0.05 for v in pl.values())))
    V.put("cells.s2.nabove", sum(1 for p in CELL_P for f in ("MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME") for m in ("hgb", "logreg")
                                 if f"sev2__{p}__{f}__{m}" in st and ("sev2", p) not in NONEST and st[f"sev2__{p}__{f}__{m}"]["ctx_ci"][0] > 0.5))
    items = [f"{fs} ({labn[lab]}, {prn[pr]})" for (lab, pr, fs), v in sorted(pl.items()) if v == min(pl.values())]
    V.put("nulllr.pnmin.list", " and ".join(items) if len(items) == 2 else "; ".join(items))
    items = [f"{fs} ({labn[lab]}, {prn[pr]})" for (lab, pr, fs), v in sorted(pns.items()) if v == pmin]
    V.put("null.pnmin.list", " and ".join(items) if len(items) == 2 else "; ".join(items))
    # positive control: planted channel of known within-stratum AUC (ceiling, among raw, among self-normalized)
    pc = json.load(open(R / "posctl.json"))
    for key, d in pc["planted"].items():
        tag = key[:key.index("sev")]; lab, pr = key[len(tag):].split("__")
        V.put(f"{TAG[tag]}{lab.replace('sev', 's')}.{PR[pr]}.chan", a2(d["channel_auc_on_target"]))
    for tag in ("sensSPK60_", "sensSPK70_"):
        for fs, nullfs in (("SPK+TIME", None), ("MOT+SPK+TIME", "MOT+TIME"), ("SELFZ+SPK+TIME", "SELFZ+TIME")):
            for mn in ("hgb", "logreg", "hgbreg"):
                if f"{tag}sev1__pooled__{fs}__{mn}" not in st:
                    continue
                res = []
                for lab in ("sev1", "sev2"):
                    for pr in ("within_seated", "within_standing", "pooled"):
                        c = st[f"{tag}{lab}__{pr}__{fs}__{mn}"]; above = None
                        if nullfs and mn == "hgb":
                            hi = nl[(nl.label == lab) & (nl.protocol == pr) & (nl.features == nullfs)].context_stratified_auc.quantile(0.975)
                            above = c["ctx_auc"] > hi
                        res.append((c["ctx_ci"][0] > 0.5, above, c["ctx_auc"]))
                k = f"{TAG[tag]}{FS[fs]}.{MN[mn]}"
                for li, lab in enumerate(("s1", "s2")):
                    rr = res[3 * li: 3 * li + 3]
                    V.put(f"{k}.{lab}.nci", sum(r[0] for r in rr)); V.put(f"{k}.{lab}.cmin", a2(min(r[2] for r in rr)))
                    V.put(f"{k}.{lab}.cmax", a2(max(r[2] for r in rr)))
                V.put(k + ".nci", sum(r[0] for r in res)); V.put(k + ".n", len(res))
                V.put(k + ".cmin", a2(min(r[2] for r in res))); V.put(k + ".cmax", a2(max(r[2] for r in res)))
                if nullfs and mn == "hgb":
                    V.put(k + ".nnull", sum(bool(r[1]) for r in res))
    from fa_run import spike_feature
    from fa_stats import Boot
    tb = np.minimum((F.t_elapsed.to_numpy() // 30).astype(int), 9).astype(str)
    strata = np.char.add((F.campaign + "|" + F.game + "|").to_numpy().astype(str), tb)
    for tag, key in (("sensSPK60_", "spk60"), ("sensSPK70_", "spk70")):
        vals = []
        for lab in (1, 2):
            y = (F.severity.to_numpy() >= lab).astype(int)
            spk = spike_feature(F, y, pc["planted"][f"{tag}sev{lab}__pooled"]["d"], seed=900 + lab)
            for pr in ("within_seated", "within_standing", "pooled"):
                t = D[pr][1]
                vals.append(Boot(F.group_id.to_numpy()[t], b=0).auc(y[t], spk[t], strata[t])[0])
        V.put(f"{key}.chanctx.min", a2(min(vals))); V.put(f"{key}.chanctx.max", a2(max(vals)))
    V.put("spk.d60", f"{pc['planted']['sensSPK60_sev2__pooled']['d']:.2f}")
    # reporting rule of the analysis plan (fixed before HGB-reg was run): where a learner recovers the planted
    # 0.70 channel among the real features (interval above 0.5) and the real-label context-stratified interval
    # lies entirely below the recovered one -> evidence against a within-context signal of that size
    for mn in ("hgbreg", "logreg", "hgb"):
        rec, ag = [], []
        for lab in (1, 2):
            for pr in ("within_seated", "within_standing", "pooled"):
                for fr, fp in (("MOT+TIME", "MOT+SPK+TIME"), ("SELFZ+TIME", "SELFZ+SPK+TIME")):
                    pl = st.get(f"sensSPK70_sev{lab}__{pr}__{fp}__{mn}"); re_ = st.get(f"sev{lab}__{pr}__{fr}__{mn}")
                    if pl is None or re_ is None:
                        continue
                    if pl["ctx_ci"][0] > 0.5:
                        rec.append((lab, pr, fr))
                        if re_["ctx_ci"][1] < pl["ctx_ci"][0]:
                            ag.append((lab, pr, fr))
        if rec:
            V.put(f"rule.{MN[mn]}.nrec", len(rec)); V.put(f"rule.{MN[mn]}.nagainst", len(ag))
            RULE[mn] = {"recovered": rec, "against": ag}
    # HGB-reg: what the nested grouped selection chose (main-table runs, every outer fold)
    sel = [r for f in sorted((R / "preds").glob("sev*__hgbreg.selection.json")) for r in json.load(open(f))]
    if sel:
        nit = np.array([r["n_iter"] for r in sel])
        V.put("hgbr.nsel", n_(len(sel))); V.put("hgbr.nit.med", f"{np.median(nit):.0f}")
        V.put("hgbr.nit.q1", f"{np.percentile(nit, 25):.0f}"); V.put("hgbr.nit.q3", f"{np.percentile(nit, 75):.0f}")
        V.put("hgbr.nit.le5", pct(np.mean(nit <= 5), 0))
        V.put("hgbr.most", pct(np.mean([(r["max_depth"], r["min_samples_leaf"], r["l2_regularization"]) == (2, 300, 10.0) for r in sel]), 0))
        V.put("hgbr.ctxcrit", pct(np.mean([r["criterion"] == "ctx" for r in sel]), 0))
    V.put("spk.d70", f"{pc['planted']['sensSPK70_sev2__pooled']['d']:.2f}")
    print("observed context-stratified AUCs outside the null range:", outside)
    cn = nl[nl.features == "CTX"]
    V.put("null.C.ctxmax", a2(max(abs(cn.context_stratified_auc - 0.5)) + 0.5))
    # first run (leave-one-group-out; archived output in results_logo/): primary-family outcome
    old = json.load(open(root / "results_logo" / "stats.json"))["tests"]
    fam = [n for n in old if "p_holm" in old[n].get("pooled", {})]
    new = json.load(open(R / "stats.json"))["tests"]
    V.put("logo.fam.nsig", sum(old[n]["pooled"]["p_holm"] < 0.05 for n in fam))
    V.put("logo.fam.ndiff", sum((old[n]["pooled"]["p_holm"] < 0.05) != (new[n]["pooled"]["p_holm"] < 0.05) for n in fam))
    q = json.load(open(R / "vrsq.json"))["correlations"]
    rh = [c["rho"] for k, c in q.items() if not k.startswith("det_")]
    V.put("vrsq.rho.min", a2(min(rh))); V.put("vrsq.rho.max", a2(max(rh)))
    V.put("vrsq.cihi.max", a2(max(c["ci"][1] for k, c in q.items() if not k.startswith("det_"))))
    dl = [c for k, c in q.items() if k.endswith("vrsq_delta_total") and not k.startswith("det_")]
    V.put("vrsq.delta.nexcl0", sum(c["ci"][0] > 0 for c in dl)); V.put("vrsq.delta.n", len(dl))
    Fr = pd.read_csv(root / "data" / "features_w30.csv.gz", usecols=["game", "CameraRotationX", "CameraRotationY", "CameraRotationZ"])
    nv = np.sqrt((Fr[["CameraRotationX", "CameraRotationY", "CameraRotationZ"]] ** 2).sum(1))
    V.put("rot.maxnorm", f"{nv.max():.4f}")
    for g, d in nv.groupby(Fr.game):
        V.put(f"rot.med.{g.lower()}", a2(d.median()))


def claims(st, R, staticframe_auc=None, demo_shared_sizes=None):
    """Every qualitative statement of the manuscript that is not itself a printed number is
    checked here against the results; a failed check means the text must change."""
    C = st["configs"]; T = st["tests"]
    a = lambda lab, pr, fs, mn="hgb": C[f"sev{lab}__{pr}__{fs}__{mn}"]["auc"]
    ca = lambda lab, pr, fs, mn="hgb": C[f"sev{lab}__{pr}__{fs}__{mn}"]["ctx_auc"]
    cci = lambda lab, pr, fs, mn="hgb": C[f"sev{lab}__{pr}__{fs}__{mn}"]["ctx_ci"]
    ho = lambda n: T[n]["pooled"]["p_holm"]
    d = lambda n: T[n]["pooled"]["delta"]
    out = {}

    def check(name, ok):
        out[name] = bool(ok)
    raw = ("MOT+TIME", "MOT+CTX+TIME")
    if demo_shared_sizes is not None:      # "pairs of groups share all five VR-LENS answers"
        check("vrlens_shared_demographic_profiles_are_pairs", len(demo_shared_sizes) > 0 and all(n == 2 for n in demo_shared_sizes))
    check("hgbreg_t2s_ctx_interval_above_0.5_s2", cci(2, "standing_to_seated", "MOT+TIME", "hgbreg")[0] > 0.5)
    check("CT_hgb_beats_all_raw_telemetry_both_learners_same_setting_s2",
          all(a(2, p, "CTX+TIME") > max(a(2, p, f, m) for f in raw for m in ("hgb", "logreg")) for p in WITHIN_P))
    check("cross_MT_hgb_below_0.5_s2", all(a(2, p, "MOT+TIME") < 0.5 for p in CROSS_P))
    check("f2r_MT_ci_below_0.5_s2", C["sev2__flight_to_race__MOT+TIME__hgb"]["ci"][1] < 0.5)
    check("TIME_best_hgb_on_every_shift_s2",
          all(a(2, p, "TIME") > max(a(2, p, f) for f in ("CTX", "CTX+TIME", "MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME")) for p in CROSS_P))
    check("TIME_only_feature_set_above_0.5_everywhere_tabmain_s2_both_learners",
          all(a(2, p, "TIME", m) > 0.5 for p in WITHIN_P + CROSS_P for m in ("hgb", "logreg")) and
          all(min(a(2, p, f, m) for p in WITHIN_P + CROSS_P) <= 0.5 for f in ("CTX+TIME", "MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME")
              for m in ("hgb", "logreg")))
    check("CT_reversed_under_every_shift_s2", all(a(2, p, "CTX+TIME") < 0.5 for p in CROSS_P))
    check("raw_MT_ctx_ci_includes_or_below_0.5_same_setting_s2_hgb",
          all(cci(2, p, "MOT+TIME")[0] <= 0.5 for p in WITHIN_P))
    gainC = {p: f"sev2_hgb_telemetry_gain_over_context__{p}" for p in ["within_seated", "within_standing", "within_flight", "within_race", "pooled"] + CROSS_P}
    check("gainC_never_positive_same_setting", all(d(gainC[p]) <= 0 for p in ["within_seated", "within_standing", "within_flight", "within_race", "pooled"]))
    out["gainC_sig_negative_same_setting"] = [p for p in ["within_seated", "within_standing", "within_flight", "within_race", "pooled"]
                                              if d(gainC[p]) < 0 and ho(gainC[p]) < 0.05]
    out["gainC_positive_cross"] = [p for p in CROSS_P if d(gainC[p]) > 0]
    out["gainC_sig_positive_cross"] = [p for p in CROSS_P if d(gainC[p]) > 0 and ho(gainC[p]) < 0.05]
    check("t2s_r2f_both_models_at_or_below_0.5",
          all(a(2, p, f) <= 0.5 for p in ("standing_to_seated", "race_to_flight") for f in ("CTX+TIME", "MOT+CTX+TIME")))
    out["shift_cost_sig_targets"] = [t for t in ("seated", "standing", "race", "flight")
                                     if ho(f"sev2_hgb_shift_cost_MOT+TIME__{t}") < 0.05]
    check("shift_costs_all_positive", all(d(f"sev2_hgb_shift_cost_MOT+TIME__{t}") > 0 for t in ("seated", "standing", "race", "flight")))
    check("ZT_above_MT_in_ws_wf_wr_s2", all(a(2, p, "SELFZ+TIME") > a(2, p, "MOT+TIME") for p in ("within_seated", "within_flight", "within_race")))
    check("ZT_above_CT_only_in_flight_s2", [p for p in WITHIN_P if a(2, p, "SELFZ+TIME") > a(2, p, "CTX+TIME")] == ["within_flight"])
    check("ZT_below_MT_within_standing_s2", a(2, "within_standing", "SELFZ+TIME") < a(2, "within_standing", "MOT+TIME"))
    check("ZT_cross_s2_all_at_or_below_0.51", all(a(2, p, "SELFZ+TIME") <= 0.51 for p in CROSS_P))
    check("cells_TIME_at_least_MT_same_learner_seatR_standF_s2",
          all(a(2, p, "TIME", m) >= a(2, p, "MOT+TIME", m) for p in ("cell_seated_race", "cell_standing_flight") for m in ("hgb", "logreg")))
    check("s1_cross_MT_close_to_same_setting",
          all(abs(a(1, WITHIN_P[1 + i], "MOT+TIME") - a(1, x, "MOT+TIME")) < 0.1 for i, x in
              [(0, "standing_to_seated"), (1, "seated_to_standing"), (3, "flight_to_race"), (2, "race_to_flight")]))
    P = st["pooled_on_target"]
    check("pooled_CT_above_pooled_MT_every_target_s2",
          all(P[f"sev2__{t}__CTX+TIME__hgb"]["auc"] > P[f"sev2__{t}__MOT+TIME__hgb"]["auc"] for t in ("seated", "standing", "race", "flight")))
    check("pooled_MT_above_cross_MT_every_target_s2",
          all(P[f"sev2__{t}__MOT+TIME__hgb"]["auc"] > a(2, x, "MOT+TIME") for t, x in
              (("seated", "standing_to_seated"), ("standing", "seated_to_standing"), ("race", "flight_to_race"), ("flight", "race_to_flight"))))
    q = json.load(open(R / "vrsq.json"))["correlations"]
    post = {k: q[f"{k}~vrsq_post_total"]["rho"] for k in ("ing_max", "ing_mean", "ing_frac_ge1", "ing_frac_ge2")}
    check("vrsq_frac_ge2_agrees_best", max(post, key=post.get) == "ing_frac_ge2")
    dlt = {k: q[f"{k}~vrsq_delta_total"]["rho"] for k in ("ing_max", "ing_mean", "ing_frac_ge1", "ing_frac_ge2")}
    check("vrsq_frac_ge2_agrees_best_with_change_too", max(dlt, key=dlt.get) == "ing_frac_ge2")
    hi = max(c["ci"][1] for k, c in q.items() if not k.startswith("det_"))
    check("vrsq_intervals_exclude_strong_not_moderate", 0.4 < hi < 0.7)
    on = json.load(open(R / "vrsq.json"))["onset"]
    check("standing_earlier_onset", on["standing"]["ge1_median_s"] < on["seated"]["ge1_median_s"] and
          on["standing"]["ge2_median_s"] < on["seated"]["ge2_median_s"])
    check("s1_wt_raw_MT_ctx_ci_above_0.5", cci(1, "within_standing", "MOT+TIME")[0] > 0.5)
    sp = lambda tag, lab, pr, fs, mn: C[f"{tag}sev{lab}__{pr}__{fs}__{mn}"]
    P3 = ("within_seated", "within_standing", "pooled")
    check("posctl_70_single_channel_found_all_six_both_learners",
          all(sp("sensSPK70_", l, p, "SPK+TIME", m)["ctx_ci"][0] > 0.5 for l in (1, 2) for p in P3 for m in ("hgb", "logreg")))
    check("posctl_60_not_reliable_even_single_channel",
          not all(sp("sensSPK60_", l, p, "SPK+TIME", m)["ctx_ci"][0] > 0.5 for l in (1, 2) for p in P3 for m in ("hgb", "logreg")))
    check("posctl_70_raw_hgb_below_0.5_seated_both_labels",
          all(sp("sensSPK70_", l, "within_seated", "MOT+SPK+TIME", "hgb")["ctx_auc"] < 0.5 for l in (1, 2)))
    check("posctl_70_raw_lr_beats_raw_hgb_count",
          sum(sp("sensSPK70_", l, p, "MOT+SPK+TIME", "logreg")["ctx_ci"][0] > 0.5 for l in (1, 2) for p in P3) >
          sum(sp("sensSPK70_", l, p, "MOT+SPK+TIME", "hgb")["ctx_ci"][0] > 0.5 for l in (1, 2) for p in P3))
    check("raw_telemetry_s2_same_setting_ctx_not_above_chance_both_learners",
          all(cci(2, p, f, m)[0] <= 0.5 for p in WITHIN_P for f in ("MOT+TIME", "MOT+CTX+TIME") for m in ("hgb", "logreg")))
    check("posctl_70_raw_s2_not_found_within_campaigns_either_learner",
          all(sp("sensSPK70_", 2, p, "MOT+SPK+TIME", m)["ctx_ci"][0] <= 0.5 for p in ("within_seated", "within_standing") for m in ("hgb", "logreg")))
    check("posctl_70_lr_s1_found_raw_and_selfz_all_three",
          all(sp("sensSPK70_", 1, p, f, "logreg")["ctx_ci"][0] > 0.5 for p in P3 for f in ("MOT+SPK+TIME", "SELFZ+SPK+TIME")))
    check("posctl_selfz_hgb_similar_for_60_and_70",
          max(abs(sp("sensSPK70_", l, p, "SELFZ+SPK+TIME", "hgb")["ctx_auc"] - sp("sensSPK60_", l, p, "SELFZ+SPK+TIME", "hgb")["ctx_auc"])
              for l in (1, 2) for p in P3) < 0.08)
    check("MT_hgb_s2_ctx_ci_below_0.5_only_within_seated", [p for p in WITHIN_P if cci(2, p, "MOT+TIME")[1] < 0.5] == ["within_seated"])
    check("lr_family_shift_costs_positive", all(d(f"sev2_logreg_shift_cost_MOT+TIME__{t}") > 0 for t in ("seated", "standing", "race", "flight")))
    check("lr_family_gainC_negative_same_setting_except_pooled",
          all(d(f"sev2_logreg_telemetry_gain_over_context__{p}") < 0 for p in ("within_seated", "within_standing", "within_flight", "within_race"))
          and d("sev2_logreg_telemetry_gain_over_context__pooled") > 0)
    check("posctl_60_single_channel_s1_found_all_three_both_learners",
          all(sp("sensSPK60_", 1, p, "SPK+TIME", m)["ctx_ci"][0] > 0.5 for p in P3 for m in ("hgb", "logreg")))
    check("posctl_60_single_channel_s2_not_all_three",
          all(sum(sp("sensSPK60_", 2, p, "SPK+TIME", m)["ctx_ci"][0] > 0.5 for p in P3) < 3 for m in ("hgb", "logreg")))
    check("lr_selfz_s2_standing_below_planted70_but_overlapping",
          sp("sensSPK70_", 2, "within_standing", "SELFZ+SPK+TIME", "logreg")["ctx_ci"][0] > 0.5
          and ca(2, "within_standing", "SELFZ+TIME", "logreg") < sp("sensSPK70_", 2, "within_standing", "SELFZ+SPK+TIME", "logreg")["ctx_auc"]
          and cci(2, "within_standing", "SELFZ+TIME", "logreg")[1] >= sp("sensSPK70_", 2, "within_standing", "SELFZ+SPK+TIME", "logreg")["ctx_ci"][0])
    check("lr_selfz_s2_planted70_two_of_three", sum(sp("sensSPK70_", 2, p, "SELFZ+SPK+TIME", "logreg")["ctx_ci"][0] > 0.5 for p in P3) == 2)
    check("cross_MT_ctx_ci_include_0.5_s2", all(cci(2, p, "MOT+TIME")[0] <= 0.5 <= cci(2, p, "MOT+TIME")[1] for p in CROSS_P))
    check("ZT_lr_s1_no_ctx_ci_above_0.5_same_setting", all(cci(1, p, "SELFZ+TIME", "logreg")[0] <= 0.5 for p in WITHIN_P))
    check("ZT_cross_s2_below_TIME", all(a(2, p, "SELFZ+TIME") < a(2, p, "TIME") for p in CROSS_P))
    check("pooled_recovers_most_of_shift_loss_s2",
          all((P[f"sev2__{t}__MOT+TIME__hgb"]["auc"] - a(2, x, "MOT+TIME")) > 0.5 * (a(2, w, "MOT+TIME") - a(2, x, "MOT+TIME"))
              for t, w, x in (("seated", "within_seated", "standing_to_seated"), ("standing", "within_standing", "seated_to_standing"),
                              ("race", "within_race", "flight_to_race"), ("flight", "within_flight", "race_to_flight"))))
    nl = pd.DataFrame(json.load(open(R / "null.json")))
    outside = sorted(f"{lab}/{pr}/{fs}" for (lab, pr, fs), dd in nl.groupby(["label", "protocol", "features"]) if fs != "CTX"
                     and not (dd.context_stratified_auc.quantile(0.025) <= C[f"{lab}__{pr}__{fs}__hgb"]["ctx_auc"]
                              <= dd.context_stratified_auc.quantile(0.975)))
    out["null_outside_range"] = outside
    pn = {}
    for (lab, pr, fs), dd in nl.groupby(["label", "protocol", "features"]):
        if fs != "CTX":
            cx = dd.context_stratified_auc.to_numpy(); pn[(lab, pr, fs)] = (1 + (cx >= C[f"{lab}__{pr}__{fs}__hgb"]["ctx_auc"]).sum()) / (1 + len(cx))
    from fa_stats import holm
    check("no_null_comparison_survives_holm", min(holm({"/".join(k): v for k, v in pn.items()}).values()) >= 0.05)
    nlr = pd.DataFrame(json.load(open(R / "null_lr.json"))); pl = {}
    for (lab, pr, fs), dd in nlr.groupby(["label", "protocol", "features"]):
        if fs != "CTX":
            cx = dd.context_stratified_auc.to_numpy(); pl["/".join((lab, pr, fs))] = (1 + (cx >= C[f"{lab}__{pr}__{fs}__logreg"]["ctx_auc"]).sum()) / (1 + len(cx))
    check("no_lr_null_comparison_survives_holm", min(holm(pl).values()) >= 0.05)
    out["null_perm_p"] = {"/".join(k): round(v, 3) for k, v in pn.items()}
    check("ctl_s2_ws_close_to_real", abs(C["control_sev2__within_seated__MOT__hgb"]["auc"] - a(2, "within_seated", "MOT")) < 0.05)
    check("sens_s2_s2t_all_below_0.5", all(C[f"{t}sev2__seated_to_standing__{f}__{m}"]["auc"] < 0.5 for t, f, m in
          (("sensFPS_", "MOT+FPS+TIME", "hgb"), ("sensMLP_", "MOT+TIME", "mlp"), ("sensDEMO_", "MOT+TIME+DEMO", "hgb"),
           ("sensCLEAN_", "MOT+TIME", "hgb"), ("sensSESSGROUP_", "MOT+TIME", "hgb"), ("sensW10_", "MOT+TIME", "hgb"),
           ("sensW60_", "MOT+TIME", "hgb"), ("sensQUAT_", "MOTQ+TIME", "hgb"))))
    check("sens_s2_ws_ctx_never_above_0.5", all(C[f"{t}sev2__within_seated__{f}__{m}"]["ctx_auc"] < 0.5 for t, f, m in
          (("sensFPS_", "MOT+FPS+TIME", "hgb"), ("sensMLP_", "MOT+TIME", "mlp"), ("sensDEMO_", "MOT+TIME+DEMO", "hgb"),
           ("sensCLEAN_", "MOT+TIME", "hgb"), ("sensSESSGROUP_", "MOT+TIME", "hgb"), ("sensW10_", "MOT+TIME", "hgb"),
           ("sensW60_", "MOT+TIME", "hgb"), ("sensQUAT_", "MOTQ+TIME", "hgb"))))
    # checks added in the final revision
    nlo = pd.DataFrame(json.load(open(R / "null.json")))

    def ovr(lab, pr, fs):
        dd = nlo[(nlo.label == f"sev{lab}") & (nlo.protocol == pr) & (nlo.features == fs)].overall_auc
        return dd.quantile(0.025), dd.quantile(0.975)
    check("null_overall_s2_MT_observed_inside_range_all_three",
          all(ovr(2, p, "MOT+TIME")[0] <= a(2, p, "MOT+TIME") <= ovr(2, p, "MOT+TIME")[1] for p in P3))
    check("evidence_of_absence_s1_lr_raw_ws_pool_not_wt",
          all(cci(1, p, "MOT+TIME", "logreg")[1] < sp("sensSPK70_", 1, p, "MOT+SPK+TIME", "logreg")["ctx_ci"][0] for p in ("within_seated", "pooled"))
          and not cci(1, "within_standing", "MOT+TIME", "logreg")[1] < sp("sensSPK70_", 1, "within_standing", "MOT+SPK+TIME", "logreg")["ctx_ci"][0])
    check("rowwise_MT_ctx_near_1", all(C[f"rowwise_sev{l}__within_curated__MOT+TIME__hgb"]["ctx_auc"] > 0.99 for l in (1, 2)))
    check("cross_MT_ctx_intervals_include_0.5_and_points_near_0.5", all(0.45 <= ca(2, p, "MOT+TIME") <= 0.65 for p in CROSS_P))
    # counts stated in words in the text
    check("words_three_cross_gains_positive_two_significant",
          len(out["gainC_positive_cross"]) == 3 and out["gainC_sig_positive_cross"] == ["standing_to_seated", "race_to_flight"])
    check("words_gain_sig_negative_seated_racing_pooled", out["gainC_sig_negative_same_setting"] == ["within_seated", "within_race", "pooled"])
    check("words_shift_cost_significant_only_seated", out["shift_cost_sig_targets"] == ["seated"])
    check("words_three_s2_cell_ctx_intervals_above_0.5",
          sum(1 for p in CELL_P for f in ("MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME") for m in ("hgb", "logreg")
              if f"sev2__{p}__{f}__{m}" in C and ("sev2", p) not in NONEST and C[f"sev2__{p}__{f}__{m}"]["ctx_ci"][0] > 0.5) == 3)
    check("appendix_lr_cross_game_exceptions_below_TIME",
          a(2, "flight_to_race", "MOT+CTX+TIME", "logreg") < a(2, "flight_to_race", "TIME", "logreg") and
          all(a(2, p, "SELFZ+TIME", "logreg") < a(2, p, "TIME", "logreg") for p in ("flight_to_race", "race_to_flight")))
    check("s2_no_telemetry_ctx_ci_above_0.5_same_setting_any_learner",
          all(cci(2, p, f, m)[0] <= 0.5 for p in WITHIN_P for f in ("MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME") for m in ("hgb", "logreg")))
    check("s2_raw_cell_ctx_above_0.5_only_standing_race_lr",
          [(p, f, m) for p in CELL_P for f in ("MOT+TIME", "MOT+CTX+TIME") for m in ("hgb", "logreg")
           if f"sev2__{p}__{f}__{m}" in C and C[f"sev2__{p}__{f}__{m}"]["ctx_ci"][0] > 0.5 and ("sev2", p) not in NONEST]
          == [("cell_standing_race", "MOT+TIME", "logreg")])
    check("s2_selfz_ctx_ci_include_0.5_same_setting", all(cci(2, p, "SELFZ+TIME")[0] <= 0.5 <= cci(2, p, "SELFZ+TIME")[1] for p in WITHIN_P))
    # ---- revision 2: statements of the rewritten Results (main learner HGB-reg)
    if "sev2__race_to_flight__MOT+CTX+TIME__hgbreg" in C:
        L3 = ("hgbreg", "hgb", "logreg")
        check("r2_CT_hgbr_beats_raw_telemetry_all_three_learners_same_setting_s2",
              all(a(2, p, "CTX+TIME", "hgbreg") > max(a(2, p, f, m) for f in raw for m in L3) for p in WITHIN_P))
        zbeat = [(p, m) for p in WITHIN_P for m in L3 if a(2, p, "SELFZ+TIME", m) > a(2, p, "CTX+TIME", "hgbreg")]
        check("r2_selfz_beats_CT_hgbr_once_flight_hgb_below_0.01",
              zbeat == [("within_flight", "hgb")] and a(2, "within_flight", "SELFZ+TIME", "hgb") - a(2, "within_flight", "CTX+TIME", "hgbreg") < 0.01)
        check("r2_curated_MT_hgbr_below_CT_hgbr_and_StaticFrame",
              a(2, "within_curated", "MOT+TIME", "hgbreg") < a(2, "within_curated", "CTX+TIME", "hgbreg")
              and (staticframe_auc is None or a(2, "within_curated", "MOT+TIME", "hgbreg") < staticframe_auc))
        check("r2_gainC_hgbr_negative_all_five_same_setting",
              all(d(f"sev2_hgbreg_telemetry_gain_over_context__{p}") < 0 for p in WITHIN_P[1:]))
        check("r2_ctl_s1_wt_below_real", C["control_sev1__within_standing__MOT__hgb"]["auc"] < a(1, "within_standing", "MOT"))
        check("r2_hgb_MT_ctx_ci_below_0.5_within_seated_s2", cci(2, "within_seated", "MOT+TIME", "hgb")[1] < 0.5)
        check("r2_shift_ctx_cost_not_positive_any_target_hgbr",
              all(T[f"sev2_hgbreg_shift_cost_MOT+TIME__{t}"]["context"]["ci"][0] <= 0 for t in ("seated", "standing", "race", "flight")))
        check("r2_shift_overall_cost_positive_all_targets_hgbr",
              all(d(f"sev2_hgbreg_shift_cost_MOT+TIME__{t}") > 0 for t in ("seated", "standing", "race", "flight")))
        check("r2_cross_CT_and_MCT_hgbr_below_0.5", all(a(2, p, f, "hgbreg") < 0.5 for p in CROSS_P for f in ("CTX+TIME", "MOT+CTX+TIME")))
        check("r2_cross_MT_hgbr_below_0.5", all(a(2, p, "MOT+TIME", "hgbreg") < 0.5 for p in CROSS_P))
        check("r2_TIME_hgbr_above_0.5_all_ten", all(a(2, p, "TIME", "hgbreg") > 0.5 for p in WITHIN_P + CROSS_P))
        out["r2_gainC_positive_cross_hgbr"] = [p for p in CROSS_P if d(f"sev2_hgbreg_telemetry_gain_over_context__{p}") > 0]
        check("r2_ctl_s2_wt_close_to_real", abs(C["control_sev2__within_standing__MOT__hgb"]["auc"] - a(2, "within_standing", "MOT")) < 0.05)
        # within-context section
        if "sensSPK70_sev2__pooled__MOT+SPK+TIME__hgbreg" in C:
            check("r2_posctl70_single_channel_all_six_all_learners",
                  all(sp("sensSPK70_", l, p, "SPK+TIME", m)["ctx_ci"][0] > 0.5 for l in (1, 2) for p in P3 for m in L3))
            check("r2_posctl60_single_channel_s1_all_three_all_learners",
                  all(sp("sensSPK60_", 1, p, "SPK+TIME", m)["ctx_ci"][0] > 0.5 for p in P3 for m in L3))
            check("r2_posctl70_raw_s2_not_found_within_campaigns_any_learner",
                  all(sp("sensSPK70_", 2, p, "MOT+SPK+TIME", m)["ctx_ci"][0] <= 0.5 for p in ("within_seated", "within_standing") for m in L3))
            check("r2_posctl70_raw_s2_hgbr_pooled_only",
                  [p for p in P3 if sp("sensSPK70_", 2, p, "MOT+SPK+TIME", "hgbreg")["ctx_ci"][0] > 0.5] == ["pooled"])
            nci = lambda m, lab: sum(sp("sensSPK70_", lab, p, "MOT+SPK+TIME", m)["ctx_ci"][0] > 0.5 for p in P3)
            check("r2_posctl_lr_as_well_as_hgbr_default_worse",
                  all(nci("logreg", l) == nci("hgbreg", l) and nci("hgb", l) < nci("hgbreg", l) for l in (1, 2)))
            check("r2_rule_hgbr_against_only_s1_pooled_raw", RULE.get("hgbreg", {}).get("against") == [(1, "pooled", "MOT+TIME")])
            check("r2_rule_lr_against_s1_raw_seated_and_pooled",
                  RULE.get("logreg", {}).get("against") == [(1, "within_seated", "MOT+TIME"), (1, "pooled", "MOT+TIME")])
        check("r2_s2_no_telemetry_ctx_ci_above_0.5_same_setting_three_learners",
              all(cci(2, p, f, m)[0] <= 0.5 for p in WITHIN_P for f in ("MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME") for m in L3))
        check("r2_s1_raw_no_ctx_ci_above_0.5_same_setting_hgbr_lr",
              all(cci(1, p, f, m)[0] <= 0.5 for p in WITHIN_P for f in raw for m in ("hgbreg", "logreg")))
        above = sorted((l, p, f) for l in (1, 2) for p in WITHIN_P + CROSS_P for f in ("MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME")
                       if cci(l, p, f, "hgbreg")[0] > 0.5)
        out["r2_hgbr_ctx_intervals_above_0.5"] = above
        check("r2_hgbr_exactly_four_signals", above == [(1, "within_flight", "SELFZ+TIME"), (1, "within_race", "SELFZ+TIME"),
                                                       (2, "standing_to_seated", "MOT+CTX+TIME"), (2, "standing_to_seated", "MOT+TIME")])
        check("r2_signals_small_below_0.6", all(ca(l, p, f, "hgbreg") < 0.6 for l, p, f in above))
        zt = lambda p: T[f"sev1_hgbreg_selfnorm_gain_over_time__{p}"]["context"]["ci"][0]
        check("r2_selfz_minus_time_excludes_zero_only_race_narrowly", zt("within_flight") <= 0 < zt("within_race") < 0.01)
        check("r2_cells_s2_above_at_most_16_groups",
              all(C[f"sev2__{p}__{f}__{m}"]["n_groups"] <= 16 for p in CELL_P for f in ("MOT+TIME", "MOT+CTX+TIME", "SELFZ+TIME")
                  for m in ("hgb", "logreg") if f"sev2__{p}__{f}__{m}" in C and ("sev2", p) not in NONEST and C[f"sev2__{p}__{f}__{m}"]["ctx_ci"][0] > 0.5))
        nlc = pd.DataFrame(json.load(open(R / "null.json")))
        check("r2_null_CTX_exactly_0.5", (nlc[nlc.features == "CTX"].context_stratified_auc - 0.5).abs().max() < 0.005)
        if "sensVRLENS_sev2__within_curated__VRLENS__stack" in C:
            v = C["sensVRLENS_sev2__within_curated__VRLENS__stack"]
            check("r2_vrlens_grouped_below_CT_hgbr", v["auc"] < a(2, "within_curated", "CTX+TIME", "hgbreg"))
            check("r2_vrlens_ctx_ci_just_above_0.5", 0.5 < v["ctx_ci"][0] < 0.52)
            vr = json.load(open(R / "vrlens.json"))
            check("r2_vrlens_rowwise_acc_close_to_published_81", abs(vr["recordwise_10fold"]["accuracy_4class"] - 0.81) < 0.06)
    json.dump(out, open(R / "claims.json", "w"), indent=1)
    bad = [k for k, v in out.items() if v is False]
    print("CLAIM CHECKS:", "all passed" if not bad else f"FAILED: {bad}")
    for k, v in out.items():
        if not isinstance(v, bool):
            print("   ", k, v)
    return out


def supp_table(st, ops):
    """Supplementary Table S1: every model of the final run."""
    L = [r"\begin{longtable}{@{}lllllcccrrrc@{}}",
         r"\caption{Every model of the earlier learners (final run of the earlier analysis, the label-shift control including the single cells, the record-wise reference, the earlier positive control, the VR-LENS re-implementation and its ablation, and the sensitivity analyses). AUC: overall (within-fold) ROC-AUC; Ctx: context-stratified AUC; both with 95\,\% participant-group bootstrap intervals. PR: PR-AUC (chance = prevalence). Rec, FPR, Prec: recall, false-positive rate and precision (\%) at the threshold set for a 10\,\% false-alarm budget on training groups. Last column: MCC at that threshold / median [IQR] of per-group (within-participant) AUCs.}\\",
         r"\toprule", r"Run & Label & Protocol & Features & Model & AUC [CI] & Ctx [CI] & PR & Rec & FPR & Prec & MCC / group AUC \\",
         r"\midrule", r"\endhead"]
    for key in sorted(st["configs"]):
        c = st["configs"][key]
        tag = next(t for t in sorted(TAG, key=len, reverse=True) if key.startswith(t))
        lab, pr, fs, mn = key[len(tag):].split("__")
        o = ops.get(key, {})
        if (lab, pr) in NONEST:
            L.append(" & ".join([(tag.rstrip("_") or "main").replace("_", r"\_"), "$\\geq$" + lab[-1], pr.replace("_", r"\_"), fs, mn,
                                 "n/e", "n/e", "--", "--", "--", "--", "-- / --"]) + r" \\")
            continue
        grp = f"{a2(o['per_group_auc_median'])} {ci2(o['per_group_auc_iqr'])}" if o.get("per_group_auc_iqr") else "--"
        op = [pct(o[k]) if k in o else "--" for k in ("op_recall", "op_fpr", "op_precision")]
        mcc = a2(o["op_mcc"]) if "op_mcc" in o else "--"
        L.append(" & ".join([(tag.rstrip("_") or "main").replace("_", r"\_"), "$\\geq$" + lab[-1], pr.replace("_", r"\_"),
                             fs, mn, *((f"{c['auc']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]", f"{c['ctx_auc']:.3f} [{c['ctx_ci'][0]:.3f}, {c['ctx_ci'][1]:.3f}]")
                                       if tag == "rowwise_" else (f"{a2(c['auc'])} {ci2(c['ci'])}", f"{a2(c['ctx_auc'])} {ci2(c['ctx_ci'])}")),
                             a2(o["pr_auc"]) if "pr_auc" in o else "--", *op, f"{mcc} / {grp}"]) + r" \\")
    L += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(L)


def supp_firstrun(st, old):
    """Supplementary Table S4: the primary family in the first (leave-one-group-out) run and in the final run."""
    L = [r"\begin{table}[h]", r"\centering", r"\footnotesize",
         r"\caption{Comparison family (\sevtwo{}, default HGB) in the first run (leave-one-group-out folds, scores pooled across folds) and in the final run "
         r"(repeated grouped 5-fold, within-fold AUC): difference in overall ROC-AUC and Holm-adjusted $p$. "
         r"The family was fixed before the final run, after these first-run values had been seen.}",
         r"\label{tab:supp_first}", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{First run} & \multicolumn{2}{c}{Final run} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Comparison & $\Delta$AUC & $p_\mathrm{Holm}$ & $\Delta$AUC & $p_\mathrm{Holm}$ \\", r"\midrule"]
    tn = {"standing": "Shift cost on standing", "seated": "Shift cost on seated", "race": "Shift cost on racing", "flight": "Shift cost on flight"}
    names = [(tn[t], f"sev2_hgb_shift_cost_MOT+TIME__{t}") for t in ("standing", "seated", "race", "flight")] + \
            [("Gain over context, " + PR_NAME[pr].lower().replace("all cells (pooled)", "pooled"), f"sev2_hgb_telemetry_gain_over_context__{pr}")
             for pr in ["within_seated", "within_standing", "within_flight", "within_race", "pooled"] + CROSS_P]
    fmt = lambda x: f"{x:+.2f}".replace("-", "$-$")
    pf = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
    for nm, k in names:
        o, n = old["tests"][k]["pooled"], st["tests"][k]["pooled"]
        mark = "$^{*}$" if (o["p_holm"] < 0.05) != (n["p_holm"] < 0.05) else ""
        L.append(f"{nm}{mark} & {fmt(o['delta'])} & {pf(o['p_holm'])} & {fmt(n['delta'])} & {pf(n['p_holm'])} \\\\")
    L += [r"\bottomrule", r"\multicolumn{5}{@{}l}{$^{*}$Significance at the 5\,\% level after Holm correction differs between the runs.}",
          r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def supp_strata(V, F, P):
    """Supplementary table (revision 2, Stage 2d): behind every context-stratified AUC of the
    main table, the number of strata (fold x campaign x game x 30-s bin) that hold both
    positive and negative target records, and the number of positive-negative pairs, per
    repeat (mean over the 5 repeats). Strata depend on the folds and labels only, so they are
    the same for every feature set and learner of a protocol."""
    tbin = np.minimum((F.t_elapsed.to_numpy() // 30).astype(int), 9).astype(str)
    ctx = np.char.add(np.char.add((F.campaign + "|" + F.game).to_numpy().astype(str), "|"), tbin)
    from fa_run import domains
    D = domains(F.assign(curated=F.curated.astype(bool)))
    L = [r"\begin{table}[h]", r"\centering", r"\footnotesize",
         r"\caption{Strata behind the context-stratified AUCs: per protocol and label, the number of strata "
         r"(test fold $\times$ campaign $\times$ game $\times$ 30-s exposure bin) that contain both positive and negative target records, "
         r"and the number of positive--negative pairs compared, per repeat (mean over the five repeats). Strata depend only on the folds "
         r"and the labels, so they are the same for every feature set and learner of a protocol.}",
         r"\label{tab:supp_strata}", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{\sevtwo{}} & \multicolumn{2}{c}{\sevone{}} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Protocol & Strata & Pairs & Strata & Pairs \\", r"\midrule"]
    out = {}
    for pr in WITHIN_P + CROSS_P:
        tgt = D[pr][1]; row = [PR_NAME[pr]]
        for lab in (2, 1):
            y = (F.severity.to_numpy() >= lab).astype(int)
            z = np.load(P / f"sev{lab}__{pr}__MOT+TIME__hgb.npz"); FO = z["fold"]
            ns, npairs = [], []
            for r in range(FO.shape[0]):
                m = tgt & (FO[r] >= 0)
                key = np.char.add(np.char.add(FO[r][m].astype(str), "|"), ctx[m]); yy = y[m]
                n = pairs = 0
                for k in np.unique(key):
                    q = key == k; a = int(yy[q].sum()); b = int(q.sum()) - a
                    if a and b:
                        n += 1; pairs += a * b
                ns.append(n); npairs.append(pairs)
            V.put(f"strata.s{lab}.{PR[pr]}.n", f"{np.mean(ns):.0f}"); V.put(f"strata.s{lab}.{PR[pr]}.pairs", n_(round(np.mean(npairs))))
            out[(lab, pr)] = (np.mean(ns), np.mean(npairs))
            row += [f"{np.mean(ns):.0f}", n_(round(np.mean(npairs)))]
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    allv = [v[0] for v in out.values()]
    V.put("strata.min", f"{min(allv):.0f}"); V.put("strata.max", f"{max(allv):.0f}")
    return "\n".join(L)


if __name__ == "__main__":
    main(*sys.argv[1:5])
