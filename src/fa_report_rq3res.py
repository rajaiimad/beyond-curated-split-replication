"""
fa_report_rq3res.py -- resolution of the primary RQ3 tests (post hoc, descriptive; no model is fitted).

Reads only existing result files: results/rev3_null_lroff.json (label-transplant null of LR+offset, MOT,
1000 permutations, models refitted), results/rev3_posctl_seeds.json (planted channel, 20 realizations) and
results/rev3_stats.json (observed AUCs). Written after all results were known, to state on the scale of the
gain how large an observed gain each test of families A and B would have needed. It changes no result, test,
label or decision rule.

For each test (label x protocol x estimand, B = context-stratified, W = within-session):
  bar for p < 0.05      the smallest observed gain with unadjusted permutation p = (1 + k)/(1 + n) < 0.05:
                        the gain must exceed the (k + 1)-th largest null gain, k the largest count with
                        (1 + k)/(1 + n) < 0.05 (k = 49 for n = 1000). No test could pass the rule below it.
  bar for p < 0.05/14   the same for p < 0.05/14 (k = 2 for n = 1000): above it, the Holm-adjusted p is
                        below 0.05 whatever the other 13 tests of the family show. The Holm criterion of a test
                        lies between the two bars, depending on how many other tests carry a signal.
  Neither bar is a power estimate (power: Table VIII), and a declaration also needs the bootstrap interval of
  the gain above 0, which the bars do not include.
  planted median        median gain over the 20 realizations of the planted channel at within-stratum AUC
                        0.70 and 0.80 (as in Table VIII).
Also: the mean overall-AUC gain under the same null (the overall gain is not a tested estimand).

Writes NEW files only:
  results/rq3_resolution.json                       full-precision rows
  results/paper/values_rq3res.json (and .tex)       printed values, keys "r3r.*" (values.json unchanged)
  results/paper/paper_keys_rq3res.json              the keys the manuscript and supplement print
  results/paper/tab_rq3_resolution.tex              Supplementary Table S38
  results/claims_rq3res.json                        qualitative statements checked at full precision
Usage: python fa_report_rq3res.py <project dir>
"""
from __future__ import annotations
import json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fa_report import PR, PR_NAME, Values, d2, pfmt  # noqa: E402

FAM = ["within_seated", "within_standing", "pooled", "cell_seated_flight", "cell_seated_race",
       "cell_standing_flight", "cell_standing_race"]
EST = {"c": "gain_ctx", "w": "gain_session"}
ALPHA, M = 0.05, 14
LEVELS = (0.70, 0.80)
ORD = {1: "largest", 2: "second-largest", 3: "third-largest", 50: "50th-largest"}


def bar(null, alpha):
    """(bar, k): the observed gain must exceed `bar` for p = (1 + k)/(1 + n) < alpha; None if unattainable."""
    v = np.sort(null)[::-1]; n = len(v)
    k = math.ceil(alpha * (1 + n) - 1) - 1                      # largest k with (1 + k)/(1 + n) < alpha
    return (None, k) if k < 0 else (float(v[k]), k)


def compute(R):
    nul = json.load(open(R / "rev3_null_lroff.json"))
    pc = json.load(open(R / "rev3_posctl_seeds.json"))["rows"]
    C = json.load(open(R / "rev3_stats.json"))["configs"]
    N = defaultdict(list); O = defaultdict(list); G = defaultdict(list)
    for r in nul:
        if r["features"] != "CTXB+MOT" or r["learner"] != "lroff":
            continue
        for e, f in EST.items():
            if r[f] is not None and np.isfinite(r[f]):
                N[(r["label"], r["protocol"], e)].append(r[f])
        if r["overall_auc"] is not None and r["base_overall_auc"] is not None and np.isfinite(r["overall_auc"] - r["base_overall_auc"]):
            O[(r["label"], r["protocol"])].append(r["overall_auc"] - r["base_overall_auc"])
    for r in pc:
        if r["learner"] == "lroff" and r["features"] == "CTXB+MOT+SPK" and round(r["level"], 2) in LEVELS:
            for e, f in EST.items():
                if r[f] is not None and np.isfinite(r[f]):
                    G[(r["label"], r["protocol"], e, round(r["level"], 2))].append(r[f])
    rows = []
    for lab in ("sev2", "sev1"):
        for pr in FAM:
            tel, base = C[f"{lab}__{pr}__CTXB+MOT__lroff"], C[f"{lab}__{pr}__CTXB__ctxb"]
            o = np.array(O[(lab, pr)])
            row = {"label": lab, "protocol": pr,
                   "overall": {"observed": tel["overall"]["auc"] - base["overall"]["auc"], "null_n": len(o),
                               "null_mean": float(o.mean()), "null_lo": float(np.percentile(o, 2.5)),
                               "null_hi": float(np.percentile(o, 97.5))}}
            for e, key in (("c", "ctx"), ("w", "session")):
                v = np.array(N[(lab, pr, e)]); b05, k05 = bar(v, ALPHA); b14, k14 = bar(v, ALPHA / M)
                row[e] = {"observed": tel[key]["auc"] - base[key]["auc"], "null_n": len(v), "null_mean": float(v.mean()),
                          "bar_p05": b05, "k_p05": k05, "bar_p05_14": b14, "k_p05_14": k14,
                          "min_attainable_p": 1 / (1 + len(v)),
                          **{f"planted_median_{int(round(100 * lv))}": float(np.median(G[(lab, pr, e, lv)])) for lv in LEVELS},
                          **{f"planted_n_{int(round(100 * lv))}": len(G[(lab, pr, e, lv)]) for lv in LEVELS}}
            rows.append(row)
    return rows


def lab_s(lab):
    return lab.replace("sev", "s")


def values(V, rows):
    V.put("r3r.alpha14", f"{ALPHA / M:.4f}")
    for row in rows:
        b = f"r3r.{lab_s(row['label'])}.{PR[row['protocol']]}"
        for e in ("c", "w"):
            x = row[e]
            V.put(f"{b}.{e}.b05", d2(x["bar_p05"]))
            V.put(f"{b}.{e}.b14", "–" if x["bar_p05_14"] is None else d2(x["bar_p05_14"]))
            V.put(f"{b}.{e}.pm70", d2(x["planted_median_70"])); V.put(f"{b}.{e}.pm80", d2(x["planted_median_80"]))
            V.put(f"{b}.{e}.nperm", str(x["null_n"])); V.put(f"{b}.{e}.k05", str(x["k_p05"]))
            V.put(f"{b}.{e}.k14", "–" if x["bar_p05_14"] is None else str(x["k_p05_14"]))
            V.put(f"{b}.{e}.minp", pfmt(x["min_attainable_p"]))
        V.put(f"r3r.o.nm.{lab_s(row['label'])}.{PR[row['protocol']]}", d2(row["overall"]["null_mean"]))
    full = [r for r in rows if r["c"]["null_n"] == 1000]
    V.put("r3r.k05", str(full[0]["c"]["k_p05"])); V.put("r3r.k14", str(full[0]["c"]["k_p05_14"]))
    V.put("r3r.rank05", ORD[full[0]["c"]["k_p05"] + 1]); V.put("r3r.rank14", ORD[full[0]["c"]["k_p05_14"] + 1])
    for e in ("c", "w"):
        b05 = [r[e]["bar_p05"] for r in rows]; b14 = [r[e]["bar_p05_14"] for r in rows if r[e]["bar_p05_14"] is not None]
        V.put(f"r3r.{e}.b05.min", d2(min(b05))); V.put(f"r3r.{e}.b05.max", d2(max(b05)))
        V.put(f"r3r.{e}.b14.min", d2(min(b14))); V.put(f"r3r.{e}.b14.max", d2(max(b14)))
        nm = [r[e]["null_mean"] for r in rows]
        V.put(f"r3r.{e}.nm.min", d2(min(nm))); V.put(f"r3r.{e}.nm.max", d2(max(nm)))
    onm = [r["overall"]["null_mean"] for r in rows]
    V.put("r3r.o.n", str(len(onm))); V.put("r3r.o.nneg", str(sum(x < 0 for x in onm)))
    V.put("r3r.o.nm.min", d2(min(onm))); V.put("r3r.o.nm.max", d2(max(onm)))


def interval_readings(R):
    """Planted tests (both offset learners): declarations under "interval above 0" (used) and "interval excludes 0"
    (the wording of the plan), the tests where they differ, and whether any count of combinations declared in at
    least half of the realizations changes."""
    pc = json.load(open(R / "rev3_posctl_seeds.json"))["rows"]
    ok = lambda x: x is not None and np.isfinite(x)
    out = {}
    for ln in ("lroff", "xgboff"):
        A = defaultdict(list); E = defaultdict(list); n = 0
        for r in pc:
            if r["learner"] != ln or r["features"] != "CTXB+MOT+SPK":
                continue
            for e in ("ctx", "session"):
                h, lo, hi = r.get(f"holm_{e}"), r.get(f"ci_lo_{e}"), r.get(f"ci_hi_{e}")
                if not ok(h):
                    continue
                n += 1; k = (r["label"], r["protocol"], e, round(r["level"], 2))
                A[k].append(h < 0.05 and ok(lo) and lo > 0)
                E[k].append(h < 0.05 and ((ok(lo) and lo > 0) or (ok(hi) and hi < 0)))
        diff = {k: (sum(E[k]) - sum(A[k]), len(A[k]), sum(A[k])) for k in A if E[k] != A[k]}
        cnt = lambda D, e, lv: sum(np.mean(D[k]) >= 0.5 for k in D if k[2] == e and k[3] == lv and k[:2] != ("sev2", "cell_seated_flight"))
        same = all(cnt(A, e, lv) == cnt(E, e, lv) for e in ("ctx", "session") for lv in (0.55, 0.6, 0.65, 0.7, 0.8))
        out[ln] = {"n": n, "diff": diff, "n50_same": same}
    return out


def claims(rows, old, R_):
    """Qualitative statements of the text, at full precision."""
    R = {(r["label"], r["protocol"]): r for r in rows}
    out = {}
    # the plan states the interval criterion as "excludes 0", the analysis used "above 0" (lower bound > 0)
    ir = interval_readings(R_)
    out["interval_readings_agree_in_every_planted_test_of_the_main_learner"] = ir["lroff"]["n"] > 0 and not ir["lroff"]["diff"]
    out["interval_readings_differ_only_in_one_boosting_entry"] = list(ir["xgboff"]["diff"]) == [("sev1", "cell_seated_flight", "session", 0.8)]
    out["interval_readings_change_no_count_of_combinations_declared_in_half"] = all(x["n50_same"] for x in ir.values())
    ph = [float(v) for k, v in old.items() if k.startswith("r3n.") and ".M.lo." in k and k.endswith((".c.ph", ".w.ph"))]
    out["no_real_primary_test_with_holm_below_005"] = len(ph) == 28 and min(ph) >= 0.05
    out["lone_bar_not_below_unadjusted_bar"] = all(r[e]["bar_p05_14"] is None or r[e]["bar_p05_14"] >= r[e]["bar_p05"]
                                                   for r in rows for e in ("c", "w"))
    out["no_between_session_gain_above_unadjusted_bar"] = all(r["c"]["observed"] <= r["c"]["bar_p05"] for r in rows)
    out["three_within_session_gains_above_unadjusted_bar"] = sum(r["w"]["observed"] > r["w"]["bar_p05"] for r in rows) == 3
    out["no_gain_above_lone_bar"] = all(r[e]["bar_p05_14"] is None or r[e]["observed"] <= r[e]["bar_p05_14"]
                                        for r in rows for e in ("c", "w"))
    out["only_sev2_seated_flight_lone_bar_unattainable"] = \
        [(r["label"], r["protocol"]) for r in rows for e in ("c", "w") if r[e]["bar_p05_14"] is None] == [("sev2", "cell_seated_flight")] * 2
    out["overall_null_mean_negative_all_14"] = all(r["overall"]["null_mean"] < 0 for r in rows)
    out["sev2_same_setting_observed_overall_inside_null_95"] = all(
        R[("sev2", p)]["overall"]["null_lo"] <= R[("sev2", p)]["overall"]["observed"] <= R[("sev2", p)]["overall"]["null_hi"]
        for p in ("within_seated", "within_standing", "pooled"))
    out["within_session_null_mean_negative_all_14"] = all(r["w"]["null_mean"] < 0 for r in rows)
    for lab in ("sev2", "sev1"):
        b = {p: R[(lab, p)]["c"]["bar_p05"] for p in FAM}
        out[f"{lab}_pooled_lowest_between_unadjusted_bar"] = b["pooled"] == min(b.values())
        out[f"{lab}_cells_not_below_campaign_protocols_between_unadjusted_bar"] = \
            min(b[p] for p in FAM[3:]) >= max(b["within_seated"], b["within_standing"])
    # consistency with Table VIII: the between-session combinations whose median planted 0.80 gain exceeds the
    # lone bar are those declared in at least half of the realizations
    above = {(r["label"], r["protocol"]) for r in rows if r["c"]["bar_p05_14"] is not None and r["c"]["planted_median_80"] > r["c"]["bar_p05_14"]}
    pw = {(r["label"], r["protocol"]) for r in rows
          if old.get(f"r3pw.lo.80.c.{lab_s(r['label'])}.{PR[r['protocol']]}", "n/e") not in ("n/e",)
          and float(old[f"r3pw.lo.80.c.{lab_s(r['label'])}.{PR[r['protocol']]}"]) >= 50}
    out["between_lone_bar_vs_planted_080_matches_table_viii"] = above == pw
    return out


PRINTED = {
    "manuscript, Section V-E": ["r3r.o.n", "r3r.o.nneg", "r3r.o.nm.s2.ws", "r3r.o.nm.s2.wt", "r3r.o.nm.s2.pool",
                                "r3pwn.lo.80.c.n50d", "r3r.s2.pool.c.b05", "r3r.s2.pool.c.b14",
                                "r3r.c.b05.min", "r3r.c.b05.max", "r3r.c.b14.min", "r3r.c.b14.max"],
    "manuscript, Table IX": ["r3pwn.lo.80.c.n50d", "r3pwn.lo.80.c.n", "r3r.c.b14.min", "r3r.c.b14.max"],
    "manuscript, Section VII": ["r3r.c.b05.min", "r3r.c.b05.max", "r3r.c.b14.min", "r3r.c.b14.max",
                                "r3pw.nseeds", "r3pwn.lo.80.c.n50d", "r3pwn.lo.80.c.n", "r3pwn.lo.80.w.n50d",
                                "r3pwn.lo.80.w.n"],
    "supplement, Note S6": ["r3r.k05", "r3r.rank05", "r3r.k14", "r3r.rank14", "r3r.alpha14", "r3r.s2.cSF.c.nperm",
                            "r3r.s2.cSF.c.minp", "r3r.w.nm.min", "r3r.w.nm.max", "r3pwn.lo.80.c.n50d",
                            "r3pwn.lo.80.c.n50perm", "r3pwn.lo.80.w.n50perm", "r3pwn.lo.80.w.n50d"],
    "supplement, Note S2": ["r3r.ir.xo.ndiff", "r3r.ir.xo.nreal", "r3r.ir.xo.pctabove", "r3r.ir.xo.pctexcl"],
    "supplement, Note S7": ["r3r.o.n", "r3r.o.nneg", "r3r.o.nm.min", "r3r.o.nm.max", "r3g.s1.pool.M.lo.w.d",
                            "r3n.s1.pool.M.lo.w.p"],
    "supplement, Table S38": ["r3r.rank05", "r3r.rank14", "r3r.s2.cSF.c.nperm", "r3r.s2.cSF.c.minp"],
}


def table(V):
    """Supplementary Table S38 (LaTeX, booktabs, as the other generated tables)."""
    cols = ("b05", "b14", "pm70", "pm80")
    L = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{4}{c}{Between sessions (B)} & \multicolumn{4}{c}{Within sessions (W)} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"Label & Protocol & $p<0.05$ & $p<0.05/14$ & 0.70 & 0.80 & $p<0.05$ & $p<0.05/14$ & 0.70 & 0.80 \\", r"\midrule"]
    for lab, lname in (("s2", r"\sevtwo{}"), ("s1", r"\sevone{}")):
        for i, pr in enumerate(FAM):
            dag = r"$^\dagger$" if (lab, pr) == ("s2", "cell_seated_flight") else ""
            cells = [V.v[f"r3r.{lab}.{PR[pr]}.{e}.{c}"] for e in ("c", "w") for c in cols]
            L.append(f"{lname if i == 0 else ''} & {PR_NAME[pr]}{dag} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if lab == "s2" else r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


def write_defs(vals, path):
    """Value definitions only: \\V itself is defined by values.tex, which must be loaded first."""
    lines = ["% generated by src/fa_report_rq3res.py -- do not edit by hand; load after values.tex, which defines \\V", "\\makeatletter"]
    lines += [f"\\expandafter\\def\\csname fa@{k}\\endcsname{{{v}}}" for k, v in sorted(vals.items())]
    lines.append("\\makeatother")
    Path(path).write_text("\n".join(lines) + "\n")
    Path(path).with_suffix(".json").write_text(json.dumps(dict(sorted(vals.items())), indent=1, ensure_ascii=False))


def main(root):
    root = Path(root); R = root / "results"; G = R / "paper"
    old = json.loads((G / "values.json").read_text())
    if (G / "values_add3.json").exists():
        old.update(json.loads((G / "values_add3.json").read_text()))
    rows = compute(R)
    V = Values(); values(V, rows)
    (k, (nd, nr, na)), = interval_readings(R)["xgboff"]["diff"].items()       # the one boosting entry (see claims)
    V.put("r3r.ir.xo.ndiff", str(nd)); V.put("r3r.ir.xo.nreal", str(nr))
    V.put("r3r.ir.xo.pctabove", f"{100 * na / nr:.0f}"); V.put("r3r.ir.xo.pctexcl", f"{100 * (na + nd) / nr:.0f}")
    clash = sorted(set(V.v) & set(old))
    assert not clash and all(k.startswith("r3r.") for k in V.v), clash
    (R / "rq3_resolution.json").write_text(json.dumps({"note": __doc__.split("Writes NEW")[0].strip(), "rows": rows}, indent=1))
    write_defs(V.v, G / "values_rq3res.tex")
    keys = defaultdict(list)
    for src, ks in PRINTED.items():
        for k in ks:
            keys[k].append(src)
    for k in V.v:                                            # every per-test value is printed in Table S38
        if k.split(".")[-1] in ("b05", "b14", "pm70", "pm80") and k.split(".")[1] in ("s1", "s2"):
            keys[k].append("supplement, Table S38")
    assert all(k in V.v or k in old for k in keys)
    (G / "paper_keys_rq3res.json").write_text(json.dumps(dict(sorted(keys.items())), indent=1, ensure_ascii=False))
    (G / "tab_rq3_resolution.tex").write_text(table(V) + "\n")
    cl = claims(rows, old, R); json.dump(cl, open(R / "claims_rq3res.json", "w"), indent=1)
    print(f"values_rq3res.json: {len(V.v)} values; printed keys {len(keys)}; "
          f"claims: {sum(cl.values())} true, {sum(not v for v in cl.values())} false {[k for k, v in cl.items() if not v]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
