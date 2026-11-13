"""
fa_report_v10.py -- values printed by manuscript and supplement v10 that no earlier report script writes.
Writes NEW files only: results/paper/values_v10.json, results/paper/values_v10.tex and
results/paper/paper_keys_v10.json. No model is fitted here.

Three kinds of value, each under its own key prefix:
  v10a.*  archive facts, counted from data/sessions.csv and (for the logged
          channels) the raw FILE.xml files: sessions by posture, game, gender and age band; field of view;
          recording dates; the logged channels; the two seated sessions outside the curated release; the
          VRSQ files of the curated release under the creators' criterion; the curated sessions against the
          creators' per-participant table ([9], Table 4).
  v10l.*  numbers quoted from the creators' papers and from the VR-LENS paper, typed once here with their
          source (table or page), so that the manuscript takes every numeral from a value file. They are
          not results of this study. Sources (read in full for the v10 revision):
            [9]    Porcino et al., Entertain. Comput. 41 (2022) 100473, read as arXiv:2606.12214 (page images)
            [ICEC] Porcino et al., ICEC 2021, LNCS 13056, pp. 115-126, read as arXiv:2608.07486
            [10]   Porcino, Trevisan, Clua, XR in Games Workshop with IMX 2021 (SBC), PDF of the authors
            [VRL]  Kundu et al., IUI 2023 (VR-LENS), author copy https://par.nsf.gov/servlets/purl/10429771
  v10r.*  the two post hoc runs of src/fa_v10.py (results/v10/v10_runs.json), if present.

Usage: python src/fa_report_v10.py <results root (parent of results/)> <data dir> [<RAW_DATABASE dir>]
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

AGE = {"18 A 36": "1836", "37 A 50": "3750", "+50": "50p"}
GENDER = {"masculino": "m", "feminino": "f"}

# ----------------------------------------------------------------------------- quoted values (not results)
LIT = {
    # [9] abstract and Table 2 (page 4): six protocols, 88 volunteers
    "v10l.c9.protocols": ("6", "[9] abstract; Table 2"), "v10l.c9.volunteers": ("88", "[9] abstract; Table 2"),
    "v10l.c9.p4.n": ("37", "[9] Table 2"), "v10l.c9.p4.m": ("30", "[9] Table 2"), "v10l.c9.p4.f": ("7", "[9] Table 2"),
    "v10l.c9.p4.race": ("12", "[9] Table 2"), "v10l.c9.p4.flight": ("25", "[9] Table 2"),
    "v10l.c9.p4.age1836": ("33", "[9] Table 2"), "v10l.c9.p4.age3750": ("4", "[9] Table 2"), "v10l.c9.p4.age50p": ("0", "[9] Table 2"),
    "v10l.c9.p4.rift": ("37", "[9] Table 2"), "v10l.c9.p4.vive": ("0", "[9] Table 2"),
    "v10l.c9.p5.n": ("35", "[9] Table 2; [ICEC] p. 5"), "v10l.c9.p5.m": ("26", "[9] Table 2; [ICEC] p. 5"), "v10l.c9.p5.f": ("9", "[9] Table 2; [ICEC] p. 5"),
    "v10l.c9.p5.race": ("15", "[9] Table 2"), "v10l.c9.p5.flight": ("20", "[9] Table 2"),
    "v10l.c9.p5.age1836": ("28", "[9] Table 2"), "v10l.c9.p5.age3750": ("4", "[9] Table 2"), "v10l.c9.p5.age50p": ("3", "[9] Table 2"),
    "v10l.c9.p5.rift": ("15", "[9] Table 2"), "v10l.c9.p5.vive": ("20", "[9] Table 2"),
    "v10l.c9.p6.n": ("2", "[9] Table 2"), "v10l.c9.p6.race": ("1", "[9] Table 2"), "v10l.c9.p6.flight": ("1", "[9] Table 2"),
    "v10l.c9.p6.age3750": ("2", "[9] Table 2"), "v10l.c9.p6.min": ("20", "[9] Table 2; Table 4 (ExtraA, ExtraB: 1200 instances)"),
    "v10l.c9.min": ("5", "[9] Table 2; [ICEC] p. 5"),
    "v10l.c9.pilot.n": ("14", "[9] Table 2: P1-P3, 4 + 4 + 6"),
    # VRSQ: [ICEC] p. 5; [9] p. 4 and Table 4
    "v10l.icec.vrsq.race.pos": ("7", "[ICEC] p. 5"), "v10l.icec.vrsq.race.n": ("15", "[ICEC] p. 5"),
    "v10l.icec.vrsq.flight.pos": ("8", "[ICEC] p. 5"), "v10l.icec.vrsq.flight.n": ("20", "[ICEC] p. 5"),
    "v10l.c9.vrsq.zero.race": ("8", "[9] p. 4"), "v10l.c9.vrsq.zero.flight": ("12", "[9] p. 4"),
    # leave-one-subject-out, average test AUC over users: [9] Tables 7 and 8; [ICEC] Tables 2 and 3
    "v10l.c9.loso.race.dt": ("0.58", "[9] Table 7"), "v10l.c9.loso.race.rf": ("0.77", "[9] Table 7"),
    "v10l.c9.loso.flight.dt": ("0.79", "[9] Table 8"), "v10l.c9.loso.flight.rf": ("0.95", "[9] Table 8"),
    "v10l.icec.loso.race.dt": ("0.51", "[ICEC] Table 2"), "v10l.icec.loso.race.rf": ("0.58", "[ICEC] Table 2"),
    "v10l.icec.loso.flight.dt": ("0.62", "[ICEC] Table 3"), "v10l.icec.loso.flight.rf": ("0.72", "[ICEC] Table 3"),
    "v10l.icec.loso.flight.n": ("20", "[ICEC] Table 3: users 1-20"),
    # [10] Section 3 (pp. 2-4)
    "v10l.c10.race.samples": ("3,993", "[10] p. 2"), "v10l.c10.race.n": ("15", "[10] p. 4"),
    "v10l.c10.race.f": ("7", "[10] pp. 3-4"), "v10l.c10.race.m": ("8", "[10] pp. 3-4"),
    "v10l.c10.race.fsamples": ("1,772", "[10] p. 3"), "v10l.c10.race.msamples": ("2,221", "[10] p. 3"),
    # VR-LENS author copy: Section 4.1.3, Section 5 and the reduced-model table
    "v10l.vrl.participants": ("87", "[VRL] Sec. 4.1.3"), "v10l.vrl.used": ("35", "[VRL] Sec. 4.1.3"),
    "v10l.vrl.reduced.acc": ("81", "[VRL] reduced-model table, gameplay row"), "v10l.vrl.reduced.nfeat": ("10", "[VRL] reduced-model table"),
    "v10l.vrl.folds": ("10", "[VRL] Sec. 5"),
}


# [9] Table 4 (page 7, read on the page image): user, gender, instances (records), VRSQ (total)
T4_RACE = [(21, "m", 300, 0), (22, "m", 104, 0), (23, "m", 300, 67.5), (24, "m", 300, 10.83), (25, "m", 300, 80),
           (26, "m", 300, 61.6), (27, "m", 300, 30), (28, "f", 24, 0), (29, "f", 300, 0), (30, "f", 287, 49.9),
           (31, "f", 250, 42.5), (32, "f", 300, 0), (33, "m", 300, 0), (34, "f", 300, 0), (35, "f", 300, 0)]
T4_FLIGHT = [(1, "m", 103, 10.83), (2, "m", 300, 0), (3, "f", 300, 0), (4, "m", 300, 7.5), (5, "m", 102, 10.83),
             (6, "m", 300, 0), (7, "m", 300, 0), (8, "m", 300, 3.3), (9, "f", 300, 4.16), (10, "m", 300, 0),
             (11, "m", 300, 0), (12, "m", 300, 10.83), (13, "m", 52, 3.33), (14, "m", 300, 0), (15, "m", 300, 0),
             (16, "m", 300, 0), (17, "m", 300, 10.83), (18, "m", 300, 0), (19, "m", 300, 0), (20, "m", 300, 0)]


def f2(x):
    return f"{x:.2f}"


def pct1(x):
    return f"{100 * x:.1f}"


def ci2(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def ci_pct(v):
    return f"[{100 * v[0]:.1f}, {100 * v[1]:.1f}]"


def archive_values(DATA, RAWDIR=None):
    V = {}; put = V.__setitem__
    S = pd.read_csv(DATA / "sessions.csv")
    fp = S.fingerprint.str.split("|", expand=True); S["g"] = fp[0].map(GENDER); S["age"] = fp[1].map(AGE)
    S["setting"] = np.where(S.curated_release, "cur", np.where(S.campaign == "standing", "stand", "seatx"))
    for name, d in (("stand", S[S.campaign == "standing"]), ("seat", S[S.campaign == "seated"]), ("cur", S[S.curated_release]),
                    ("seatx", S[(S.campaign == "seated") & ~S.curated_release])):
        put(f"v10a.{name}.n", str(len(d)))
        put(f"v10a.{name}.race", str(int((d.game == "Race").sum()))); put(f"v10a.{name}.flight", str(int((d.game == "Flight").sum())))
        put(f"v10a.{name}.m", str(int((d.g == "m").sum()))); put(f"v10a.{name}.f", str(int((d.g == "f").sum())))
        for a in AGE.values():
            put(f"v10a.{name}.age{a}", str(int((d.age == a).sum())))
    # age differences against the creators' Table 2 (number of participants in other bands)
    for name, p in (("stand", "p4"), ("cur", "p5")):
        diff = sum(abs(int(V[f"v10a.{name}.age{a}"]) - int(LIT[f"v10l.c9.{p}.age{a}"][0])) for a in AGE.values()) // 2
        put(f"v10a.{name}.agediff", str(diff))
    # field of view and recording dates per cell
    for (c, g), d in S.groupby(["campaign", "game"]):
        key = f"v10a.{c}.{g.lower()}"
        vals = sorted(d.fov.round(1).unique())
        put(f"{key}.fov", f"{d.fov.round(1).mode()[0]:.1f}"); put(f"{key}.nfov", str(len(vals)))
        put(f"{key}.ndates", str(d.date.nunique()))
    st = S[S.campaign == "standing"]
    put("v10a.stand.fov.other", f"{st.fov[st.fov.round(1) != st.fov.round(1).mode()[0]].round(0).iloc[0]:.0f}")
    put("v10a.stand.fov.nother", str(int((st.fov.round(1) != st.fov.round(1).mode()[0]).sum())))
    # the two seated sessions outside the curated release
    X = S[(S.campaign == "seated") & ~S.curated_release]
    put("v10a.seatx.allflight", str(bool((X.game == "Flight").all()))); put("v10a.seatx.allmale", str(bool((X.g == "m").all())))
    put("v10a.seatx.allyoung", str(bool((X.age == "1836").all())))
    put("v10a.seatx.len.min", f"{X.duration_s.min():.0f}"); put("v10a.seatx.len.max", f"{X.duration_s.max():.0f}")
    put("v10a.seatx.nvrsq", str(int(X.vrsq_post_total.notna().sum())))
    put("v10a.seatx.nflag", str(int((X.flag_gap | X.flag_config | X.flag_short).sum())))
    put("v10a.seatx.gap", f"{X.max_gap_s.max():.0f}")
    put("v10a.maxlen", f"{S.duration_s.max():.0f}")
    put("v10a.nflagged", str(int((S.flag_gap | S.flag_config | S.flag_short).sum())))
    # VRSQ files of the curated release under the creators' criterion (|post - pre| > 0) and with a non-zero item
    C = S[S.curated_release].copy()
    C["has"] = C.vrsq_post_total.notna()
    C["allzero"] = (C.vrsq_post_raw_sum == 0) & (C.vrsq_pre_total == 0)
    C["pos"] = (C.vrsq_post_total - C.vrsq_pre_total).abs() > 0
    for g, d in C.groupby("game"):
        k = f"v10a.cur.{g.lower()}"
        put(f"{k}.vrsq.files", str(int(d.has.sum()))); put(f"{k}.vrsq.nofile", str(int((~d.has).sum())))
        put(f"{k}.vrsq.nonzero", str(int((d.has & ~d.allzero).sum()))); put(f"{k}.vrsq.pos", str(int((d.has & d.pos).sum())))
    # the creators' per-participant table ([9] Table 4) against the curated sessions in start-time order:
    # gender, number of records (within 3) and the VRSQ score |post - pre| (a missing file counts as no score)
    C = C.assign(t0=pd.to_datetime(C.date + " " + C.start)).sort_values("t0")
    for g, rows in (("race", T4_RACE), ("flight", T4_FLIGHT)):
        d = C[C.game.str.lower() == g]; assert len(d) == len(rows)
        same_g = sum(GENDER[x.split("|")[0]] == r[1] for x, r in zip(d.fingerprint, rows))
        same_n = sum(abs(n - r[2]) <= 3 for n, r in zip(d.n_rows, rows))
        vr = (d.vrsq_post_total - d.vrsq_pre_total).abs().round(2).to_numpy()
        same_v = sum((not np.isnan(v)) and abs(v - r[3]) < 0.6 for v, r in zip(vr, rows))
        V[f"v10a.t4.{g}.n"] = str(len(rows)); V[f"v10a.t4.{g}.gender"] = str(same_g); V[f"v10a.t4.{g}.records"] = str(same_n)
        V[f"v10a.t4.{g}.vrsq"] = str(same_v)
    V["v10a.cur.race.date"] = "6 December 2019"; V["v10a.cur.flight.dates"] = "9 and 10 December 2019"
    assert sorted(C[C.game == "Race"].date.unique()) == ["2019-12-06"]
    assert sorted(C[C.game == "Flight"].date.unique()) == ["2019-12-09", "2019-12-10"]
    # logged channels (raw files): position fields absent, rotation single-valued, per campaign
    if RAWDIR is not None and Path(RAWDIR).is_dir():
        import xml.etree.ElementTree as ET
        nopos = {"seated": 0, "standing": 0}; constrot = {"seated": 0, "standing": 0}
        camp = S.set_index("session_id").campaign
        for sid in S.session_id:
            recs = [{c.tag: (c.text or "") for c in d} for d in ET.parse(Path(RAWDIR) / sid / "FILE.xml").getroot()]
            df = pd.DataFrame(recs)
            nopos[camp[sid]] += int(not any(c.startswith("PlayerPosition") for c in df.columns))
            constrot[camp[sid]] += int(all(df[c].nunique() == 1 for c in ("CameraRotationX", "CameraRotationY", "CameraRotationZ")))
        for c in ("seated", "standing"):
            put(f"v10a.{c}.nopos", str(nopos[c])); put(f"v10a.{c}.constrot", str(constrot[c]))
    return V


def run_values(R):
    f = R / "v10" / "v10_runs.json"; V = {}
    if not f.exists():
        return V
    res = json.load(open(f))
    if "vrlens" in res:
        v = res["vrlens"]; V["v10r.vrl.majority"] = pct1(v["majority_class_share"])
        for unit, u in (("groups", "grp"), ("sessions", "sess")):
            d = v[unit]; k = f"v10r.vrl.{u}"
            V[f"{k}.units"] = str(d["units"]); V[f"{k}.acc"] = pct1(d["accuracy_4class"]); V[f"{k}.acc.ci"] = ci_pct(d["accuracy_4class_ci"])
            for lab in (1, 2):
                s = d[f"sev{lab}"]
                V[f"{k}.s{lab}.auc"] = f2(s["auc"]); V[f"{k}.s{lab}.ci"] = ci2(s["ci"]); V[f"{k}.s{lab}.ctx"] = f2(s["ctx_auc"])
                V[f"{k}.s{lab}.ctxci"] = ci2(s["ctx_ci"])
    if "tab6sess" in res:
        V["v10r.t6s.units"] = str(res["tab6sess"]["units"])
        for key, s in res["tab6sess"]["configs"].items():
            lab = key.split("sev")[1][0]; fs = key.split("__")[2]; k = f"v10r.t6s.{fs}.s{lab}"
            V[f"{k}.auc"] = f2(s["auc"]); V[f"{k}.ci"] = ci2(s["ci"]); V[f"{k}.ctx"] = f2(s["ctx_auc"])
    return V


PRINTED = {}   # filled by register(); key -> list of print locations (written to paper_keys_v10.json)


def register(locations: dict):
    for loc, keys in locations.items():
        for k in keys:
            PRINTED.setdefault(k, []).append(loc)


def main(ROOT, DATA, RAWDIR=None):
    R = ROOT / "results"; G = R / "paper"
    V = archive_values(DATA, RAWDIR)
    V.update({k: v for k, (v, _) in LIT.items()})
    V.update(run_values(R))
    old = {}
    for f in ("values.json", "values_add3.json", "values_rq3res.json", "values_sample.json"):
        if (G / f).exists():
            old.update(json.load(open(G / f)))
    clash = sorted(set(V) & set(old)); assert not clash, clash
    keys_file = ROOT / "results" / "paper" / "paper_keys_v10.json"
    printed = json.load(open(keys_file)) if keys_file.exists() else {}
    missing = [k for k in printed if k not in V and k not in old]
    assert not missing, f"printed keys without a value: {missing}"
    (G / "values_v10.json").write_text(json.dumps(dict(sorted(V.items())), indent=1, ensure_ascii=False))
    (G / "values_v10_sources.json").write_text(json.dumps({k: s for k, (_, s) in sorted(LIT.items())}, indent=1, ensure_ascii=False))
    lines = ["% generated by src/fa_report_v10.py -- do not edit by hand; load after values.tex, which defines \\V", "\\makeatletter"]
    lines += [f"\\expandafter\\def\\csname fa@{k}\\endcsname{{{v}}}" for k, v in sorted(V.items())]
    (G / "values_v10.tex").write_text("\n".join(lines + ["\\makeatother"]) + "\n")
    print(f"values_v10.json: {len(V)} values ({sum(k.startswith('v10a.') for k in V)} archive, "
          f"{sum(k.startswith('v10l.') for k in V)} quoted, {sum(k.startswith('v10r.') for k in V)} from the v10 runs); "
          f"printed keys listed: {len(printed)}")
    return V


if __name__ == "__main__":
    a = [Path(x) for x in sys.argv[1:]]
    main(a[0], a[1], a[2] if len(a) > 2 else None)
