"""
fa_report_sample.py -- the archive's demographic answers, counted (descriptive; no model is fitted).

Reads only data/sessions.csv (written by fa_data.py). Each session carries the answers to the seven demographic
items of the archive, joined in its `fingerprint` column in the order of fa_data.DEMO; identical answers define the
inferred participant groups. The number of people is unknown, so answers are counted by session (all, seated,
standing) and by answer profile (inferred group), never as people. Written after all results were known, to
describe the sample; it changes no result, test, label or decision rule.

Writes NEW files only:
  results/paper/values_sample.json (and .tex)   printed values, keys "smp.*" (values.json unchanged)
  results/paper/paper_keys_sample.json          the keys the manuscript and supplement print
  results/paper/tab_sample.tex                  Supplementary Table S39
  results/claims_sample.json                    qualitative statements checked at full precision
Usage: python fa_report_sample.py <project dir>
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fa_data import DEMO  # noqa: E402
from fa_report import Values  # noqa: E402

# item -> (short key, printed name, [(archive answer, key, printed answer)]); answers translated from Portuguese
ITEMS = {
    "UserGenere": ("gen", "Gender", [("masculino", "m", "male"), ("feminino", "f", "female")]),
    "UserAge": ("age", "Age band (years)", [("18 A 36", "1836", "18–36"), ("37 A 50", "3750", "37–50"),
                                            ("+50", "50p", "over 50")]),
    "UserExperience": ("exp", "Previous VR experience", [("NENHUMA", "none", "none"), ("ALGUMA", "some", "some"),
                                                         ("MUITA", "much", "much")]),
    "UserFlicker": ("flk", "Flicker (UserFlicker)", [("NENHUM", "none", "none"), ("ALGUM", "some", "some")]),
    "UserGlassesUse": ("gls", "Wears glasses", [("SIM", "yes", "yes"), ("NÃO", "no", "no")]),
    "UserVisionProblems": ("vis", "Vision problems", [
        ("NENHUM", "none", "none"), ("MIOPIA", "my", "myopia"), ("ASTIGMATISMO/MIOPIA", "asmy", "astigmatism and myopia"),
        ("ASTIGMATISMO", "as", "astigmatism"), ("HIPERMETROPIA", "hy", "hyperopia"),
        ("ASTIGMATISMO/HIPERMETROPIA", "ashy", "astigmatism and hyperopia")]),
    "UserEyeDominance": ("eye", "Dominant eye", [("DIREITO", "r", "right"), ("ESQUERDO", "l", "left")]),
}
COLS = (("s", "all sessions"), ("st", "seated sessions"), ("sd", "standing sessions"), ("g", "answer profiles"))


def counts(S):
    """{(item key, answer key): {col: count}} and the column totals."""
    A = S.fingerprint.str.split("|", expand=True); A.columns = DEMO
    S = pd.concat([S.reset_index(drop=True), A], axis=1)
    sub = {"s": S, "st": S[S.campaign == "seated"], "sd": S[S.campaign == "standing"], "g": S.drop_duplicates("group_id")}
    assert S.groupby("group_id").fingerprint.nunique().max() == 1          # one answer profile per inferred group
    out = {}
    for item, (ik, _, answers) in ITEMS.items():
        assert set(S[item]) == {a for a, _, _ in answers}, (item, set(S[item]))
        for a, ak, _ in answers:
            out[(ik, ak)] = {c: int((d[item] == a).sum()) for c, d in sub.items()}
    return out, {c: len(d) for c, d in sub.items()}


def values(V, C, N):
    for c, n in N.items():
        V.put(f"smp.n.{c}", n)
    for (ik, ak), d in C.items():
        for c, n in d.items():
            V.put(f"smp.{ik}.{ak}.{c}", n)


PRINTED = {
    "manuscript, Abstract": ["smp.n.s", "smp.gen.m.s", "smp.age.1836.s", "smp.exp.none.s"],
    "manuscript, Section III-B": ["smp.n.s", "smp.gen.m.s", "smp.age.1836.s", "smp.exp.none.s"],
    "supplement, Note S4": ["smp.n.s", "smp.gen.m.s", "smp.age.1836.s", "smp.exp.none.s", "smp.exp.much.st",
                            "smp.exp.much.sd"],
}


def claims(C, N, S):
    """Qualitative statements of the text, at full precision."""
    s = lambda ik, ak, c="s": C[(ik, ak)][c]
    big = S[S.group_id == S.groupby("group_id").size().idxmax()]
    return {
        "largest_profile_spans_both_campaigns_and_both_games": big.campaign.nunique() == 2 and big.game.nunique() == 2,
        "sessions_mostly_male": s("gen", "m") > N["s"] / 2,
        "sessions_mostly_youngest_age_band": s("age", "1836") > N["s"] / 2,
        "sessions_mostly_without_previous_vr_experience": s("exp", "none") > N["s"] / 2,
        "much_vr_experience_only_in_standing_campaign": s("exp", "much", "st") == 0 < s("exp", "much", "sd"),
        "every_item_sums_to_its_column_total": all(sum(C[(ik, ak)][c] for (ik2, ak) in C if ik2 == ik) == N[c]
                                                   for ik in {k[0] for k in C} for c in N),
    }


def table(C, N):
    """Supplementary Table S39 (LaTeX, booktabs, as the other generated tables)."""
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         "Item & Answer & " + " & ".join(f"{name} ({N[c]})" for c, name in COLS) + r" \\", r"\midrule"]
    for item, (ik, iname, answers) in ITEMS.items():
        for i, (_, ak, aname) in enumerate(answers):
            L.append(f"{iname if i == 0 else ''} & {aname} & " + " & ".join(str(C[(ik, ak)][c]) for c, _ in COLS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def write_defs(vals, path):
    """Value definitions only: \\V itself is defined by values.tex, which must be loaded first."""
    lines = ["% generated by src/fa_report_sample.py -- do not edit by hand; load after values.tex, which defines \\V", "\\makeatletter"]
    lines += [f"\\expandafter\\def\\csname fa@{k}\\endcsname{{{v}}}" for k, v in sorted(vals.items())]
    lines.append("\\makeatother")
    Path(path).write_text("\n".join(lines) + "\n")
    Path(path).with_suffix(".json").write_text(json.dumps(dict(sorted(vals.items())), indent=1, ensure_ascii=False))


def main(root):
    root = Path(root); G = root / "results" / "paper"
    old = json.loads((G / "values.json").read_text())
    for f in ("values_add3.json", "values_rq3res.json"):
        if (G / f).exists():
            old.update(json.loads((G / f).read_text()))
    S = pd.read_csv(root / "data" / "sessions.csv")
    C, N = counts(S)
    V = Values(); values(V, C, N)
    clash = sorted(set(V.v) & set(old))
    assert not clash and all(k.startswith("smp.") for k in V.v), clash
    write_defs(V.v, G / "values_sample.tex")
    keys = defaultdict(list)
    for src, ks in PRINTED.items():
        for k in ks:
            keys[k].append(src)
    for k in V.v:                                            # every count is printed in Table S39
        keys[k].append("supplement, Table S39")
    assert all(k in V.v for k in keys)
    (G / "paper_keys_sample.json").write_text(json.dumps(dict(sorted(keys.items())), indent=1, ensure_ascii=False))
    (G / "tab_sample.tex").write_text(table(C, N) + "\n")
    cl = claims(C, N, S); json.dump(cl, open(root / "results" / "claims_sample.json", "w"), indent=1)
    print(f"values_sample.json: {len(V.v)} values; printed keys {len(keys)}; "
          f"claims: {sum(cl.values())} true, {sum(not v for v in cl.values())} false {[k for k, v in cl.items() if not v]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
