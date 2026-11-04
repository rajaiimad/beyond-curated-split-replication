"""
fa_add3_outputs.py -- the two named outputs of addendum 3 (docs/addendum_3.md) that the run did not
write as separate files: results/add3_family6.json (P2: the real-data families of 6 tests) and
results/add3_sess.json (P4: every session as its own group). Both are extracted from
results/add3_stats.json; nothing is refitted or recomputed. Added after the audit of 16 October 2026
(docs/audit_addendum3.md).

The overall-AUC gain is not a tested estimand (addendum 3, P1), so neither file contains it. A target
file that already exists is never overwritten: it must equal what would be written, or the step stops.

Usage: python fa_add3_outputs.py <results dir>
"""
from __future__ import annotations
import json, sys
from pathlib import Path

P6 = ["within_seated", "within_standing", "pooled"]
RULE = "Holm-adjusted one-sided permutation p, p = (1 + k)/(1 + n), below 0.05 within the family AND 95 % group-bootstrap interval above 0"
SUMMARY = ("m", "n_perm_sims", "holm_floor", "n_declared", "n_holm_lt_05", "n_raw_lt_05", "min_p_holm", "min_p_perm")


def _summary(fam):
    return {k: fam[k] for k in SUMMARY}


def build(A):
    F = A["families"]
    f6 = {k: v for k, v in F.items() if k.endswith("|6") and k.split("|")[0] in ("P1", "P2")}
    assert not any("|overall|" in k for k in f6)
    family6 = {
        "status": "post hoc (docs/addendum_3.md, P2); extracted from results/add3_stats.json, not recomputed",
        "rule": RULE, "protocols": P6, "labels": ["sev1", "sev2"],
        "note": "The real-data outcome of these families was determined by p-values known before the family was defined "
                "(addendum 3, 'Consequence for P2'). The families of the two P1 learners are the P2 scoring specified in P1.",
        "families": f6, "summary": {k: _summary(v) for k, v in f6.items()},
        "n_declared_total": int(sum(v["n_declared"] for v in f6.values())),
    }
    fs = {k: v for k, v in F.items() if k.startswith("P4|")}
    p2, p4 = A["power_p2"]["counts"], A["power_p4"]["counts"]
    comp_real = {st: {"session_grouping": _summary(F[f"P4|sess|lroff|CTXB+MOT|{st}|6"]),
                      "inferred_groups": _summary(F[f"P2|lroff|CTXB+MOT|{st}|6"])} for st in ("ctx", "session")}
    comp_pow = {}
    for k, v in p4.items():
        fam, ln, lev, st = k.split("|")
        comp_pow[f"{lev}|{st}"] = {"n_est": v["n_est"], "n50_session_grouping": v["n50"],
                                   "n50_inferred_groups": p2[k]["n50"] if k in p2 else None}
    sess = {
        "status": "post hoc (docs/addendum_3.md, P4); extracted from results/add3_stats.json, not recomputed",
        "grouping": "every session is its own group (group_id = session codes) for the folds, the transplant's donor rule and the bootstrap clusters",
        "rule": RULE, "protocols": P6, "labels": ["sev1", "sev2"], "learner": "lroff", "features": "CTXB+MOT",
        "real_gains_source": "stored session-grouping predictions (results/preds/sess_*); intervals with session clusters (results/rev3_stats.json)",
        "families": fs, "power": A["power_p4"],
        "comparison_with_inferred_groups": {
            "real_data": comp_real,
            "same_number_declared": all(v["session_grouping"]["n_declared"] == v["inferred_groups"]["n_declared"] for v in comp_real.values()),
            "power_combinations_declared_in_at_least_half_of_realizations": comp_pow,
            "power_max_abs_difference_n50": max(abs(v["n50_session_grouping"] - v["n50_inferred_groups"])
                                                for v in comp_pow.values() if v["n50_inferred_groups"] is not None),
        },
    }
    return {"add3_family6.json": family6, "add3_sess.json": sess}


def main(R):
    R = Path(R); A = json.load(open(R / "add3_stats.json"))
    for name, obj in build(A).items():
        f = R / name; text = json.dumps(obj, indent=1)
        if f.exists():
            if f.read_text() != text:
                sys.exit(f"{f} exists and differs from the extraction; not overwritten")
            print(f"{name}: exists, identical")
        else:
            f.write_text(text); print(f"{name}: written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
