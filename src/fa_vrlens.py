"""
fa_vrlens.py -- revision 2, Stage 4: re-implementation of the reduced VR-LENS detector
(Kundu et al., IUI 2023, arXiv:2302.01985) on the curated UFFCSData release.

This is a re-implementation from the paper, not the authors' code. What the paper gives:
  * features (Table 6, gameplay dataset, reduced model): PlayerPositionZ, UserGlassesUse,
    UserGenere, UserAge, PlayerSpeed, UserVisionProblems, UserExperience, PlayerPositionX,
    StaticFrame, CameraRotationY -- columns of the released dataset.csv;
  * label: DiscomfortLevel 0-3 as four classes (none, low, medium, high);
  * model: stacking "super learner" with base learners SVC, KNN, RF, ET, GBC, XGB and a
    logistic-regression meta-learner; reduced-model hyperparameters (Table 4): KNN
    n_neighbors=2, SVC kernel=poly, ET n_estimators=50, RF n_estimators=10,
    XGB n_estimators=30, GBC learning_rate=0.05, everything else at library defaults;
  * evaluation: 10-fold cross-validation over records (reported accuracy 81 %).
What the paper does not give, and what we assume (stated in the manuscript):
  * standardisation: StandardScaler inside the pipeline, fitted on training records only;
  * stacking: scikit-learn StackingClassifier with a shuffled, stratified internal 5-fold CV
    (StratifiedKFold(5, shuffle=True, random_state=0)), predict_proba passed on. The records of
    dataset.csv are ordered by session, so scikit-learn's default (unshuffled) internal folds would
    split the training records by session; a first run with that default reached only 51 % record-wise
    accuracy (paper: 81 %), because the meta-learner then learns from session-disjoint base-learner
    predictions. Shuffled internal folds match the record-wise design of the original evaluation
    (results/vrlens_attempt1.json keeps the first run);
  * record-wise folds: StratifiedKFold(10, shuffle=True, random_state=0);
  * random_state=0 for every stochastic learner; XGBoost with its library defaults otherwise.
Evaluations (Stage 4 of the revision brief):
  (i)   the original record-wise 10-fold split: 4-class accuracy, and ROC-AUC for severity>=2
        and >=1 from the class probabilities (P(2)+P(3) and P(1)+P(2)+P(3));
  (ii)  participant-disjoint: the study's grouped 5-fold x 5 repeats of the curated protocol;
  (iii) its context-stratified AUC (computed by fa_stats.py from the saved predictions);
  (iv)  the CTX+TIME baseline on the same records is already part of the main results.
Outputs: results/vrlens.json and results/preds/{rowwise_,sensVRLENS_}sev{1,2}__within_curated__VRLENS__stack.npz

Usage: python fa_vrlens.py <RAW_DATABASE dir> <DATABASE_SELECTED_DATA dir> <dataset.csv> <data dir> <results dir>
Requires xgboost (not installable in the environment used for revision 2; see the report).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))

FEATURES = ["PlayerPositionZ", "UserGlassesUse", "UserGenere", "UserAge", "PlayerSpeed", "UserVisionProblems",
            "UserExperience", "PlayerPositionX", "StaticFrame", "CameraRotationY"]
MATCH_KEYS = ["CameraRotationX", "CameraRotationY", "CameraRotationZ", "TimeStamp", "PlayerSpeed", "DiscomfortLevel"]


def vrlens_reduced(cv=None):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        sys.exit("xgboost is required for a faithful VR-LENS re-implementation (pip install xgboost).")
    base = [("svc", make_pipeline(StandardScaler(), SVC(kernel="poly", probability=True, random_state=0))),
            ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=2))),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=0)),
            ("et", ExtraTreesClassifier(n_estimators=50, random_state=0)),
            ("gbc", GradientBoostingClassifier(learning_rate=0.05, random_state=0)),
            ("xgb", XGBClassifier(n_estimators=30, random_state=0))]
    return StackingClassifier(base, final_estimator=LogisticRegression(max_iter=3000),
                              cv=StratifiedKFold(5, shuffle=True, random_state=0) if cv is None else cv,
                              stack_method="predict_proba", n_jobs=-1)


def curated_rows(raw, csv):
    """Map every dataset.csv row to its raw record (exact match on rotation, time, speed, label)."""
    D = pd.read_csv(csv); r = raw.reset_index(drop=True).reset_index().rename(columns={"index": "row"})
    for k in MATCH_KEYS:
        r[k] = pd.to_numeric(r[k]).round(6); D[k] = D[k].round(6)
    m = D.reset_index().merge(r[MATCH_KEYS + ["row"]], on=MATCH_KEYS, how="left")
    assert len(m) == len(D) and m.row.notna().all() and not m.duplicated("index").any(), "dataset.csv rows do not map 1:1"
    return D, m.sort_values("index").row.to_numpy().astype(int)


ABLATIONS = {  # revision 3 (referee Q5): drop the inputs that identify participants or the game
    "sensVRLENSA_": [f for f in FEATURES if not f.startswith("User")],                       # no demographic answers
    "sensVRLENSB_": [f for f in FEATURES if not f.startswith(("User", "PlayerPosition"))],   # ... and no player position
    "sensVRLENSC_": [f for f in FEATURES if not f.startswith(("User", "PlayerPosition", "StaticFrame"))],  # ... and no game alias
}


def main(RAWDIR, SELDIR, CSV, DATA, R, only_ablation=False):
    from fa_data import add_context, build_features, curated_sessions, load_archive
    from fa_run import R_REP, domains, make_folds
    raw, _ = load_archive(RAWDIR)
    cur = set(curated_sessions(SELDIR, RAWDIR).values()); raw["curated"] = raw.session_id.isin(cur)
    F = add_context(build_features(raw)); F["curated"] = raw["curated"].to_numpy()
    D, rows = curated_rows(raw, CSV)
    assert F.curated.to_numpy()[rows].all()
    X = D[FEATURES].to_numpy(float); y4 = D.DiscomfortLevel.to_numpy().astype(int)
    out = {"features": FEATURES, "n_records": int(len(D)), "assumptions": __doc__.split("What the paper does not give")[1].split("Evaluations")[0].strip()}
    P = R / "preds"; P.mkdir(parents=True, exist_ok=True)
    import sklearn, xgboost
    out["versions"] = {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    print("versions", out["versions"], flush=True)

    if only_ablation == "groupcv":
        return groupcv(F, D, rows, y4, R, P)
    if only_ablation:
        return ablation(F, D, rows, y4, R, P)
    # (i) original protocol: record-wise 10-fold
    t0 = time.time(); proba = np.zeros((len(D), 4)); fold = np.zeros(len(D), np.int8)
    for k, (a, b) in enumerate(StratifiedKFold(10, shuffle=True, random_state=0).split(X, y4)):
        proba[b] = vrlens_reduced().fit(X[a], y4[a]).predict_proba(X[b]); fold[b] = k
        print(f"record-wise fold {k + 1}/10 done ({time.time() - t0:.0f}s)", flush=True)
    pred = proba.argmax(1)
    out["recordwise_10fold"] = {"accuracy_4class": float(accuracy_score(y4, pred)), "macro_f1_4class": float(f1_score(y4, pred, average="macro")),
                                "secs": round(time.time() - t0)}
    for lab in (1, 2):
        s = proba[:, lab:].sum(1); yy = (y4 >= lab).astype(int)
        out["recordwise_10fold"][f"auc_sev{lab}"] = float(roc_auc_score(yy, s))
        np.savez_compressed(P / f"rowwise_sev{lab}__within_curated__VRLENS__stack.npz", p=s, idx=rows, fold=fold)
    print(out["recordwise_10fold"], flush=True)

    # (ii) participant-disjoint: the study's grouped 5-fold x 5 repeats (curated protocol)
    src, tgt = domains(F)["within_curated"]; g = F.group_id.to_numpy()
    pos_of = np.full(len(F), -1); pos_of[rows] = np.arange(len(rows))
    PP = {lab: np.full((R_REP, len(F)), np.nan) for lab in (1, 2)}; FOLD = np.full((R_REP, len(F)), -1, np.int8)
    acc = []
    for r, f, tr, te in make_folds(g, src, tgt):
        assert not (set(g[tr]) & set(g[te]))
        a, b = pos_of[tr], pos_of[te]; assert (a >= 0).all() and (b >= 0).all()
        pr = vrlens_reduced().fit(X[a], y4[a]).predict_proba(X[b])
        for lab in (1, 2):
            PP[lab][r, te] = pr[:, lab:].sum(1)
        FOLD[r, te] = f; acc.append((r, accuracy_score(y4[b], pr.argmax(1)), len(b)))
        print(f"grouped repeat {r + 1}/5 fold {f + 1}/5 done", flush=True)
    for lab in (1, 2):
        np.savez_compressed(P / f"sensVRLENS_sev{lab}__within_curated__VRLENS__stack.npz", p=PP[lab], thr=np.full_like(PP[lab], np.nan), fold=FOLD)
    rep_acc = [sum(a * n for rr, a, n in acc if rr == r) / sum(n for rr, a, n in acc if rr == r) for r in range(R_REP)]
    out["grouped_5x5"] = {"accuracy_4class_mean_over_repeats": float(np.mean(rep_acc))}
    json.dump(out, open(R / "vrlens.json", "w"), indent=1)
    print(out, flush=True)


def ablation(F, D, rows, y4, R, P):
    """Revision 3: the detector under the participant-disjoint protocol without the inputs that
    identify participants (demographic answers), their position, or the game (StaticFrame)."""
    from fa_run import R_REP, domains, make_folds
    src, tgt = domains(F)["within_curated"]; g = F.group_id.to_numpy()
    pos_of = np.full(len(F), -1); pos_of[rows] = np.arange(len(rows)); out = {}
    for tag, feats in ABLATIONS.items():
        X = D[feats].to_numpy(float); t0 = time.time()
        PP = {lab: np.full((R_REP, len(F)), np.nan) for lab in (1, 2)}; FOLD = np.full((R_REP, len(F)), -1, np.int8); acc = []
        for r, f, tr, te in make_folds(g, src, tgt):
            assert not (set(g[tr]) & set(g[te]))
            a, b = pos_of[tr], pos_of[te]
            pr = vrlens_reduced().fit(X[a], y4[a]).predict_proba(X[b])
            for lab in (1, 2):
                PP[lab][r, te] = pr[:, lab:].sum(1)
            FOLD[r, te] = f; acc.append((r, accuracy_score(y4[b], pr.argmax(1)), len(b)))
        for lab in (1, 2):
            np.savez_compressed(P / f"{tag}sev{lab}__within_curated__VRLENS__stack.npz", p=PP[lab], thr=np.full_like(PP[lab], np.nan), fold=FOLD)
        rep_acc = [sum(a_ * n for rr, a_, n in acc if rr == r) / sum(n for rr, a_, n in acc if rr == r) for r in range(R_REP)]
        out[tag] = {"features": feats, "accuracy_4class_mean_over_repeats": float(np.mean(rep_acc)), "secs": round(time.time() - t0)}
        print(tag, out[tag], flush=True)
    json.dump(out, open(R / "rev3_vrlens_ablation.json", "w"), indent=1)


def groupcv(F, D, rows, y4, R, P):
    """Addendum 2 (post hoc, docs/ANALYSIS_PLAN_REVISION3_ADDENDUM2.md): the group-disjoint run with
    group-aware internal stacking folds, so that the meta-learner never sees base-learner predictions
    for records of a group that was in the base learners' training part. Everything else unchanged."""
    from fa_run import R_REP, domains, make_folds
    src, tgt = domains(F)["within_curated"]; g = F.group_id.to_numpy(); gD = g[rows]
    pos_of = np.full(len(F), -1); pos_of[rows] = np.arange(len(rows))
    X = D[FEATURES].to_numpy(float); t0 = time.time()
    PP = {lab: np.full((R_REP, len(F)), np.nan) for lab in (1, 2)}; FOLD = np.full((R_REP, len(F)), -1, np.int8); acc = []
    for r, f, tr, te in make_folds(g, src, tgt):
        assert not (set(g[tr]) & set(g[te]))
        a, b = pos_of[tr], pos_of[te]; assert (a >= 0).all() and (b >= 0).all()
        inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X[a], y4[a], groups=gD[a]))
        assert all(not (set(gD[a][i]) & set(gD[a][j])) for i, j in inner)
        pr = vrlens_reduced(cv=inner).fit(X[a], y4[a]).predict_proba(X[b])
        for lab in (1, 2):
            PP[lab][r, te] = pr[:, lab:].sum(1)
        FOLD[r, te] = f; acc.append((r, accuracy_score(y4[b], pr.argmax(1)), len(b)))
        print(f"group-aware stacking: repeat {r + 1}/5 fold {f + 1}/5 done ({time.time() - t0:.0f}s)", flush=True)
    for lab in (1, 2):
        np.savez_compressed(P / f"sensVRLENSG_sev{lab}__within_curated__VRLENS__stack.npz", p=PP[lab], thr=np.full_like(PP[lab], np.nan), fold=FOLD)
    rep_acc = [sum(a_ * n for rr, a_, n in acc if rr == r) / sum(n for rr, a_, n in acc if rr == r) for r in range(R_REP)]
    out = {"features": FEATURES, "inner_cv": "StratifiedGroupKFold(5, shuffle=True, random_state=0) over inferred groups",
           "status": "post hoc (addendum 2)", "accuracy_4class_mean_over_repeats": float(np.mean(rep_acc)), "secs": round(time.time() - t0)}
    json.dump(out, open(R / "rev3_vrlens_groupcv.json", "w"), indent=1)
    print(out, flush=True)


if __name__ == "__main__":
    mode = sys.argv[6] if len(sys.argv) > 6 else ""
    main(*map(Path, sys.argv[1:6]), only_ablation=mode if mode in ("ablation", "groupcv") else False)
