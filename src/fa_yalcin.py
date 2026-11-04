"""
fa_yalcin.py -- revision 3, Stage 6: the three evaluation-design findings of Table X on a second,
independent public archive (Yalcin et al., 2024, Front. Virtual Real. 5:1364207;
github.com/m1237/automatic-cybersickness-detection, MIT licence). Pre-processing and tests are
fixed in docs/ANALYSIS_PLAN_REVISION3.md (Stage 6).

  D1  record-wise stratified 5-fold vs participant-disjoint grouped 5-fold x 5 repeats, and
      participant identification from the features under the record-wise split;
  D2  a prevalence score and default HGB evaluated leave-one-participant-out with scores pooled
      across folds, against pairs counted within a fold model;
  D3  the label-shift control under participant-disjoint folds (overall and within-session AUC).

Usage: python fa_yalcin.py <repo dir of the Yalcin archive> <results dir>
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from fa_run import make_folds  # noqa: E402
from fa_stats import Boot, ci  # noqa: E402
from fa_rev3 import strat_auc  # noqa: E402

EXCLUDE = {16: "byte-identical copy of folder 13 (eye-tracking and button files)"}
WIN = 10                                    # trailing window, seconds (= records)
HGB = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, class_weight="balanced", random_state=0)
LR = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
BASE = ["pupil_mean", "pupil_sd", "gaze_speed", "gaze_disp", "blink_frac", "valid_frac"]


def _ts(s):
    """Timestamps written as yyyymmddHHMMSS followed by the fraction of a second with its trailing
    zeros dropped (e.g. '...161122' + '18' = 22.18 s; a whole second has no fraction at all), so the
    fraction is right-padded. (A single strptime format would read the 14-digit '...134014' as
    13:40:01.4.)"""
    s = s.astype(str).str.strip()
    frac = s.str[14:].str.ljust(6, "0").str[:6].astype(int)
    return pd.to_datetime(s.str[:14], format="%Y%m%d%H%M%S") + pd.to_timedelta(frac, unit="us")


def _episodes(tc, gap=1.0):
    """Button-press episodes: timestamps less than gap seconds apart (the file logs while held)."""
    tc = np.sort(tc)
    return int(len(tc) > 0) + int((np.diff(tc) >= gap).sum())


def load_participant(d: Path):
    et = pd.read_csv(next(d.glob("*eyetracking_data.csv")), sep=";")
    cs = pd.read_csv(next(d.glob("*cs_timestamps.csv"))).cs_timestamp
    t = _ts(et.current_timestamp)
    order = np.argsort(t.to_numpy(), kind="stable")                    # samples in time order
    et = et.iloc[order].reset_index(drop=True); t = t.iloc[order].reset_index(drop=True); t0 = t.min()
    ts = (t - t0).dt.total_seconds().to_numpy()
    tf = lambda c: et[c].astype(str).str.lower().eq("true").to_numpy()
    lv, rv = tf("left_pupil_diameter_valid"), tf("right_pupil_diameter_valid")
    lp = np.where(lv, et.left_pupil_diameter_value.astype(float), np.nan)
    rp = np.where(rv, et.right_pupil_diameter_value.astype(float), np.nan)
    with np.errstate(all="ignore"):
        pupil = np.nanmean(np.c_[lp, rp], axis=1)
    gl, gr = tf("left_gazeray_valid"), tf("right_gazeray_valid")
    dl = et[[f"left_gazeray_direction_{a}" for a in "xyz"]].to_numpy(float)
    dr = et[[f"right_gazeray_direction_{a}" for a in "xyz"]].to_numpy(float)
    gdir = np.where(gl[:, None] & gr[:, None], (dl + dr) / 2, np.where(gl[:, None], dl, np.where(gr[:, None], dr, np.nan)))
    gdir = gdir / np.linalg.norm(gdir, axis=1, keepdims=True)
    dot = np.clip((gdir[1:] * gdir[:-1]).sum(1), -1, 1); dt = np.diff(ts)
    with np.errstate(all="ignore"):
        speed = np.r_[np.nan, np.degrees(np.arccos(dot)) / np.where(dt > 0, dt, np.nan)]
    blink = tf("left_blinking") | tf("right_blinking")
    S = pd.DataFrame({"bin": np.floor(ts).astype(int), "pupil": pupil, "speed": speed, "gx": gdir[:, 0], "gy": gdir[:, 1],
                      "gz": gdir[:, 2], "blink": blink.astype(float), "valid": (gl & gr).astype(float)})
    g = S.groupby("bin")
    B = pd.DataFrame({"pupil_mean": g.pupil.mean(), "pupil_sd": g.pupil.std(), "gaze_speed": g.speed.mean(),
                      "gaze_disp": g.gx.std() + g.gy.std() + g.gz.std(), "blink_frac": g.blink.mean(),
                      "valid_frac": g.valid.mean()})
    n_occupied = len(B)
    B = B.reindex(range(int(S.bin.max()) + 1))                         # empty bins stay as records
    tc = (_ts(cs) - t0).dt.total_seconds().to_numpy() if len(cs) else np.array([])
    inside = (tc >= 0) & (tc < len(B))
    lab = np.zeros(len(B), int); lab[np.floor(tc[inside]).astype(int)] = 1
    B = B.ffill()                                                      # past-only fill
    out = {"t_elapsed": np.arange(len(B), dtype=float)}
    for c in BASE:
        x = B[c]
        out[c] = x.to_numpy(); out[f"{c}_m{WIN}"] = x.rolling(WIN, min_periods=1).mean().to_numpy()
        out[f"{c}_s{WIN}"] = x.rolling(WIN, min_periods=2).std().fillna(0).to_numpy()
    R = pd.DataFrame(out).fillna(0.0); R["y"] = lab
    info = {"n_presses": int(len(tc)), "presses_outside_span": int((~inside).sum()), "duration_s": float(ts.max()),
            "positive_seconds": int(lab.sum()), "press_episodes": _episodes(tc),
            "press_episodes_outside_span": _episodes(tc[~inside]),
            "empty_bins": int(len(B) - n_occupied),                      # bins without a sample (features carried forward)
            "first_gap_s": float(np.diff(ts)[0]) if len(ts) > 1 else 0.0,
            "samples_per_s": float(len(ts) / ts.max()) if ts.max() > 0 else float("nan")}
    return R, info


def load_all(repo: Path):
    rows, info = [], {}
    for d in sorted((repo / "data").glob("*/*"), key=lambda p: int(p.name)):
        pid = int(d.name)
        if not d.is_dir():
            continue
        if pid in EXCLUDE:
            info[pid] = {"excluded": EXCLUDE[pid]}; continue
        R, inf = load_participant(d); R["pid"] = pid; rows.append(R); info[pid] = inf
    F = pd.concat(rows, ignore_index=True)
    med = np.median([v["duration_s"] for v in info.values() if "duration_s" in v])
    for pid, v in info.items():
        if "duration_s" in v:
            v["flag_long"] = bool(v["duration_s"] > 2 * med)
    return F, info


def feats(F):
    return [c for c in F.columns if c not in ("y", "pid", "t_elapsed")]


def grouped_scores(model, X, y, g, y_train=None):
    """Participant-disjoint grouped 5-fold x 5 repeats (the study's make_folds)."""
    y_train = y if y_train is None else y_train; allm = np.ones(len(y), bool)
    P = np.full((5, len(y)), np.nan); FO = np.full((5, len(y)), -1)
    for r, f, tr, te in make_folds(g, allm, allm):
        if len(set(y_train[tr])) < 2:
            continue
        P[r, te] = clone(model).fit(X[tr], y_train[tr]).predict_proba(X[te])[:, 1]; FO[r, te] = f
    return P, FO


def boot_metrics(y, P, FO, g, sess):
    bt = Boot(g); ov, ws = [], []
    for r in range(P.shape[0]):
        v = FO[r] >= 0; fs = FO[r].astype(str)
        ov.append(bt.auc(y, P[r], fs, v)); ws.append(bt.auc(y, P[r], np.char.add(np.char.add(fs, "|"), sess.astype(str)), v, nested=True))
    with np.errstate(invalid="ignore"):
        a, b = np.nanmean(ov, 0), np.nanmean(ws, 0)
    return {"auc": float(a[0]), "ci": ci(a), "within_session_auc": float(b[0]), "within_session_ci": ci(b)}


def offset_gain(X, t, y, g):
    """LR with a time-only baseline (t, t^2; fitted on the training participants) as fixed offset,
    grouped 5-fold x 5 repeats; overall and within-participant AUC of baseline and offset model and
    their difference with the same bootstrap draws."""
    from fa_rev3 import fit_offset_lr, _std
    B = np.c_[t / 60, (t / 60) ** 2]; allm = np.ones(len(y), bool)
    Pb = np.full((5, len(y)), np.nan); Pf = Pb.copy(); FO = np.full((5, len(y)), -1)
    for r, f, tr, te in make_folds(g, allm, allm):
        if len(set(y[tr])) < 2:
            continue
        Bs, Bt = _std(B[tr], B[te]); b0, be = fit_offset_lr(Bs, y[tr], np.zeros(len(tr)))
        oa, ob = b0 + Bs @ be, b0 + Bt @ be
        Xs, Xt = _std(X[tr], X[te]); c0, ce = fit_offset_lr(Xs, y[tr], oa)
        Pb[r, te] = ob; Pf[r, te] = ob + c0 + Xt @ ce; FO[r, te] = f
    bt = Boot(g); res = {}
    arr = {}
    for name, P in (("baseline", Pb), ("offset", Pf)):
        ov, ws = [], []
        for r in range(5):
            v = FO[r] >= 0; fs = FO[r].astype(str)
            ov.append(bt.auc(y, P[r], fs, v)); ws.append(bt.auc(y, P[r], np.char.add(np.char.add(fs, "|"), g.astype(str)), v, nested=True))
        with np.errstate(invalid="ignore"):
            arr[name] = (np.nanmean(ov, 0), np.nanmean(ws, 0))
        res[name] = {"auc": float(arr[name][0][0]), "ci": ci(arr[name][0]),
                     "within_participant_auc": float(arr[name][1][0]), "within_participant_ci": ci(arr[name][1])}
    for k, i in (("gain_overall", 0), ("gain_within_participant", 1)):
        d = arr["offset"][i] - arr["baseline"][i]; res[k] = {"delta": float(d[0]), "ci": ci(d)}
    return res


def shifted(F, y, seed=1):
    rng = np.random.default_rng(seed); ys = y.copy()
    for pid, idx in sorted(F.groupby("pid").indices.items()):
        n = len(idx); k = int(rng.integers(n // 4, max(n // 4 + 1, (3 * n) // 4 + 1)))
        ys[idx] = np.roll(y[idx], k)
    return ys


def analyse(F):
    y = F.y.to_numpy(); g = F.pid.to_numpy(); X = F[feats(F) + ["t_elapsed"]].to_numpy(float)
    T = F[["t_elapsed"]].to_numpy(float); out = {}
    # D1 -- record-wise vs participant-disjoint
    for name, m in (("hgb", HGB), ("logreg", LR)):
        p = np.zeros(len(y)); fo = np.zeros(len(y), int)
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
            p[b] = clone(m).fit(X[a], y[a]).predict_proba(X[b])[:, 1]; fo[b] = k
        out[f"D1_recordwise_{name}"] = {"auc": float(strat_auc(y, p, fo.astype(str)))}
        P, FO = grouped_scores(m, X, y, g)
        out[f"D1_grouped_{name}"] = boot_metrics(y, P, FO, g, g)
    P, FO = grouped_scores(LR, T, y, g); out["D1_grouped_TIME"] = boot_metrics(y, P, FO, g, g)
    # participant identification from the features, record-wise split
    acc = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, g):
        acc.append(float((HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=0)
                          .fit(X[a], g[a]).predict(X[b]) == g[b]).mean()))
    out["D1_participant_identification"] = {"accuracy": float(np.mean(acc)), "n_classes": int(len(np.unique(g))),
                                            "majority_share": float(pd.Series(g).value_counts(normalize=True).max())}
    # D2 -- leave-one-participant-out, scores pooled across folds vs within-fold pairs
    loo_prev = np.zeros(len(y)); loo_hgb = np.zeros(len(y))
    for k in np.unique(g):
        tr, te = g != k, g == k
        loo_prev[te] = y[tr].mean(); loo_hgb[te] = clone(HGB).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    P, FO = grouped_scores(HGB, X, y, g)
    out["D2"] = {"prevalence_score_LOPO_pooled_auc": float(roc_auc_score(y, loo_prev)),
                 "hgb_LOPO_pooled_auc": float(roc_auc_score(y, loo_hgb)),
                 "hgb_LOPO_within_participant_auc": float(strat_auc(y, loo_hgb, g.astype(str))),
                 "hgb_grouped5_within_fold_auc": float(np.nanmean([strat_auc(y[FO[r] >= 0], P[r][FO[r] >= 0], FO[r][FO[r] >= 0].astype(str)) for r in range(5)]))}
    # D3 -- label-shift control under participant-disjoint folds
    ys = shifted(F, y)
    for name, m in (("hgb", HGB), ("logreg", LR)):
        P, FO = grouped_scores(m, X, y, g); out[f"D3_true_{name}"] = boot_metrics(y, P, FO, g, g)
        P, FO = grouped_scores(m, X, y, g, y_train=ys); out[f"D3_shifted_{name}"] = boot_metrics(y, P, FO, g, g)
    out["D3_label_agreement_after_shift"] = float((ys == y).mean())
    # addendum B7: the study's offset design -- eye-tracking features added to a time-only baseline
    out["offset"] = offset_gain(F[feats(F)].to_numpy(float), F.t_elapsed.to_numpy(float), y, g)
    out["n_records"] = int(len(y)); out["n_participants"] = int(len(np.unique(g))); out["prevalence"] = float(y.mean())
    out["participants_with_positives"] = int(F.groupby("pid").y.max().sum())
    return out


if __name__ == "__main__":
    repo, R = Path(sys.argv[1]), Path(sys.argv[2])
    F, info = load_all(repo)
    res = {"source": "github.com/m1237/automatic-cybersickness-detection",
           "commit": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "participants": {str(k): v for k, v in info.items()},
           "duplicate_check": {"13_vs_16_eyetracking_sha256_equal":
                               hashlib.sha256(next((repo / "data/train/13").glob("*eyetracking_data.csv")).read_bytes()).hexdigest() ==
                               hashlib.sha256(next((repo / "data/train/16").glob("*eyetracking_data.csv")).read_bytes()).hexdigest()},
           "main": analyse(F)}
    long_ = [int(k) for k, v in info.items() if v.get("flag_long")]
    res["without_long_recordings"] = {"excluded": long_, **analyse(F[~F.pid.isin(long_)].reset_index(drop=True))}
    json.dump(res, open(R / "rev3_yalcin.json", "w"), indent=1)
    print(json.dumps(res["main"], indent=1))
