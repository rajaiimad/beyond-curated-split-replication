"""
fa_data.py -- UFFCSData FULL archive (RAW_DATABASE, 74 sessions) -> leakage-checked tables.

Order of operations (fixed):
  1. parse every raw session file (comma decimals) and its VRSQ file
  2. harmonise to the channels present in BOTH campaigns
  3. participant group = identical answers to the 7 demographic items, across
     the whole archive (so a person who appears in both campaigns is one group)
  4. per-session, past-only features; nothing is scaled or encoded here
  5. assert_no_leakage() before any model sees the data

Usage: python fa_data.py <RAW_DATABASE dir> <out dir>
"""
from __future__ import annotations
import hashlib, json, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

DEMO = ["UserGenere", "UserAge", "UserExperience", "UserFlicker",
        "UserGlassesUse", "UserVisionProblems", "UserEyeDominance"]
MOTION = ["PlayerSpeed", "PlayerAcceleration", "CameraRotationX", "CameraRotationY", "CameraRotationZ"]
FPS = ["GameFps"]
ROI = ["background", "middleground", "foreground"]    # RegionOfInterest, varies within a session
VRSQ_O = ["GeneralDiscomfort", "Fatigue", "EyeStraing", "DifficultyFocusing"]
VRSQ_D = ["Headache", "FulnessOfHead", "BlurredVision", "DizzyEyesClosed", "Vertigo"]


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="raise")


def vrsq_scores(item: dict) -> dict:
    """VRSQ (Kim et al., 2018): oculomotor = sum(4 items)/12*100,
    disorientation = sum(5 items)/15*100, total = mean of the two."""
    o = sum(item[k] for k in VRSQ_O) / 12 * 100
    d = sum(item[k] for k in VRSQ_D) / 15 * 100
    return {"oculomotor": o, "disorientation": d, "total": (o + d) / 2}


def load_archive(raw_dir: Path):
    rows, sess = [], []
    for d in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir()):
        f = d / "FILE.xml"
        recs = [{c.tag: (c.text or "").strip() for c in e} for e in ET.parse(f).getroot().findall("Data")]
        df = pd.DataFrame(recs)
        for c in MOTION + FPS + ["TimeStamp", "DiscomfortLevel", "CameraFieldOfView"]:
            df[c] = _num(df[c])
        df = df.sort_values("TimeStamp", kind="stable").reset_index(drop=True)
        m = re.match(r"DATA_EXPORT_(Flight|Race)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", d.name)
        posture = df["UserPosture"].iloc[0].upper()
        campaign = "seated" if "SENTADO" in posture else "standing"
        info = dict(session_id=d.name, game=m.group(1), campaign=campaign,
                    date=f"{m.group(2)}-{int(m.group(3)):02d}-{int(m.group(4)):02d}",
                    start=f"{int(m.group(5)):02d}:{int(m.group(6)):02d}:{int(m.group(7)):02d}",
                    fingerprint="|".join(df[c].iloc[0] for c in DEMO),
                    n_rows=len(df), duration_s=float(df.TimeStamp.iloc[-1] - df.TimeStamp.iloc[0]),
                    max_gap_s=float(df.TimeStamp.diff().max()), fov=float(df.CameraFieldOfView.iloc[0]),
                    fps_mean=float(df.GameFps.mean()), sha16=hashlib.sha256(f.read_bytes()).hexdigest()[:16])
        # VRSQ: block 0 = before exposure, block 1 = after. Two files repeat the pair; verify.
        v = d / "VRSQ_FILE.xml"
        if v.exists():
            blocks = [{t: int(x) for t, x in re.findall(r"<(\w+)>\s*(-?\d+)\s*</\1>", b)}
                      for b in re.findall(r"<Session[^>]*>(.*?)</Session>", v.read_text(encoding="utf-8-sig"), re.S)]
            if len(blocks) == 4:
                assert blocks[0] == blocks[2] and blocks[1] == blocks[3], f"{d.name}: VRSQ blocks differ"
            pre, post = vrsq_scores(blocks[0]), vrsq_scores(blocks[1])
            info.update({f"vrsq_pre_{k}": v_ for k, v_ in pre.items()})
            info.update({f"vrsq_post_{k}": v_ for k, v_ in post.items()})
            info["vrsq_post_raw_sum"] = sum(blocks[1].values())
        sess.append(info)
        df["session_id"] = d.name
        rows.append(df)
    raw = pd.concat(rows, ignore_index=True)
    S = pd.DataFrame(sess)
    S["group_id"] = S.groupby("fingerprint", sort=True).ngroup()
    S["flag_gap"] = S.max_gap_s > 5
    S["flag_config"] = (S.fov < 80) | (S.fps_mean < 40)     # one session: FOV 60, ~33 fps
    S["flag_short"] = S.duration_s < 60
    raw = raw.merge(S[["session_id", "campaign", "game", "group_id"]], on="session_id")
    return raw, S


def curated_sessions(selected_dir: Path, raw_dir: Path) -> dict:
    """Map each curated DATABASE_SELECTED_DATA/<k> file to the raw session it copies
    (exact match of the whole camera-rotation series)."""
    rawv = {}
    for d in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir()):
        r = ET.parse(d / "FILE.xml").getroot().findall("Data")
        rawv[d.name] = np.array([float(e.find("CameraRotationY").text.replace(",", ".")) for e in r])
    out = {}
    for k in sorted((p for p in Path(selected_dir).iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        a = np.array([float(e.find("CameraRotationY").text) for e in ET.parse(k / "FILE.xml").getroot().findall("Data")])
        hits = [n for n, b in rawv.items() if len(a) == len(b) and np.allclose(a, b, atol=1e-6)]
        assert len(hits) == 1, (k.name, hits)
        out[int(k.name)] = hits[0]
    return out


def session_features(g: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Past-only features for ONE session (rows sorted by time). Windows are in
    SECONDS of recorded time, not rows, so recording gaps do not stretch a window."""
    t = (g.TimeStamp - g.TimeStamp.iloc[0]).to_numpy(float)
    idx = pd.to_timedelta(t, unit="s"); win = f"{int(window)}s"
    out = {"t_elapsed": t}

    def block(name, x):                      # value, trailing mean / SD, first difference
        x = pd.Series(np.asarray(x, float), index=idx)
        out[name] = x.values
        out[f"{name}_mean"] = x.rolling(win, min_periods=1).mean().values
        out[f"{name}_std"] = x.rolling(win, min_periods=2).std().fillna(0).values
        out[f"{name}_diff"] = x.diff().fillna(0).values
    for c in MOTION + FPS:
        block(c, g[c])
    # the three logged rotation fields are the vector part (x, y, z) of a unit quaternion
    # (all |v| <= 1); used as logged in MOT, and quaternion-aware in the q_ block below
    rot = g[["CameraRotationX", "CameraRotationY", "CameraRotationZ"]].to_numpy(float)
    step = pd.Series(np.r_[0.0, np.linalg.norm(np.diff(rot, axis=0), axis=1)], index=idx)
    out["rot_step"] = step.values
    out["rot_instability"] = step.rolling(win, min_periods=2).std().fillna(0).values
    roi = g["RegionOfInterest"].str.strip().str.lower().to_numpy()
    for lvl in ROI:
        r = pd.Series((roi == lvl).astype(float), index=idx)
        out[f"roi_{lvl}"] = r.values
        out[f"roi_{lvl}_frac"] = r.rolling(win, min_periods=1).mean().values
    # quaternion-aware rotation features (sign-invariant): angle from the reference
    # orientation, theta = 2 asin|v|, and angular change between consecutive records,
    # 2 acos|q_t . q_t-1| with the scalar part reconstructed as w = sqrt(1 - |v|^2)
    nv = np.clip(np.linalg.norm(rot, axis=1), 0, 1)
    q = np.c_[rot, np.sqrt(np.clip(1 - nv ** 2, 0, 1))]
    block("q_theta", 2 * np.arcsin(nv))
    dot = np.r_[1.0, np.abs((q[1:] * q[:-1]).sum(1))]
    ang = pd.Series(2 * np.arccos(np.clip(dot, 0, 1)), index=idx)
    out["q_angstep"] = ang.values
    out["q_angstep_mean"] = ang.rolling(win, min_periods=1).mean().values
    out["q_angstep_std"] = ang.rolling(win, min_periods=2).std().fillna(0).values
    # per-session self-normalisation: every MOT feature as a z-score against this
    # session's own past (expanding mean / SD up to and including the record; 0 while
    # the SD is undefined; clipped to +-10 so a near-constant start cannot explode)
    for c in [k for k in list(out) if _is_mot(k)]:
        x = pd.Series(out[c], index=idx)
        mu = x.expanding().mean(); sd = x.expanding().std()
        out[f"z_{c}"] = ((x - mu) / sd.where(sd > 1e-9)).fillna(0).clip(-10, 10).values
    return pd.DataFrame(out, index=g.index)


def _is_mot(c: str) -> bool:
    return any(c.startswith(m) for m in MOTION + ["rot_", "roi_"])


def build_features(raw: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    F = pd.concat([session_features(g, window) for _, g in raw.groupby("session_id", sort=True)]).loc[raw.index]
    for c in ["session_id", "campaign", "game", "group_id"] + DEMO:
        F[c] = raw[c].values
    F["severity"] = raw["DiscomfortLevel"].astype(int).values
    return F


def feature_sets(F: pd.DataFrame) -> dict:
    mot = [c for c in F.columns if _is_mot(c)]                                   # 28 telemetry features
    fps = [c for c in F.columns if c.startswith("GameFps")]
    selfz = [f"z_{c}" for c in mot]                                               # the same 28, self-normalised
    motq = [c for c in mot if c.startswith(("PlayerSpeed", "PlayerAcceleration", "roi_"))] + \
           [c for c in F.columns if c.startswith("q_")]                           # rotation made quaternion-aware
    assert len(mot) == 28 and all(c in F.columns for c in selfz), (len(mot), mot)
    fs = {"TIME": ["t_elapsed"], "MOT": mot, "MOT+TIME": mot + ["t_elapsed"],
          "MOT+FPS+TIME": mot + fps + ["t_elapsed"], "MOTQ+TIME": motq + ["t_elapsed"],
          "SELFZ+TIME": selfz + ["t_elapsed"]}
    # Context known at deployment (which game, seated or standing): session-level
    # indicators from the file name / posture field, never from telemetry.
    if {"ctx_race", "ctx_standing"} <= set(F.columns):
        ctx = ["ctx_race", "ctx_standing"]
        fs.update({"CTX": ctx, "CTX+TIME": ctx + ["t_elapsed"], "MOT+CTX+TIME": mot + ctx + ["t_elapsed"]})
    return fs


def add_context(F: pd.DataFrame) -> pd.DataFrame:
    F = F.copy()
    F["ctx_race"] = (F.game == "Race").astype(float)
    F["ctx_standing"] = (F.campaign == "standing").astype(float)
    return F


def assert_no_leakage(raw: pd.DataFrame, F: pd.DataFrame, fold_sets: dict) -> dict:
    checks = {}
    assert (F.groupby("session_id").group_id.nunique() == 1).all()
    checks["session_in_one_group"] = True
    for name, folds in fold_sets.items():
        for tr, te in folds:
            assert not (set(F.group_id.values[tr]) & set(F.group_id.values[te])), name
            assert not (set(F.session_id.values[tr]) & set(F.session_id.values[te])), name
    checks["train_test_group_and_session_disjoint_every_fold"] = sorted(fold_sets)
    rng = np.random.default_rng(0)
    for sid in raw.session_id.unique():               # every session
        g = raw[raw.session_id == sid].reset_index(drop=True)
        k = len(g) // 2
        if k < 3:
            continue
        fut = g.iloc[k + 1:].copy()
        perm = rng.permutation(len(fut))
        fut[MOTION + FPS] = fut[MOTION + FPS].values[perm] * 3.7 + 11
        fut["RegionOfInterest"] = fut["RegionOfInterest"].values[perm[::-1]]
        a = session_features(g).iloc[:k + 1].values
        b = session_features(pd.concat([g.iloc[:k + 1], fut])).iloc[:k + 1].values
        assert np.allclose(a, b, equal_nan=True), f"future rows changed past features in {sid}"
    checks["features_past_only"] = True
    assert len(F) == len(raw)
    checks["one_row_per_raw_record_no_oversampling"] = True
    return checks


if __name__ == "__main__":
    raw_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    sel_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    raw, S = load_archive(raw_dir)
    if sel_dir is not None:
        cm = curated_sessions(sel_dir, raw_dir)
        S["curated_release"] = S.session_id.isin(set(cm.values()))
        S["curated_id"] = S.session_id.map({v: k for k, v in cm.items()})
    F = build_features(raw)
    F.to_csv(out / "features_w30.csv.gz", index=False)
    S.to_csv(out / "sessions.csv", index=False)
    print(S.groupby(["campaign", "game"]).agg(sessions=("session_id", "size"), rows=("n_rows", "sum"),
          groups=("group_id", "nunique")).to_string())
    print("groups:", S.group_id.nunique(), "| in both campaigns:",
          int((S.groupby("group_id").campaign.nunique() > 1).sum()))
    print("flags:", S[["flag_gap", "flag_config", "flag_short"]].sum().to_dict())
    print("VRSQ available:", int(S.vrsq_post_total.notna().sum()))
