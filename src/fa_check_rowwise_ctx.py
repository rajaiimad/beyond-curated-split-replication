"""
fa_check_rowwise_ctx.py -- post hoc check (14 November 2026) of why the record-wise CTX+TIME model (default HGB,
curated release, severity >= 2; manuscript Table 6) has a context-stratified AUC below 0.5. It fits no model and
writes no file: it reads the stored record-wise predictions and compares them with the share of high-rated records
at the same game and second of exposure among the training records of the other folds.

Usage: python src/fa_check_rowwise_ctx.py <repository dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

R = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
F = pd.read_csv(R / "data" / "features_w30.csv.gz", usecols=["session_id", "campaign", "game", "severity", "t_elapsed"])
z = np.load(R / "results" / "preds" / "rowwise_sev2__within_curated__CTX+TIME__hgb.npz")
G = F.iloc[z["idx"]].reset_index(drop=True); p = z["p"]; fold = z["fold"].astype(int)
y = (G.severity.to_numpy() >= 2).astype(int)
tbin = np.minimum((G.t_elapsed.to_numpy() // 30).astype(int), 9)          # 30-s bins; the last holds t >= 270 s
tsec = np.round(G.t_elapsed.to_numpy()).astype(int)
strata = (pd.Series(fold).astype(str) + "|" + G.campaign + "|" + G.game + "|" + pd.Series(tbin).astype(str)).to_numpy()


def strat_auc(y, s, strata):
    """Pairs of a positive and a negative record of the same stratum, pooled over strata; ties count one half."""
    num = den = 0.0
    for st in np.unique(strata):
        m = strata == st; yy = y[m]; npos = int(yy.sum()); nneg = len(yy) - npos
        if npos and nneg:
            r = rankdata(s[m]); num += r[yy == 1].sum() - npos * (npos + 1) / 2; den += npos * nneg
    return num / den


key = (G.game + "|" + pd.Series(tsec).astype(str)).to_numpy()
df = pd.DataFrame({"k": key, "f": fold, "y": y})
tot = df.groupby("k").y.agg(["sum", "count"]); byf = df.groupby(["k", "f"]).y.agg(["sum", "count"])
s_k = tot.loc[df.k, "sum"].to_numpy(); c_k = tot.loc[df.k, "count"].to_numpy()
idx = pd.MultiIndex.from_arrays([df.k, df.f]); s_kf = byf.loc[idx, "sum"].to_numpy(); c_kf = byf.loc[idx, "count"].to_numpy()
ntr = c_k - c_kf
share = np.where(ntr > 0, (s_k - s_kf) / np.maximum(ntr, 1), np.nan)       # training share, own fold left out
ok = (fold >= 0) & ~np.isnan(share)
print(f"stored score: context-stratified AUC {strat_auc(y[fold >= 0], p[fold >= 0], strata[fold >= 0]):.3f}")
print(f"training share of high-rated records at the same game and second: context-stratified AUC "
      f"{strat_auc(y[ok], share[ok], strata[ok]):.3f}")
print(f"Spearman correlation of the two scores: {spearmanr(p[ok], share[ok]).correlation:.3f}")
