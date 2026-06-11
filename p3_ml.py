# -*- coding: utf-8 -*-
"""P3-10/11: ML feature selection (LASSO-logistic + RandomForest) + Optuna check.
NOTE ON RELEVANCE: P0-1 showed the strategies' edge lives in predicting next-day REALIZED
variance (the proxy), which is untradable in isolation. This analysis therefore answers
"which features predict next-day realized-variance richness" (research value), NOT
"what should we trade". Purged walk-forward, honest deflation framing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

D = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)
df = pd.read_csv(os.path.join(D, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(D, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
skew_ix = pd.read_csv(os.path.join(D, "SKEW_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
ret = df["Close"].pct_change()
idx = ret.index.intersection(vxn.index)
df, ret, vxn = df.loc[idx], ret.loc[idx], vxn.loc[idx]

park = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
p10, p21, p42 = park(10), park(21), park(42)
rng = np.log(df["High"]/df["Low"])*100
F = pd.DataFrame({
    "richness": vxn - p21, "vol_trend": p10 - p42, "range_be": (rng/(vxn/SQ)).shift(0),
    "park10": p10, "park21": p21, "park42": p42,
    "vxn": vxn, "vxn_z63": (vxn - vxn.rolling(63).mean())/vxn.rolling(63).std(),
    "dvxn1": vxn.diff(), "dvxn5": vxn.diff(5),
    "volofvol": vxn.diff().rolling(21).std(), "vxn_term": vxn - vxn.rolling(63).mean(),
    "ret1": ret, "ret5": ret.rolling(5).sum(), "ret21": ret.rolling(21).sum(),
    "absret_ema": ret.abs().ewm(span=10).mean(), "dow": pd.Series(idx.dayofweek, index=idx),
    "dd21": df["Close"]/df["Close"].rolling(21).max() - 1,
    "skew_z": ((skew_ix - skew_ix.rolling(252).mean())/skew_ix.rolling(252).std()).reindex(idx),
}).shift(1)                                        # ALL features lagged one day (causal)
iv = (vxn.shift(1)/100)**2/252
y = ((iv - ret**2) > 0).astype(int)                # next-day short-vol pnl sign (target at t)

data = pd.concat([F, y.rename("y")], axis=1).dropna(subset=[c for c in F.columns if c != "skew_z"] + ["y"])
data["skew_z"] = data["skew_z"].fillna(0)
TRAIN, TEST, EMB = 1260, 252, 21
sel_counts = {c: 0 for c in F.columns}; rf_imp = pd.Series(0.0, index=F.columns)
oos_pnl = []
pnl_raw = (iv - ret**2).reindex(data.index)
blocks = 0; start = TRAIN
while start + TEST <= len(data):
    tr = data.iloc[max(0, start-TRAIN):start-EMB]   # 21d embargo purge
    te = data.iloc[start:start+TEST]
    Xtr = StandardScaler().fit(tr[F.columns])
    las = LogisticRegression(penalty="l1", C=0.05, solver="liblinear", max_iter=2000)
    las.fit(Xtr.transform(tr[F.columns]), tr["y"])
    for c, w in zip(F.columns, las.coef_[0]):
        if abs(w) > 1e-6: sel_counts[c] += 1
    rf = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=7, n_jobs=-1)
    rf.fit(tr[F.columns], tr["y"])
    rf_imp += pd.Series(rf.feature_importances_, index=F.columns)
    prob = rf.predict_proba(te[F.columns])[:, 1]
    s = np.where(prob > 0.60, 1.0, np.where(prob < 0.40, -1.0, 0.0))
    oos_pnl.append(pd.Series(s, index=te.index) * pnl_raw.reindex(te.index))
    blocks += 1; start += TEST
oos = pd.concat(oos_pnl).dropna()
rf_imp /= blocks
print(f"purged walk-forward: {blocks} blocks, OOS {oos.index.min().date()}..{oos.index.max().date()}")
print("\nLASSO selection frequency (blocks selected / total):")
for c, n in sorted(sel_counts.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {c:<12} {n}/{blocks}")
print("\nRandomForest mean importance (top 8):")
for c, v in rf_imp.sort_values(ascending=False).head(8).items():
    print(f"  {c:<12} {v:.3f}")
sh = oos.mean()/oos.std()*SQ if oos.std() > 0 else float("nan")
print(f"\nML-gated strategy OOS Sharpe (proxy): {sh:.2f}  (hand-crafted blend on same era: ~1.7)")
print("HONESTY: ~20 features x continuous fitting = far more effective trials than the hand grid;")
print("any ML edge must clear a much higher deflation bar. And P0-1 showed the predicted quantity")
print("(next-day realized variance vs strike) is NOT tradable in isolation — research value only.")
