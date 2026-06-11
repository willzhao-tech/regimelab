import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import ls_harness as H
import pandas as pd

# ---- Data: daily 6m ----
df = H.load_daily6m()
ret = df["Close"].pct_change()

# ---- Causal split: first 3m TRAIN, last 3m TEST (50/50 by time) ----
df_tr, df_te = H.split(df, 0.5)
# pct_change leaves a leading NaN on the first bar of each segment. In H.backtest,
# (1 + NaN) poisons the entire compounded equity curve (ret/maxdd come back NaN).
# That first bar carries no tradable info (signal is shift(1)-ed, so pos=0 there),
# so we drop it. This is purely numerical hygiene, NOT look-ahead.
ret_tr = df_tr["Close"].pct_change().dropna()
ret_te = df_te["Close"].pct_change().dropna()

print("Full range:", df.index.min().date(), "->", df.index.max().date(), "n=", len(df))
print("TRAIN range:", df_tr.index.min().date(), "->", df_tr.index.max().date(), "n=", len(df_tr))
print("TEST  range:", df_te.index.min().date(), "->", df_te.index.max().date(), "n=", len(df_te))

def make_signal(close, n):
    # Long-short trend: +1 if Close>SMA(n) else -1.
    # SMA(n) at bar t uses Close[t-n+1..t] (causal, includes current close only).
    # H.backtest shifts signal by 1, so signal(t) is applied to ret(t+1). No leakage.
    sma = close.rolling(n).mean()
    sig = pd.Series(0.0, index=close.index)
    sig[close > sma] = 1.0
    sig[close <= sma] = -1.0
    sig[sma.isna()] = 0.0  # warm-up: flat until SMA defined
    return sig

# ---- Grid search on TRAIN only ----
grid = [5, 10, 20, 50]
combos = 0
results = []
for n in grid:
    combos += 1
    sig_tr = make_signal(df_tr["Close"], n)
    bt = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=252)
    results.append((n, bt))
    print(f"TRAIN n={n:>3}  sharpe={bt['sharpe']:.4f}  ret={bt['ret']:.4f}  maxdd={bt['maxdd']:.4f}")

# pick best TRAIN sharpe (nan-safe)
def keyfn(r):
    s = r[1]["sharpe"]
    return s if s == s else -1e9  # nan -> very low
best = max(results, key=keyfn)
best_n = best[0]
print(f"\nBEST TRAIN params: n={best_n}  (sharpe={best[1]['sharpe']:.4f})")
print(f"#param-combos tried: {combos}")

# ---- Report TRAIN and TEST at 1x on chosen params ----
sig_tr_best = make_signal(df_tr["Close"], best_n)
sig_te_best = make_signal(df_te["Close"], best_n)
bt_tr = H.backtest(sig_tr_best, ret_tr, leverage=1.0, cost_bps=1.0, ppy=252)
bt_te = H.backtest(sig_te_best, ret_te, leverage=1.0, cost_bps=1.0, ppy=252)
print("\n=== 1x ===")
print("TRAIN:", bt_tr)
print("TEST :", bt_te)

# ---- Leverage table on TEST segment ----
print("\n=== LEVERAGE TABLE (TEST) ===")
lt = H.lev_table(sig_te_best, ret_te, cost_bps=1.0, ppy=252)
for L, d in lt.items():
    print(f"{L:>2}x  ret={d['ret']:.4f}  sharpe={d['sharpe']:.4f}  maxdd={d['maxdd']:.4f}  ruined={d['ruined']}  ruin_dt={d['ruin_dt']}")
