# -*- coding: utf-8 -*-
# Family: breakout_ls  --  Donchian long-short on NQ 1h bars.
# +1 on break above rolling-max(n), -1 on break below rolling-min(n), hold otherwise.
# CAUSAL: channel at bar t uses bars strictly < t (shifted rolling window).
#         signal is in [-1,1] at bar t; H.backtest shifts it by 1 so it is applied to ret(t+1).
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import ls_harness as H
import numpy as np, pandas as pd

PPY = 252 * 23  # 1h bars, ~23 trading hours/day

df = H.load_1h()
close = df["Close"].astype(float)
ret = close.pct_change().fillna(0.0)          # bar return, aligned to close-to-close

def donchian_signal(close, n):
    """
    Causal Donchian long-short signal in {-1,0,+1}, then forward-filled (hold).
    Channel for bar t is built ONLY from closes at t-n .. t-1 (note the .shift(1)),
    so deciding the break at bar t uses no information from bar t itself except its own close,
    which is known at the close of t. The position is then applied to the NEXT bar's return
    (H.backtest does signal.shift(1)).  No future leakage.
    """
    roll_max = close.shift(1).rolling(n).max()   # highest close over the PRIOR n bars
    roll_min = close.shift(1).rolling(n).min()   # lowest  close over the PRIOR n bars
    sig = pd.Series(np.nan, index=close.index)
    sig[close > roll_max] = 1.0                  # breakout up  -> long
    sig[close < roll_min] = -1.0                 # breakout dn  -> short
    sig = sig.ffill().fillna(0.0)                # hold last state; flat until first break
    return sig.clip(-1, 1)

# ---- split 50/50: first 3m TRAIN, last 3m TEST ----
tr_close, te_close = H.split(close)
tr_ret,   te_ret   = H.split(ret)

grid = [12, 24, 48, 96]
combos = 0
results = []
for n in grid:
    combos += 1
    sig_tr = donchian_signal(tr_close, n)
    bt = H.backtest(sig_tr, tr_ret, leverage=1.0, cost_bps=1.0, ppy=PPY)
    results.append((n, bt["sharpe"], bt["ret"]))
    print(f"[TRAIN] n={n:3d}  sharpe={bt['sharpe']:.3f}  ret={bt['ret']:.4f}")

# pick best by TRAIN sharpe (NaN-safe)
def keyf(r):
    s = r[1]
    return -1e9 if (s != s) else s
best = max(results, key=keyf)
best_n = best[0]
print(f"\n#param-combos tried = {combos}")
print(f"BEST (by TRAIN sharpe): n={best_n}  TRAIN sharpe={best[1]:.3f}")

# ---- report TRAIN and TEST at 1x on chosen params ----
sig_tr_best = donchian_signal(tr_close, best_n)
sig_te_best = donchian_signal(te_close, best_n)

bt_train = H.backtest(sig_tr_best, tr_ret, leverage=1.0, cost_bps=1.0, ppy=PPY)
bt_test  = H.backtest(sig_te_best, te_ret, leverage=1.0, cost_bps=1.0, ppy=PPY)

print("\n=== CHOSEN PARAMS:  n =", best_n, "===")
print(f"TRAIN 1x: sharpe={bt_train['sharpe']:.3f}  ret={bt_train['ret']:.4f}  maxdd={bt_train['maxdd']:.4f}  n={bt_train['n']}")
print(f"TEST  1x: sharpe={bt_test['sharpe']:.3f}  ret={bt_test['ret']:.4f}  maxdd={bt_test['maxdd']:.4f}  n={bt_test['n']}")

# ---- leverage table on TEST ----
print("\n=== LEVERAGE TABLE on TEST (n =", best_n, ") ===")
lt = H.lev_table(sig_te_best, te_ret, cost_bps=1.0, ppy=PPY)
for L in (5, 10, 15, 20):
    d = lt[L]
    print(f"{L:2d}x: ret={d['ret']:+.4f}  sharpe={d['sharpe']:.3f}  maxdd={d['maxdd']:.4f}  "
          f"RUINED={'Y' if d['ruined'] else 'N'}  ruin_dt={d['ruin_dt'] if d['ruined'] else '-'}")

print("\nTEST index range:", te_close.index.min(), "->", te_close.index.max())
print("TRAIN index range:", tr_close.index.min(), "->", tr_close.index.max())
