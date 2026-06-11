# -*- coding: utf-8 -*-
"""
LONG-SHORT NQ strategy, family: vol_scaled_mom
Volatility-scaled momentum on 1h bars.

  signal_t = clip( trailing_return(n)_t / trailing_vol(n)_t , -1, 1 )

All inputs use ONLY data up to and including bar t (causal). The harness
(H.backtest) shifts the signal by 1 bar, so signal_t is applied to ret_{t+1}.
We never use a full-sample statistic to normalize per-bar.

Protocol:
  - load 1h NQ (~6 months)
  - split 50/50 by time: first 3m = TRAIN, last 3m = TEST
  - grid-search n (and the vol window) on TRAIN, pick best TRAIN Sharpe
  - report TRAIN and TEST 1x backtest + TEST leverage table (5/10/15/20x)
"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import ls_harness as H

PPY = 252 * 23  # 1h bars: ~23 trading hours/day

# ---------------------------------------------------------------------------
# Causal signal builder
# ---------------------------------------------------------------------------
def build_signal(close, n, volwin):
    """
    Volatility-scaled momentum, fully causal.

    ret_n_t  = pct change of close over the trailing n bars, known at bar t.
    vol_t    = rolling std of 1-bar log/pct returns over `volwin` bars, known at t.
    signal_t = clip(ret_n_t / (vol_t * sqrt(n)), -1, 1)

    The sqrt(n) makes the trailing return and the per-bar vol comparable in
    scale (an n-bar return has std ~ vol*sqrt(n) under random walk), so the
    raw ratio is ~O(1) and clipping does not saturate trivially.
    """
    # 1-bar simple returns, causal
    r1 = close.pct_change()
    # trailing n-bar return ending at t (uses close_t and close_{t-n}, both <= t)
    ret_n = close.pct_change(n)
    # trailing realized vol of 1-bar returns over volwin, ending at t
    vol = r1.rolling(volwin).std()
    denom = vol * np.sqrt(n)
    sig = ret_n / denom
    sig = sig.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return sig.clip(-1, 1)


def main():
    df = H.load_1h()
    close = df["Close"]
    ret = close.pct_change().fillna(0.0)  # bar return; H.backtest applies pos.shift(1)

    # 50/50 time split
    close_tr, close_te = H.split(close)
    ret_tr, ret_te = H.split(ret)

    print("=" * 70)
    print("DATA")
    print("=" * 70)
    print(f"Full:  {df.index[0]}  ->  {df.index[-1]}  ({len(df)} bars)")
    print(f"TRAIN: {close_tr.index[0]}  ->  {close_tr.index[-1]}  ({len(close_tr)} bars)")
    print(f"TEST:  {close_te.index[0]}  ->  {close_te.index[-1]}  ({len(close_te)} bars)")
    print(f"ppy = {PPY}")

    # -----------------------------------------------------------------------
    # Grid search on TRAIN ONLY
    # -----------------------------------------------------------------------
    n_grid = [4, 8, 12, 16, 24, 36, 48, 72, 96]        # trailing-return window (bars)
    volwin_grid = [24, 48, 72, 96, 120]                 # trailing-vol window (bars)
    grid = list(itertools.product(n_grid, volwin_grid))

    results = []
    for n, volwin in grid:
        sig_tr = build_signal(close_tr, n, volwin)
        bt = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=PPY)
        results.append((n, volwin, bt["sharpe"], bt["ret"]))

    res_df = pd.DataFrame(results, columns=["n", "volwin", "train_sharpe", "train_ret"])
    res_df = res_df.sort_values("train_sharpe", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print(f"GRID SEARCH ON TRAIN  ({len(grid)} param combos)")
    print("=" * 70)
    print(res_df.head(12).to_string(index=False))

    best = res_df.iloc[0]
    best_n, best_vw = int(best["n"]), int(best["volwin"])
    print(f"\nBEST (by TRAIN Sharpe): n={best_n}, volwin={best_vw}, "
          f"train_sharpe={best['train_sharpe']:.3f}")

    # -----------------------------------------------------------------------
    # Re-evaluate chosen params on TRAIN and TEST (1x)
    # -----------------------------------------------------------------------
    sig_tr = build_signal(close_tr, best_n, best_vw)
    sig_te = build_signal(close_te, best_n, best_vw)

    bt_tr = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=PPY)
    bt_te = H.backtest(sig_te, ret_te, leverage=1.0, cost_bps=1.0, ppy=PPY)

    print("\n" + "=" * 70)
    print("CHOSEN PARAMS  (1x backtest)")
    print("=" * 70)
    print(f"params: n={best_n}, volwin={best_vw}")
    print(f"TRAIN 1x: sharpe={bt_tr['sharpe']:.3f}  ret={bt_tr['ret']*100:.2f}%  "
          f"maxdd={bt_tr['maxdd']*100:.2f}%  n={bt_tr['n']}")
    print(f"TEST  1x: sharpe={bt_te['sharpe']:.3f}  ret={bt_te['ret']*100:.2f}%  "
          f"maxdd={bt_te['maxdd']*100:.2f}%  n={bt_te['n']}   <-- THE REAL NUMBER")

    # -----------------------------------------------------------------------
    # Leverage table on TEST
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LEVERAGE TABLE ON TEST  (5/10/15/20x)")
    print("=" * 70)
    lev = H.lev_table(sig_te, ret_te, cost_bps=1.0, ppy=PPY)
    print(f"{'lev':>4} {'ret%':>10} {'sharpe':>8} {'maxdd%':>9} {'ruined':>7} {'ruin_dt':>22}")
    for L in (5, 10, 15, 20):
        d = lev[L]
        rd = d["ruin_dt"] if d["ruined"] else "-"
        print(f"{L:>4} {d['ret']*100:>10.2f} {d['sharpe']:>8.3f} "
              f"{d['maxdd']*100:>9.2f} {('YES' if d['ruined'] else 'no'):>7} {rd:>22}")

    # -----------------------------------------------------------------------
    # Signal sanity / look-ahead audit helpers
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SIGNAL SANITY")
    print("=" * 70)
    print(f"TEST signal range: [{sig_te.min():.3f}, {sig_te.max():.3f}], "
          f"mean={sig_te.mean():.3f}, frac nonzero={float((sig_te!=0).mean()):.3f}")
    # first `max(n,volwin)` bars must be 0 (insufficient history -> NaN -> 0)
    warm = max(best_n, best_vw)
    print(f"First {warm} TEST signal values all zero (warmup)? "
          f"{bool((sig_te.iloc[:warm].abs() < 1e-12).all())}")

    return dict(best_n=best_n, best_vw=best_vw,
                train_sharpe=bt_tr["sharpe"], test_sharpe=bt_te["sharpe"],
                test_ret=bt_te["ret"], lev=lev, ncombos=len(grid))


if __name__ == "__main__":
    main()
