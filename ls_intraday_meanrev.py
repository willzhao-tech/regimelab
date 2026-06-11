"""
LONG-SHORT NQ intraday mean-reversion strategy.
Family: intraday_meanrev. 1h bars.

Signal: long when RSI(n) < lo, short when RSI(n) > hi, flat otherwise. Values in {-1,0,+1}.
Grid: n in {4,8,14}, (lo,hi) in {(30,70),(20,80),(10,90)}  -> 9 combos.
Optimize on TRAIN (first 3m, 50%), pick best TRAIN Sharpe, report held-out TEST (last 3m).
ppy = 252*23.

CAUSAL DESIGN / LOOK-AHEAD AUDIT
--------------------------------
- RSI(n) at bar t is computed purely from closes up to and including bar t (Wilder's
  smoothing via ewm). No future bars enter the calc.
- The signal series is built from RSI on the FULL series, but each entry signal[t]
  depends only on info available at the close of bar t. We then split AFTER building,
  so TEST signals use only TEST-and-earlier RSI -> no cross-split leakage of *parameters*
  is the only thing the grid touches, and the grid only ever reads TRAIN.
- H.backtest does pos = signal.shift(1): the signal decided at bar t is applied to the
  return of bar t+1. So we never trade on the same bar's return we used to decide.
- ret is the close-to-close pct_change (return of bar t = Close[t]/Close[t-1]-1), and it
  is shifted by the harness so signal[t] meets ret[t+1]. Confirmed causal.
- The only parameter selection (n, lo, hi) is chosen by max Sharpe on TRAIN ONLY.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import ls_harness as H
import numpy as np
import pandas as pd

PPY = 252 * 23


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder's RSI, causal (only past+current closes)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder smoothing == ewm with alpha=1/n
    roll_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down
    out = 100.0 - 100.0 / (1.0 + rs)
    return out


def build_signal(close: pd.Series, n: int, lo: float, hi: float) -> pd.Series:
    r = rsi(close, n)
    sig = pd.Series(0.0, index=close.index)
    sig[r < lo] = 1.0    # oversold -> long
    sig[r > hi] = -1.0   # overbought -> short
    sig[r.isna()] = 0.0  # warmup: stay flat
    return sig


def main():
    df = H.load_1h()
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    # Split 50/50 by time: first 3m TRAIN, last 3m TEST.
    close_tr, close_te = H.split(close)
    ret_tr, ret_te = H.split(ret)
    print(f"TRAIN bars: {len(close_tr)}  {close_tr.index.min()} -> {close_tr.index.max()}")
    print(f"TEST  bars: {len(close_te)}  {close_te.index.min()} -> {close_te.index.max()}")

    ns = [4, 8, 14]
    bands = [(30, 70), (20, 80), (10, 90)]

    results = []
    for n in ns:
        for lo, hi in bands:
            sig_tr = build_signal(close_tr, n, lo, hi)
            bt = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=PPY)
            results.append((n, lo, hi, bt["sharpe"], bt["ret"], bt["maxdd"]))

    combos = len(results)
    print(f"\n#param-combos tried on TRAIN: {combos}")
    print("\nTRAIN grid (n, lo, hi -> sharpe, ret, maxdd):")
    for n, lo, hi, sh, rr, dd in results:
        print(f"  n={n:>2} lo={lo:>2} hi={hi:>2}  sharpe={sh:7.3f}  ret={rr:8.4f}  maxdd={dd:7.4f}")

    # Pick best TRAIN Sharpe (NaN-safe).
    valid = [r for r in results if not (r[3] != r[3])]  # drop NaN sharpe
    best = max(valid, key=lambda r: r[3])
    bn, blo, bhi = best[0], best[1], best[2]
    print(f"\nBEST TRAIN params: n={bn}, lo={blo}, hi={bhi}  (TRAIN sharpe={best[3]:.4f})")

    # Report TRAIN + TEST at 1x on chosen params.
    sig_tr = build_signal(close_tr, bn, blo, bhi)
    sig_te = build_signal(close_te, bn, blo, bhi)
    bt_tr = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=PPY)
    bt_te = H.backtest(sig_te, ret_te, leverage=1.0, cost_bps=1.0, ppy=PPY)

    print("\n=== CHOSEN PARAMS @ 1x ===")
    print(f"TRAIN: sharpe={bt_tr['sharpe']:.4f}  ret={bt_tr['ret']:.4f}  maxdd={bt_tr['maxdd']:.4f}  n={bt_tr['n']}")
    print(f"TEST : sharpe={bt_te['sharpe']:.4f}  ret={bt_te['ret']:.4f}  maxdd={bt_te['maxdd']:.4f}  n={bt_te['n']}")

    # Leverage table on TEST.
    print("\n=== LEVERAGE TABLE on TEST (5/10/15/20x) ===")
    lt = H.lev_table(sig_te, ret_te, cost_bps=1.0, ppy=PPY)
    for L in (5, 10, 15, 20):
        d = lt[L]
        flag = "Y" if d["ruined"] else "N"
        rdt = d["ruin_dt"] if d["ruined"] else "-"
        print(f"  {L:>2}x: ret={d['ret']:10.4f}  sharpe={d['sharpe']:7.3f}  maxdd={d['maxdd']:7.4f}  RUINED={flag}  ruin_dt={rdt}")

    # Activity diagnostics on TEST.
    nz_te = int((sig_te != 0).sum())
    print(f"\nTEST signal activity: {nz_te}/{len(sig_te)} bars nonzero ({100*nz_te/len(sig_te):.1f}%)")


if __name__ == "__main__":
    main()
