# -*- coding: utf-8 -*-
"""
LONG-SHORT NQ strategy, family: intraday_trend.

Causal trend signal on 1h bars:
    raw = (Close - EMA(n)) / ATR(n)         # trend distance, vol-normalized
    signal = tanh(raw)   in [-1,1]          # OR sign() for the binary variant

All quantities at bar t use only data through bar t (EMA/ATR are causal
rolling/ewm with no centering). H.backtest shifts the signal by 1 bar, so the
position taken on the close of bar t earns ret of bar t+1. No full-sample stats.

Protocol:
  - split 50/50 by time: first 3m = TRAIN, last 3m = TEST.
  - grid-search params on TRAIN ONLY, pick best TRAIN Sharpe.
  - report TRAIN & TEST 1x, plus lev_table(5/10/15/20x) on TEST with ruin flags.
  - the TEST number is the only one that counts.

ppy = 252*23 (23 RTH-ish trading hours per day for 1h bars).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import ls_harness as H

PPY = 252 * 23


def ema(x, n):
    # causal exponential moving average; only past+current bars
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df, n):
    # causal ATR (Wilder-ish via ewm on true range); uses bars up to t only
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=n).mean()


def build_signal(df, n, mode):
    """Causal long-short trend signal in [-1,1].
    mode='sign'  -> sign(Close - EMA(n))
    mode='tanh'  -> tanh((Close - EMA(n)) / ATR(n))
    """
    e = ema(df["Close"], n)
    dist = df["Close"] - e
    if mode == "sign":
        sig = np.sign(dist)
    elif mode == "tanh":
        a = atr(df, n)
        sig = np.tanh(dist / a.replace(0, np.nan))
    else:
        raise ValueError(mode)
    return sig.reindex(df.index).fillna(0.0).clip(-1, 1)


def main():
    df = H.load_1h()
    ret = df["Close"].pct_change()
    # Drop the single leading bar whose pct_change is NaN. (pct_change()[0] is
    # always NaN; a NaN return poisons the harness equity product -> ret/maxdd
    # come back NaN. Dropping bar 0 is causal: we just don't trade the first bar.)
    df = df.iloc[1:]
    ret = ret.iloc[1:]
    assert not ret.isna().any(), "ret still has NaN after dropping leading bar"

    # 50/50 time split: first 3m TRAIN, last 3m TEST
    df_tr, df_te = H.split(df, 0.5)
    ret_tr, ret_te = H.split(ret, 0.5)

    print("== data ==")
    print(f"all bars: {len(df)}  {df.index[0]} .. {df.index[-1]}")
    print(f"TRAIN   : {len(df_tr)} bars  {df_tr.index[0]} .. {df_tr.index[-1]}")
    print(f"TEST    : {len(df_te)} bars  {df_te.index[0]} .. {df_te.index[-1]}")

    # ---- grid search on TRAIN only ----
    ns = [6, 12, 24, 48]
    modes = ["sign", "tanh"]
    combos = [(n, m) for n in ns for m in modes]

    rows = []
    best = None
    for n, m in combos:
        sig_tr = build_signal(df_tr, n, m)
        bt = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=PPY)
        rows.append((n, m, bt["sharpe"], bt["ret"], bt["maxdd"]))
        sh = bt["sharpe"]
        if best is None or (np.isfinite(sh) and sh > best[2]):
            best = (n, m, sh, bt["ret"])

    print("\n== TRAIN grid (n, mode, sharpe, ret, maxdd) ==")
    for r in rows:
        print(f"  n={r[0]:>2} {r[1]:>4}  sharpe={r[2]:+.3f}  ret={r[3]:+.4f}  maxdd={r[4]:+.4f}")
    print(f"\n#param-combos tried = {len(combos)}")

    n_best, m_best, sh_tr, ret_tr_best = best
    print(f"\nBEST TRAIN params: n={n_best} mode={m_best}  TRAIN sharpe={sh_tr:+.3f}")

    # ---- apply chosen params: build signal on TRAIN and TEST segments separately ----
    sig_tr = build_signal(df_tr, n_best, m_best)
    sig_te = build_signal(df_te, n_best, m_best)

    bt_tr = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=PPY)
    bt_te = H.backtest(sig_te, ret_te, leverage=1.0, cost_bps=1.0, ppy=PPY)

    print("\n== chosen params, 1x ==")
    print(f"TRAIN: sharpe={bt_tr['sharpe']:+.3f}  ret={bt_tr['ret']:+.4f}  maxdd={bt_tr['maxdd']:+.4f}  ruined={bt_tr['ruined']}")
    print(f"TEST : sharpe={bt_te['sharpe']:+.3f}  ret={bt_te['ret']:+.4f}  maxdd={bt_te['maxdd']:+.4f}  ruined={bt_te['ruined']}")

    # ---- leverage table on TEST ----
    lt = H.lev_table(sig_te, ret_te, cost_bps=1.0, ppy=PPY)
    print("\n== TEST leverage table (5/10/15/20x) ==")
    for L in (5, 10, 15, 20):
        d = lt[L]
        rr = "Y" if d["ruined"] else "N"
        print(f"  {L:>2}x  ret={d['ret']:+.4f}  sharpe={d['sharpe']:+.3f}  maxdd={d['maxdd']:+.4f}  ruined={rr}  ruin_dt={d['ruin_dt'] if d['ruined'] else '-'}")

    # machine-readable summary line for the orchestrator
    print("\n== SUMMARY ==")
    print(f"FAMILY=intraday_trend")
    print(f"BEST_PARAMS=n={n_best},mode={m_best}")
    print(f"TRAIN_SHARPE={bt_tr['sharpe']:.4f}")
    print(f"TEST_SHARPE={bt_te['sharpe']:.4f}")
    print(f"TEST_RET_1X={bt_te['ret']:.6f}")
    print(f"COMBOS={len(combos)}")
    for L in (5, 10, 15, 20):
        d = lt[L]
        print(f"LEV_{L}=ret={d['ret']:.6f},ruined={int(d['ruined'])},ruin_dt={d['ruin_dt'] if d['ruined'] else '-'}")


if __name__ == "__main__":
    main()
