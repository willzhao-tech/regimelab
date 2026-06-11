"""
Example 8 — OUT-OF-SAMPLE validation of the vol-targeting (+trend) edge.

07 was in-sample, one path, NQ only, and tuning the params would be a search. This
script closes those gaps two ways:

  PART A — CROSS-ASSET, fixed a-priori config (no fitting): apply ONE config
           (63d vol window, 15% target, optional 200d-MA trend brake) UNCHANGED to
           all six tradables. If the edge is real it should generalize, not just fit NQ.

  PART B — WALK-FORWARD parameter selection (no look-ahead): each test year, pick the
           config with the best Sharpe on the PRIOR 5y of data only, then apply it to
           the next 1y out-of-sample; accumulate. Tests that the edge survives honest,
           peek-free parameter choice — not just one lucky setting.

Config grid (Part B): vol_win in {21,63,126} x trend in {none, 200d}.
All causal (1-day delay), monthly leverage rebal, 5bps/switch, leverage capped 3x.

Run:  python examples/08_oos_validation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
FILES = {"NQ": "NQ_F_all_history.csv", "A50": "A50_all_history.csv",
         "US10Y": "US10Y_all_history.csv", "WTI": "WTI_all_history.csv",
         "XAU": "XAU_all_history.csv", "EURUSD": "EURUSD_all_history.csv"}
TARGET, COST, MAXLEV = 0.15, 0.0005, 3.0
GRID = [(vw, tr) for vw in (21, 63, 126) for tr in (None, 200)]   # 6 configs


def load_prices():
    out = {}
    for name, f in FILES.items():
        s = pd.read_csv(os.path.join(DATA_DIR, f), parse_dates=["Date"]).set_index("Date")["Close"]
        out[name] = s.sort_index().dropna()
    return out


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return {"sharpe": np.nan, "cagr": np.nan, "maxdd": np.nan, "vol": np.nan, "term": np.nan}
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"sharpe": r.mean() / r.std() * np.sqrt(252), "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "maxdd": float((eq / eq.cummax() - 1).min()), "vol": r.std() * np.sqrt(252),
            "term": float(eq.iloc[-1])}


def vt_overlay(ret, price, vol_win, trend_ma, target=TARGET, cost=COST, max_lev=MAXLEV):
    """Causal vol-targeted position on one asset; optional trend brake; monthly rebal."""
    vol = ret.rolling(vol_win).std() * np.sqrt(252)
    sig = (price > price.rolling(trend_ma).mean()).astype(float) if trend_ma else pd.Series(1.0, index=ret.index)
    desired = ((target / vol).clip(upper=max_lev) * sig).shift(1).fillna(0.0)
    pos, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        pos.append(cur); pm = d
    pos = pd.Series(pos, index=desired.index)
    return pos * ret - cost * pos.diff().abs().fillna(0.0)


def main():
    prices = load_prices()

    # ---- PART A: cross-asset, fixed a-priori config ----------------------------
    print("=" * 104)
    print(f"PART A — CROSS-ASSET, fixed config (vol_win=63, target={TARGET:.0%}, no fitting)")
    print("=" * 104)
    print(f"{'asset':<8}{'buyhold Sharpe':>16}{'buyhold maxDD':>15}"
          f"{'VT Sharpe':>11}{'VT maxDD':>10}{'VT+trend Sh':>13}{'VT+trend DD':>13}")
    impr_vt, impr_tr = [], []
    for name, px in prices.items():
        ret = px.pct_change(fill_method=None).dropna()
        bh = perf(ret)
        vt = perf(vt_overlay(ret, px, 63, None))
        tr = perf(vt_overlay(ret, px, 63, 200))
        impr_vt.append(vt["sharpe"] - bh["sharpe"]); impr_tr.append(tr["sharpe"] - bh["sharpe"])
        print(f"{name:<8}{bh['sharpe']:>16.2f}{bh['maxdd']*100:>14.0f}%"
              f"{vt['sharpe']:>11.2f}{vt['maxdd']*100:>9.0f}%{tr['sharpe']:>13.2f}{tr['maxdd']*100:>12.0f}%")
    print("-" * 104)
    print(f"avg Sharpe improvement vs buy&hold:  vol-target {np.mean(impr_vt):+.2f}  "
          f"(better in {sum(x>0 for x in impr_vt)}/{len(impr_vt)});  "
          f"+trend {np.mean(impr_tr):+.2f} (better in {sum(x>0 for x in impr_tr)}/{len(impr_tr)})")

    # ---- PART B: walk-forward parameter selection (no look-ahead) ---------------
    TRAIN, TEST = 1260, 252                       # 5y train, 1y OOS test, rolling
    print("\n" + "=" * 104)
    print(f"PART B — WALK-FORWARD selection (train {TRAIN}d -> pick best-Sharpe config -> "
          f"apply next {TEST}d OOS)")
    print("=" * 104)
    print(f"{'asset':<8}{'OOS span':>22}{'buyhold Sh':>12}{'buyhold DD':>12}"
          f"{'WF-OOS Sh':>11}{'WF-OOS DD':>11}{'WF CAGR':>9}{'most-picked cfg':>18}")
    agg = []
    for name, px in prices.items():
        ret = px.pct_change(fill_method=None).dropna()
        series = {cfg: vt_overlay(ret, px, cfg[0], cfg[1]) for cfg in GRID}   # precompute causal
        idx = ret.index
        oos_parts, picks = [], []
        start = TRAIN
        while start + TEST <= len(idx):
            tr_sl = idx[start - TRAIN:start]; te_sl = idx[start:start + TEST]
            best = max(GRID, key=lambda c: perf(series[c].loc[tr_sl])["sharpe"]
                       if np.isfinite(perf(series[c].loc[tr_sl])["sharpe"]) else -1e9)
            picks.append(best)
            oos_parts.append(series[best].loc[te_sl])
            start += TEST
        oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype="float64")
        if len(oos) < 250:
            print(f"{name:<8}  (too short for walk-forward)"); continue
        bh = perf(ret.loc[oos.index]); wf = perf(oos)
        from collections import Counter
        mp = Counter(picks).most_common(1)[0][0]
        agg.append((wf["sharpe"] - bh["sharpe"], wf["maxdd"] - bh["maxdd"]))
        print(f"{name:<8}{oos.index.min().date().isoformat()+'..'+oos.index.max().date().isoformat():>22}"
              f"{bh['sharpe']:>12.2f}{bh['maxdd']*100:>11.0f}%{wf['sharpe']:>11.2f}{wf['maxdd']*100:>10.0f}%"
              f"{wf['cagr']*100:>8.1f}%{f'vw{mp[0]},tr{mp[1]}':>18}")
    print("-" * 104)
    dsh = [a[0] for a in agg]; ddd = [a[1] for a in agg]
    print(f"OOS Sharpe vs buy&hold: avg {np.mean(dsh):+.2f}, better in {sum(x>0 for x in dsh)}/{len(dsh)} assets")
    print(f"OOS maxDD vs buy&hold:  avg {np.mean(ddd)*100:+.0f}pp, shallower in "
          f"{sum(x>0 for x in ddd)}/{len(ddd)} assets  (positive = less deep)")
    print("\nVerdict: the edge is real OOS only if it improves Sharpe/drawdown across MOST assets")
    print("under peek-free parameter selection — not just on in-sample NQ.")


if __name__ == "__main__":
    main()
