"""
Example 18 — the variance risk premium (VRP) and a short-vol strategy, with tail accounting.

Range vol (16) showed implied (VIX) sits above realized most days — a harvestable premium.
Here we measure it and build the canonical short-variance strategy from VIX + NQ, then do the
honest part: the TAIL. Short-vol is famously 'picking up pennies in front of a steamroller'.

  - VRP = VIX(t) minus the subsequent 21d realized vol (ex-post premium).
  - Short-variance daily P&L ~ implied_var(t-1) - realized_var(t)  [sell vol, pay realized],
    normalized to 15% vol so CAGR/maxDD are comparable to the equity strategies.
  - Tail accounting: skew, worst days (2008/2018/2020), and a tail-CAPPED variant
    (proxy for buying cheap wings) — how much premium survives once you hedge the tail.

Run:  python examples/18_vrp_shortvol.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"


def perf(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "skew": float(pd.Series(r).skew()),
            "maxDD": float((eq / eq.cummax() - 1).min()), "term": float(eq.iloc[-1])}


def main():
    nq = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    # VXN = CBOE Nasdaq-100 volatility index = the MATCHED implied vol for NQ (not VIX/S&P)
    vix = pd.read_csv(os.path.join(DATA_DIR, "VXN_all_history.csv"),
                      parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = nq.pct_change().dropna()
    idx = ret.index.intersection(vix.index)
    ret, vix = ret.loc[idx], vix.loc[idx]
    vix0 = vix                                       # real VXN (no proxy needed)

    # ---- 1) the premium ------------------------------------------------------
    realized_fwd = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100   # next-21d realized vol
    vrp = (vix - realized_fwd).dropna()
    print(f"NQ/VXN {idx.min().date()}..{idx.max().date()}   (matched implied = real VXN)")
    print("=" * 72)
    print(f"  VARIANCE RISK PREMIUM (VXN minus subsequent 21d realized vol):")
    print(f"    mean VXN {vix.mean():.1f}  mean realized {realized_fwd.mean():.1f}  "
          f"VRP {vrp.mean():+.1f} vol pts, positive {100*(vrp>0).mean():.0f}% of days")

    # ---- 2) short-variance daily P&L (ADDITIVE, per unit variance notional) ---
    iv = (vix.shift(1) / 100) ** 2 / 252          # implied daily variance (causal strike)
    rvar = ret ** 2                                # realized daily variance
    pnl = (iv - rvar).dropna()                      # short-variance daily P&L

    def vol_equiv(var_mean):                        # ann variance -> ann vol %
        return np.sqrt(var_mean * 252) * 100
    print("\n" + "=" * 72)
    print("SHORT-VARIANCE P&L (sell implied variance, pay realized) — VARIANCE vs VOL terms")
    print("=" * 72)
    print(f"  in VOL terms (median day): implied richer {100*(vrp>0).mean():.0f}% of days -> looks like a premium")
    print(f"  in VARIANCE terms (what you actually pay): implied-equiv {vol_equiv(iv.mean()):.1f}% "
          f"vs realized-equiv {vol_equiv(rvar.mean()):.1f}%")
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
    net_gap = vol_equiv(iv.mean()) - vol_equiv(rvar.mean())     # +ve = implied exceeds realized (premium)
    verdict = "NET POSITIVE" if net_gap > 0 else "NET NEGATIVE"
    print(f"  => implied exceeds realized by {net_gap:+.1f} vol pts -> short-vol is {verdict} "
          f"(P&L Sharpe {sharpe:+.2f}, skew {pnl.skew():.1f})")
    print(f"  the premium is real BUT the skew is catastrophic: a few huge-move days dominate the tail.")

    # ---- 3) tail accounting --------------------------------------------------
    print("\n  TAIL: 5 worst days (variance spikes — note they're huge moves, UP or down):")
    worst = pnl.nsmallest(5)
    daily_prem = pnl.mean()
    for d, v in worst.items():
        print(f"    {d.date()}  loss = {abs(v/daily_prem):,.0f}x an average day's premium   "
              f"(VXN {vix0.get(d, float('nan')):.0f}, NQ {ret.get(d,0)*100:+.1f}%)")

    # ---- 4) fair-value tail hedge: own the wings above a cap -----------------
    print("\n" + "=" * 72)
    print("TAIL-HEDGED: own options that cap realized variance at a daily move `cap`")
    print("  (fair-value wing cost subtracted) — fixes the skew, doesn't conjure a premium")
    print("=" * 72)
    print(f"  {'cap (daily move)':>18}{'P&L Sharpe':>12}{'skew':>8}{'mean P&L':>10}")
    print(f"  {'unhedged':>18}{sharpe:>12.2f}{pnl.skew():>8.1f}{('positive' if pnl.mean()>0 else 'negative'):>10}")
    for mv in (0.07, 0.05):
        capv = mv ** 2
        wing_cost = np.maximum(rvar - capv, 0).mean()          # fair premium of the wing
        hedged = (iv - np.minimum(rvar, capv) - wing_cost).dropna()
        sign = "negative" if hedged.mean() < 0 else "positive"
        print(f"  {f'+/-{mv*100:.0f}%':>18}{hedged.mean()/hedged.std()*np.sqrt(252):>12.2f}"
              f"{hedged.skew():>8.1f}{sign:>10}")
    print("\n  Read (with REAL VXN — the matched NQ implied vol): the VRP on NQ is REAL and")
    print("  harvestable — VXN exceeds realized +3.6 vol pts (79% of days), net positive even in")
    print("  variance terms, raw short-vol P&L Sharpe +1.3. The catch is the SKEW (-10): a handful")
    print("  of huge-move days (2008, 2020) define the tail. TAIL-HEDGING (owning wings) is the")
    print("  right way to run it — it lifts Sharpe to ~2.2 and tames skew to ~-1 while KEEPING a")
    print("  positive mean. NB: an earlier VIX*1.15 PROXY wrongly showed no premium — the real VXN")
    print("  flips it. Caveats: in-sample, simplified variance-swap proxy, no option transaction")
    print("  costs/slippage, and it's still a negatively-skewed insurance business (size for the tail).")


if __name__ == "__main__":
    main()
