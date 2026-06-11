"""Independent verification of P0-1: does the vol-arb alpha die on the rolled 21d var swap?"""
import os, numpy as np, pandas as pd
D = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

df = pd.read_csv(os.path.join(D, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(D, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
ret = df["Close"].pct_change()
idx = ret.index.intersection(vxn.index)
df, ret, vxn = df.loc[idx], ret.loc[idx], vxn.loc[idx]

V = (vxn/100.0)**2
rv1 = 252.0*ret**2
gamma_leg = (1/21)*(V.shift(1) - rv1)          # realized/carry leg
vega_leg = (20/21)*(V.shift(1) - V)            # mark-to-market leg
rolled_short = (gamma_leg + vega_leg).dropna()
proxy_short = ((vxn.shift(1)/100)**2/252 - ret**2).dropna()

def sh(p): p = p.dropna(); return p.mean()/p.std()*SQ
print("MECHANISM CHECK")
print(f"  vol(vega leg)/vol(gamma leg) = {vega_leg.std()/gamma_leg.std():.2f}   (claim: 1.79)")
print(f"  corr(rolled_short, proxy_short) = {rolled_short.corr(proxy_short.reindex(rolled_short.index)):.2f}   (claim: 0.57)")
print(f"  static rolled short-vol Sharpe (full hist) = {sh(rolled_short):.2f}  skew {rolled_short.skew():.1f}")
print(f"  static proxy short-vol Sharpe  (full hist) = {sh(proxy_short):.2f}")

# fixed-param family A on both instruments (no walk-forward, dominant params 2/-2/0)
park = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
rich = vxn - park(21); trend = park(10) - park(42)
sA = pd.Series(0.0, index=idx)
sA[((rich >= 2) & (trend <= 0)).fillna(False)] = 1.0
sA[((rich <= -2) & (trend >= 0)).fillna(False)] = -1.0
posA = sA.shift(1).fillna(0.0)
print("\nFAMILY A FIXED (2,-2,0), full history, no costs:")
print(f"  on PROXY : Sharpe {sh(posA*proxy_short.reindex(idx)):+.2f}")
print(f"  on ROLLED: Sharpe {sh(posA*rolled_short.reindex(idx)):+.2f}")

# does the signal predict the vega leg at all?
print("\nDOES THE SIGNAL PREDICT THE LEGS? (corr of position with next-day leg pnl)")
print(f"  corr(posA, gamma leg short pnl) = {posA.corr(gamma_leg.reindex(idx)):+.3f}")
print(f"  corr(posA, vega  leg short pnl) = {posA.corr(vega_leg.reindex(idx)):+.3f}")
