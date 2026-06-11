# -*- coding: utf-8 -*-
"""
P1-5 ATTRIBUTION: decompose the blend OOS P&L from saved volarb ledgers.

Inputs (all OOS, 2006-02-13 .. 2026-03):
    C:/Users/ASUS/Desktop/claude doc/1/volarb_ledger_blend.csv
    C:/Users/ASUS/Desktop/claude doc/1/volarb_ledger_A.csv
    C:/Users/ASUS/Desktop/claude doc/1/volarb_ledger_B.csv

Decompositions (descriptive only -- no parameters fitted here):
  (a) by position state: short-vol days (pos>0), long-vol days (pos<0), flat (pos==0)
  (b) by VXN regime at signal time (VXN.shift(1)): <15, 15-25, >25
  (c) by calendar year
  (d) crisis episodes vs static short-vol (pos=+1 always, pnl = iv - rvar, no
      trading cost since the static book never trades)

Outputs:
    C:/Users/ASUS/Desktop/claude doc/1/p1_attribution.csv
    C:/Users/ASUS/Desktop/claude doc/1/p1_attribution.png

Honesty notes:
  * P&L is the OLD variance-swap proxy P&L (daily variance units), same as the
    ledgers; nothing is re-simulated, no tails removed, no clipping.
  * Bucket Sharpes are conditional Sharpes (mean/std * sqrt(252) of daily pnl
    *within* the bucket). They describe where the unconditional Sharpe comes
    from; they are NOT achievable standalone Sharpes because the bucket
    membership is only known ex post (year, VXN regime) or is the strategy's
    own choice (position state).
  * VXN regime uses VXN.shift(1) -- the value that priced the iv strike for
    that day's pnl -- so the bucketing is causal w.r.t. the pnl it explains.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = r"C:\Users\ASUS\Desktop\claude doc\1"
ANN = np.sqrt(252.0)


def load_ledger(name):
    df = pd.read_csv(os.path.join(ART, f"volarb_ledger_{name}.csv"),
                     parse_dates=["Date"]).set_index("Date").sort_index()
    return df


def sharpe(pnl):
    pnl = pd.Series(pnl).dropna()
    if len(pnl) < 20 or pnl.std(ddof=1) == 0:
        return np.nan
    return float(pnl.mean() / pnl.std(ddof=1) * ANN)


def bucket_row(section, bucket, pnl_bucket, total_pnl, extra=None):
    row = {
        "section": section,
        "bucket": bucket,
        "n_days": int(len(pnl_bucket)),
        "total_pnl": float(pnl_bucket.sum()),
        "pnl_share_pct": float(100.0 * pnl_bucket.sum() / total_pnl),
        "ann_sharpe_in_bucket": sharpe(pnl_bucket),
    }
    if extra:
        row.update(extra)
    return row


def main():
    blend = load_ledger("blend")
    led_a = load_ledger("A")
    led_b = load_ledger("B")

    pnl = blend["daily_pnl"]
    pos = blend["position"]
    total = pnl.sum()
    full_sharpe = sharpe(pnl)
    print(f"Blend OOS: {blend.index[0].date()} .. {blend.index[-1].date()}, "
          f"{len(blend)} days, total pnl {total:.6f}, Sharpe {full_sharpe:.2f}")

    # Static short-vol benchmark from the same ledger columns:
    # pos = +1 always -> pnl = iv - rvar, dpos = 0 -> no cost.
    static_pnl = blend["iv_strike_daily_var"] - blend["realized_var"]
    print(f"Static short-vol: total pnl {static_pnl.sum():.6f}, "
          f"Sharpe {sharpe(static_pnl):.2f}")

    rows = []
    rows.append(bucket_row("overall", "blend_all_days", pnl, total))
    rows.append({"section": "overall", "bucket": "static_short_vol_all_days",
                 "n_days": int(len(static_pnl)),
                 "total_pnl": float(static_pnl.sum()),
                 "pnl_share_pct": np.nan,
                 "ann_sharpe_in_bucket": sharpe(static_pnl)})

    # ---------------- (a) position state ----------------
    states = {
        "short_vol (pos>0)": pos > 0,
        "long_vol (pos<0)": pos < 0,
        "flat (pos==0)": pos == 0,
    }
    for name, mask in states.items():
        rows.append(bucket_row("a_position_state", name, pnl[mask], total))

    # ---------------- (b) VXN regime (signal-time VXN.shift(1)) ----------------
    vxn_lag = blend["VXN"].shift(1)
    regimes = {
        "VXN<15": vxn_lag < 15,
        "VXN 15-25": (vxn_lag >= 15) & (vxn_lag <= 25),
        "VXN>25": vxn_lag > 25,
    }
    for name, mask in regimes.items():
        mask = mask.fillna(False)
        rows.append(bucket_row("b_vxn_regime", name, pnl[mask], total))

    # ---------------- (c) calendar year ----------------
    for yr, grp in pnl.groupby(pnl.index.year):
        rows.append(bucket_row("c_year", str(yr), grp, total))

    # ---------------- (d) crisis episodes vs static short-vol ----------------
    episodes = [
        ("2008-09..2009-03 (GFC)", "2008-09-01", "2009-03-31"),
        ("2011-08..10 (US dgrade)", "2011-08-01", "2011-10-31"),
        ("2018-02 (Volmageddon)", "2018-02-01", "2018-02-28"),
        ("2018-10..12 (Q4 selloff)", "2018-10-01", "2018-12-31"),
        ("2020-02..04 (COVID)", "2020-02-01", "2020-04-30"),
        ("2022 (full year)", "2022-01-01", "2022-12-31"),
        ("2025-04 (tariff shock)", "2025-04-01", "2025-04-30"),
    ]
    for name, d0, d1 in episodes:
        m = (blend.index >= d0) & (blend.index <= d1)
        pb, ps = pnl[m], static_pnl[m]
        rows.append(bucket_row(
            "d_crisis", name, pb, total,
            extra={"static_total_pnl": float(ps.sum()),
                   "blend_minus_static": float(pb.sum() - ps.sum()),
                   "pos_share_short_pct": float(100 * (pos[m] > 0).mean()),
                   "pos_share_long_pct": float(100 * (pos[m] < 0).mean())}))

    out = pd.DataFrame(rows)
    csv_path = os.path.join(ART, "p1_attribution.csv")
    out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"saved {csv_path}")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))

    # cross-check: A and B totals for context
    print(f"\ncontext: A total {led_a['daily_pnl'].sum():.6f} "
          f"(Sharpe {sharpe(led_a['daily_pnl']):.2f}), "
          f"B total {led_b['daily_pnl'].sum():.6f} "
          f"(Sharpe {sharpe(led_b['daily_pnl']):.2f})")
    # blend identity check
    blend_chk = 0.5 * (led_a["daily_pnl"] + led_b["daily_pnl"])
    print(f"blend == 0.5*(A+B)? max abs diff "
          f"{(blend_chk - pnl).abs().max():.2e}")

    # ---------------- chart ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "P1-5 blend OOS P&L attribution (variance-units proxy P&L, "
        f"{blend.index[0].date()}..{blend.index[-1].date()}, "
        f"Sharpe {full_sharpe:.2f})", fontsize=13)

    def annotate(ax, bars, vals, fmt="{:.1f}"):
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(fmt.format(v),
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center",
                            va="bottom" if b.get_height() >= 0 else "top",
                            fontsize=8)

    # (a) position state: share + sharpe
    sub = out[out.section == "a_position_state"]
    ax = axes[0, 0]
    x = np.arange(len(sub))
    bars = ax.bar(x - 0.2, sub.pnl_share_pct, 0.4, label="pnl share %",
                  color="#1f77b4")
    annotate(ax, bars, sub.pnl_share_pct.values)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.2, sub.ann_sharpe_in_bucket, 0.4,
                 label="cond. Sharpe", color="#ff7f0e")
    annotate(ax2, b2, sub.ann_sharpe_in_bucket.values, "{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(sub.bucket, sub.n_days)],
                       fontsize=8)
    ax.set_ylabel("pnl share %"); ax2.set_ylabel("conditional Sharpe")
    ax.set_title("(a) by position state")
    ax.axhline(0, color="k", lw=0.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

    # (b) VXN regime
    sub = out[out.section == "b_vxn_regime"]
    ax = axes[0, 1]
    x = np.arange(len(sub))
    bars = ax.bar(x - 0.2, sub.pnl_share_pct, 0.4, label="pnl share %",
                  color="#1f77b4")
    annotate(ax, bars, sub.pnl_share_pct.values)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.2, sub.ann_sharpe_in_bucket, 0.4,
                 label="cond. Sharpe", color="#ff7f0e")
    annotate(ax2, b2, sub.ann_sharpe_in_bucket.values, "{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(sub.bucket, sub.n_days)],
                       fontsize=8)
    ax.set_ylabel("pnl share %"); ax2.set_ylabel("conditional Sharpe")
    ax.set_title("(b) by lagged VXN regime")
    ax.axhline(0, color="k", lw=0.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

    # (c) calendar year pnl
    sub = out[out.section == "c_year"]
    ax = axes[1, 0]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in sub.total_pnl]
    ax.bar(sub.bucket, sub.total_pnl * 1e4, color=colors)
    ax.set_title("(c) pnl by calendar year (1e-4 variance units)")
    ax.axhline(0, color="k", lw=0.5)
    ax.tick_params(axis="x", rotation=70, labelsize=8)

    # (d) crisis: blend vs static
    sub = out[out.section == "d_crisis"]
    ax = axes[1, 1]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub.total_pnl * 1e4, 0.4, label="blend", color="#1f77b4")
    ax.bar(x + 0.2, sub.static_total_pnl * 1e4, 0.4, label="static short-vol",
           color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels([b.split(" (")[0] for b in sub.bucket],
                       rotation=30, ha="right", fontsize=8)
    ax.set_title("(d) crisis episodes: blend vs static short-vol "
                 "(1e-4 variance units)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = os.path.join(ART, "p1_attribution.png")
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
