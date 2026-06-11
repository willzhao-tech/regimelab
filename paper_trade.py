"""Live paper-trading harness: risk-managed equity overlay + pre-FOMC tilt.

Strategy (validated across examples 04-13):
  - vol-target (63d) + 200d-trend brake on NQ & A50, 15% vol target, 3x cap, monthly rebal;
  - PLUS a pre-FOMC tilt on the NQ sleeve: +1.0x extra long on the trading day before and the
    day of each SCHEDULED FOMC announcement (cap 4x). FOMC dates are known months ahead, so
    this is NOT look-ahead. The tilt added +2.9%/yr (t 2.3) in backtest, lifting the overlay to
    buy&hold-like return at ~1/3 the drawdown.
  - traded sleeve: 50/50 NQ/A50. Risk-managed equity + a real catalyst edge — modest, honest.

PAPER ONLY — no real orders. Recomputes from the auto-updating CSVs each run (idempotent);
writes paper_ledger.csv; prints a dashboard incl. upcoming FOMC windows. Schedule daily.

    python paper_trade.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
ASSETS = {"NQ": "NQ_F_all_history.csv", "A50": "A50_all_history.csv"}
WEIGHTS = {"NQ": 0.5, "A50": 0.5}
VOL_WIN, TREND, TARGET, MAXLEV, COST = 63, 200, 0.15, 3.0, 0.0005
FOMC_BOOST, CAP_TILT = 1.0, 4.0               # NQ sleeve only
INCEPTION = "2026-06-09"
LEDGER = os.path.join(DATA_DIR, "paper_ledger.csv")
LEDGER_START = "2022-01-01"
FOMC_FILE = os.path.join(DATA_DIR, "FOMC_dates.csv")


def _positions(ret, px):
    vol = ret.rolling(VOL_WIN).std() * np.sqrt(252)
    sig = (px > px.rolling(TREND).mean()).astype(float)
    desired = ((TARGET / vol).clip(upper=MAXLEV) * sig).shift(1).fillna(0.0)
    pos, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        pos.append(cur); pm = d
    return pd.Series(pos, index=desired.index)


def _scheduled_fomc():
    if not os.path.exists(FOMC_FILE):
        return pd.DatetimeIndex([])
    f = pd.read_csv(FOMC_FILE, parse_dates=["fomc_date"])["fomc_date"].sort_values()
    kept, last = [], None
    for d in f:                                # drop emergency intermeeting (min 20-day gap)
        if last is None or (d - last).days >= 20:
            kept.append(d); last = d
    return pd.DatetimeIndex(kept)


def _fomc_window(idx, sched):
    """True on trading day -1 and day 0 of each scheduled FOMC date within idx."""
    mask = pd.Series(False, index=idx)
    in_range = sched[(sched >= idx.min()) & (sched <= idx.max())]
    locs = idx.get_indexer(in_range, method="bfill")
    for p in locs:
        if 1 <= p < len(idx):
            mask.iloc[p - 1] = True; mask.iloc[p] = True
    return mask


def _stats(r):
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return {"ret": float((1 + r).prod() - 1) if len(r) else 0.0,
                "ann": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan, "n": len(r)}
    eq = (1 + r).cumprod(); yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1 / 365.25)
    return {"ret": float(eq.iloc[-1] - 1), "ann": eq.iloc[-1] ** (1 / yrs) - 1,
            "vol": r.std() * np.sqrt(252), "sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxdd": float((eq / eq.cummax() - 1).min()), "n": len(r)}


def main():
    sched = _scheduled_fomc()
    per_asset, ledger_cols = {}, {}
    for name, fn in ASSETS.items():
        px = pd.read_csv(os.path.join(DATA_DIR, fn), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
        ret = px.pct_change(fill_method=None).dropna()
        pos = _positions(px.pct_change().dropna(), px).reindex(ret.index).fillna(0.0)
        win = _fomc_window(ret.index, sched)
        if name == "NQ":                       # pre-FOMC tilt on the US-equity sleeve only
            pos = pos.where(~win, np.minimum(pos + FOMC_BOOST, CAP_TILT))
        pnl = pos * ret - COST * pos.diff().abs().fillna(0.0)
        ma = px.rolling(TREND).mean()
        per_asset[name] = {"px": px, "ret": ret, "pos": pos, "pnl": pnl, "ma": ma, "win": win}
        ledger_cols[f"{name}_price"] = px
        ledger_cols[f"{name}_pos"] = pos
        ledger_cols[f"{name}_pnl"] = pnl

    pnls = pd.concat({k: v["pnl"] for k, v in per_asset.items()}, axis=1).dropna()
    combo = sum(WEIGHTS[k] * pnls[k] for k in ASSETS)
    ledger_cols["combo_pnl"] = combo
    ledger_cols["combo_equity"] = (1 + combo).cumprod()
    led = pd.DataFrame(ledger_cols).sort_index()
    led.loc[led.index >= pd.to_datetime(LEDGER_START)].to_csv(LEDGER, index_label="Date")

    today = combo.index.max()
    inc = pd.to_datetime(INCEPTION)
    live = combo.loc[combo.index >= inc]
    ly = combo.loc[combo.index >= today - pd.Timedelta(days=365)]

    print("=" * 86)
    print(f"PAPER DASHBOARD — equity overlay + pre-FOMC tilt (50/50 NQ/A50)   asof {today.date()}")
    print("=" * 86)
    print(f"{'asset':<6}{'last px':>12}{'vs 200dMA':>12}{'hold (lev)':>12}{'FOMC win':>10}{'state':>9}")
    for name in ASSETS:
        a = per_asset[name]
        last_px = a["px"].iloc[-1]; ma = a["ma"].iloc[-1]; lev = a["pos"].iloc[-1]
        fw = "BOOST" if a["win"].iloc[-1] else "-"
        state = "ABOVE" if last_px > ma else "below"
        print(f"{name:<6}{last_px:>12.1f}{(last_px/ma-1)*100:>11.1f}%{lev:>12.2f}{fw:>10}{state:>9}")

    # upcoming FOMC windows
    upcoming = sched[sched > today][:3]
    print("\n  Upcoming scheduled FOMC announcements (NQ sleeve tilts +1x on the prior day + the day):")
    for d in upcoming:
        print(f"    {d.date()}  (in {(d - today).days} days; tilt active {(d - pd.Timedelta(days=1)).date()}..{d.date()})")
    if len(per_asset["NQ"]["win"]) and per_asset["NQ"]["win"].iloc[-1]:
        print("    >> TODAY is inside an FOMC window — NQ sleeve is boosted.")

    print(f"\n  LIVE since inception {INCEPTION} ({len(live)} trading days):")
    s = _stats(live)
    if s["n"] <= 1:
        print("    (inception is today — live curve starts accruing from the next update)")
    else:
        print(f"    return {s['ret']*100:+.1f}%  ann {s['ann']*100:+.1f}%  vol {s['vol']*100:.0f}%  "
              f"Sharpe {s['sharpe']:.2f}  maxDD {s['maxdd']*100:.0f}%")
    ctx, full = _stats(ly), _stats(combo)
    print(f"  trailing 1y (backtest context): ann {ctx['ann']*100:+.1f}%  Sharpe {ctx['sharpe']:.2f}  maxDD {ctx['maxdd']*100:.0f}%")
    print(f"  full backtest {combo.index.min().date()}..{today.date()}: ann {full['ann']*100:+.1f}%  "
          f"Sharpe {full['sharpe']:.2f}  maxDD {full['maxdd']*100:.0f}%")
    print(f"\n  ledger -> {LEDGER}")
    print("  NOTE: paper only — no real orders. Risk-managed equity + a small real catalyst edge.")

    try:
        volbook_section()
    except Exception as e:                       # vol-book add-on must never break the dashboard
        print(f"\n  [vol-book section failed: {type(e).__name__}: {e}]")


# ---------------------------------------------------------------------------
# VOL-ARB PAPER SIGNAL (P3-12) — regime-gated long-short variance on NQ/VXN.
# Fixed dominant walk-forward params (A: r_hi=2, r_lo=-2, d=0; B: b1=1.0, b2=1.6).
# Variance-swap PROXY P&L; tracked on paper only. Inception = first run date.
# ---------------------------------------------------------------------------
VOLARB_LEDGER = os.path.join(DATA_DIR, "paper_volarb_ledger.csv")
VOLARB_INCEPTION = "2026-06-10"


def volarb_section():
    dfq = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                      parse_dates=["Date"]).set_index("Date").sort_index()
    vxn = pd.read_csv(os.path.join(DATA_DIR, "VXN_all_history.csv"),
                      parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = dfq["Close"].pct_change()
    idx = ret.index.intersection(vxn.index)
    dfq, ret, vxn = dfq.loc[idx], ret.loc[idx], vxn.loc[idx]
    SQ = np.sqrt(252.0)

    def park(w):                                  # trailing, causal (shifted)
        return (np.sqrt((np.log(dfq["High"]/dfq["Low"])**2).rolling(w).mean()
                        / (4*np.log(2)))*SQ*100).shift(1)

    rich = vxn - park(21)
    trend = park(10) - park(42)
    rng = np.log(dfq["High"]/dfq["Low"])*100.0
    be = vxn / SQ
    sA = pd.Series(0.0, index=idx)
    sA[((rich >= 2.0) & (trend <= 0.0)).fillna(False)] = 1.0
    sA[((rich <= -2.0) & (trend >= 0.0)).fillna(False)] = -1.0
    sB = pd.Series(0.0, index=idx)
    ok = rng.notna() & be.notna()
    sB[ok & (rng < 1.0*be)] = 1.0
    sB[ok & (rng > 1.6*be)] = -1.0
    blend_sig = 0.5*sA + 0.5*sB

    iv = (vxn.shift(1)/100.0)**2/252.0
    pos = blend_sig.shift(1).fillna(0.0)
    cost = (2*vxn.shift(1)*0.5/1e4/252).fillna(0.0)*pos.diff().abs().fillna(0.0)
    pnl = (pos*(iv - ret**2) - cost).dropna()

    led = pd.DataFrame({"VXN": vxn.reindex(pnl.index), "richness": rich.reindex(pnl.index).round(2),
                        "vol_trend": trend.reindex(pnl.index).round(2),
                        "sigA": sA.reindex(pnl.index), "sigB": sB.reindex(pnl.index),
                        "position": pos.reindex(pnl.index), "daily_pnl": pnl.round(9),
                        "cum_pnl": pnl.cumsum().round(7)})
    led.loc[led.index >= "2024-01-01"].to_csv(VOLARB_LEDGER, index_label="Date")

    today = pnl.index.max()
    state = {1.0: "SHORT VOL", 0.5: "HALF SHORT", -0.5: "HALF LONG", -1.0: "LONG VOL", 0.0: "FLAT"}
    cur = blend_sig.iloc[-1]
    print("\n" + "=" * 86)
    print(f"VOL-ARB PAPER SIGNAL (regime-gated long-short variance, NQ/VXN proxy)   asof {today.date()}")
    print("=" * 86)
    print(f"  VXN {vxn.iloc[-1]:.1f} | richness {rich.iloc[-1]:+.1f} vol-pts | vol-trend {trend.iloc[-1]:+.1f} "
          f"| range/breakeven {rng.iloc[-1]/be.iloc[-1]:.2f}")
    print(f"  signal A {sA.iloc[-1]:+.1f}  signal B {sB.iloc[-1]:+.1f}  ->  BLEND TOMORROW: "
          f"{state.get(cur, f'{cur:+.1f}')}")
    live = pnl.loc[pnl.index >= pd.to_datetime(VOLARB_INCEPTION)]
    if len(live) > 1:
        sh = live.mean()/live.std()*SQ if live.std() > 0 else float("nan")
        print(f"  LIVE since {VOLARB_INCEPTION}: {len(live)} days, cum pnl {live.sum()*1e4:+.2f}e-4/unit, "
              f"Sharpe {sh:.2f}")
    else:
        print(f"  LIVE since {VOLARB_INCEPTION}: accrues from the next data update.")
    tr1y = pnl.loc[pnl.index >= today - pd.Timedelta(days=365)]
    print(f"  trailing 1y (backtest context): Sharpe {tr1y.mean()/tr1y.std()*SQ:.2f}, "
          f"days short {100*(pos.reindex(tr1y.index)>0).mean():.0f}% / long "
          f"{100*(pos.reindex(tr1y.index)<0).mean():.0f}%")
    print(f"  ledger -> {VOLARB_LEDGER}")
    print("  NOTE: variance-swap PROXY pnl — paper tracking for live-vs-backtest deviation (P3-12).")


# ---------------------------------------------------------------------------
# VOL-BOOK PAPER TRACKER (P3-12, CURRENT) — the multi-market floor book.
# Selection-free floor (invvol x cost-coverage), 1-DTE straddle instrument, FULL L4 frictions.
# Replaces the deprecated variance-swap proxy (volarb_section, kept below for reference only).
# Tracks the actual production strategy: per-market state today + live calendar-basis P&L.
# ---------------------------------------------------------------------------
VOLBOOK_LEDGER = os.path.join(DATA_DIR, "paper_volbook_ledger.csv")
VOLBOOK_HALT = os.path.join(DATA_DIR, "volbook_halt.json")
VOLBOOK_INCIDENTS = os.path.join(DATA_DIR, "volbook_incidents.log")
# desk knobs live in volbook_config.py (versioned; change deliberately, then re-gauntlet)
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volbook_config as CFG
VOLBOOK_INCEPTION = CFG.VOLBOOK_INCEPTION
STALE_DAYS = CFG.STALE_DAYS
PNL_BOUND = CFG.PNL_BOUND


def _volbook_kill_switches(book, sleeves):
    """Kill-switch-first (Optiver doctrine): detect anomalies AUTOMATICALLY and halt
    BEFORE diagnosing. Checks: (1) stale feeds, (2) daily P&L bound, (3) HARD data
    corruption in book inputs. Returns a list of incident strings (empty = healthy)."""
    import bookopt_harness as H
    incidents = []
    today = pd.Timestamp.now().normalize()

    age = (today - book.index.max()).days
    if age > STALE_DAYS:
        incidents.append(f"STALE BOOK: tracker output ends {book.index.max().date()} "
                         f"({age}d behind today; limit {STALE_DAYS}d)")
    H._load()                                    # raw FEED staleness (not walk-forward truncation)
    for n in sleeves:
        raw_last = H._DATA[n][1].index.max()
        a = (today - raw_last).days
        if a > STALE_DAYS:
            incidents.append(f"STALE FEED {n}: last raw bar {raw_last.date()} ({a}d old) — "
                             f"instrument not on the daily update schedule?")

    last_r = float(book.iloc[-1])
    if abs(last_r) > PNL_BOUND:
        incidents.append(f"PNL BOUND: last daily book return {last_r*100:+.1f}% exceeds "
                         f"{PNL_BOUND*100:.0f}% (likely data error or regime anomaly)")

    # operational live-drawdown policy (governance, NOT an alpha rule — the alpha DD-brake
    # was tested and rejected as a vol-shrinkage artifact; this is a review trigger)
    live = book[book.index >= pd.Timestamp(VOLBOOK_INCEPTION)]
    if len(live) > 20:
        eq = (1 + live).cumprod()
        live_dd = float((eq / eq.cummax() - 1).min())
        if live_dd < CFG.LIVE_MAXDD_HALT:
            incidents.append(f"LIVE MAXDD: {live_dd*100:.0f}% breaches operational limit "
                             f"{CFG.LIVE_MAXDD_HALT*100:.0f}% (backtest maxDD -20%; model under review)")

    try:
        from data_quality import audit_ohlcv
        for _, uf, vf, _ in H.PAIRS:
            for stem in (uf, vf):
                p = os.path.join(DATA_DIR, stem + "_all_history.csv")
                if not os.path.exists(p):
                    continue
                df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
                hard = [i for i in audit_ohlcv(df, stem) if i.startswith("HARD")]
                incidents.extend(hard)
    except ImportError:
        pass
    return incidents


def _halt(incidents):
    """Write the halt state + incident log. Tracking stays suspended until --rearm."""
    import json
    from datetime import datetime
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(VOLBOOK_HALT, "w", encoding="utf-8") as f:
        json.dump({"halted_at": stamp, "incidents": incidents}, f, indent=1)
    with open(VOLBOOK_INCIDENTS, "a", encoding="utf-8") as f:
        for i in incidents:
            f.write(f"{stamp}  {i}\n")


def volbook_section():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bookopt_harness as H
    import bookopt_floor as F
    SQ = np.sqrt(252.0)

    sleeves, infos = {}, {}
    for name, _, _, _ in H.PAIRS:
        # emit_partial: include the trailing partial walk-forward window (params from the
        # last completed train) so the tracker follows the PRESENT, not the last full block
        blend, _static, info = H.market(name, return_pos=True, emit_partial=True)
        if blend is None:
            continue
        sleeves[name] = blend; infos[name] = info
    W = {n: F.invvol(sleeves[n]) * F.coverage_gate(n).reindex(sleeves[n].index) for n in sleeves}
    book = H.book_of(sleeves, W)

    # calendar-basis live curve (idle capital = 0 when the gate steps the book out)
    allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[m].index) for m in sleeves])))
    cal = book.reindex(allidx).fillna(0.0)
    today = book.index.max()
    state = {1.0: "SHORT VOL", 0.5: "HALF SHORT", -0.5: "HALF LONG", -1.0: "LONG VOL", 0.0: "FLAT"}

    print("\n" + "=" * 86)
    print(f"VOL-BOOK PAPER TRACKER — multi-market floor book (1-DTE straddle, L4 frictions)   asof {today.date()}")
    print("=" * 86)

    # ---- kill switches: halt FIRST, diagnose second -----------------------------------
    if os.path.exists(VOLBOOK_HALT):
        import json
        h = json.load(open(VOLBOOK_HALT, encoding="utf-8"))
        print(f"  ## HALTED since {h['halted_at']} — tracking SUSPENDED (book flat). Incidents:")
        for i in h["incidents"]:
            print(f"     - {i}")
        print(f"  ## Fix the cause, then re-arm:  python paper_trade.py --rearm")
        return
    incidents = _volbook_kill_switches(book, sleeves)
    if incidents:
        _halt(incidents)
        print(f"  ## KILL SWITCH TRIPPED — {len(incidents)} incident(s); book HALTED (flat), "
              f"ledger update suspended:")
        for i in incidents:
            print(f"     - {i}")
        print(f"  ## incidents -> {VOLBOOK_INCIDENTS} | re-arm after fixing: python paper_trade.py --rearm")
        return
    raw_today = {n: (float(W[n].reindex([today]).fillna(0.0).iloc[0]) if today in W[n].index else 0.0)
                 for n in sleeves}
    tot = sum(v for v in raw_today.values() if v > 0) or 1.0     # normalize to book fractions
    print(f"  {'market':<7}{'book wt':>9}{'gate':>7}{'tomorrow':>13}")
    n_active = 0
    for n in sleeves:
        w_frac = raw_today[n] / tot if raw_today[n] > 0 else 0.0
        sig = 0.5 * infos[n]["posA"].iloc[-1] + 0.5 * infos[n]["posB"].iloc[-1]   # last applied position
        on = raw_today[n] > 0 and sig != 0
        n_active += int(on)
        print(f"  {n:<7}{w_frac*100:>8.0f}%{('ON' if raw_today[n]>0 else 'out'):>7}{state.get(sig, f'{sig:+.1f}'):>13}")
    print(f"  -> {n_active}/{len(sleeves)} markets active tomorrow "
          f"(book steps OUT of sleeves whose trailing edge stops covering frictions)")

    # ---- drift telemetry: did history change under our feet since the last run? --------
    # Same code path as the backtest, so any change in past ledger rows = data revision
    # (or a code change). Alert loudly + log; do NOT halt (revisions are legitimate-but-
    # must-be-visible). This is the live-vs-backtest deviation record (P3-12 doctrine).
    led = pd.DataFrame({"book_ret": cal}).loc[CFG.LEDGER_START:]
    led["cum_equity"] = (1 + led["book_ret"]).cumprod()
    if os.path.exists(VOLBOOK_LEDGER):
        old = pd.read_csv(VOLBOOK_LEDGER, parse_dates=["Date"]).set_index("Date")
        common_d = old.index.intersection(led.index)
        if len(common_d):
            diff = (led.loc[common_d, "book_ret"] - old.loc[common_d, "book_ret"]).abs()
            drifted = diff[diff > 1e-6]
            if len(drifted):
                from datetime import datetime
                stamp = datetime.now().isoformat(timespec="seconds")
                print(f"  !! DRIFT: {len(drifted)} past ledger day(s) changed since last run "
                      f"(max |d|={drifted.max():.2e} on {drifted.idxmax().date()}) — "
                      f"data revision or code change; see {os.path.basename(VOLBOOK_INCIDENTS)}")
                with open(VOLBOOK_INCIDENTS, "a", encoding="utf-8") as f:
                    f.write(f"{stamp}  DRIFT {len(drifted)} day(s), max {drifted.max():.2e} "
                            f"on {drifted.idxmax().date()}\n")
    led.to_csv(VOLBOOK_LEDGER, index_label="Date")

    # ---- decision log: append today's per-market state (replayable audit trail) --------
    dec_path = os.path.join(DATA_DIR, "volbook_decisions.csv")
    rows = []
    for n in sleeves:
        sig = 0.5 * infos[n]["posA"].iloc[-1] + 0.5 * infos[n]["posB"].iloc[-1]
        rows.append({"Date": today.date().isoformat(), "market": n,
                     "gate": "ON" if raw_today[n] > 0 else "out",
                     "weight_frac": round(raw_today[n] / tot if raw_today[n] > 0 else 0.0, 4),
                     "signal": sig})
    dec = pd.DataFrame(rows)
    if os.path.exists(dec_path):
        prev = pd.read_csv(dec_path)
        dec = pd.concat([prev[prev["Date"] != dec["Date"].iloc[0]], dec], ignore_index=True)
    dec.to_csv(dec_path, index=False)

    live = cal.loc[cal.index >= pd.to_datetime(VOLBOOK_INCEPTION)]
    if len(live) > 1 and live.std() > 0:
        print(f"  LIVE since {VOLBOOK_INCEPTION}: {len(live)} days, cum {((1+live).prod()-1)*100:+.2f}%, "
              f"Sharpe {live.mean()/live.std()*SQ:.2f}")
    else:
        print(f"  LIVE since {VOLBOOK_INCEPTION}: accrues from the next data update.")
    ctx = cal.loc[cal.index >= today - pd.Timedelta(days=365)]
    print(f"  trailing 1y (backtest context, calendar basis): Sharpe {ctx.mean()/ctx.std()*SQ:.2f}, "
          f"deployed {100*(ctx!=0).mean():.0f}% of days")
    print(f"  full backtest Sharpe: active-days {H.sharpe(book):.2f} | calendar {H.sharpe(cal):.2f} "
          f"(deployed {len(book)/len(allidx)*100:.0f}%)")
    print(f"  ledger -> {VOLBOOK_LEDGER}")
    print("  NOTE: paper only. Straddle premium is a MODEL (k=0.82); biggest live risk = real k & spreads.")


if __name__ == "__main__":
    if "--rearm" in sys.argv:
        if os.path.exists(VOLBOOK_HALT):
            os.remove(VOLBOOK_HALT)
            print("vol-book RE-ARMED (halt cleared by operator). Tracking resumes next run.")
        else:
            print("vol-book is not halted; nothing to re-arm.")
        sys.exit(0)
    main()
