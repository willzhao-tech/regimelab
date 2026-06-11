# -*- coding: utf-8 -*-
"""regimelab desktop widget — a small always-on-top monitor for the vol-book platform.
Reads ONLY local state files (results.json, gauntlet_report.json, volbook_halt.json,
paper_volbook_ledger.csv, volbook_decisions.csv, event_calendar.csv, incidents log);
no network, no pandas — stdlib only, so it starts instantly.

Launch (no console window):
    .venv\\Scripts\\pythonw.exe volbook_widget.pyw
Drag the title bar to move - pin toggles always-on-top - auto-refreshes every 60 s."""
import os, json, csv
import tkinter as tk
from datetime import datetime, date

DATA = os.environ.get("REGIMELAB_DATA_DIR", r"C:\Users\ASUS\Desktop\claude doc\1")
INCEPTION = date(2026, 6, 11)
BG, CARD, FG, MUT = "#16181d", "#1f2229", "#e8eaf0", "#8a8f9c"
GREEN, AMBER, RED, BLUE = "#34c98e", "#e8a33d", "#e0564f", "#5aa2e0"
REFRESH_MS = 60_000


def _read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def load_state():
    s = {"dot": GREEN, "status": "HEALTHY"}
    try:
        r = json.load(open(os.path.join(DATA, "results.json"), encoding="utf-8"))
        s["cal"] = f'{r["floor"]["sharpe_calendar"]:.2f}'
        s["act"] = f'{r["floor"]["sharpe_active"]:.2f}'
        s["dd"] = f'{r["floor"]["maxdd"]*100:.0f}%'
        s["commit"] = r.get("vintage", {}).get("code_commit") or "?"
    except Exception:
        s.update(cal="?", act="?", dd="?", commit="?")
    try:
        g = json.load(open(os.path.join(DATA, "gauntlet_report.json"), encoding="utf-8"))
        n = sum(1 for c in g["checks"].values() if c["pass"])
        s["gauntlet"] = f'{g["verdict"]} {n}/{len(g["checks"])}'
        s["g_ok"] = g["verdict"] == "PROMOTED"
    except Exception:
        s["gauntlet"], s["g_ok"] = "no report", False

    halt = os.path.join(DATA, "volbook_halt.json")
    s["halted"] = os.path.exists(halt)
    if s["halted"]:
        try:
            h = json.load(open(halt, encoding="utf-8"))
            s["halt_why"] = (h["incidents"][0][:46] + "…") if h["incidents"] else ""
        except Exception:
            s["halt_why"] = ""

    led = _read_csv(os.path.join(DATA, "paper_volbook_ledger.csv"))
    if led:
        s["led_last"] = led[-1]["Date"]
        eq = 1.0
        for row in led:
            if row["Date"] >= INCEPTION.isoformat():
                try:
                    eq *= 1.0 + float(row["book_ret"])
                except ValueError:
                    pass
        s["live"] = f"{(eq-1)*100:+.2f}%"
        try:
            s["last_ret"] = f'{float(led[-1]["book_ret"])*100:+.2f}%'
        except ValueError:
            s["last_ret"] = "—"
    else:
        s.update(led_last="—", live="—", last_ret="—")

    dec = _read_csv(os.path.join(DATA, "volbook_decisions.csv"))
    s["mkts"] = []
    if dec:
        last_d = max(r["Date"] for r in dec)
        smap = {"1.0": "short", "0.5": "½short", "-0.5": "½long", "-1.0": "long", "0.0": "flat"}
        for r in dec:
            if r["Date"] == last_d:
                s["mkts"].append((r["market"], r["gate"] == "ON",
                                  smap.get(r["signal"], r["signal"]),
                                  float(r["weight_frac"] or 0)))
        s["dec_date"] = last_d

    s["events"] = []
    today = date.today().isoformat()
    for r in _read_csv(os.path.join(DATA, "event_calendar.csv")):
        if r["date"] >= today and len(s["events"]) < 3:
            s["events"].append((r["date"][5:], r["event"] + ("~" if r["source"] == "projected" else "")))

    try:
        with open(os.path.join(DATA, "volbook_incidents.log"), encoding="utf-8") as f:
            s["incidents"] = sum(1 for _ in f)
    except OSError:
        s["incidents"] = 0

    try:
        age = (datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(os.path.join(DATA, "SPX_all_history.csv")))).days
        s["fresh"] = f"{age}d ago" if age else "today"
        stale = age > 4
    except OSError:
        s["fresh"], stale = "?", True

    if s["halted"]:
        s["dot"], s["status"] = RED, "HALTED"
    elif (not s["g_ok"]) or stale:
        s["dot"], s["status"] = AMBER, "ATTENTION"
    return s


class Widget(tk.Tk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=BG)
        self.geometry("+{}+60".format(self.winfo_screenwidth() - 360))
        self.pinned = True
        self._build()
        self.refresh()

    def _row(self, parent, **kw):
        f = tk.Frame(parent, bg=kw.pop("bg", BG))
        f.pack(fill="x", **kw)
        return f

    def _build(self):
        pad = dict(padx=12)
        hdr = self._row(self, pady=(10, 6), **pad)
        self.dot = tk.Label(hdr, text="●", bg=BG, fg=GREEN, font=("Segoe UI", 11))
        self.dot.pack(side="left")
        t = tk.Label(hdr, text=" regimelab", bg=BG, fg=FG, font=("Segoe UI Semibold", 11))
        t.pack(side="left")
        self.stat = tk.Label(hdr, text="", bg=BG, fg=MUT, font=("Segoe UI", 9))
        self.stat.pack(side="left", padx=6)
        for txt, cmd in (("✕", self.destroy), ("📌", self.toggle_pin), ("↻", self.refresh),
                         ("▤", self.open_dash), ("📅", self.open_calendar)):
            b = tk.Label(hdr, text=txt, bg=BG, fg=MUT, font=("Segoe UI", 10), cursor="hand2")
            b.pack(side="right", padx=3)
            b.bind("<Button-1>", lambda e, c=cmd: c())
        for w in (hdr, t):
            w.bind("<Button-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

        card = tk.Frame(self, bg=CARD)
        card.pack(fill="x", padx=10, pady=(0, 6))
        self.metrics = tk.Label(card, bg=CARD, fg=FG, font=("Consolas", 10), justify="left", anchor="w")
        self.metrics.pack(fill="x", padx=10, pady=8)

        self.mkt = tk.Label(self, bg=BG, fg=FG, font=("Consolas", 9), justify="left", anchor="w")
        self.mkt.pack(fill="x", padx=22, pady=(0, 4))
        self.ev = tk.Label(self, bg=BG, fg=MUT, font=("Segoe UI", 9), justify="left", anchor="w")
        self.ev.pack(fill="x", padx=22, pady=(0, 8))
        self.foot = tk.Label(self, bg=BG, fg=MUT, font=("Segoe UI", 8), anchor="w")
        self.foot.pack(fill="x", padx=22, pady=(0, 8))

    def _drag_start(self, e):
        self._ox, self._oy = e.x_root - self.winfo_x(), e.y_root - self.winfo_y()

    def _drag_move(self, e):
        self.geometry(f"+{e.x_root - self._ox}+{e.y_root - self._oy}")

    def toggle_pin(self):
        self.pinned = not self.pinned
        self.attributes("-topmost", self.pinned)

    def open_dash(self):
        p = os.path.join(DATA, "volbook_dashboard.html")
        if os.path.exists(p):
            os.startfile(p)

    def open_calendar(self):
        """Open the master big-move calendar; generate it first if it doesn't exist yet."""
        p = os.path.join(DATA, "master_calendar.html")
        if not os.path.exists(p):
            import subprocess, sys
            pkg = os.path.dirname(os.path.abspath(__file__))
            py = os.path.join(pkg, ".venv", "Scripts", "python.exe")
            exe = py if os.path.exists(py) else sys.executable
            try:
                subprocess.run([exe, os.path.join(pkg, "build_master_calendar.py")],
                               timeout=180, creationflags=0x08000000)   # no console window
            except Exception:
                return
        if os.path.exists(p):
            os.startfile(p)

    def refresh(self):
        s = load_state()
        self.dot.config(fg=s["dot"])
        self.stat.config(text=s["status"])
        self.metrics.config(text=(
            f'Sharpe cal {s["cal"]}  act {s["act"]}  maxDD {s["dd"]}\n'
            f'gauntlet  {s["gauntlet"]}\n'
            f'live since {INCEPTION:%b %d}: {s["live"]}   last day {s["last_ret"]}\n'
            f'ledger through {s["led_last"]}'
            + (f'\n!! {s["halt_why"]}' if s.get("halted") else "")))
        if s["mkts"]:
            on = [m for m in s["mkts"] if m[1]]
            line = "  ".join(f'{m[0]}:{m[2]}({m[3]*100:.0f}%)' for m in on) or "all sleeves stepped out"
            self.mkt.config(text=f'gate ON {len(on)}/{len(s["mkts"])}:  {line}')
        else:
            self.mkt.config(text="no decisions logged yet")
        self.ev.config(text="next: " + "   ".join(f"{d} {e}" for d, e in s["events"]))
        self.foot.config(text=f'data {s["fresh"]} · incidents {s["incidents"]} · {s["commit"]} · '
                              f'{datetime.now():%H:%M} (auto 60s)')
        self.after(REFRESH_MS, self.refresh)


if __name__ == "__main__":
    Widget().mainloop()
