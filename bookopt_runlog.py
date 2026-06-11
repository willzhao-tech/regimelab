# -*- coding: utf-8 -*-
"""Experiment log: every research/build/gauntlet run appends one JSON line to runs.jsonl
(timestamp, code commit, data-vintage digest, params, headline metrics). The solo-scale
analogue of an experiment-tracking registry — makes "what did we run in March?" answerable."""
import os, json, hashlib, subprocess
from datetime import datetime

DATA_DIR = os.environ.get("REGIMELAB_DATA_DIR", r"C:\Users\ASUS\Desktop\claude doc\1")
RUNS = os.path.join(DATA_DIR, "runs.jsonl")
_PKG = os.path.dirname(os.path.abspath(__file__))


def code_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_PKG,
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def vintage_digest(stems):
    """One short hash over the content-hashes of the named input CSVs (order-independent)."""
    parts = []
    for s in sorted(stems):
        p = os.path.join(DATA_DIR, s + "_all_history.csv")
        if not os.path.exists(p):
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        parts.append(s + ":" + h.hexdigest()[:12])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def log_run(kind, metrics, params=None, note=None):
    """Append one run record. kind: 'build_results' | 'gauntlet' | 'example' | ..."""
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind,
           "code": code_commit(), "params": params or {}, "metrics": metrics}
    if note:
        rec["note"] = note
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
