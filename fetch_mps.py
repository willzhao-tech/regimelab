# -*- coding: utf-8 -*-
"""Fetch / parse the Bauer-Swanson (2023) high-frequency FOMC monetary-policy surprises.
Source: SF Fed, https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx
A FREE, deep-history (1988-2023), SIGNED & GRADED FOMC surprise series (30-min futures move around
each announcement) - the magnitude our binary FOMC-date flag lacks. MPS = raw 1st-PC surprise;
MPS_ORTH = orthogonalized vs pre-announcement public info (the 'pure' surprise). Units: % (≈ ±8bp max).

Writes FOMC_MPS.csv (Date, unscheduled, mps, mps_orth). Re-download via the China proxy if missing.
    python fetch_mps.py"""
import os
import pandas as pd

DATA_DIR = os.environ.get("REGIMELAB_DATA_DIR", r"C:\Users\ASUS\Desktop\claude doc\1")
XLSX = os.path.join(DATA_DIR, "bauer_swanson_mps.xlsx")
URL = "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx"
OUT = os.path.join(DATA_DIR, "FOMC_MPS.csv")


def _download():
    import urllib.request
    proxy = urllib.request.ProxyHandler({"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"})
    opener = urllib.request.build_opener(proxy)
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(URL, XLSX)


def main():
    if not os.path.exists(XLSX):
        print("xlsx missing — downloading via proxy ...")
        _download()
    df = pd.read_excel(XLSX, sheet_name="FOMC (update 2023)")
    df = df[["Date", "Unscheduled", "MPS", "MPS_ORTH"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["MPS"]).rename(columns={"Unscheduled": "unscheduled",
                                                    "MPS": "mps", "MPS_ORTH": "mps_orth"})
    df.to_csv(OUT, index=False)
    print(f"FOMC_MPS.csv -> {OUT}  ({len(df)} announcements {df['Date'].min().date()}..{df['Date'].max().date()})")
    print(f"  scheduled {int((df['unscheduled']==0).sum())} / unscheduled {int((df['unscheduled']==1).sum())}")
    print(f"  |MPS_ORTH| terciles (bp): "
          f"{(df['mps_orth'].abs().quantile([.33,.67])*100).round(2).tolist()}")


if __name__ == "__main__":
    main()
