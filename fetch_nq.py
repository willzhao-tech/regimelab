"""Fetch NQ (Nasdaq-100 continuous futures) ALL history via regimelab.

The requested Stooq symbol `NQ.F` is now behind a captcha-gated API key (its CSV
endpoint requires a key obtained by solving a captcha), so it can't be pulled
headlessly. We fetch the same instrument from Yahoo (`NQ=F`) via YahooSource;
Yahoo serves the full continuous-future history, which begins 2000-09-18.
"""
import regimelab as rl

# start well before the contract's existence so Yahoo returns everything it has
src = rl.data.YahooSource(tickers={"NQ": "NQ=F"}, start="1999-01-01")
panel = rl.data.build_panel([src])

prices = panel.prices
print("rows:", len(prices))
print("date range:", prices.index.min().date(), "->", prices.index.max().date())
print(prices.tail())

out = "NQ_F_all_history.csv"
prices.to_csv(out)
print("saved ->", out)
