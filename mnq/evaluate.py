"""Valutazione finale della strategia scelta (parametri fissati PRIMA di guardare il 2023+).

Uso: python evaluate.py data/ndx_1m.csv.gz
"""
import sys

import numpy as np
import pandas as pd

from backtest import POINT_VALUE, Params, intraday_momentum, load_rth, stats

P = Params(lookback=14, band_mult=1.0, step=30, first=30, use_vwap=True)   # default della letteratura
OOS_START = pd.Timestamp("2023-01-01")


def table(tr, days, by):
    rows = {}
    for k, g in tr.groupby(by(tr["day"])):
        dd = days[np.asarray(by(pd.Series(days))) == k]
        s = stats(g, dd)
        s["USD/contratto"] = round(s["total"] * POINT_VALUE)
        s["ret% (vol target)"] = round(100 * g["ret"].sum(), 1)
        rows[k] = s
    return pd.DataFrame(rows).T


def main(path):
    pd.set_option("display.width", 220)
    d = load_rth(path)
    days = d["days"][P.lookback + 1:]
    tr = intraday_momentum(d, Params(**{**P.__dict__, "vol_target": 0.02}))
    period = lambda s: np.where(pd.to_datetime(s) < OOS_START, "1) in-sample 2019-22", "2) OUT-OF-SAMPLE 2023+")
    print("== Strategia base, costo 1.5 punti a trade ==")
    print(table(tr, days, period).to_string())
    print("\n== Per anno ==")
    print(table(tr, days, lambda s: pd.to_datetime(s).dt.year).to_string())

    oos = {k: v[d["days"] >= OOS_START - pd.Timedelta(days=30)] for k, v in d.items()}
    od = oos["days"][oos["days"] >= OOS_START]
    f = lambda t: t[t["day"] >= OOS_START]
    print("\n== Out-of-sample: sensibilita' ai costi (punti per trade A/R) ==")
    print(pd.DataFrame({c: stats(f(intraday_momentum(oos, P, cost=c)), od) for c in (0, 1.5, 3, 5)}).T.to_string())
    print("\n== Out-of-sample: robustezza ai parametri ==")
    var = {
        "base": P,
        "mult 0.8": Params(band_mult=0.8), "mult 1.2": Params(band_mult=1.2),
        "lookback 10": Params(lookback=10), "lookback 30": Params(lookback=30),
        "step 15": Params(step=15), "step 60": Params(step=60),
        "senza VWAP": Params(use_vwap=False), "solo long": Params(long_only=True),
    }
    print(pd.DataFrame({k: stats(f(intraday_momentum(oos, p)), od) for k, p in var.items()}).T.to_string())
    lg, sh = tr[(tr.side > 0) & (tr.day >= OOS_START)], tr[(tr.side < 0) & (tr.day >= OOS_START)]
    print("\n== Out-of-sample: long vs short ==")
    print(pd.DataFrame({"long": stats(lg, od), "short": stats(sh, od)}).T.to_string())
    tr.to_csv("trades.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
