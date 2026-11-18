"""Grafico a 1, 5, 15 minuti e decisioni ogni 15/30 minuti, con i parametri di default della strategia NinjaTrader.

Uso: python timeframe.py data/ndx_1m.csv.gz
"""
import sys

import pandas as pd

from backtest import Params, load_rth
from drawdown import daily, summary
from backtest import intraday_momentum

DEFAULT = dict(stop_pct=0.4, protect="giveback", protect_act=0.5, protect_val=0.5, protect_sides="short",
               protect_rules=(("riduci", 0.25, 0.2, "entrambi"),))


def main(path):
    pd.set_option("display.width", 250)
    d = load_rth(path)
    days = d["days"][15:]
    rows = {}
    for name, extra in (("grafico 1 min, decisioni ogni 30 min (attuale)", {}),
                        ("grafico 5 min, decisioni ogni 30 min", dict(bar_min=5)),
                        ("grafico 15 min, decisioni ogni 30 min", dict(bar_min=15)),
                        ("grafico 15 min, decisioni ogni 15 min", dict(bar_min=15, step=15)),
                        ("grafico 1 min, decisioni ogni 15 min", dict(step=15))):
        for lab, base in (("", DEFAULT), (" | solo stop 0,4%", dict(stop_pct=0.4))):
            tr = intraday_momentum(d, Params(**{**base, **extra}))
            r = summary(daily(d, tr, days))
            r["trade"] = len(tr)
            rows[name + lab] = r
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
