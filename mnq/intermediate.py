"""Gestione intermedia del trade: cosa succede ai trade che arrivano a +0,25/0,35% e regole per proteggerli.

Uso: python intermediate.py data/ndx_1m.csv.gz
"""
import sys

import numpy as np
import pandas as pd

from backtest import POINT_VALUE, Params, intraday_momentum, load_rth
from protect import mfe, row

GB = ("giveback", 0.5, 0.5)


def main(path):
    pd.set_option("display.width", 260)
    d = load_rth(path)
    level = d["close"][-1, -1]
    tr = intraday_momentum(d, Params(stop_pct=0.4))
    tr["usd"] = tr.points / tr.entry * level * POINT_VALUE
    tr["mfe"] = mfe(d, tr)
    print("== Senza protezione: che fine fanno i trade in base al massimo guadagno raggiunto ==")
    bins = [-1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 99]
    g = tr.groupby(pd.cut(tr.mfe, bins), observed=True).usd
    print(pd.DataFrame({"trade": g.size(), "chiusi in perdita %": g.apply(lambda x: round(100 * (x <= 0).mean())),
                        "risultato medio $": g.mean().round(), "totale $": g.sum().round()}).to_string())

    s = lambda **k: Params(stop_pct=0.4, **k)
    short_gb = dict(protect="giveback", protect_act=0.5, protect_val=0.5, protect_sides="short")
    both_gb = dict(protect="giveback", protect_act=0.5, protect_val=0.5)
    extra = {"": (),
             " + pareggio dopo +0,3%": (("pareggio", 0.3, 2.0, "entrambi"),),
             " + stop a -0,1% dopo +0,3%": (("riduci", 0.3, 0.1, "entrambi"),),
             " + stop a -0,2% dopo +0,3%": (("riduci", 0.3, 0.2, "entrambi"),),
             " + stop a -0,2% dopo +0,25%": (("riduci", 0.25, 0.2, "entrambi"),),
             " + restituisce 75% dopo +0,3%": (("giveback", 0.3, 0.75, "entrambi"),)}
    rows = {"nessuna protezione": row(d, s())}
    for name, base in (("protezione 50%/+0,5% su entrambi", both_gb), ("protezione 50%/+0,5% solo short", short_gb)):
        for k, rules in extra.items():
            rows[name + k] = row(d, s(**base, protect_rules=rules))
    keys = ["trade", "vincenti %", "vincita media $", "perdita media $", "in utile >0,3% poi chiusi in perdita",
            "utile/anno $", "maxDD $", "sharpe 2019-22", "sharpe 2023+"]
    print("\n== Long + short, stop 0,4% ==")
    print(pd.DataFrame(rows).T[keys].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
