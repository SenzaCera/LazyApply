"""Analisi aggiuntive: stop loss, frequenza dei controlli, capitale necessario.

Uso: python extras.py data/ndx_1m.csv.gz
"""
import sys

import numpy as np
import pandas as pd

from backtest import POINT_VALUE, Params, intraday_momentum, load_rth, stats

OOS = pd.Timestamp("2023-01-01")


def both(d, p):
    tr = intraday_momentum(d, p)
    days = d["days"][p.lookback + 1:]
    a = stats(tr[tr.day < OOS], days[days < OOS])
    b = stats(tr[tr.day >= OOS], days[days >= OOS])
    keys = ["trades", "win%", "PF", "total", "sharpe", "maxDD"]
    return {**{f"IS {k}": a[k] for k in keys}, **{f"OOS {k}": b[k] for k in keys}}


def main(path):
    pd.set_option("display.width", 250)
    d = load_rth(path)
    print("== Stop loss (in % del prezzo d'ingresso) ==")
    rows = {"nessuno": both(d, Params())}
    for s in (0.15, 0.25, 0.4, 0.6, 1.0):
        rows[f"stop {s}%"] = both(d, Params(stop_pct=s))
    print(pd.DataFrame(rows).T.to_string())
    print("\n== Frequenza dei controlli (minuti) ==")
    print(pd.DataFrame({f"ogni {s} min": both(d, Params(step=s)) for s in (5, 10, 15, 30, 60)}).T.to_string())

    # Capitale: P&L giornaliero in % del nozionale, riportato al livello attuale dell'indice
    tr = intraday_momentum(d, Params())
    days = d["days"][15:]
    level = d["close"][-1, -1]
    pct = (tr["points"] / tr["entry"]).groupby(tr["day"]).sum().reindex(days, fill_value=0.0)
    usd = pct.to_numpy() * level * POINT_VALUE
    eq = np.cumsum(usd)
    print(f"\n== Capitale per 1 MNQ (indice a {level:.0f}, 1 contratto = {level * POINT_VALUE:,.0f} USD di nozionale) ==")
    print(f"P&L medio annuo {usd.mean() * 252:,.0f} USD | giorno peggiore {usd.min():,.0f} | drawdown storico {(eq - np.maximum.accumulate(eq)).min():,.0f}")
    rng = np.random.default_rng(0)
    block, n = 10, len(usd)
    for horizon, label in ((252, "1 anno"), (n, "tutto il periodo (~8 anni)")):
        dds, finals = [], []
        for _ in range(5000):                               # bootstrap a blocchi di 10 giorni
            starts = rng.integers(0, n - block, horizon // block + 1)
            x = np.concatenate([usd[s:s + block] for s in starts])[:horizon]
            e = np.cumsum(x)
            dds.append((e - np.maximum.accumulate(e)).min())
            finals.append(e[-1])
        dds, finals = np.array(dds), np.array(finals)
        print(f"{label}: drawdown mediano {np.median(dds):,.0f} | 95% {np.percentile(dds, 5):,.0f} | 99% {np.percentile(dds, 1):,.0f} | prob. di chiudere in perdita {100 * (finals < 0).mean():.0f}%")


if __name__ == "__main__":
    main(sys.argv[1])
