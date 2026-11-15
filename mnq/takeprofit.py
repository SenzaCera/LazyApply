"""Effetto di un take profit. Uso: python takeprofit.py data/ndx_1m.csv.gz"""
import sys

import pandas as pd

from backtest import POINT_VALUE, Params, intraday_momentum, load_rth, stats

OOS = pd.Timestamp("2023-01-01")


def row(d, p):
    level, days = d["close"][-1, -1], d["days"][p.lookback + 1:]
    tr = intraday_momentum(d, p)
    tr["usd"] = tr.points / tr.entry * level * POINT_VALUE          # riportato al livello attuale dell'indice
    dl = tr.groupby("day").usd.sum().reindex(days, fill_value=0)
    eq, w, l = dl.cumsum(), tr[tr.usd > 0], tr[tr.usd <= 0]
    a, b = stats(tr[tr.day < OOS], days[days < OOS]), stats(tr[tr.day >= OOS], days[days >= OOS])
    return dict(trades=len(tr), win=round(100 * len(w) / len(tr), 1), avgW=round(w.usd.mean()), avgL=round(l.usd.mean()),
                net_yr=round(tr.usd.sum() / (len(days) / 252)), worst_day=round(dl.min()), maxDD=round((eq - eq.cummax()).min()),
                sh_IS=a["sharpe"], sh_OOS=b["sharpe"], PF_IS=a["PF"], PF_OOS=b["PF"])


def main(path):
    pd.set_option("display.width", 250)
    d = load_rth(path)
    base = dict(stop_pct=0.4, after_stop="no_stessa_direzione")
    rows = {"stop 0.4 no-rientro, senza TP": row(d, Params(**base))}
    for mode in ("rientra", "no_stessa_direzione"):
        for tp in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
            rows[f"TP {tp}% poi {mode}"] = row(d, Params(**base, take_pct=tp, after_take=mode))
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
