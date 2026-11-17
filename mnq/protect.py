"""Protezione del profitto: chiudere un trade in utile prima che torni in perdita.

Confronta pareggio (breakeven), trailing stop e "giveback" sulla strategia momentum MNQ,
sia long+short sia solo long, con stop 0,4%. USD per 1 contratto al livello attuale dell'indice.

Uso: python protect.py data/ndx_1m.csv.gz
"""
import sys

import numpy as np
import pandas as pd

from backtest import POINT_VALUE, Params, intraday_momentum, load_rth, stats

OOS = pd.Timestamp("2023-01-01")


def mfe(d, tr):
    """Massimo guadagno (in %) raggiunto da ogni trade fra ingresso e uscita."""
    out = []
    for r in tr.itertuples():
        i = d["days"].get_loc(r.day)
        seg = slice(r.t_in, max(r.t_out, r.t_in + 1))
        best = d["high"][i, seg].max() if r.side == 1 else d["low"][i, seg].min()
        out.append((best - r.entry) * r.side / r.entry * 100)
    return np.array(out)


def row(d, p):
    level, days = d["close"][-1, -1], d["days"][p.lookback + 1:]
    tr = intraday_momentum(d, p)
    tr["usd"] = tr.points / tr.entry * level * POINT_VALUE
    dl = tr.groupby("day").usd.sum().reindex(days, fill_value=0)
    eq = dl.cumsum()
    w, lo = tr[tr.usd > 0], tr[tr.usd <= 0]
    m = mfe(d, tr)
    a, b = stats(tr[tr.day < OOS], days[days < OOS]), stats(tr[tr.day >= OOS], days[days >= OOS])
    return {"trade": len(tr), "vincenti %": round(100 * len(w) / len(tr), 1), "vincita media $": round(w.usd.mean()),
            "perdita media $": round(lo.usd.mean()),
            "in utile >0,3% poi chiusi in perdita": int(((m >= 0.3) & (tr.usd <= 0)).sum()),
            "utile/anno $": round(tr.usd.sum() / (len(days) / 252)), "maxDD $": round((eq - eq.cummax()).min()),
            "sharpe 2019-22": a["sharpe"], "sharpe 2023+": b["sharpe"], "PF 2019-22": a["PF"], "PF 2023+": b["PF"]}


def variants(base):
    v = {"nessuna protezione": {}}
    for act in (0.2, 0.3, 0.5):
        v[f"pareggio dopo +{act}%"] = dict(protect="pareggio", protect_act=act, protect_val=2.0)
    for act, dist in ((0.3, 0.2), (0.5, 0.3), (0.5, 0.5), (1.0, 0.5)):
        v[f"trailing {dist}% dopo +{act}%"] = dict(protect="trailing", protect_act=act, protect_val=dist)
    for act, gb in ((0.3, 0.5), (0.5, 0.5), (0.5, 0.33), (1.0, 0.5)):
        v[f"restituisce {int(gb * 100)}% dopo +{act}%"] = dict(protect="giveback", protect_act=act, protect_val=gb)
    v["pareggio dopo +0,3%, no rientro"] = dict(protect="pareggio", protect_act=0.3, protect_val=2.0, after_protect="no_stessa_direzione")
    v["restituisce 50% dopo +0,5%, no rientro"] = dict(protect="giveback", protect_act=0.5, protect_val=0.5, after_protect="no_stessa_direzione")
    return {k: Params(**{**base, **x}) for k, x in v.items()}


def main(path):
    pd.set_option("display.width", 260)
    d = load_rth(path)
    for name, base in (("LONG + SHORT, stop 0,4%", dict(stop_pct=0.4)), ("SOLO LONG, stop 0,4%", dict(stop_pct=0.4, long_only=True))):
        print(f"\n== {name} ==")
        print(pd.DataFrame({k: row(d, p) for k, p in variants(base).items()}).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
