"""Backtest di IbsSwingDaily su MNQ con barre a 1 minuto (CFD Nasdaq-100 Dukascopy come proxy, 2019-2026).

Meccanica replicata dal NinjaScript:
  * IBS sulla finestra cash 09:30-16:00 ET; segnale se IBS < soglia a fine sessione e si e' flat
  * ingresso a mercato dopo la prima barra della sessione successiva (apertura della barra 09:31)
  * controllo d'uscita sulla barra che chiude alle 15:58: Close > massimo della sessione precedente,
    oppure 5a sessione in posizione -> uscita all'apertura della barra successiva
  * stop catastrofe 8% dal prezzo d'ingresso (controllato solo in sessione cash e sull'apertura)
1 contratto, 2 USD a punto, costo 1,5 punti a trade.

Uso: python ibs_mnq.py ../mnq/data/ndx_1m.csv.gz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mnq"))
from backtest import COST_POINTS, POINT_VALUE, load_rth  # noqa: E402


def backtest(d, thr=0.10, hold=5, stop=0.08, cost=COST_POINTS, check=387):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    n = len(d["days"])
    sh, sl, sc = h.max(axis=1), l.min(axis=1), c[:, -1]
    ibs = (sc - sl) / (sh - sl)
    pnl = np.zeros(n)                                        # P&L giornaliero in punti (mark-to-market)
    trades, in_pos, entry, held, pending, i_in = [], False, 0.0, 0, False, 0
    for i in range(1, n):
        exited = False
        start = 0
        if pending:
            pending, in_pos, entry, held, i_in, start = False, True, o[i, 1], 0, i, 1
        if in_pos:
            ref = entry if i == i_in else sc[i - 1]
            px = None
            if stop:
                lvl = entry * (1 - stop)
                if start == 0 and o[i, 0] <= lvl:
                    px = o[i, 0]                             # apre gia' sotto lo stop
                else:
                    hit = l[i, start:] <= lvl
                    if hit.any() and start + int(np.argmax(hit)) <= check:
                        px = lvl
            if px is None:
                held += 1
                if c[i, check] > sh[i - 1] or held >= hold:
                    px = o[i, check + 1]
            if px is not None:
                pnl[i] = px - ref - cost
                trades.append((d["days"][i_in], d["days"][i], entry, px, i - i_in + 1))
                in_pos, exited = False, True
            else:
                pnl[i] = sc[i] - ref
        if not in_pos and not exited and ibs[i] < thr:
            pending = True
    tr = pd.DataFrame(trades, columns=["in", "out", "entry", "exit", "days"])
    tr["points"] = tr["exit"] - tr["entry"] - cost
    return tr, pd.Series(pnl, index=d["days"]), pd.Series(sc, index=d["days"])


def stats(tr, pnl, close, level=None):
    """level=None: dollari storici; altrimenti tutto riportato a quel livello dell'indice."""
    if level:
        usd_t = tr.points / tr.entry * level * POINT_VALUE
        usd_d = pnl / close.shift(1).bfill() * level * POINT_VALUE
    else:
        usd_t, usd_d = tr.points * POINT_VALUE, pnl * POINT_VALUE
    eq = usd_d.cumsum()
    w, lo = usd_t[usd_t > 0], usd_t[usd_t <= 0]
    yrs = len(pnl) / 252
    return {"trade": len(tr), "trade/anno": round(len(tr) / yrs, 1), "vincenti %": round(100 * len(w) / len(tr), 1),
            "vincita media $": round(w.mean()), "perdita media $": round(lo.mean()), "media trade $": round(usd_t.mean()),
            "peggiore $": round(usd_t.min()), "migliore $": round(usd_t.max()), "PF": round(w.sum() / -lo.sum(), 2),
            "utile/anno $": round(usd_d.sum() / yrs), "maxDD $": round((eq - eq.cummax()).min()),
            "giorno peggiore $": round(usd_d.min()), "sharpe": round(usd_d.mean() / usd_d.std() * np.sqrt(252), 2),
            "t-stat trade": round(usd_t.mean() / usd_t.std() * np.sqrt(len(tr)), 1),
            "durata media (sessioni)": round(tr.days.mean(), 1), "max perdite di fila": int(max_streak(usd_t <= 0))}


def max_streak(b):
    best = cur = 0
    for x in b:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def main(path):
    pd.set_option("display.width", 250)
    d = load_rth(path)
    level = d["close"][-1, -1]
    tr, pnl, close = backtest(d)
    print(f"{len(d['days'])} sessioni, {d['days'][0].date()} -> {d['days'][-1].date()}, indice attuale {level:.0f}")
    print("\n== Default (IBS < 0,10, 5 sessioni, stop 8%), 1 contratto MNQ ==")
    print(pd.DataFrame({"dollari storici": stats(tr, pnl, close), "al livello attuale": stats(tr, pnl, close, level)}).to_string())
    usd = pnl / close.shift(1).bfill() * level * POINT_VALUE
    bh = close.diff().fillna(0) / close.shift(1).bfill() * level * POINT_VALUE
    print("\n== Per anno, USD al livello attuale (strategia / tenere 1 contratto sempre) ==")
    print(pd.DataFrame({"strategia": usd.groupby(usd.index.year).sum(), "sempre long": bh.groupby(bh.index.year).sum(),
                        "trade": tr.groupby(tr["in"].dt.year).size()}).round().T.to_string())
    print("\n== Varianti (al livello attuale) ==")
    keys = ["trade", "vincenti %", "media trade $", "peggiore $", "PF", "utile/anno $", "maxDD $", "sharpe", "t-stat trade"]
    var = {"base": {}, "senza stop": dict(stop=None), "stop 4%": dict(stop=0.04), "soglia 0,05": dict(thr=0.05),
           "soglia 0,15": dict(thr=0.15), "soglia 0,20": dict(thr=0.20), "hold 3": dict(hold=3), "hold 10": dict(hold=10),
           "costo 5 punti": dict(cost=5.0)}
    print(pd.DataFrame({k: stats(*backtest(d, **v), level) for k, v in var.items()}).loc[keys].to_string())
    eq = bh.cumsum()
    print(f"\nSempre long 1 contratto: utile/anno {bh.sum() / (len(bh) / 252):,.0f} USD, maxDD {(eq - eq.cummax()).min():,.0f} USD")


if __name__ == "__main__":
    main(sys.argv[1])
