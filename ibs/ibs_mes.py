"""IbsSwingDaily su MES (Micro E-mini S&P 500, 5 USD a punto), stimato dai dati giornalieri di SPY.

SPY copre esattamente la finestra cash 09:30-16:00 ET usata dalla strategia; 1 punto S&P ~ 0,10 USD di SPY.
Prezzi NON corretti per i dividendi (il future non li incassa). Costo: 0,02% a trade (~1 punto).
Per controllo, lo stesso metodo e' applicato a QQQ -> MNQ, da confrontare con il test a 1 minuto (ibs_mnq.py).

Uso: python ibs_mes.py
"""
import numpy as np
import pandas as pd

from ibs_backtest import backtest, load

SPECS = {"MES (da SPY)": ("SPY", 10.0, 5.0), "MNQ (da QQQ)": ("QQQ", 41.0, 2.0)}   # simbolo, punti indice per USD di ETF, USD a punto


def run(sym, a="1990", b="2030", level_mult=10.0, pv=5.0, **kw):
    df = load(sym, adjust=False)
    notional = df["close"].iloc[-1] * level_mult * pv        # valore attuale di 1 contratto
    tr, daily = backtest(df, cost=0.0002, **kw)
    tr, daily = tr[(tr["in"] >= a) & (tr["in"] < b)], daily[(daily.index >= a) & (daily.index < b)]
    usd_t, usd_d = tr.ret * notional, daily * notional
    eq = usd_d.cumsum()
    w, lo = usd_t[usd_t > 0], usd_t[usd_t <= 0]
    yrs = len(daily) / 252
    bh = df["close"].pct_change().reindex(daily.index).fillna(0) * notional
    beq = bh.cumsum()
    return {"nozionale 1 contratto $": round(notional), "trade": len(tr), "trade/anno": round(len(tr) / yrs, 1),
            "vincenti %": round(100 * len(w) / len(tr), 1), "vincita media $": round(w.mean()), "perdita media $": round(lo.mean()),
            "media trade $": round(usd_t.mean()), "peggiore $": round(usd_t.min()), "PF": round(w.sum() / -lo.sum(), 2),
            "utile/anno $": round(usd_d.sum() / yrs), "maxDD $": round((eq - eq.cummax()).min()),
            "giorno peggiore $": round(usd_d.min()), "sharpe": round(usd_d.mean() / usd_d.std() * np.sqrt(252), 2),
            "t-stat trade": round(usd_t.mean() / usd_t.std() * np.sqrt(len(tr)), 1),
            "sempre long: utile/anno $": round(bh.sum() / yrs), "sempre long: maxDD $": round((beq - beq.cummax()).min())}, usd_d, tr


def main():
    pd.set_option("display.width", 250)
    mes = lambda **k: run("SPY", level_mult=10.0, pv=5.0, **k)
    print("== 1 contratto, USD al livello attuale dell'indice, default (IBS<0,10, 5 sessioni, stop 8%) ==")
    cols = {"MES 2019-2026": mes(a="2019")[0], "MNQ 2019-2026 (controllo)": run("QQQ", a="2019", level_mult=41.0, pv=2.0)[0],
            "MES 1993-2026": mes()[0], "MES 1993-2012": mes(b="2013")[0], "MES 2013-2018": mes(a="2013", b="2019")[0]}
    print(pd.DataFrame(cols).to_string())
    _, d, tr = mes()
    print("\n== MES per anno, USD al livello attuale ==")
    print(pd.DataFrame({"strategia": d.groupby(d.index.year).sum(), "trade": tr.groupby(tr["in"].dt.year).size()}).round().T.to_string())
    print("\n== MES 2019-2026: varianti ==")
    keys = ["trade", "vincenti %", "media trade $", "peggiore $", "PF", "utile/anno $", "maxDD $", "sharpe"]
    var = {"base": {}, "senza stop": dict(stop=None), "soglia 0,05": dict(thr=0.05), "soglia 0,20": dict(thr=0.20), "hold 3": dict(hold=3)}
    print(pd.DataFrame({k: mes(a="2019", **v)[0] for k, v in var.items()}).loc[keys].to_string())


if __name__ == "__main__":
    main()
