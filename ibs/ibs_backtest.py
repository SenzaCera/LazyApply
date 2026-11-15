"""Backtest della strategia IbsSwingDaily (NinjaScript) su dati giornalieri Yahoo.

Regole replicate:
  * segnale: IBS = (Close-Low)/(High-Low) < soglia a fine sessione, solo se flat e senza uscita nello stesso giorno
  * ingresso all'apertura della sessione successiva
  * uscita in chiusura se Close > massimo della sessione precedente, oppure alla 5a sessione (quella d'ingresso conta)
  * stop catastrofe all'8% dal prezzo d'ingresso (eseguito allo stop, o all'apertura se apre gia' sotto)
Approssimazione: il controllo d'uscita avviene 2 minuti prima della chiusura; qui si usa il prezzo di chiusura.
Prezzi corretti per dividendi e split (rendimento totale). Costo: 0,05% andata e ritorno per trade.

Uso: python ibs_backtest.py            (legge data/<SIMBOLO>.json scaricati da Yahoo)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

COST = 0.0005
SYMBOLS = ["QQQ", "SPY", "IWM", "DIA"]


def load(sym, adjust=True):
    r = json.loads((Path(__file__).parent / "data" / f"{sym}.json").read_text())["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                      index=pd.to_datetime(r["timestamp"], unit="s").normalize())
    df["adj"] = r["indicators"]["adjclose"][0]["adjclose"]
    df = df.dropna()
    f = df["adj"] / df["close"] if adjust else 1.0         # fattore dividendi (no per i future)
    for c in ("open", "high", "low", "close"):
        df[c] *= f
    return df[df.high > df.low]


def backtest(df, thr=0.10, hold=5, stop=0.08, cost=COST):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    ibs = (c - l) / (h - l)
    n = len(df)
    daily = np.zeros(n)                                      # rendimento giornaliero della strategia
    trades, in_pos, entry, held, pending, i_in = [], False, 0.0, 0, False, 0
    for i in range(1, n):
        exited = False
        if pending:
            pending, in_pos, entry, held, i_in = False, True, o[i], 0, i
        if in_pos:
            ref = entry if i == i_in else c[i - 1]           # base per il mark-to-market di oggi
            px = None
            if stop and l[i] <= entry * (1 - stop):
                px = min(o[i], entry * (1 - stop))
            else:
                held += 1
                if c[i] > h[i - 1] or held >= hold:
                    px = c[i]
            if px is not None:
                daily[i] = px / ref - 1 - cost
                trades.append((df.index[i_in], df.index[i], px / entry - 1 - cost, i - i_in + 1))
                in_pos, exited = False, True
            else:
                daily[i] = c[i] / ref - 1
        if not in_pos and not exited and ibs[i] < thr:
            pending = True
    tr = pd.DataFrame(trades, columns=["in", "out", "ret", "days"])
    return tr, pd.Series(daily, index=df.index)


def stats(tr, daily, df):
    yrs = len(daily) / 252
    eq = (1 + daily).cumprod()
    bh = df["close"].reindex(daily.index)
    bh_ret = bh.pct_change().fillna(0)
    bh_eq = (1 + bh_ret).cumprod()
    w, lo = tr.ret[tr.ret > 0], tr.ret[tr.ret <= 0]
    return {
        "trade": len(tr), "trade/anno": round(len(tr) / yrs, 1), "vincenti %": round(100 * len(w) / max(len(tr), 1), 1),
        "media trade %": round(100 * tr.ret.mean(), 2), "vincita media %": round(100 * w.mean(), 2),
        "perdita media %": round(100 * lo.mean(), 2), "peggiore %": round(100 * tr.ret.min(), 1),
        "PF": round(w.sum() / -lo.sum(), 2), "CAGR %": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 1),
        "maxDD %": round(100 * (eq / eq.cummax() - 1).min(), 1),
        "sharpe": round(daily.mean() / daily.std() * np.sqrt(252), 2),
        "t-stat trade": round(tr.ret.mean() / tr.ret.std() * np.sqrt(len(tr)), 1),
        "tempo a mercato %": round(100 * (daily != 0).mean()),
        "B&H CAGR %": round(100 * (bh_eq.iloc[-1] ** (1 / yrs) - 1), 1),
        "B&H maxDD %": round(100 * (bh_eq / bh_eq.cummax() - 1).min(), 1),
    }


def sliced(tr, daily, df, a, b):
    m = (daily.index >= a) & (daily.index < b)
    return stats(tr[(tr["in"] >= a) & (tr["in"] < b)], daily[m], df[m])


def main():
    pd.set_option("display.width", 250)
    data = {s: load(s) for s in SYMBOLS}
    res = {s: backtest(df) for s, df in data.items()}
    print("== Parametri di default (IBS < 0,10, 5 sessioni, stop 8%) - intero storico ==")
    print(pd.DataFrame({f"{s} dal {data[s].index[0].year}": stats(*res[s], data[s]) for s in SYMBOLS}).to_string())
    for name, a, b in (("fino al 2012", "1990", "2013"), ("2013-2019", "2013", "2020"), ("2020-2026", "2020", "2030")):
        print(f"\n== {name} ==")
        keys = ["trade", "vincenti %", "media trade %", "PF", "CAGR %", "maxDD %", "sharpe", "t-stat trade", "B&H CAGR %"]
        print(pd.DataFrame({s: sliced(*res[s], data[s], a, b) for s in SYMBOLS}).loc[keys].to_string())
    print("\n== QQQ per anno (rendimento % strategia / buy&hold) ==")
    d, bh = res["QQQ"][1], data["QQQ"]["close"].pct_change().fillna(0)
    yr = pd.DataFrame({"strategia": ((1 + d).groupby(d.index.year).prod() - 1) * 100,
                       "buy&hold": ((1 + bh).groupby(bh.index.year).prod() - 1) * 100}).round(1)
    print(yr.T.to_string())
    print("\n== QQQ: sensibilita' ai parametri (intero storico) ==")
    keys = ["trade", "vincenti %", "media trade %", "PF", "CAGR %", "maxDD %", "sharpe"]
    var = {"base": {}, "soglia 0,05": dict(thr=0.05), "soglia 0,15": dict(thr=0.15), "soglia 0,20": dict(thr=0.20),
           "soglia 0,30": dict(thr=0.30), "hold 3": dict(hold=3), "hold 10": dict(hold=10),
           "senza stop": dict(stop=None), "stop 4%": dict(stop=0.04), "costo 0,2%": dict(cost=0.002)}
    print(pd.DataFrame({k: stats(*backtest(data["QQQ"], **v), data["QQQ"]) for k, v in var.items()}).loc[keys].to_string())


if __name__ == "__main__":
    main()
