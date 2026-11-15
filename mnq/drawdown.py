"""Leve per ridurre il drawdown. Uso: python drawdown.py data/ndx_1m.csv.gz

Tutto per 1 contratto "medio", in USD riportati al livello attuale dell'indice.
La colonna utile/DD e' quella che conta: il drawdown in dollari si riduce sempre operando piu' piccolo,
una leva e' utile solo se migliora il rapporto fra utile annuo e drawdown massimo.
"""
import sys

import numpy as np
import pandas as pd

from backtest import POINT_VALUE, Params, intraday_momentum, load_rth

OOS = pd.Timestamp("2023-01-01")


def daily(d, tr, days, size=None, loss_limit=None):
    level = d["close"][-1, -1]
    tr = tr.copy()
    tr["usd"] = tr.points / tr.entry * level * POINT_VALUE
    if loss_limit:                                          # stop giornaliero: dopo -X USD realizzati, basta per oggi
        keep = []
        for _, g in tr.sort_values("t_in").groupby("day"):
            cum, cut = 0.0, None
            for idx, r in g.iterrows():
                if cut is not None and r.t_in >= cut:
                    continue
                keep.append(idx)
                cum += r.usd
                if cum <= -loss_limit and cut is None:
                    cut = r.t_out
        tr = tr.loc[keep]
    dl = tr.groupby("day").usd.sum().reindex(days, fill_value=0.0)
    return dl if size is None else dl * size


def breaker(dl, dd_limit=None, losing_days=None, factor=0.5):
    """Riduce la dimensione (x factor) quando la curva a dimensione piena e' in drawdown oltre dd_limit
    o dopo N giornate operative negative di fila. Decisione presa sui dati fino a ieri."""
    x = dl.to_numpy()
    eq = np.cumsum(x)
    dd = eq - np.maximum.accumulate(eq)
    size = np.ones(len(x))
    streak = 0
    for i in range(1, len(x)):
        if x[i - 1] < 0:
            streak += 1
        elif x[i - 1] > 0:
            streak = 0
        if (dd_limit and dd[i - 1] < -dd_limit) or (losing_days and streak >= losing_days):
            size[i] = factor
    return dl * size


def summary(dl):
    def one(x):
        eq = x.cumsum()
        return x.mean() * 252, (eq - eq.cummax()).min(), x.mean() / x.std() * np.sqrt(252)
    net, dd, _ = one(dl)
    _, dd_is, sh_is = one(dl[dl.index < OOS])
    net_o, dd_oos, sh_oos = one(dl[dl.index >= OOS])
    eq = dl.cumsum()
    under = (eq < eq.cummax()).astype(int)
    longest = under.groupby((under == 0).cumsum()).sum().max()
    return {"utile/anno": round(net), "maxDD": round(dd), "utile/DD": round(net / -dd, 2), "giorno peggiore": round(dl.min()),
            "giorni sott'acqua max": int(longest), "sharpe IS": round(sh_is, 2), "sharpe OOS": round(sh_oos, 2),
            "utile/DD OOS": round(net_o / -dd_oos, 2)}


def main(path):
    pd.set_option("display.width", 250)
    d = load_rth(path)
    days = d["days"][15:]
    run = lambda **k: intraday_momentum(d, Params(**k))
    base = run(stop_pct=0.4)
    dl = daily(d, base, days)
    # sizing a volatilita' costante, normalizzato a dimensione media 1
    ret = pd.Series(d["close"][:, -1]).pct_change()
    vol = pd.Series(ret.rolling(14).std().shift(1).to_numpy(), index=d["days"]).reindex(days)
    inv = (1 / vol).clip(upper=(1 / vol).quantile(0.95))
    size = inv / inv.mean()
    rows = {
        "BASE: stop 0,4% con rientro": summary(dl),
        "senza stop": summary(daily(d, run(), days)),
        "stop 0,25% con rientro": summary(daily(d, run(stop_pct=0.25), days)),
        "stop 0,4%, no rientro stessa direzione": summary(daily(d, run(stop_pct=0.4, after_stop="no_stessa_direzione"), days)),
        "stop 0,4% + TP 1% no rientro": summary(daily(d, run(stop_pct=0.4, after_stop="no_stessa_direzione", take_pct=1.0, after_take="no_stessa_direzione"), days)),
        "limite perdita giornaliera 300 USD": summary(daily(d, base, days, loss_limit=300)),
        "limite perdita giornaliera 500 USD": summary(daily(d, base, days, loss_limit=500)),
        "solo long": summary(daily(d, run(stop_pct=0.4, long_only=True), days)),
        "controlli ogni 60 min": summary(daily(d, run(stop_pct=0.4, step=60), days)),
        "bande 1,2x": summary(daily(d, run(stop_pct=0.4, band_mult=1.2), days)),
        "sizing a volatilita' costante": summary(daily(d, base, days, size=size)),
        "meta' size se drawdown > 2.000 USD": summary(breaker(dl, dd_limit=2000)),
        "meta' size se drawdown > 3.000 USD": summary(breaker(dl, dd_limit=3000)),
        "fermo se drawdown > 3.000 USD": summary(breaker(dl, dd_limit=3000, factor=0.0)),
        "meta' size dopo 5 giorni negativi di fila": summary(breaker(dl, losing_days=5)),
        "no rientro + sizing a volatilita'": summary(daily(d, run(stop_pct=0.4, after_stop="no_stessa_direzione"), days, size=size)),
    }
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
