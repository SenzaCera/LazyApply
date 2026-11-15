"""Filtri di conferma sull'ingresso (indicatori classici). Uso: python confirm.py data/ndx_1m.csv.gz

Ogni filtro usa solo dati disponibili alla chiusura della barra del controllo.
Base: stop 0,4% con rientro (la configurazione provata su NinjaTrader).
"""
import sys

import numpy as np
import pandas as pd

from backtest import BARS, POINT_VALUE, Params, intraday_momentum, load_rth, stats

OOS = pd.Timestamp("2023-01-01")
BASE = Params(stop_pct=0.4)


def bars30(d):
    """Barre a 30 minuti RTH concatenate (13 al giorno) -> DataFrame con indice (giorno, slot)."""
    n = len(d["days"])
    r = lambda a, f: f(a.reshape(n, 13, 30), axis=2).ravel()
    return pd.DataFrame({"high": r(d["high"], np.max), "low": r(d["low"], np.min),
                         "close": d["close"].reshape(n, 13, 30)[:, :, -1].ravel(),
                         "volume": r(d["volume"], np.sum)})


def to_matrix(series, n):
    """Valore dell'indicatore alla chiusura di ogni barra da 30' -> matrice giorni x 390 (valido ai minuti 29, 59, ...)."""
    m = np.full((n, BARS), np.nan)
    m[:, 29::30] = np.asarray(series).reshape(n, 13)
    return m


def wilder(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def adx(b, n=14):
    up, dn = b.high.diff(), -b.low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0))
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr, n)
    pdi, mdi = 100 * wilder(pdm, n) / atr, 100 * wilder(mdm, n) / atr
    return wilder(100 * (pdi - mdi).abs() / (pdi + mdi), n), pdi, mdi


def rsi(c, n=14):
    ch = c.diff()
    return 100 - 100 / (1 + wilder(ch.clip(lower=0), n) / wilder(-ch.clip(upper=0), n))


def filters(d):
    n = len(d["days"])
    b = bars30(d)
    T, F = np.ones((n, BARS), bool), np.zeros((n, BARS), bool)
    f = {}
    a, pdi, mdi = (to_matrix(x, n) for x in adx(b))
    for th in (20, 25, 30):
        f[f"ADX(14) 30' > {th}"] = (a > th, a > th)
    f["DI+ > DI- (long) / DI- > DI+ (short)"] = (pdi > mdi, mdi > pdi)
    r = to_matrix(rsi(b.close), n)
    for th in (55, 60, 70):
        f[f"RSI(14) 30' > {th} long / < {100 - th} short"] = (r > th, r < 100 - th)
    f["RSI non in eccesso (<75 long / >25 short)"] = (r < 75, r > 25)
    e_f, e_s = to_matrix(b.close.ewm(span=9, adjust=False).mean(), n), to_matrix(b.close.ewm(span=21, adjust=False).mean(), n)
    f["EMA9 > EMA21 a 30' (long) / < (short)"] = (e_f > e_s, e_f < e_s)
    macd = b.close.ewm(span=12, adjust=False).mean() - b.close.ewm(span=26, adjust=False).mean()
    h = to_matrix(macd - macd.ewm(span=9, adjust=False).mean(), n)
    f["MACD istogramma > 0 (long) / < 0 (short)"] = (h > 0, h < 0)
    # trend giornaliero: chiusura di ieri contro la sua media mobile
    dc = pd.Series(d["close"][:, -1])
    for k in (20, 50, 200):
        up = (dc.shift(1) > dc.rolling(k).mean().shift(1)).to_numpy()[:, None]
        f[f"Trend giornaliero SMA{k} (solo nella direzione)"] = (T & up, T & ~up)
    # volume relativo: volume cumulato oggi contro la media alla stessa ora degli ultimi 14 giorni
    cv = np.cumsum(d["volume"], axis=1)
    rv = cv / pd.DataFrame(cv).rolling(14).mean().shift(1).to_numpy()
    for th in (1.0, 1.2):
        f[f"Volume relativo > {th}"] = (rv > th, rv > th)
    # direzione della prima mezz'ora
    first = (d["close"][:, 29] > d["open"][:, 0])[:, None]
    f["Prima mezz'ora nella stessa direzione"] = (T & first, T & ~first)
    # volatilita': range della seduta finora contro ATR giornaliero a 14 giorni
    rng = np.maximum.accumulate(d["high"], axis=1) - np.minimum.accumulate(d["low"], axis=1)
    atr_d = pd.Series(d["high"].max(axis=1) - d["low"].min(axis=1)).rolling(14).mean().shift(1).to_numpy()[:, None]
    f["Range di oggi < 100% ATR giornaliero"] = (rng < atr_d, rng < atr_d)
    # orario
    early = np.arange(BARS)[None, :] < 240
    f["Nuovi ingressi solo prima delle 13:30"] = (T & early, T & early)
    late = np.arange(BARS)[None, :] >= 60
    f["Nuovi ingressi solo dalle 10:30"] = (T & late, T & late)
    return f


def row(d, allow):
    level, days = d["close"][-1, -1], d["days"][201:]       # stesso periodo per tutti (SMA200 richiede 200 giorni)
    tr = intraday_momentum(d, BASE, allow=allow)
    tr = tr[tr.day >= days[0]].copy()
    tr["usd"] = tr.points / tr.entry * level * POINT_VALUE
    dl = tr.groupby("day").usd.sum().reindex(days, fill_value=0)
    eq = dl.cumsum()
    a, b = stats(tr[tr.day < OOS], days[days < OOS]), stats(tr[tr.day >= OOS], days[days >= OOS])
    return dict(trades=len(tr), win=round(100 * (tr.usd > 0).mean(), 1), avg_trade=round(tr.usd.mean(), 1),
                net_yr=round(tr.usd.sum() / (len(days) / 252)), maxDD=round((eq - eq.cummax()).min()),
                sh_IS=a["sharpe"], sh_OOS=b["sharpe"], PF_IS=a["PF"], PF_OOS=b["PF"])


def main(path):
    pd.set_option("display.width", 250)
    d = load_rth(path)
    rows = {"NESSUN FILTRO (base)": row(d, None)}
    for name, allow in filters(d).items():
        rows[name] = row(d, allow)
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
