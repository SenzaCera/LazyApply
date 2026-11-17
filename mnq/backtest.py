"""Backtest di strategie intraday su MNQ (dati 1 minuto del Nasdaq-100, sessione RTH 09:30-16:00 ET).

Regole anti-autoinganno:
  * segnale calcolato sulla chiusura della barra t, eseguito all'apertura della barra t+1;
  * costi sottratti a ogni trade (commissioni + slippage), espressi in punti indice;
  * nessuna posizione overnight: tutto chiuso alle 16:00 ET;
  * gli indicatori usano solo dati precedenti (medie con shift di un giorno).

Risultati in punti indice per 1 contratto. 1 punto MNQ = 2 USD.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POINT_VALUE = 2.0          # USD per punto, Micro E-mini Nasdaq-100
COST_POINTS = 1.5          # costo andata+ritorno: ~1.2$ commissioni + 1 tick di slippage per lato
BARS = 390                 # minuti RTH


def load_rth(path: str) -> dict[str, np.ndarray]:
    """Restituisce matrici (giorni x 390 minuti) open/high/low/close/volume della sessione RTH."""
    df = pd.read_csv(path, parse_dates=["ts"])
    ts = df["ts"].dt.tz_convert("America/New_York")
    mins = ts.dt.hour * 60 + ts.dt.minute - 570            # 0 = barra 09:30
    df = df[(mins >= 0) & (mins < BARS)].assign(day=ts.dt.date, m=mins)
    out = {}
    days = np.array(sorted(df["day"].unique()))
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = df.pivot(index="day", columns="m", values=c).reindex(index=days, columns=range(BARS)).to_numpy()
    full = np.isnan(out["close"]).sum(axis=1) <= 5          # scarta festivi / sedute corte / buchi
    for c in out:
        a = out[c][full]
        if c == "volume":
            a = np.nan_to_num(a)
        else:                                               # riempi i rari buchi con l'ultimo prezzo
            a = pd.DataFrame(a).ffill(axis=1).bfill(axis=1).to_numpy()
        out[c] = a
    out["days"] = pd.to_datetime(days[full])
    return out


@dataclass
class Params:
    lookback: int = 14        # giorni per la "noise area"
    band_mult: float = 1.0    # ampiezza bande in multipli del movimento medio
    step: int = 30            # minuti fra una decisione e l'altra
    first: int = 30           # prima decisione: 10:00
    use_vwap: bool = True     # VWAP come filtro d'ingresso e trailing stop
    long_only: bool = False
    vol_target: float | None = None   # es. 0.02 = rischio giornaliero 2%; None = 1 contratto fisso
    max_lev: float = 4.0
    stop_pct: float | None = None     # stop loss intrabar, in % del prezzo d'ingresso (es. 0.5)
    after_stop: str = "rientra"       # dopo uno stop: "rientra", "no_stessa_direzione", "fermo" (basta per oggi)
    take_pct: float | None = None     # take profit intrabar, in % del prezzo d'ingresso
    after_take: str = "rientra"       # dopo un take profit: stesse opzioni di after_stop
    # Protezione del profitto (attiva solo dopo che il trade ha guadagnato almeno protect_act %):
    #   "pareggio": stop spostato all'ingresso + protect_val punti
    #   "trailing": stop a protect_val % sotto il massimo raggiunto (sopra il minimo per gli short)
    #   "giveback": chiude se restituisce protect_val (frazione, es. 0.5) del massimo guadagno raggiunto
    protect: str | None = None
    protect_act: float = 0.5
    protect_val: float = 0.0
    after_protect: str = "rientra"
    protect_sides: str = "entrambi"   # a quali posizioni si applica la protezione: "entrambi", "long", "short"
    # Regole aggiuntive, ognuna (modo, attivazione %, valore, lati). Modo extra: "riduci" = stop a valore % dall'ingresso
    # (es. 0.1 = stop a -0,1%, -0.05 = stop a +0,05%). Fra le regole attive vale lo stop piu' stretto.
    protect_rules: tuple = ()


def noise_bands(d, p: Params):
    """Bande della noise area: apertura +/- movimento medio |close_t/open-1| degli ultimi N giorni."""
    o = d["open"][:, [0]]
    move = np.abs(d["close"] / o - 1)
    sig = pd.DataFrame(move).rolling(p.lookback).mean().shift(1).to_numpy()
    prev_close = np.r_[np.nan, d["close"][:-1, -1]][:, None]
    ub = np.fmax(o, prev_close) * (1 + p.band_mult * sig)
    lb = np.fmin(o, prev_close) * (1 - p.band_mult * sig)
    return ub, lb


def _intrabar_exit(d, i, pos, entry, t_in, t_end, p):
    """Cerca stop loss / take profit fra t_in e t_end (incluso). Ritorna (barra, prezzo, tipo) o None.
    Se stop e target cadono nella stessa barra si assume lo stop (ipotesi prudente)."""
    lo, hi, op = d["low"][i, t_in:t_end + 1], d["high"][i, t_in:t_end + 1], d["open"][i, t_in:t_end + 1]
    best = None
    if p.stop_pct:
        stop = entry * (1 - pos * p.stop_pct / 100)
        hit = lo <= stop if pos == 1 else hi >= stop
        if hit.any():
            k = int(np.argmax(hit))
            px = min(stop, op[k]) if pos == 1 else max(stop, op[k])      # gap oltre lo stop
            best = (t_in + k, px, "stop")
    if p.take_pct:
        tgt = entry * (1 + pos * p.take_pct / 100)
        hit = hi >= tgt if pos == 1 else lo <= tgt
        if hit.any():
            k = int(np.argmax(hit))
            if best is None or t_in + k < best[0]:
                best = (t_in + k, tgt, "take")
    rules = list(p.protect_rules) + ([(p.protect, p.protect_act, p.protect_val, p.protect_sides)] if p.protect else [])
    rules = [r for r in rules if r[3] == "entrambi" or r[3] == ("long" if pos == 1 else "short")]
    if rules:
        # miglior prezzo raggiunto PRIMA di ogni barra (nella barra stessa non si sa se viene prima il massimo o il minimo)
        fav = hi if pos == 1 else lo
        acc = np.maximum.accumulate if pos == 1 else np.minimum.accumulate
        best_px = acc(np.r_[entry, fav[:-1]])
        mfe = (best_px - entry) * pos / entry * 100              # massimo guadagno in % finora
        eff = np.full(len(best_px), -np.inf if pos == 1 else np.inf)
        for mode, act, val, _ in rules:
            if mode == "pareggio":
                lvl = np.full(len(best_px), entry + pos * val)
            elif mode == "riduci":
                lvl = np.full(len(best_px), entry * (1 - pos * val / 100))
            elif mode == "trailing":
                lvl = best_px * (1 - pos * val / 100)
            else:                                                # giveback
                lvl = entry + (best_px - entry) * (1 - val)
            on = mfe >= act
            eff = np.where(on, np.maximum(eff, lvl), eff) if pos == 1 else np.where(on, np.minimum(eff, lvl), eff)
        hit = (lo <= eff) if pos == 1 else (hi >= eff)
        if hit.any():
            k = int(np.argmax(hit))
            px = min(eff[k], op[k]) if pos == 1 else max(eff[k], op[k])
            if best is None or t_in + k < best[0]:
                best = (t_in + k, px, "protect")
    return best


def _block(mode, pos):
    return {pos} if mode == "no_stessa_direzione" else ({1, -1} if mode == "fermo" else set())


def intraday_momentum(d, p: Params = Params(), cost=COST_POINTS, allow=None) -> pd.DataFrame:
    """Momentum intraday su rottura della noise area. Ritorna una riga per trade.

    allow = (ok_long, ok_short): matrici booleane giorni x minuti; un NUOVO ingresso e' ammesso solo se
    il filtro di conferma e' vero alla barra del controllo (le posizioni gia' aperte non ne risentono)."""
    ub, lb = noise_bands(d, p)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vol = np.maximum(d["volume"], 1e-9)
    vwap = np.cumsum(tp * vol, axis=1) / np.cumsum(vol, axis=1)
    ret = pd.Series(d["close"][:, -1]).pct_change()
    dvol = ret.rolling(14).std().shift(1).to_numpy()        # volatilita' giornaliera per il sizing
    trades = []
    checks = range(p.first - 1, BARS - 1, p.step)           # barra che chiude a hh:00 / hh:30
    intrabar = p.stop_pct or p.take_pct or p.protect or p.protect_rules
    for i in range(p.lookback + 1, len(d["days"])):
        pos, entry, t_in, blocked = 0, 0.0, 0, set()
        size = 1.0 if p.vol_target is None else min(p.max_lev, p.vol_target / dvol[i]) if dvol[i] > 0 else 0.0
        for t in checks:
            c = d["close"][i, t]
            up, dn = ub[i, t], lb[i, t]
            if p.use_vwap:
                up, dn = max(up, vwap[i, t]), min(dn, vwap[i, t])
            want = 1 if c > up else (-1 if c < dn and not p.long_only else 0)
            if pos and intrabar:                            # stop/target toccato fra l'ingresso e questo controllo?
                ex = _intrabar_exit(d, i, pos, entry, t_in, t, p)
                if ex:
                    trades.append((d["days"][i], t_in, ex[0], pos, entry, ex[1], size))
                    blocked |= _block({"stop": p.after_stop, "take": p.after_take}.get(ex[2], p.after_protect), pos)
                    pos = 0
            if want in blocked:
                want = 0
            if allow is not None and want != 0 and want != pos and not allow[0 if want == 1 else 1][i, t]:
                want = 0                                    # conferma mancante: niente nuovo ingresso
            if want != pos:
                px = d["open"][i, t + 1]
                if pos:
                    trades.append((d["days"][i], t_in, t + 1, pos, entry, px, size))
                pos, entry, t_in = want, px, t + 1
        if pos and intrabar:                                # ultimo tratto fino alla chiusura
            ex = _intrabar_exit(d, i, pos, entry, t_in, BARS - 1, p)
            if ex:
                trades.append((d["days"][i], t_in, ex[0], pos, entry, ex[1], size))
                pos = 0
        if pos:
            trades.append((d["days"][i], t_in, BARS - 1, pos, entry, d["close"][i, -1], size))
    tr = pd.DataFrame(trades, columns=["day", "t_in", "t_out", "side", "entry", "exit", "size"])
    tr["points"] = (tr["exit"] - tr["entry"]) * tr["side"] - cost
    tr["ret"] = tr["points"] / tr["entry"] * tr["size"]    # rendimento su nozionale * leva
    return tr


def orb(d, minutes=5, cost=COST_POINTS, target_r: float | None = None) -> pd.DataFrame:
    """Opening Range Breakout: direzione della prima candela, stop sull'estremo opposto, uscita a fine seduta."""
    trades = []
    for i in range(len(d["days"])):
        o, c = d["open"][i, 0], d["close"][i, minutes - 1]
        if c == o:
            continue
        side = 1 if c > o else -1
        hi, lo = d["high"][i, :minutes].max(), d["low"][i, :minutes].min()
        entry = d["open"][i, minutes]
        stop = lo if side == 1 else hi
        risk = (entry - stop) * side
        if risk <= 0:
            continue
        exit_px, t_out = d["close"][i, -1], BARS - 1
        for t in range(minutes, BARS):
            if (side == 1 and d["low"][i, t] <= stop) or (side == -1 and d["high"][i, t] >= stop):
                exit_px, t_out = stop, t
                break
            if target_r:
                tgt = entry + side * target_r * risk
                if (side == 1 and d["high"][i, t] >= tgt) or (side == -1 and d["low"][i, t] <= tgt):
                    exit_px, t_out = tgt, t
                    break
        trades.append((d["days"][i], minutes, t_out, side, entry, exit_px, 1.0))
    tr = pd.DataFrame(trades, columns=["day", "t_in", "t_out", "side", "entry", "exit", "size"])
    tr["points"] = (tr["exit"] - tr["entry"]) * tr["side"] - cost
    tr["ret"] = tr["points"] / tr["entry"]
    return tr


def stats(tr: pd.DataFrame, all_days: pd.DatetimeIndex, col="points") -> dict:
    """Metriche su P&L giornaliero (i giorni senza trade contano come 0)."""
    daily = tr.groupby("day")[col].sum().reindex(all_days, fill_value=0.0)
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    sd = daily.std()
    n = len(tr)
    wins = tr[col] > 0
    gp, gl = tr.loc[wins, col].sum(), -tr.loc[~wins, col].sum()
    return {
        "trades": n,
        "win%": round(100 * wins.mean(), 1) if n else 0,
        "avg": round(tr[col].mean(), 4) if n else 0,
        "PF": round(gp / gl, 2) if gl > 0 else np.nan,
        "total": round(daily.sum(), 3),
        "sharpe": round(daily.mean() / sd * np.sqrt(252), 2) if sd > 0 else 0,
        "t": round(daily.mean() / sd * np.sqrt(len(daily)), 2) if sd > 0 else 0,
        "maxDD": round(dd, 3),
        "days": len(daily),
    }


def split(d, start, end):
    m = (d["days"] >= start) & (d["days"] < end)
    return {k: v[m] for k, v in d.items()}
