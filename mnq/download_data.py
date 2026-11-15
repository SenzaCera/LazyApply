"""Scarica barre a 1 minuto del Nasdaq-100 (CFD Dukascopy USATECHIDXUSD, proxy di NQ/MNQ)
e le salva in un CSV unico con timestamp UTC.

Uso: python download_data.py 2019-01-01 2026-11-13 data/ndx_1m.csv.gz
"""
import datetime as dt
import lzma
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://datafeed.dukascopy.com/datafeed/USATECHIDXUSD/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
REC = struct.Struct(">5if")  # sec offset, open, close, low, high, volume
SCALE = 1000.0
PAUSE = 0.6             # secondi fra una richiesta e l'altra (il server limita la frequenza)
_tl = threading.local()


def session():
    # keep-alive: riusare la connessione rende il download molto piu' affidabile
    if not hasattr(_tl, "s"):
        _tl.s = requests.Session()
    return _tl.s


def fetch_day(day: dt.date, cache: Path):
    f = cache / f"{day}.bi5"
    if not f.exists():
        url = URL.format(y=day.year, m=day.month - 1, d=day.day)  # mesi 0-based
        for attempt in range(25):
            try:
                r = session().get(url, timeout=20)
                if r.status_code == 404:
                    f.write_bytes(b"")
                    break
                if r.status_code == 429:          # rate limit: aspetta e riprova
                    time.sleep(60)
                    continue
                r.raise_for_status()
                f.write_bytes(r.content)
                time.sleep(PAUSE)
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("FAILED", day, file=sys.stderr)
            return None
    raw = f.read_bytes()
    if not raw:
        return None
    buf = lzma.decompress(raw)
    a = np.array(list(REC.iter_unpack(buf)), dtype=float)
    if a.size == 0:
        return None
    base = pd.Timestamp(day, tz="UTC")
    df = pd.DataFrame({
        "ts": base + pd.to_timedelta(a[:, 0], unit="s"),
        "open": a[:, 1] / SCALE, "close": a[:, 2] / SCALE,
        "low": a[:, 3] / SCALE, "high": a[:, 4] / SCALE, "volume": a[:, 5],
    })
    return df[df.volume > 0]  # le barre a volume 0 sono riempitivi a mercato chiuso


def main(start, end, out):
    out = Path(out)
    cache = out.parent / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    days = [d.date() for d in pd.date_range(start, end, freq="D") if d.weekday() != 5]
    with ThreadPoolExecutor(1) as ex:
        parts = [p for p in ex.map(lambda d: fetch_day(d, cache), days) if p is not None]
    df = pd.concat(parts).sort_values("ts")[["ts", "open", "high", "low", "close", "volume"]]
    df.to_csv(out, index=False)
    print(len(df), "barre", df.ts.iloc[0], "->", df.ts.iloc[-1])


if __name__ == "__main__":
    main(*sys.argv[1:4])
