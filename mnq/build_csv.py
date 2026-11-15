"""Ricostruisce il CSV dai file gia' in cache (senza rete). Uso: python build_csv.py data/ndx_1m.csv.gz"""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

import download_data as dd

out = Path(sys.argv[1])
cache = out.parent / "raw"
parts = [dd.fetch_day(dt.date.fromisoformat(f.stem), cache) for f in sorted(cache.glob("*.bi5")) if f.stat().st_size]
df = pd.concat([p for p in parts if p is not None]).sort_values("ts")[["ts", "open", "high", "low", "close", "volume"]]
df.to_csv(out, index=False)
print(len(df), "barre", df.ts.iloc[0], "->", df.ts.iloc[-1])
