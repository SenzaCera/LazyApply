"""Ricerca in-sample (2019-2022): confronto fra ipotesi candidate. NON tocca il periodo 2023+.

Uso: python research.py data/ndx_1m.csv.gz
"""
import sys

import pandas as pd

from backtest import Params, intraday_momentum, load_rth, orb, split, stats

IS = ("2019-01-01", "2023-01-01")


def main(path):
    d = split(load_rth(path), *IS)
    days = d["days"]
    print("giorni in-sample:", len(days), days[0].date(), "->", days[-1].date())
    rows = {}
    for m in (5, 15, 30):
        rows[f"ORB {m}m"] = stats(orb(d, m), days)
        rows[f"ORB {m}m target 2R"] = stats(orb(d, m, target_r=2), days)
    for vw in (True, False):
        for mult in (0.8, 1.0, 1.2):
            for lb in (14,):
                p = Params(lookback=lb, band_mult=mult, use_vwap=vw)
                rows[f"MOM mult={mult} vwap={vw}"] = stats(intraday_momentum(d, p), days[lb + 1:])
    rows["MOM long-only"] = stats(intraday_momentum(d, Params(long_only=True)), days[15:])
    rows["MOM step=15"] = stats(intraday_momentum(d, Params(step=15, first=30)), days[15:])
    rows["MOM step=60"] = stats(intraday_momentum(d, Params(step=60, first=30)), days[15:])
    rows["MOM lookback=30"] = stats(intraday_momentum(d, Params(lookback=30)), days[31:])
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
