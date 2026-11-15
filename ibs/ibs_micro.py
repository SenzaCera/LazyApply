"""IbsSwingDaily sui quattro micro future azionari, stimati dagli ETF giornalieri (vedi ibs_mes.py per il metodo).

MES = SPY x10 x 5$   MNQ = QQQ x41 x 2$   MYM = DIA x100 x 0,5$   M2K = IWM x10 x 5$

Uso: python ibs_micro.py
"""
import pandas as pd

from ibs_mes import run

SPECS = {"MYM": ("DIA", 100.0, 0.5), "M2K": ("IWM", 10.0, 5.0), "MES": ("SPY", 10.0, 5.0), "MNQ": ("QQQ", 41.0, 2.0)}


def main():
    pd.set_option("display.width", 250)
    r = lambda k, **kw: run(SPECS[k][0], level_mult=SPECS[k][1], pv=SPECS[k][2], **kw)
    print("== 2019-2026, 1 contratto, USD al livello attuale ==")
    print(pd.DataFrame({k: r(k, a="2019")[0] for k in SPECS}).to_string())
    print("\n== Intero storico (dal 2000 per confronto omogeneo) ==")
    print(pd.DataFrame({k: r(k, a="2001")[0] for k in SPECS}).to_string())
    keys = ["trade", "vincenti %", "PF", "utile/anno $", "maxDD $", "sharpe", "t-stat trade"]
    for name, a, b in (("2001-2012", "2001", "2013"), ("2013-2018", "2013", "2019")):
        print(f"\n== {name} ==")
        print(pd.DataFrame({k: r(k, a=a, b=b)[0] for k in SPECS}).loc[keys].to_string())
    print("\n== Per anno, USD al livello attuale ==")
    print(pd.DataFrame({k: (lambda d: d.groupby(d.index.year).sum())(r(k, a="2001")[1]) for k in ("MYM", "M2K")}).round().T.to_string())
    print("\n== Senza stop, 2019-2026 ==")
    print(pd.DataFrame({k: r(k, a="2019", stop=None)[0] for k in ("MYM", "M2K")}).loc[keys + ["peggiore $"]].to_string())


if __name__ == "__main__":
    main()
