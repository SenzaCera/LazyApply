# Strategia intraday su MNQ: momentum su rottura della "noise area"

Strategia di day trading solo su Micro E-mini Nasdaq-100 (MNQ), nessuna posizione overnight.
L'idea viene dalla letteratura sul momentum intraday (Zarattini, Aziz, Barbon, 2024): quando il prezzo
esce dalla fascia di oscillazione "normale" per quell'ora del giorno, tende a proseguire fino alla chiusura.

## Regole

Sessione: 09:30-16:00 ET. Tutti i calcoli su barre a 1 minuto.

1. **Noise area.** Per ogni minuto della seduta, `sigma(t)` = media degli ultimi 14 giorni di
   `|prezzo(t) / apertura 09:30 - 1|`.
   - Banda alta = `max(apertura, chiusura di ieri) * (1 + sigma(t))`
   - Banda bassa = `min(apertura, chiusura di ieri) * (1 - sigma(t))`
2. **Decisioni solo alle :00 e :30**, dalle 10:00 alle 15:30.
   - Long se il prezzo chiude sopra `max(banda alta, VWAP)`.
   - Short se chiude sotto `min(banda bassa, VWAP)`.
   - Altrimenti flat: la stessa soglia fa da trailing stop (si esce al controllo successivo in cui il prezzo e' rientrato).
3. **Esecuzione** a mercato sulla barra successiva. **Uscita obbligata alle 16:00 ET.**
4. **Sizing** (facoltativo): contratti = `2% del capitale / volatilita' giornaliera a 14 giorni`, leva massima 4x.

Parametri (14 giorni, bande 1x, passo 30 minuti) presi dalla letteratura e fissati prima di guardare il periodo di verifica.

## Risultati

Costo ipotizzato: 1,5 punti (3 USD) a trade andata e ritorno. 1 punto MNQ = 2 USD. Valori per 1 contratto fisso.

| Periodo | Trade | Vincenti | Profit factor | Punti | USD | Sharpe | t-stat | Max drawdown |
|---|---|---|---|---|---|---|---|---|
| In-sample 2019-2022 | 872 | 39% | 1,21 | +3.846 | +7.691 | 0,95 | 1,87 | -897 pt |
| **Fuori campione 2023-nov 2026** | 795 | 42% | 1,23 | +5.827 | +11.655 | 0,93 | 1,80 | -2.095 pt |
| Totale | 1.667 | | | +9.673 | +19.346 | 0,91 | 2,52 | -2.095 pt |

Con il sizing a volatilita': circa +12,7% annuo sul capitale, Sharpe 0,93, drawdown massimo -17%, giorno peggiore -4,1%.

Per anno (punti, 1 contratto): 2019 -178, 2020 +406, 2021 +1.492, 2022 +2.126, 2023 +1.688, 2024 +1.635, 2025 +1.289, 2026 +1.216.

Fuori campione, variando un parametro alla volta lo Sharpe resta positivo (0,49-0,93); con costi a 3 punti scende a 0,74, a 5 punti a 0,48.
Long 0,74, short 0,55. L'opening range breakout (5/15/30 minuti) e' stato scartato in-sample (Sharpe 0,1-0,5).

## Cosa aspettarsi

- Circa 0,9 trade al giorno, durata media ~2 ore, si opera nel 59% delle sedute.
- **Solo il 26% delle giornate chiude in positivo**: molte piccole perdite, poche giornate di trend che pagano tutto.
- Drawdown di 2.000 punti (4.000 USD a contratto) e' gia' successo: va dimensionato di conseguenza.

## Limiti

- L'edge e' modesto: t-stat 2,5 su tutto il campione, 1,8 sul solo fuori campione. Non e' una certezza statistica.
- I dati sono il CFD Nasdaq-100 di Dukascopy, non il future. Sulle ultime 60 sedute la correlazione dei rendimenti a 5 minuti con MNQ reale e' 0,988, ma il VWAP qui usa volumi tick; senza VWAP il risultato e' quasi identico (Sharpe 0,92).
- 2019 negativo (con l'indice a 7.000 il costo fisso pesava di piu') e 2025-26 piu' deboli a parita' di rischio.
- Slippage reale su notizie e aperture non e' modellato oltre il costo fisso. Fare paper trading prima di usare denaro vero.

## Riprodurre

```
pip install pandas numpy requests
python download_data.py 2019-01-01 2026-11-13 data/ndx_1m.csv.gz   # ~1-2 ore, il server limita la frequenza
python research.py data/ndx_1m.csv.gz      # confronto ipotesi, solo 2019-2022
python evaluate.py data/ndx_1m.csv.gz      # in-sample, fuori campione, costi, robustezza
```

`backtest.py` contiene il motore: segnale sulla chiusura della barra, esecuzione all'apertura della successiva, costi su ogni trade.

## NinjaTrader 8

`ninjatrader/MnqNoiseAreaMomentum.cs` e' la stessa strategia in NinjaScript (non compilata ne' provata in questo ambiente).

1. Copia il file in `Documenti\NinjaTrader 8\bin\Custom\Strategies\` e compila dal NinjaScript Editor (F5).
2. Applicala a un grafico **MNQ a 1 minuto** con almeno 30 giorni di storico caricati (servono 14 sedute complete prima del primo segnale).
3. Imposta commissioni e 1 tick di slippage nello Strategy Analyzer per confrontare con i risultati sopra.

La dimensione e' fissa (parametro "Contratti"); il sizing a volatilita' non e' incluso.
Parametri facoltativi (tutti disattivati di default): "Stop loss %", "Take profit %" e le due opzioni per non rientrare nella stessa direzione dopo uno stop o un take profit.

## Stop loss, frequenza dei controlli, capitale (`extras.py`)

Queste prove sono state fatte dopo aver gia' visto il periodo 2023+, quindi valgono meno della verifica principale.

| Variante | Sharpe 2019-22 | Sharpe 2023+ | Max drawdown 2023+ |
|---|---|---|---|
| Nessuno stop (base) | 0,95 | 0,93 | -2.095 pt |
| Stop 0,25% | 1,16 | 1,03 | -1.605 pt |
| Stop 0,4% | 1,16 | 1,05 | -1.796 pt |
| Stop 0,6% | 1,14 | 0,94 | -1.978 pt |
| Controlli ogni 5 min | 0,08 | 0,49 | -1.710 pt |
| Controlli ogni 15 min | 0,73 | 0,51 | -2.441 pt |
| Controlli ogni 60 min | 1,01 | 0,91 | -1.617 pt |

Capitale per 1 MNQ con l'indice a ~28.000 (bootstrap a blocchi sul P&L giornaliero): utile medio ~4.300 USD/anno,
giorno peggiore -2.200, drawdown in un anno -2.700 mediano, -6.600 al 95%, -8.600 al 99%; sull'intero periodo -11.300 al 95%.

## Take profit (`takeprofit.py`)

Provato sopra la variante stop 0,4% senza rientro nella stessa direzione. Un target stretto (0,3%) dimezza l'utile; fra 0,5% e 2%
i risultati oscillano senza un andamento regolare (Sharpe 2023+ fra 0,8 e 1,35), segno che le differenze sono in buona parte rumore.
Il migliore sulla carta e' 1% senza rientro (utile ~4.500 USD/anno contro ~3.800), ma e' un valore scelto a posteriori.
