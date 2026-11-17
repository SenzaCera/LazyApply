#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
#endregion

// Momentum intraday su rottura della "noise area" - solo MNQ, grafico a 1 minuto.
// Regole (vedi mnq/README.md):
//   * sigma(t) = media, sugli ultimi N giorni, di |prezzo(t) / apertura 09:30 - 1|
//   * banda alta  = max(apertura, chiusura di ieri) * (1 + mult * sigma(t))
//   * banda bassa = min(apertura, chiusura di ieri) * (1 - mult * sigma(t))
//   * ogni 30 minuti dalle 10:00 alle 15:30 ET: long sopra max(banda alta, VWAP),
//     short sotto min(banda bassa, VWAP), altrimenti flat
//   * ordini a mercato sulla barra successiva, tutto chiuso entro le 16:00 ET
namespace NinjaTrader.NinjaScript.Strategies
{
    public class MnqNoiseAreaMomentum : Strategy
    {
        private const int SessionMinutes = 390;      // 09:30 -> 16:00 ET
        private const int FirstDecision  = 30;       // 10:00 ET

        private TimeZoneInfo    eastern;
        private Queue<double[]> history;             // scostamenti per minuto dei giorni precedenti
        private double[]        todayMove;           // indice = minuti dalle 09:30 alla chiusura della barra (1..390)
        private DateTime        curDate = DateTime.MinValue;
        private int             lastM;
        private double          dayOpen, prevClose, lastClose, cumPV, cumV;
        private double          dayStartPnl;             // utile realizzato cumulato a inizio seduta
        private bool            blockLong, blockShort;   // direzioni fermate dallo stop per oggi
        private double          bestPx;                  // miglior prezzo raggiunto dalla posizione aperta

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                         = "MnqNoiseAreaMomentum";
                Description                  = "Momentum intraday su rottura della noise area (MNQ, 1 minuto).";
                Calculate                    = Calculate.OnBarClose;
                EntriesPerDirection          = 1;
                EntryHandling                = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds    = 30;
                BarsRequiredToTrade          = 1;
                IsInstantiatedOnEachOptimizationIteration = true;

                Lookback    = 14;
                BandMult    = 1.0;
                StepMinutes = 30;
                UseVwap     = true;
                AllowShort  = true;
                Quantity    = 1;
                StopPct     = 0;
                NoReentrySameDirection = false;
                TakePct     = 0;
                NoReentryAfterTarget   = false;
                DailyLossLimit = 0;
                ProtectActivationPct = 0;
                ProtectGivebackPct   = 50;
                ProtectLong          = false;
                ProtectShort         = true;
                ReduceRiskActivationPct = 0;
                ReduceRiskStopPct       = 0.2;

                AddPlot(Brushes.SeaGreen, "BandaAlta");
                AddPlot(Brushes.IndianRed, "BandaBassa");
                AddPlot(Brushes.Goldenrod, "VWAP");
            }
            else if (State == State.Configure)
            {
                if (StopPct > 0)                      // stop loss intrabar, in % del prezzo d'ingresso
                    SetStopLoss(CalculationMode.Percent, StopPct / 100.0);
                if (TakePct > 0)                      // take profit, in % del prezzo d'ingresso
                    SetProfitTarget(CalculationMode.Percent, TakePct / 100.0);
            }
            else if (State == State.DataLoaded)
            {
                eastern = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                history = new Queue<double[]>();
                if (BarsPeriod.BarsPeriodType != BarsPeriodType.Minute || BarsPeriod.Value != 1)
                    Print("MnqNoiseAreaMomentum: ATTENZIONE, la strategia e' pensata per barre a 1 minuto.");
            }
        }

        protected override void OnBarUpdate()
        {
            // Ora di New York della chiusura della barra, qualunque sia il fuso impostato in NinjaTrader
            DateTime et = TimeZoneInfo.ConvertTime(Time[0], Core.Globals.GeneralOptions.TimeZoneInfo, eastern);
            int m = (int)Math.Round((et.TimeOfDay - new TimeSpan(9, 30, 0)).TotalMinutes);
            if (m < 1 || m > SessionMinutes)
                return;                               // fuori dalla sessione regolare

            if (et.Date != curDate)                   // prima barra RTH di una nuova seduta
            {
                if (todayMove != null)
                {
                    prevClose = lastClose;
                    if (lastM == SessionMinutes)      // solo le sedute complete entrano nella media
                    {
                        history.Enqueue(todayMove);
                        while (history.Count > Lookback)
                            history.Dequeue();
                    }
                }
                curDate   = et.Date;
                todayMove = new double[SessionMinutes + 1];
                dayOpen   = Open[0];
                cumPV     = 0;
                cumV      = 0;
                lastM     = 0;
                dayStartPnl = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit;
                blockLong  = false;
                blockShort = false;
            }

            // Scostamento dall'apertura; eventuali minuti senza scambi ereditano l'ultimo valore
            for (int k = lastM + 1; k < m; k++)
                todayMove[k] = todayMove[lastM];
            todayMove[m] = Math.Abs(Close[0] / dayOpen - 1.0);
            lastM        = m;
            lastClose    = Close[0];

            cumPV += (High[0] + Low[0] + Close[0]) / 3.0 * Volume[0];
            cumV  += Volume[0];
            double vwap = cumV > 0 ? cumPV / cumV : Close[0];
            Values[2][0] = vwap;

            if (history.Count < Lookback || prevClose <= 0)
                return;                               // servono N sedute complete di storico

            double sigma = 0;
            foreach (double[] day in history)
                sigma += day[m];
            sigma /= history.Count;

            double upper = Math.Max(dayOpen, prevClose) * (1.0 + BandMult * sigma);
            double lower = Math.Min(dayOpen, prevClose) * (1.0 - BandMult * sigma);
            Values[0][0] = upper;
            Values[1][0] = lower;

            if (m >= SessionMinutes - 1)              // 15:59 ET: chiudi tutto, nessun nuovo ingresso
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong();
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort();
                return;
            }

            // Miglior prezzo raggiunto dalla posizione (serve alla protezione del profitto)
            if (Position.MarketPosition == MarketPosition.Long)
                bestPx = Math.Max(bestPx, High[0]);
            else if (Position.MarketPosition == MarketPosition.Short)
                bestPx = Math.Min(bestPx, Low[0]);

            bool acted = false;
            if (m >= FirstDecision && (m - FirstDecision) % StepMinutes == 0)   // si decide solo alle :00 e :30 (10:00 - 15:30)
            {
                double up = UseVwap ? Math.Max(upper, vwap) : upper;
                double dn = UseVwap ? Math.Min(lower, vwap) : lower;
                int want  = Close[0] > up ? 1 : (Close[0] < dn && AllowShort ? -1 : 0);
                // Limite di perdita giornaliera: superato sul realizzato di oggi, niente nuovi ingressi fino a domani
                double dayPnl = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit - dayStartPnl;
                if (DailyLossLimit > 0 && dayPnl <= -DailyLossLimit && want != 0
                    && Position.MarketPosition != (want == 1 ? MarketPosition.Long : MarketPosition.Short))
                    want = 0;
                if ((want == 1 && blockLong) || (want == -1 && blockShort))
                    want = 0;                             // direzione gia' fermata oggi

                if (want == 1)
                {
                    if (Position.MarketPosition != MarketPosition.Long)
                        { EnterLong(Quantity, "Long"); acted = true; }      // se short, NinjaTrader inverte la posizione
                }
                else if (want == -1)
                {
                    if (Position.MarketPosition != MarketPosition.Short)
                        { EnterShort(Quantity, "Short"); acted = true; }
                }
                else
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        { ExitLong(); acted = true; }
                    else if (Position.MarketPosition == MarketPosition.Short)
                        { ExitShort(); acted = true; }
                }
            }

            if (!acted)
                ManageProfitProtection();
        }

        // Gestione della posizione aperta, con due regole che si sommano (vale lo stop piu' stretto):
        //  1) protezione del profitto: dopo +ProtectActivationPct %, chiude se restituisce ProtectGivebackPct %
        //     del massimo guadagno (attivabile separatamente su long e short);
        //  2) riduzione del rischio: dopo +ReduceRiskActivationPct %, lo stop sale a -ReduceRiskStopPct % dall'ingresso.
        // L'ordine stop vale per la barra successiva e viene ricalcolato a ogni barra.
        private void ManageProfitProtection()
        {
            if (Position.MarketPosition == MarketPosition.Flat)
                return;
            double entry   = Position.AveragePrice;
            int    dir     = Position.MarketPosition == MarketPosition.Long ? 1 : -1;
            double gain    = (bestPx - entry) * dir;
            double gainPct = gain / entry * 100.0;
            double level   = double.NaN;

            bool sideOn = dir == 1 ? ProtectLong : ProtectShort;
            if (sideOn && ProtectActivationPct > 0 && gainPct >= ProtectActivationPct)
                level = entry + dir * gain * (1.0 - ProtectGivebackPct / 100.0);

            if (ReduceRiskActivationPct > 0 && gainPct >= ReduceRiskActivationPct)
            {
                double r = entry * (1.0 - dir * ReduceRiskStopPct / 100.0);
                if (double.IsNaN(level) || (dir == 1 ? r > level : r < level))
                    level = r;                        // tieni lo stop piu' stretto
            }
            if (double.IsNaN(level))
                return;                               // nessuna regola attiva

            level = Instrument.MasterInstrument.RoundToTickSize(level);
            if (dir == 1)
            {
                if (Close[0] <= level)
                    ExitLong(Position.Quantity, "Protezione", "Long");          // gia' sotto il livello: esci subito
                else
                    ExitLongStopMarket(0, false, Position.Quantity, level, "Protezione", "Long");
            }
            else
            {
                if (Close[0] >= level)
                    ExitShort(Position.Quantity, "Protezione", "Short");
                else
                    ExitShortStopMarket(0, false, Position.Quantity, level, "Protezione", "Short");
            }
        }

        // Dopo uno stop loss (o un take profit), niente nuovi ingressi nella stessa direzione fino a domani
        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (execution.Order == null)
                return;
            if ((execution.Order.Name == "Long" || execution.Order.Name == "Short") && execution.Order.OrderState == OrderState.Filled)
                bestPx = execution.Order.AverageFillPrice;   // nuova posizione: riparte il conteggio del massimo
            bool stop   = NoReentrySameDirection && execution.Order.Name == "Stop loss";
            bool target = NoReentryAfterTarget && execution.Order.Name == "Profit target";
            if (!stop && !target)
                return;
            if (execution.Order.OrderAction == OrderAction.Sell)
                blockLong = true;                     // chiusa una posizione long
            else if (execution.Order.OrderAction == OrderAction.BuyToCover)
                blockShort = true;                    // chiusa una posizione short
        }

        #region Parametri
        [NinjaScriptProperty]
        [Range(2, 100)]
        [Display(Name = "Giorni per la noise area", Order = 1, GroupName = "Parametri")]
        public int Lookback { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 5.0)]
        [Display(Name = "Moltiplicatore bande", Order = 2, GroupName = "Parametri")]
        public double BandMult { get; set; }

        [NinjaScriptProperty]
        [Range(1, 180)]
        [Display(Name = "Minuti fra le decisioni", Order = 3, GroupName = "Parametri")]
        public int StepMinutes { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Usa VWAP", Order = 4, GroupName = "Parametri")]
        public bool UseVwap { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Consenti short", Order = 5, GroupName = "Parametri")]
        public bool AllowShort { get; set; }

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Contratti", Order = 6, GroupName = "Parametri")]
        public int Quantity { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 5.0)]
        [Display(Name = "Stop loss % (0 = disattivato)", Order = 7, GroupName = "Parametri")]
        public double StopPct { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Dopo lo stop non rientra nella stessa direzione", Order = 8, GroupName = "Parametri")]
        public bool NoReentrySameDirection { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 10.0)]
        [Display(Name = "Take profit % (0 = disattivato)", Order = 9, GroupName = "Parametri")]
        public double TakePct { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Dopo il take profit non rientra nella stessa direzione", Order = 10, GroupName = "Parametri")]
        public bool NoReentryAfterTarget { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 1000000.0)]
        [Display(Name = "Limite perdita giornaliera $ (0 = disattivato)", Order = 11, GroupName = "Parametri")]
        public double DailyLossLimit { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 10.0)]
        [Display(Name = "Protezione profitto: attiva dopo +% (0 = disattivata)", Order = 12, GroupName = "Parametri")]
        public double ProtectActivationPct { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 100.0)]
        [Display(Name = "Protezione profitto: chiudi se restituisce % del massimo guadagno", Order = 13, GroupName = "Parametri")]
        public double ProtectGivebackPct { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Protezione profitto sui long", Order = 14, GroupName = "Parametri")]
        public bool ProtectLong { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Protezione profitto sugli short", Order = 15, GroupName = "Parametri")]
        public bool ProtectShort { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 10.0)]
        [Display(Name = "Riduci rischio: attiva dopo +% (0 = disattivato)", Order = 16, GroupName = "Parametri")]
        public double ReduceRiskActivationPct { get; set; }

        [NinjaScriptProperty]
        [Range(-5.0, 5.0)]
        [Display(Name = "Riduci rischio: nuovo stop a -% dall'ingresso", Order = 17, GroupName = "Parametri")]
        public double ReduceRiskStopPct { get; set; }
        #endregion
    }
}
