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

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                         = "MnqNoiseAreaMomentum";
                Description                  = "Momentum intraday su rottura della noise area (MNQ, 1 minuto).";
                Calculate                    = Calculate.OnBarClose;
                EntriesPerDirection          = 1;
                EntryHandling                = EntryHandling.AllEntries;
                IsExitOnSessionClose         = true;
                ExitOnSessionCloseSeconds    = 30;
                BarsRequiredToTrade          = 1;
                IsInstantiatedOnEachOptimizationIteration = true;

                Lookback    = 14;
                BandMult    = 1.0;
                StepMinutes = 30;
                UseVwap     = true;
                AllowShort  = true;
                Quantity    = 1;

                AddPlot(Brushes.SeaGreen, "BandaAlta");
                AddPlot(Brushes.IndianRed, "BandaBassa");
                AddPlot(Brushes.Goldenrod, "VWAP");
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

            if (m < FirstDecision || (m - FirstDecision) % StepMinutes != 0)
                return;                               // si decide solo alle :00 e :30 (10:00 - 15:30)

            double up = UseVwap ? Math.Max(upper, vwap) : upper;
            double dn = UseVwap ? Math.Min(lower, vwap) : lower;
            int want  = Close[0] > up ? 1 : (Close[0] < dn && AllowShort ? -1 : 0);

            if (want == 1)
            {
                if (Position.MarketPosition != MarketPosition.Long)
                    EnterLong(Quantity, "Long");      // se short, NinjaTrader inverte la posizione
            }
            else if (want == -1)
            {
                if (Position.MarketPosition != MarketPosition.Short)
                    EnterShort(Quantity, "Short");
            }
            else
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong();
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort();
            }
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
        #endregion
    }
}
