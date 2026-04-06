//+------------------------------------------------------------------+
//|                                         EMA_Pullback_BTC.mq5     |
//|                        EMA Pullback Trend Rider - BTC Strategy    |
//|                        Backtested: +20.8% (2022-2026), DD 32.6%  |
//+------------------------------------------------------------------+
#property copyright "Pablo Hernandez - Trading Bot"
#property version   "1.00"
#property description "EMA Pullback Trend Rider Strategy"
#property description "Enters on pullback to EMA21 in strong trend"
#property description "Multi-timeframe confirmation (15m + 1H + 4H)"

#include <Trade\Trade.mqh>

//--- Input parameters
input group "=== Estrategia Principal ==="
input int      InpADX_Period    = 14;        // ADX Period
input int      InpADX_Min       = 25;        // ADX Minimum (trend strength)
input int      InpEMA_Fast      = 9;         // EMA Fast period
input int      InpEMA_Mid       = 21;        // EMA Mid period (pullback target)
input int      InpEMA_Slow      = 50;        // EMA Slow period
input double   InpTouchMargin   = 0.3;       // Pullback touch margin (x ATR)
input double   InpAwayMargin    = 0.5;       // Away from EMA margin (x ATR)

input group "=== Risk Management ==="
input double   InpRiskPct       = 1.0;       // Risk per trade (% of balance)
input double   InpTP_Ratio      = 2.0;       // Take Profit ratio (x SL distance)
input double   InpSL_ATR_Min    = 1.0;       // Minimum SL (x ATR)
input double   InpSL_EMA_Buffer = 0.3;       // SL buffer beyond EMA50 (x ATR)
input int      InpMaxBars       = 100;       // Max bars in trade (auto-close)

input group "=== Filtros ==="
input bool     InpLongOnly      = true;      // LONG only mode
input int      InpCooldownBars  = 12;        // Cooldown between trades (bars on 15m = 3h)
input double   InpRSI_Min       = 35.0;      // RSI minimum for entry
input double   InpRSI_Max       = 65.0;      // RSI maximum for entry
input int      InpADX_1H_Min    = 20;        // ADX minimum on 1H for multi-TF

input group "=== General ==="
input int      InpMagicNumber   = 20260406;  // Magic Number
input string   InpComment       = "EMA_PB";  // Trade comment

//--- Global variables
CTrade trade;
int handle_ema9, handle_ema21, handle_ema50;
int handle_adx, handle_rsi, handle_atr;
int handle_ema9_1h, handle_ema21_1h, handle_adx_1h;
int handle_ema9_4h, handle_ema21_4h;

bool wasAwayLong = false;
bool wasAwayShort = false;
int lastTradeBars = -999;
int barCount = 0;
int positionOpenBar = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   // 15-minute indicators
   handle_ema9  = iMA(_Symbol, PERIOD_M15, InpEMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handle_ema21 = iMA(_Symbol, PERIOD_M15, InpEMA_Mid, 0, MODE_EMA, PRICE_CLOSE);
   handle_ema50 = iMA(_Symbol, PERIOD_M15, InpEMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   handle_adx   = iADX(_Symbol, PERIOD_M15, InpADX_Period);
   handle_rsi   = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   handle_atr   = iATR(_Symbol, PERIOD_M15, 14);

   // 1H indicators (multi-TF)
   handle_ema9_1h  = iMA(_Symbol, PERIOD_H1, InpEMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handle_ema21_1h = iMA(_Symbol, PERIOD_H1, InpEMA_Mid, 0, MODE_EMA, PRICE_CLOSE);
   handle_adx_1h   = iADX(_Symbol, PERIOD_H1, InpADX_Period);

   // 4H indicators (multi-TF)
   handle_ema9_4h  = iMA(_Symbol, PERIOD_H4, InpEMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handle_ema21_4h = iMA(_Symbol, PERIOD_H4, InpEMA_Mid, 0, MODE_EMA, PRICE_CLOSE);

   // Verify all handles
   if(handle_ema9 == INVALID_HANDLE || handle_ema21 == INVALID_HANDLE ||
      handle_ema50 == INVALID_HANDLE || handle_adx == INVALID_HANDLE ||
      handle_rsi == INVALID_HANDLE || handle_atr == INVALID_HANDLE ||
      handle_ema9_1h == INVALID_HANDLE || handle_ema21_1h == INVALID_HANDLE ||
      handle_adx_1h == INVALID_HANDLE ||
      handle_ema9_4h == INVALID_HANDLE || handle_ema21_4h == INVALID_HANDLE)
   {
      Print("Error creating indicator handles!");
      return(INIT_FAILED);
   }

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(50);  // Slippage tolerance

   Print("EMA Pullback BTC Strategy initialized");
   Print("Mode: ", InpLongOnly ? "LONG ONLY" : "LONG + SHORT");
   Print("R:R Ratio: 1:", DoubleToString(InpTP_Ratio, 1));

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(handle_ema9);
   IndicatorRelease(handle_ema21);
   IndicatorRelease(handle_ema50);
   IndicatorRelease(handle_adx);
   IndicatorRelease(handle_rsi);
   IndicatorRelease(handle_atr);
   IndicatorRelease(handle_ema9_1h);
   IndicatorRelease(handle_ema21_1h);
   IndicatorRelease(handle_adx_1h);
   IndicatorRelease(handle_ema9_4h);
   IndicatorRelease(handle_ema21_4h);
}

//+------------------------------------------------------------------+
//| Get indicator value                                               |
//+------------------------------------------------------------------+
double GetIndicator(int handle, int buffer, int shift)
{
   double val[1];
   if(CopyBuffer(handle, buffer, shift, 1, val) != 1)
      return(0);
   return(val[0]);
}

//+------------------------------------------------------------------+
//| Check if we have an open position                                 |
//+------------------------------------------------------------------+
bool HasPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber &&
            PositionGetString(POSITION_SYMBOL) == _Symbol)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Close position if max duration exceeded                           |
//+------------------------------------------------------------------+
void CheckMaxDuration()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber &&
            PositionGetString(POSITION_SYMBOL) == _Symbol)
         {
            int barsInTrade = barCount - positionOpenBar;
            if(barsInTrade >= InpMaxBars)
            {
               trade.PositionClose(PositionGetTicket(i));
               Print("Position closed: max duration (", barsInTrade, " bars)");
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                  |
//+------------------------------------------------------------------+
double CalculateLotSize(double slDistance)
{
   if(slDistance <= 0) return 0;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * InpRiskPct / 100.0;

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double lotStep   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(tickValue <= 0 || tickSize <= 0) return minLot;

   double lots = riskAmount / (slDistance / tickSize * tickValue);
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);

   return(NormalizeDouble(lots, 2));
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only process on new 15m bar
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, PERIOD_M15, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;
   barCount++;

   // Check max duration on existing positions
   if(HasPosition())
   {
      CheckMaxDuration();
      return;  // Don't open new trades while in position
   }

   // Cooldown check
   if(barCount - lastTradeBars < InpCooldownBars)
      return;

   // ══════════════════════════════════════════════════
   // GET ALL INDICATOR VALUES
   // ══════════════════════════════════════════════════

   // 15m indicators (shift 1 = completed bar)
   double ema9_val  = GetIndicator(handle_ema9, 0, 1);
   double ema21_val = GetIndicator(handle_ema21, 0, 1);
   double ema50_val = GetIndicator(handle_ema50, 0, 1);
   double adx_val   = GetIndicator(handle_adx, 0, 1);   // Main ADX line
   double rsi_val   = GetIndicator(handle_rsi, 0, 1);
   double atr_val   = GetIndicator(handle_atr, 0, 1);
   double price     = iClose(_Symbol, PERIOD_M15, 1);

   // 1H multi-TF
   double ema9_1h  = GetIndicator(handle_ema9_1h, 0, 1);
   double ema21_1h = GetIndicator(handle_ema21_1h, 0, 1);
   double adx_1h   = GetIndicator(handle_adx_1h, 0, 1);

   // 4H multi-TF
   double ema9_4h  = GetIndicator(handle_ema9_4h, 0, 1);
   double ema21_4h = GetIndicator(handle_ema21_4h, 0, 1);

   // Validate data
   if(atr_val <= 0 || adx_val <= 0) return;

   // ══════════════════════════════════════════════════
   // STRATEGY CONDITIONS
   // ══════════════════════════════════════════════════

   // 1. ADX minimum (trend exists)
   if(adx_val < InpADX_Min)
   {
      wasAwayLong = false;
      wasAwayShort = false;
      return;
   }

   // 2. EMA Ribbon alignment
   bool ribbonBull = (ema9_val > ema21_val) && (ema21_val > ema50_val);
   bool ribbonBear = (ema9_val < ema21_val) && (ema21_val < ema50_val);

   // 3. Multi-TF confirmation
   bool tf_bull = false;
   bool tf_bear = false;
   if(adx_1h > InpADX_1H_Min)
   {
      bool bull_1h = ema9_1h > ema21_1h;
      bool bull_4h = ema9_4h > ema21_4h;
      tf_bull = bull_1h && bull_4h;
      tf_bear = !bull_1h && !bull_4h;
   }

   // 4. Track pullback (price must first move away from EMA21, then return)
   double touchMargin = atr_val * InpTouchMargin;
   double awayMargin  = atr_val * InpAwayMargin;

   if(ribbonBull && price > ema21_val + awayMargin)
      wasAwayLong = true;
   if(ribbonBear && price < ema21_val - awayMargin)
      wasAwayShort = true;

   bool pullbackLong  = (MathAbs(price - ema21_val) < touchMargin) && wasAwayLong;
   bool pullbackShort = (MathAbs(price - ema21_val) < touchMargin) && wasAwayShort;

   // 5. RSI filter
   bool rsiOK = (rsi_val > InpRSI_Min && rsi_val < InpRSI_Max);

   // ══════════════════════════════════════════════════
   // ENTRY SIGNALS
   // ══════════════════════════════════════════════════

   // ── LONG SIGNAL ──
   if(ribbonBull && tf_bull && pullbackLong && rsiOK)
   {
      double slDist = MathAbs(price - ema50_val) + atr_val * InpSL_EMA_Buffer;
      slDist = MathMax(slDist, atr_val * InpSL_ATR_Min);
      double tpDist = slDist * InpTP_Ratio;

      double sl = price - slDist;
      double tp = price + tpDist;

      // Normalize prices
      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      sl = NormalizeDouble(sl, digits);
      tp = NormalizeDouble(tp, digits);
      double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      double lots = CalculateLotSize(slDist);
      if(lots > 0)
      {
         if(trade.Buy(lots, _Symbol, askPrice, sl, tp, InpComment))
         {
            lastTradeBars = barCount;
            positionOpenBar = barCount;
            wasAwayLong = false;
            Print("LONG opened at ", askPrice, " SL:", sl, " TP:", tp, " Lots:", lots);
         }
      }
   }

   // ── SHORT SIGNAL ──
   if(!InpLongOnly && ribbonBear && tf_bear && pullbackShort && rsiOK)
   {
      double slDist = MathAbs(ema50_val - price) + atr_val * InpSL_EMA_Buffer;
      slDist = MathMax(slDist, atr_val * InpSL_ATR_Min);
      double tpDist = slDist * InpTP_Ratio;

      double sl = price + slDist;
      double tp = price - tpDist;

      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      sl = NormalizeDouble(sl, digits);
      tp = NormalizeDouble(tp, digits);
      double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);

      double lots = CalculateLotSize(slDist);
      if(lots > 0)
      {
         if(trade.Sell(lots, _Symbol, bidPrice, sl, tp, InpComment))
         {
            lastTradeBars = barCount;
            positionOpenBar = barCount;
            wasAwayShort = false;
            Print("SHORT opened at ", bidPrice, " SL:", sl, " TP:", tp, " Lots:", lots);
         }
      }
   }
}
//+------------------------------------------------------------------+
