//+------------------------------------------------------------------+
//|                                       Golden_Cross_Gold.mq5      |
//|                   Golden Cross Trend Rider - XAUUSD Strategy      |
//|                   Optimized for Gold (EMA 50/200)                 |
//+------------------------------------------------------------------+
#property copyright "Pablo Hernandez - Trading Bot"
#property version   "1.00"
#property description "Golden Cross Trend Rider for XAUUSD"
#property description "Enters on pullback to EMA50 during Golden/Death Cross"
#property description "Session filter: London + New York only"

#include <Trade\Trade.mqh>

//--- Input parameters
input group "=== Estrategia Golden Cross ==="
input int      InpEMA_Fast      = 50;        // EMA Fast (Golden Cross)
input int      InpEMA_Slow      = 200;       // EMA Slow (Golden Cross)
input int      InpEMA_Signal    = 9;         // EMA Signal (for ribbon)
input int      InpEMA_Mid       = 21;        // EMA Mid (trend confirmation)
input int      InpADX_Period    = 14;        // ADX Period
input int      InpADX_Min       = 20;        // ADX Minimum
input double   InpTouchMargin   = 0.5;       // Pullback touch margin (x ATR)
input double   InpAwayMargin    = 0.8;       // Away from EMA margin (x ATR)

input group "=== Risk Management ==="
input double   InpRiskPct       = 1.0;       // Risk per trade (% of balance)
input double   InpSL_ATR        = 1.5;       // Stop Loss (x ATR)
input double   InpTP_Ratio      = 2.5;       // Take Profit ratio (x SL)
input int      InpMaxBars       = 200;       // Max bars in trade (auto-close)

input group "=== Filtros ==="
input bool     InpLongOnly      = false;     // LONG only mode
input int      InpCooldownBars  = 32;        // Cooldown between trades (bars, 32=8h on M15)
input double   InpRSI_Min_Long  = 30.0;      // RSI min for LONG
input double   InpRSI_Max_Long  = 65.0;      // RSI max for LONG
input double   InpRSI_Min_Short = 35.0;      // RSI min for SHORT
input double   InpRSI_Max_Short = 70.0;      // RSI max for SHORT
input bool     InpUseSessionFilter = true;   // Use session filter (London+NY)
input int      InpSessionStart  = 7;         // Session start hour (UTC)
input int      InpSessionEnd    = 20;        // Session end hour (UTC)

input group "=== General ==="
input int      InpMagicNumber   = 20260407;  // Magic Number
input string   InpComment       = "GC_Gold"; // Trade comment

//--- Global variables
CTrade trade;
int handle_ema50, handle_ema200, handle_ema9, handle_ema21;
int handle_adx, handle_rsi, handle_atr;

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
   handle_ema9   = iMA(_Symbol, PERIOD_M15, InpEMA_Signal, 0, MODE_EMA, PRICE_CLOSE);
   handle_ema21  = iMA(_Symbol, PERIOD_M15, InpEMA_Mid, 0, MODE_EMA, PRICE_CLOSE);
   handle_ema50  = iMA(_Symbol, PERIOD_M15, InpEMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handle_ema200 = iMA(_Symbol, PERIOD_M15, InpEMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   handle_adx    = iADX(_Symbol, PERIOD_M15, InpADX_Period);
   handle_rsi    = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   handle_atr    = iATR(_Symbol, PERIOD_M15, 14);

   if(handle_ema9 == INVALID_HANDLE || handle_ema21 == INVALID_HANDLE ||
      handle_ema50 == INVALID_HANDLE || handle_ema200 == INVALID_HANDLE ||
      handle_adx == INVALID_HANDLE || handle_rsi == INVALID_HANDLE ||
      handle_atr == INVALID_HANDLE)
   {
      Print("Error creating indicator handles!");
      return(INIT_FAILED);
   }

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(50);

   Print("Golden Cross Gold Strategy initialized");
   Print("Mode: ", InpLongOnly ? "LONG ONLY" : "LONG + SHORT");
   Print("R:R Ratio: 1:", DoubleToString(InpTP_Ratio, 1));
   Print("Session filter: ", InpUseSessionFilter ? "ON" : "OFF");

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(handle_ema9);
   IndicatorRelease(handle_ema21);
   IndicatorRelease(handle_ema50);
   IndicatorRelease(handle_ema200);
   IndicatorRelease(handle_adx);
   IndicatorRelease(handle_rsi);
   IndicatorRelease(handle_atr);
}

//+------------------------------------------------------------------+
double GetIndicator(int handle, int buffer, int shift)
{
   double val[1];
   if(CopyBuffer(handle, buffer, shift, 1, val) != 1)
      return(0);
   return(val[0]);
}

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

   // Check max duration
   if(HasPosition())
   {
      CheckMaxDuration();
      return;
   }

   // Cooldown
   if(barCount - lastTradeBars < InpCooldownBars)
      return;

   // Session filter (UTC hours)
   MqlDateTime dt;
   TimeToStruct(currentBarTime, dt);
   if(InpUseSessionFilter)
   {
      if(dt.hour < InpSessionStart || dt.hour >= InpSessionEnd)
         return;
   }

   // ══════════════════════════════════════════════════
   // GET INDICATOR VALUES
   // ══════════════════════════════════════════════════

   double ema9_val   = GetIndicator(handle_ema9, 0, 1);
   double ema21_val  = GetIndicator(handle_ema21, 0, 1);
   double ema50_val  = GetIndicator(handle_ema50, 0, 1);
   double ema200_val = GetIndicator(handle_ema200, 0, 1);
   double adx_val    = GetIndicator(handle_adx, 0, 1);
   double rsi_val    = GetIndicator(handle_rsi, 0, 1);
   double atr_val    = GetIndicator(handle_atr, 0, 1);
   double price      = iClose(_Symbol, PERIOD_M15, 1);

   if(atr_val <= 0 || adx_val <= 0 || ema200_val <= 0) return;

   // ══════════════════════════════════════════════════
   // GOLDEN CROSS / DEATH CROSS DETECTION
   // ══════════════════════════════════════════════════

   bool goldenCross = ema50_val > ema200_val;  // Bullish regime
   bool deathCross  = ema50_val < ema200_val;  // Bearish regime

   // ADX minimum
   if(adx_val < InpADX_Min)
   {
      wasAwayLong = false;
      wasAwayShort = false;
      return;
   }

   // ══════════════════════════════════════════════════
   // PULLBACK DETECTION TO EMA50
   // ══════════════════════════════════════════════════

   double touchMargin = atr_val * InpTouchMargin;
   double awayMargin  = atr_val * InpAwayMargin;

   // Track if price moved away from EMA50
   if(goldenCross && price > ema50_val + awayMargin)
      wasAwayLong = true;
   if(deathCross && price < ema50_val - awayMargin)
      wasAwayShort = true;

   bool pullbackLong  = goldenCross && (MathAbs(price - ema50_val) < touchMargin) && wasAwayLong;
   bool pullbackShort = deathCross && (MathAbs(price - ema50_val) < touchMargin) && wasAwayShort;

   // Additional confirmation: EMA9 > EMA21 for trend direction
   bool shortTermBull = ema9_val > ema21_val;
   bool shortTermBear = ema9_val < ema21_val;

   // ══════════════════════════════════════════════════
   // ENTRY SIGNALS
   // ══════════════════════════════════════════════════

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   // ── LONG: Golden Cross + pullback to EMA50 + short-term bull ──
   if(pullbackLong && shortTermBull && rsi_val > InpRSI_Min_Long && rsi_val < InpRSI_Max_Long)
   {
      double slDist = atr_val * InpSL_ATR;
      double tpDist = slDist * InpTP_Ratio;

      double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = NormalizeDouble(askPrice - slDist, digits);
      double tp = NormalizeDouble(askPrice + tpDist, digits);

      double lots = CalculateLotSize(slDist);
      if(lots > 0)
      {
         if(trade.Buy(lots, _Symbol, askPrice, sl, tp, InpComment))
         {
            lastTradeBars = barCount;
            positionOpenBar = barCount;
            wasAwayLong = false;
            Print("LONG at ", askPrice, " SL:", sl, " TP:", tp,
                  " EMA50:", ema50_val, " EMA200:", ema200_val,
                  " ADX:", adx_val, " RSI:", rsi_val);
         }
      }
   }

   // ── SHORT: Death Cross + pullback to EMA50 + short-term bear ──
   if(!InpLongOnly && pullbackShort && shortTermBear &&
      rsi_val > InpRSI_Min_Short && rsi_val < InpRSI_Max_Short)
   {
      double slDist = atr_val * InpSL_ATR;
      double tpDist = slDist * InpTP_Ratio;

      double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = NormalizeDouble(bidPrice + slDist, digits);
      double tp = NormalizeDouble(bidPrice - tpDist, digits);

      double lots = CalculateLotSize(slDist);
      if(lots > 0)
      {
         if(trade.Sell(lots, _Symbol, bidPrice, sl, tp, InpComment))
         {
            lastTradeBars = barCount;
            positionOpenBar = barCount;
            wasAwayShort = false;
            Print("SHORT at ", bidPrice, " SL:", sl, " TP:", tp,
                  " EMA50:", ema50_val, " EMA200:", ema200_val,
                  " ADX:", adx_val, " RSI:", rsi_val);
         }
      }
   }
}
//+------------------------------------------------------------------+
