"""
backtest_v3.py - Backtesting BTC/USDT: EMA Pullback variants + Scalping Momentum

Estrategias (6 variantes total):
1. EMA Pullback Original (L+S y LONG only) - Copia exacta de v2 para confirmar
2. EMA Pullback v2 Improved (L+S y LONG only) - ADX 28, vol filter, trailing stop, session filter
3. Scalping Momentum (L+S y LONG only) - R:R 1:1.5, MACD + RSI + EMA + ADX + vol

Datos: BTC/USDT 15m desde 2022 hasta hoy
Uso:   python3 backtest_v3.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ccxt
import time

# ── Configuración global ──────────────────────────────────────
SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 10_000.0
RISK_PCT = 0.01
FEE_RATE = 0.0004
STOP_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
START_DATE = "2022-01-01"


# ══════════════════════════════════════════════════════════════
# DESCARGA DE DATOS
# ══════════════════════════════════════════════════════════════

def fetch_all_candles(timeframe="15m"):
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    since = exchange.parse8601(f"{START_DATE}T00:00:00Z")
    all_candles = []
    print(f"Descargando datos {timeframe} desde {START_DATE}...")

    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, timeframe, since=since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            print(f"  {len(all_candles)} velas...", end="\r")
            time.sleep(0.3)
            if candles[-1][0] / 1000 > time.time() - 86400:
                break
        except Exception as e:
            print(f"Error: {e}. Reintentando...")
            time.sleep(5)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="last")]
    print(f"\nTotal: {len(df)} velas de {df.index[0].date()} a {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════
# INDICADORES
# ══════════════════════════════════════════════════════════════

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_val = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr_val.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1/period).mean()

def bollinger_bands(series, period=20, std_dev=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bandwidth = (upper - lower) / mid * 100  # % bandwidth
    return mid, upper, lower, bandwidth

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_indicators(df):
    d = df.copy()
    d["ema9"] = ema(d["close"], 9)
    d["ema21"] = ema(d["close"], 21)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["ema9_prev"] = d["ema9"].shift(1)
    d["ema21_prev"] = d["ema21"].shift(1)
    d["rsi"] = rsi(d["close"])
    d["adx"] = adx(d)
    d["atr"] = atr(d)
    d["vol_ma"] = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma"]
    d["ema_above"] = d["ema9"] > d["ema21"]
    d["ema_cross_up"] = (d["ema9_prev"] <= d["ema21_prev"]) & (d["ema9"] > d["ema21"])
    d["ema_cross_down"] = (d["ema9_prev"] >= d["ema21_prev"]) & (d["ema9"] < d["ema21"])

    # Bollinger Bands
    d["bb_mid"], d["bb_upper"], d["bb_lower"], d["bb_bandwidth"] = bollinger_bands(d["close"])
    d["bb_pct"] = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])

    # MACD
    d["macd"], d["macd_signal"], d["macd_hist"] = macd(d["close"])
    d["macd_cross_up"] = (d["macd"].shift(1) < d["macd_signal"].shift(1)) & (d["macd"] > d["macd_signal"])
    d["macd_cross_down"] = (d["macd"].shift(1) > d["macd_signal"].shift(1)) & (d["macd"] < d["macd_signal"])

    # Momentum
    d["momentum_10"] = d["close"].pct_change(10) * 100  # % cambio últimas 10 velas
    d["momentum_40"] = d["close"].pct_change(40) * 100  # % cambio últimas 40 velas (10h)

    return d


# ══════════════════════════════════════════════════════════════
# MOTOR DE BACKTEST
# ══════════════════════════════════════════════════════════════

class BacktestEngine:
    def __init__(self, name, max_duration=100):
        self.name = name
        self.balance = INITIAL_BALANCE
        self.peak_balance = INITIAL_BALANCE
        self.trades = []
        self.equity_curve = []
        self.position = None
        self.max_duration = max_duration  # Max candles in a position

    def open_trade(self, direction, price, sl_dist, tp_dist, ts,
                   trailing_stop_atr=None, atr_value=None):
        if self.position or self.balance < 100:
            return

        if direction == "LONG":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        qty = (self.balance * RISK_PCT) / sl_dist
        if qty * price < 10:
            return

        fee = price * qty * FEE_RATE
        self.balance -= fee

        self.position = {
            "direction": direction, "entry": price,
            "sl": sl, "tp": tp, "qty": qty, "open_ts": ts,
            "trailing_stop_atr": trailing_stop_atr,
            "atr_value": atr_value,
            "trailing_activated": False,
        }

    def check_position(self, row, ts):
        if not self.position:
            return
        pos = self.position
        high, low = row["high"], row["low"]

        # ── Trailing stop logic ──
        # If trailing_stop_atr is set and price moved 1.5x ATR in profit,
        # move SL to breakeven (entry price)
        if (pos.get("trailing_stop_atr") is not None
                and pos.get("atr_value") is not None
                and not pos.get("trailing_activated", False)):
            threshold = pos["trailing_stop_atr"] * pos["atr_value"]
            if pos["direction"] == "LONG":
                # Check if price reached entry + 1.5*ATR at some point (use high)
                if high >= pos["entry"] + threshold:
                    pos["sl"] = pos["entry"]  # Move SL to breakeven
                    pos["trailing_activated"] = True
            else:
                # SHORT: check if price dropped below entry - 1.5*ATR (use low)
                if low <= pos["entry"] - threshold:
                    pos["sl"] = pos["entry"]  # Move SL to breakeven
                    pos["trailing_activated"] = True

        hit_tp = hit_sl = False
        if pos["direction"] == "LONG":
            if low <= pos["sl"]:
                hit_sl, exit_price = True, pos["sl"]
            elif high >= pos["tp"]:
                hit_tp, exit_price = True, pos["tp"]
        else:
            if high >= pos["sl"]:
                hit_sl, exit_price = True, pos["sl"]
            elif low <= pos["tp"]:
                hit_tp, exit_price = True, pos["tp"]

        # Max duration
        if not hit_sl and not hit_tp:
            duration = len(self.equity_curve) - next(
                (i for i, e in enumerate(self.equity_curve) if e["ts"] >= pos["open_ts"]),
                len(self.equity_curve)
            )
            if duration > self.max_duration:
                hit_sl = True
                exit_price = row["close"]

        if hit_sl or hit_tp:
            if pos["direction"] == "LONG":
                pnl = (exit_price - pos["entry"]) * pos["qty"]
            else:
                pnl = (pos["entry"] - exit_price) * pos["qty"]
            fee = exit_price * pos["qty"] * FEE_RATE
            pnl -= fee
            self.balance += pnl
            self.peak_balance = max(self.peak_balance, self.balance)
            self.trades.append({
                "ts": ts, "direction": pos["direction"],
                "entry": pos["entry"], "exit": exit_price,
                "pnl": pnl, "pnl_pct": pnl / (pos["entry"] * pos["qty"]),
                "reason": "TP" if hit_tp else "SL/MAX",
            })
            self.position = None

        self.equity_curve.append({"ts": ts, "balance": self.balance})

    def results(self):
        if not self.trades:
            return {"name": self.name, "trades": 0, "total_pnl": 0, "final_balance": INITIAL_BALANCE}
        t = pd.DataFrame(self.trades)
        wins = (t["pnl"] > 0).sum()
        losses = len(t) - wins
        total_pnl = t["pnl"].sum()
        gp = t[t["pnl"]>0]["pnl"].sum()
        gl = abs(t[t["pnl"]<0]["pnl"].sum())
        eq = pd.DataFrame(self.equity_curve)
        eq["peak"] = eq["balance"].cummax()
        eq["dd"] = (eq["peak"] - eq["balance"]) / eq["peak"]
        pnl_arr = t["pnl_pct"].values
        return {
            "name": self.name,
            "trades": len(t), "wins": int(wins), "losses": int(losses),
            "win_rate": wins/len(t),
            "total_pnl": total_pnl,
            "final_balance": INITIAL_BALANCE + total_pnl,
            "return_pct": total_pnl / INITIAL_BALANCE * 100,
            "max_drawdown": eq["dd"].max(),
            "profit_factor": round(gp/gl, 2) if gl > 0 else float("inf"),
            "sharpe": round((pnl_arr.mean()/pnl_arr.std()*np.sqrt(252)), 4) if pnl_arr.std()>0 else 0,
            "avg_win": t[t["pnl"]>0]["pnl"].mean() if wins>0 else 0,
            "avg_loss": t[t["pnl"]<0]["pnl"].mean() if losses>0 else 0,
            "best_trade": t["pnl"].max(), "worst_trade": t["pnl"].min(),
        }


# ══════════════════════════════════════════════════════════════
# ESTRATEGIA 1: EMA PULLBACK ORIGINAL (copia exacta de v2)
# ══════════════════════════════════════════════════════════════
# - Tendencia fuerte (ADX > 25, EMA ribbon 9>21>50 / 9<21<50)
# - Multi-TF (1h + 4h EMAs alineadas)
# - Pullback a EMA21 (touch_margin = ATR * 0.3)
# - SL debajo de EMA50, TP = 2x SL
# - Cooldown 3h

def run_ema_pullback_original(df_15m, long_only=False):
    name = "EMA Pullback Original" + (" LONG only" if long_only else " (L+S)")
    engine = BacktestEngine(name, max_duration=100)

    df_1h = df_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    df_4h = df_15m.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    ind = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)
    ind_4h = compute_indicators(df_4h)

    adx_1h = ind_1h["adx"]
    ema_above_1h = ind_1h["ema_above"].astype(float)
    ema_above_4h = ind_4h["ema_above"].astype(float)

    last_trade_ts = None
    cooldown_secs = 3 * 3600  # 3 horas
    was_away_from_ema = {}

    print(f"Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 60 or pd.isna(row["adx"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        # ADX fuerte en 15m
        if row["adx"] < 25:
            was_away_from_ema["long"] = False
            was_away_from_ema["short"] = False
            continue

        # EMAs alineadas en 15m
        ema_ribbon_bull = row["ema9"] > row["ema21"] > row["ema50"]
        ema_ribbon_bear = row["ema9"] < row["ema21"] < row["ema50"]

        # Confirmar tendencia en timeframes superiores
        try:
            idx_1h = ema_above_1h.index.asof(ts)
            idx_4h = ema_above_4h.index.asof(ts)
            tf_bull = False
            tf_bear = False
            if idx_1h is not pd.NaT and idx_4h is not pd.NaT:
                a1h = adx_1h.get(idx_1h, 0)
                bull_1h = ema_above_1h.get(idx_1h, 0) > 0.5
                bull_4h = ema_above_4h.get(idx_4h, 0) > 0.5
                if a1h > 20:
                    tf_bull = bull_1h and bull_4h
                    tf_bear = not bull_1h and not bull_4h
        except Exception:
            continue

        # Detectar pullback
        price = row["close"]
        ema21 = row["ema21"]
        ema50 = row["ema50"]
        touch_margin = row["atr"] * 0.3

        if ema_ribbon_bull and price > ema21 + row["atr"] * 0.5:
            was_away_from_ema["long"] = True
        if ema_ribbon_bear and price < ema21 - row["atr"] * 0.5:
            was_away_from_ema["short"] = True

        pullback_long = (abs(price - ema21) < touch_margin) and was_away_from_ema.get("long", False)
        pullback_short = (abs(price - ema21) < touch_margin) and was_away_from_ema.get("short", False)

        rsi_val = row["rsi"]

        # LONG
        if ema_ribbon_bull and tf_bull and pullback_long and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(price - ema50) + row["atr"] * 0.3
                sl_dist = max(sl_dist, row["atr"] * 1.0)
                tp_dist = sl_dist * 2.0
                engine.open_trade("LONG", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts
                was_away_from_ema["long"] = False

        # SHORT
        elif not long_only and ema_ribbon_bear and tf_bear and pullback_short and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(ema50 - price) + row["atr"] * 0.3
                sl_dist = max(sl_dist, row["atr"] * 1.0)
                tp_dist = sl_dist * 2.0
                engine.open_trade("SHORT", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts
                was_away_from_ema["short"] = False

    return engine


# ══════════════════════════════════════════════════════════════
# ESTRATEGIA 2: EMA PULLBACK v2 IMPROVED
# ══════════════════════════════════════════════════════════════
# Mejoras sobre el original:
# - ADX threshold 28 (más selectivo, solo tendencias fuertes)
# - Volume confirmation: vol_ratio >= 1.3
# - Trailing stop: si precio mueve 1.5x ATR en profit, SL → breakeven
# - Session filter: solo 08:00-20:00 UTC
# - Tighter pullback: touch_margin = ATR * 0.2 (antes 0.3)
# - Mismo R:R 1:2

def run_ema_pullback_v2_improved(df_15m, long_only=False):
    name = "EMA Pullback v2 Imp" + (" LONG only" if long_only else " (L+S)")
    engine = BacktestEngine(name, max_duration=100)

    df_1h = df_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    df_4h = df_15m.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    ind = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)
    ind_4h = compute_indicators(df_4h)

    adx_1h = ind_1h["adx"]
    ema_above_1h = ind_1h["ema_above"].astype(float)
    ema_above_4h = ind_4h["ema_above"].astype(float)

    last_trade_ts = None
    cooldown_secs = 3 * 3600  # 3 horas
    was_away_from_ema = {}

    print(f"Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 60 or pd.isna(row["adx"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        # ── SESSION FILTER: only trade 08:00-20:00 UTC ──
        if ts.hour < 8 or ts.hour >= 20:
            continue

        # ADX threshold raised to 28 (stronger trends only)
        if row["adx"] < 28:
            was_away_from_ema["long"] = False
            was_away_from_ema["short"] = False
            continue

        # Volume confirmation: vol_ratio >= 1.3
        if row["vol_ratio"] < 1.3:
            continue

        # EMAs alineadas en 15m
        ema_ribbon_bull = row["ema9"] > row["ema21"] > row["ema50"]
        ema_ribbon_bear = row["ema9"] < row["ema21"] < row["ema50"]

        # Multi-TF confirmation
        try:
            idx_1h = ema_above_1h.index.asof(ts)
            idx_4h = ema_above_4h.index.asof(ts)
            tf_bull = False
            tf_bear = False
            if idx_1h is not pd.NaT and idx_4h is not pd.NaT:
                a1h = adx_1h.get(idx_1h, 0)
                bull_1h = ema_above_1h.get(idx_1h, 0) > 0.5
                bull_4h = ema_above_4h.get(idx_4h, 0) > 0.5
                if a1h > 20:
                    tf_bull = bull_1h and bull_4h
                    tf_bear = not bull_1h and not bull_4h
        except Exception:
            continue

        # Pullback detection with TIGHTER margin (ATR * 0.2 instead of 0.3)
        price = row["close"]
        ema21 = row["ema21"]
        ema50 = row["ema50"]
        touch_margin = row["atr"] * 0.2  # Tighter pullback

        if ema_ribbon_bull and price > ema21 + row["atr"] * 0.5:
            was_away_from_ema["long"] = True
        if ema_ribbon_bear and price < ema21 - row["atr"] * 0.5:
            was_away_from_ema["short"] = True

        pullback_long = (abs(price - ema21) < touch_margin) and was_away_from_ema.get("long", False)
        pullback_short = (abs(price - ema21) < touch_margin) and was_away_from_ema.get("short", False)

        rsi_val = row["rsi"]
        current_atr = row["atr"]

        # LONG
        if ema_ribbon_bull and tf_bull and pullback_long and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(price - ema50) + current_atr * 0.3
                sl_dist = max(sl_dist, current_atr * 1.0)
                tp_dist = sl_dist * 2.0
                engine.open_trade("LONG", price, sl_dist, tp_dist, ts,
                                  trailing_stop_atr=1.5, atr_value=current_atr)
                last_trade_ts = ts
                was_away_from_ema["long"] = False

        # SHORT
        elif not long_only and ema_ribbon_bear and tf_bear and pullback_short and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(ema50 - price) + current_atr * 0.3
                sl_dist = max(sl_dist, current_atr * 1.0)
                tp_dist = sl_dist * 2.0
                engine.open_trade("SHORT", price, sl_dist, tp_dist, ts,
                                  trailing_stop_atr=1.5, atr_value=current_atr)
                last_trade_ts = ts
                was_away_from_ema["short"] = False

    return engine


# ══════════════════════════════════════════════════════════════
# ESTRATEGIA 3: SCALPING MOMENTUM
# ══════════════════════════════════════════════════════════════
# Concepto: R:R 1:1.5 con mayor win rate (~40% necesario)
# - MACD histogram positivo y creciendo (LONG) / negativo y cayendo (SHORT)
# - RSI 40-60 (zona de momentum, no extremos)
# - EMA9 > EMA21 (LONG) / EMA9 < EMA21 (SHORT)
# - ADX > 20 (hay tendencia)
# - Vol ratio > 1.2 (actividad por encima del promedio)
# - Precio > EMA50 para LONG, < EMA50 para SHORT
# - SL: 1.0x ATR, TP: 1.5x ATR (ratio 1:1.5)
# - Cooldown: 2 horas
# - Max duración: 40 velas (10 horas)

def run_scalping_momentum(df_15m, long_only=False):
    name = "Scalping Momentum" + (" LONG only" if long_only else " (L+S)")
    engine = BacktestEngine(name, max_duration=40)  # 40 candles = 10 hours

    ind = compute_indicators(df_15m)

    last_trade_ts = None
    cooldown_secs = 2 * 3600  # 2 horas

    print(f"Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 60 or pd.isna(row["adx"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue
        if pd.isna(row["macd_hist"]) or pd.isna(row["rsi"]):
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        # ── CONDITIONS ──

        # ADX > 20 (there IS a trend)
        if row["adx"] < 20:
            continue

        # Volume ratio > 1.2
        if pd.isna(row["vol_ratio"]) or row["vol_ratio"] < 1.2:
            continue

        # RSI between 40-60 (momentum zone)
        rsi_val = row["rsi"]
        if rsi_val < 40 or rsi_val > 60:
            continue

        # MACD histogram direction
        if i < 2:
            continue
        prev_hist = ind["macd_hist"].iloc[i - 1]
        if pd.isna(prev_hist):
            continue

        macd_hist_val = row["macd_hist"]
        macd_bull = macd_hist_val > 0 and macd_hist_val > prev_hist   # Positive and growing
        macd_bear = macd_hist_val < 0 and macd_hist_val < prev_hist   # Negative and falling

        # EMA basic trend
        ema9_above_21 = row["ema9"] > row["ema21"]
        ema9_below_21 = row["ema9"] < row["ema21"]

        # Price vs EMA50 (trend context)
        price = row["close"]
        price_above_ema50 = price > row["ema50"]
        price_below_ema50 = price < row["ema50"]

        current_atr = row["atr"]
        sl_dist = current_atr * 1.0    # SL: 1x ATR
        tp_dist = current_atr * 1.5    # TP: 1.5x ATR  (ratio 1:1.5)

        # ── LONG SIGNAL ──
        if macd_bull and ema9_above_21 and price_above_ema50:
            if not engine.position:
                engine.open_trade("LONG", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts

        # ── SHORT SIGNAL ──
        elif not long_only and macd_bear and ema9_below_21 and price_below_ema50:
            if not engine.position:
                engine.open_trade("SHORT", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts

    return engine


# ══════════════════════════════════════════════════════════════
# INFORME
# ══════════════════════════════════════════════════════════════

def print_report(results):
    print("\n" + "=" * 70)
    print("  BACKTEST v3 — COMPARATIVA DE ESTRATEGIAS")
    print(f"  Periodo: {START_DATE} -> hoy | Capital: ${INITIAL_BALANCE:,.0f}")
    print("=" * 70)
    print("  Variantes probadas:")
    print("    1. EMA Pullback Original (copia exacta v2)")
    print("    2. EMA Pullback v2 Improved (ADX28, vol, trailing, session)")
    print("    3. Scalping Momentum (R:R 1:1.5, MACD+RSI+EMA+ADX+vol)")
    print("=" * 70)

    for r in results:
        if r.get("trades", 0) == 0:
            print(f"\n{r['name']}: Sin trades")
            continue
        icon = "+" if r["total_pnl"] > 0 else "-"
        print(f"""
{'─'*70}
 {r['name']}
{'─'*70}
  Capital final:    ${r['final_balance']:>10,.2f}  ({r['return_pct']:+.1f}%)  [{icon}]
  PnL total:        ${r['total_pnl']:>+10,.2f}
  Total trades:     {r['trades']:>10}
  Wins / Losses:    {r['wins']:>5} / {r['losses']}   (WR: {r['win_rate']:.1%})
  Profit Factor:    {r['profit_factor']:>10.2f}
  Sharpe Ratio:     {r['sharpe']:>10.4f}
  Max Drawdown:     {r['max_drawdown']:>9.2%}
  Avg Win:          ${r['avg_win']:>+10,.2f}
  Avg Loss:         ${r['avg_loss']:>+10,.2f}
  Mejor trade:      ${r['best_trade']:>+10,.2f}
  Peor trade:       ${r['worst_trade']:>+10,.2f}""")

    print("\n" + "=" * 70)
    best = max(results, key=lambda x: x.get("total_pnl", float("-inf")))
    print(f"  GANADOR: {best['name']} ({best.get('return_pct', 0):+.1f}%)")
    print("=" * 70 + "\n")

    pd.DataFrame(results).to_csv("backtest_v3_results.csv", index=False)
    print("Resultados guardados en backtest_v3_results.csv")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  BACKTEST v3 — 6 ESTRATEGIAS (3 tipos x L+S / LONG only)")
    print("=" * 70)

    df_15m = fetch_all_candles("15m")

    strategies = []

    print("\n── Estrategia 1: EMA Pullback Original (replica v2) ──")
    strategies.append(run_ema_pullback_original(df_15m, long_only=False))
    strategies.append(run_ema_pullback_original(df_15m, long_only=True))

    print("\n── Estrategia 2: EMA Pullback v2 Improved ──")
    strategies.append(run_ema_pullback_v2_improved(df_15m, long_only=False))
    strategies.append(run_ema_pullback_v2_improved(df_15m, long_only=True))

    print("\n── Estrategia 3: Scalping Momentum ──")
    strategies.append(run_scalping_momentum(df_15m, long_only=False))
    strategies.append(run_scalping_momentum(df_15m, long_only=True))

    results = [s.results() for s in strategies]
    print_report(results)
