"""
backtest_v2.py - Backtesting mejorado de estrategias BTC/USDT

Estrategias (8 variantes total):
1. Sniper v2 (mejorado)         - Score>=5, cooldown 4h, requiere catalizador
2. BB Squeeze + MACD Momentum   - Baja volatilidad → breakout con confirmación
3. RSI Mean Reversion           - Comprar en sobreventa con tendencia alcista
4. EMA Pullback Trend Rider     - Pullback a EMA21 en tendencia fuerte (patrón MT5)

Cada una se prueba en LONG+SHORT y LONG only.

Datos: BTC/USDT 15m desde 2022 hasta hoy
Uso:   python3 backtest_v2.py
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
    signal_line = ema_line = ema(macd_line, signal)
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
    def __init__(self, name):
        self.name = name
        self.balance = INITIAL_BALANCE
        self.peak_balance = INITIAL_BALANCE
        self.trades = []
        self.equity_curve = []
        self.position = None

    def open_trade(self, direction, price, sl_dist, tp_dist, ts):
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
        }

    def check_position(self, row, ts):
        if not self.position:
            return
        pos = self.position
        high, low = row["high"], row["low"]

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

        # Max duration: 100 candles (25h en 15m)
        if not hit_sl and not hit_tp:
            duration = len(self.equity_curve) - next(
                (i for i, e in enumerate(self.equity_curve) if e["ts"] >= pos["open_ts"]),
                len(self.equity_curve)
            )
            if duration > 100:
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
# ESTRATEGIA 1: SNIPER v2 (mejorado)
# ══════════════════════════════════════════════════════════════
# Mejoras vs v1:
# - Score mínimo 5 (antes 4)
# - Cooldown 4 horas (antes 15 min)
# - Requiere cruce EMA O multi-TF alineado (no puntos sueltos)
# - ATR mínimo para evitar mercados sin movimiento

def run_sniper_v2(df_15m, long_only=False):
    name = "Sniper v2 LONG only" if long_only else "Sniper v2 (LONG+SHORT)"
    engine = BacktestEngine(name)

    df_1h = df_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    df_4h = df_15m.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

    ind_15 = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)
    ind_4h = compute_indicators(df_4h)

    # Pre-compute 1h/4h lookups para velocidad
    ema_above_1h = ind_1h["ema_above"].astype(float)
    adx_1h = ind_1h["adx"]
    ema_above_4h = ind_4h["ema_above"].astype(float)
    adx_4h = ind_4h["adx"]

    last_trade_ts = None
    last_signal_dir = None
    cooldown_secs = 4 * 3600  # 4 horas

    print(f"Ejecutando {name}...")

    for ts, row in ind_15.iterrows():
        if pd.isna(row["adx"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        # Cooldown 4h
        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        # ADX mínimo
        if row["adx"] < 20:
            last_signal_dir = None
            continue

        # ATR mínimo (evitar mercados muertos)
        atr_pct = row["atr"] / row["close"] * 100
        if atr_pct < 0.3:
            continue

        # Scoring
        buy_s = sell_s = 0
        has_ema_cross = False

        # 1. ADX fuerte (1pt)
        if row["adx"] >= 30:
            buy_s += 1; sell_s += 1

        # 2. EMA cross (2pt) - CRUCIAL
        if row["ema_cross_up"]:
            buy_s += 2; has_ema_cross = True
        if row["ema_cross_down"]:
            sell_s += 2; has_ema_cross = True

        # 3. EMA alineadas (1pt)
        if row["ema_above"]:
            buy_s += 1
        else:
            sell_s += 1

        # 4. RSI favorable (1pt)
        rsi_val = row["rsi"]
        if 35 <= rsi_val <= 50:
            buy_s += 1
        if 50 <= rsi_val <= 65:
            sell_s += 1

        if rsi_val > 70: buy_s = 0
        if rsi_val < 30: sell_s = 0

        # 5. Volumen spike (1pt)
        if row["vol_ratio"] >= 2.0:
            buy_s += 1; sell_s += 1

        # 6. Multi-TF (2pt)
        bullish = bearish = 0
        try:
            idx_1h = ema_above_1h.index.asof(ts)
            idx_4h = ema_above_4h.index.asof(ts)
            if idx_1h is not pd.NaT:
                a1h = adx_1h.get(idx_1h, 0)
                if a1h > 20:
                    if ema_above_1h.get(idx_1h, 0) > 0.5: bullish += 1
                    else: bearish += 1
            if idx_4h is not pd.NaT:
                a4h = adx_4h.get(idx_4h, 0)
                if a4h > 20:
                    if ema_above_4h.get(idx_4h, 0) > 0.5: bullish += 1
                    else: bearish += 1
        except Exception:
            pass

        if bullish >= 2: buy_s += 2
        if bearish >= 2: sell_s += 2

        # 7. Momentum confirmación (1pt extra)
        if not pd.isna(row.get("momentum_10", np.nan)):
            if row["momentum_10"] > 1.0: buy_s += 1
            elif row["momentum_10"] < -1.0: sell_s += 1

        # Decisión: score >= 5 Y (tiene cruce EMA O multi-TF alineado)
        min_score = 5
        need_catalyst = has_ema_cross or bullish >= 2 or bearish >= 2

        sl_dist = row["atr"] * STOP_ATR_MULT
        tp_dist = row["atr"] * TP_ATR_MULT

        if buy_s >= min_score and buy_s > sell_s and need_catalyst:
            if last_signal_dir != "LONG" and not engine.position:
                engine.open_trade("LONG", row["close"], sl_dist, tp_dist, ts)
                last_trade_ts = ts; last_signal_dir = "LONG"
        elif not long_only and sell_s >= min_score and sell_s > buy_s and need_catalyst:
            if last_signal_dir != "SHORT" and not engine.position:
                engine.open_trade("SHORT", row["close"], sl_dist, tp_dist, ts)
                last_trade_ts = ts; last_signal_dir = "SHORT"
        else:
            if buy_s < 3 and sell_s < 3:
                last_signal_dir = None

    return engine


# ══════════════════════════════════════════════════════════════
# ESTRATEGIA 2: BOLLINGER SQUEEZE + MACD MOMENTUM
# ══════════════════════════════════════════════════════════════
# Lógica:
# - Detecta cuando Bollinger Bands se comprimen (squeeze = baja volatilidad)
# - Espera a que el precio rompa la banda + MACD confirme dirección
# - Usa EMA 50/200 para contexto de tendencia mayor
# - Solo opera en dirección de la tendencia mayor
# - SL más ajustado (1x ATR), TP más grande (3.5x ATR) = ratio 1:3.5

def run_bollinger_momentum(df_15m, long_only=False):
    name = "BB Squeeze+MACD" + (" LONG only" if long_only else "")
    engine = BacktestEngine(name)

    df_1h = df_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    ind = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)

    # Pre-compute lookups
    trend_1h = ind_1h["ema_above"].astype(float)
    ema50_1h = ind_1h["ema50"]

    last_trade_ts = None
    cooldown_secs = 3 * 3600  # 3 horas entre trades
    squeeze_lookback = 40     # Mirar squeeze en últimas 40 velas (10h)

    print(f"Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 50 or pd.isna(row["bb_bandwidth"]) or pd.isna(row["macd_hist"]):
            continue

        engine.check_position(row, ts)

        # Cooldown
        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        # ATR mínimo
        if pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        # ── CONDICIONES ──

        # 1. Detectar squeeze: bandwidth actual en el 20% inferior de las últimas 40 velas
        bw_window = ind["bb_bandwidth"].iloc[max(0,i-squeeze_lookback):i+1]
        if len(bw_window) < 20:
            continue
        bw_percentile = (row["bb_bandwidth"] - bw_window.min()) / (bw_window.max() - bw_window.min()) if bw_window.max() != bw_window.min() else 0.5

        # 2. Precio rompe banda
        price_breaks_upper = row["close"] > row["bb_upper"]
        price_breaks_lower = row["close"] < row["bb_lower"]

        # 3. MACD confirma (relajado: histogram positivo/negativo + creciendo)
        macd_bullish = False
        macd_bearish = False
        if row["macd_hist"] > 0:
            if row.get("macd_cross_up", False):
                macd_bullish = True
            elif i > 1:
                prev_hist = ind["macd_hist"].iloc[i-1]
                if not pd.isna(prev_hist) and row["macd_hist"] > prev_hist:
                    macd_bullish = True  # Histogram positivo y creciendo
        if row["macd_hist"] < 0:
            if row.get("macd_cross_down", False):
                macd_bearish = True
            elif i > 1:
                prev_hist = ind["macd_hist"].iloc[i-1]
                if not pd.isna(prev_hist) and row["macd_hist"] < prev_hist:
                    macd_bearish = True  # Histogram negativo y cayendo

        # 4. Tendencia mayor (EMA 50 en 1h)
        try:
            idx_1h = trend_1h.index.asof(ts)
            trend_up = trend_1h.get(idx_1h, 0.5) > 0.5 if idx_1h is not pd.NaT else None
        except Exception:
            trend_up = None

        # 5. Volumen confirma
        vol_ok = row["vol_ratio"] >= 1.5

        # 6. RSI no extremo
        rsi_val = row["rsi"]

        # Squeeze detectado = bandwidth en percentil bajo Y precio rompe banda
        squeeze_long = bw_percentile < 0.35 and price_breaks_upper
        squeeze_short = bw_percentile < 0.35 and price_breaks_lower
        # También permitir breakout fuerte sin squeeze estricto
        strong_breakout_long = price_breaks_upper and row["vol_ratio"] >= 2.0
        strong_breakout_short = price_breaks_lower and row["vol_ratio"] >= 2.0

        # ── SEÑAL LONG ──
        if (squeeze_long or strong_breakout_long) and macd_bullish and trend_up and rsi_val < 75:
            if not engine.position:
                sl_dist = row["atr"] * 1.2
                tp_dist = row["atr"] * 3.0   # Ratio 1:2.5
                engine.open_trade("LONG", row["close"], sl_dist, tp_dist, ts)
                last_trade_ts = ts

        # ── SEÑAL SHORT ──
        elif not long_only and (squeeze_short or strong_breakout_short) and macd_bearish and trend_up == False and rsi_val > 25:
            if not engine.position:
                sl_dist = row["atr"] * 1.2
                tp_dist = row["atr"] * 3.0
                engine.open_trade("SHORT", row["close"], sl_dist, tp_dist, ts)
                last_trade_ts = ts

    return engine


# ══════════════════════════════════════════════════════════════
# ESTRATEGIA 3: RSI MEAN REVERSION + TENDENCIA
# ══════════════════════════════════════════════════════════════
# Lógica basada en estrategias probadas de crypto:
# - Compra cuando RSI está en sobreventa (< 30) en tendencia alcista mayor
# - Vende cuando RSI está en sobrecompra (> 70) en tendencia bajista mayor
# - Usa EMA 50/200 para determinar la tendencia mayor
# - Confirmación con volumen y MACD
# - SL debajo del mínimo reciente, TP con ratio 1:2

def run_rsi_mean_reversion(df_15m, long_only=False):
    name = "RSI MeanRev" + (" LONG only" if long_only else "")
    engine = BacktestEngine(name)

    df_1h = df_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    ind = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)

    ema50_1h = ind_1h["ema50"]
    ema200_1h = ind_1h["ema200"]

    last_trade_ts = None
    cooldown_secs = 2 * 3600  # 2 horas

    print(f"Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 200 or pd.isna(row["rsi"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        rsi_val = row["rsi"]

        # Determinar tendencia mayor (1h: EMA50 vs EMA200)
        try:
            idx_1h = ema50_1h.index.asof(ts)
            if idx_1h is pd.NaT:
                continue
            e50 = ema50_1h.get(idx_1h, None)
            e200 = ema200_1h.get(idx_1h, None)
            if e50 is None or e200 is None or pd.isna(e50) or pd.isna(e200):
                continue
            uptrend_major = e50 > e200
        except Exception:
            continue

        # Mínimo/máximo recientes para SL dinámico
        recent_low = ind["low"].iloc[max(0, i-10):i+1].min()
        recent_high = ind["high"].iloc[max(0, i-10):i+1].max()

        # ── SEÑAL LONG: RSI sobreventa en tendencia alcista ──
        if rsi_val < 30 and uptrend_major:
            # Confirmaciones adicionales
            macd_turning = row["macd_hist"] > ind["macd_hist"].iloc[i-1] if i > 0 and not pd.isna(ind["macd_hist"].iloc[i-1]) else False
            price_near_ema = row["close"] <= row["ema50"] * 1.02  # Precio cerca o debajo de EMA50

            if macd_turning or price_near_ema:
                if not engine.position:
                    sl_dist = max(row["close"] - recent_low, row["atr"] * 0.8)
                    tp_dist = sl_dist * 2.5  # Ratio 1:2.5
                    engine.open_trade("LONG", row["close"], sl_dist, tp_dist, ts)
                    last_trade_ts = ts

        # ── SEÑAL SHORT: RSI sobrecompra en tendencia bajista ──
        elif not long_only and rsi_val > 70 and not uptrend_major:
            macd_turning = row["macd_hist"] < ind["macd_hist"].iloc[i-1] if i > 0 and not pd.isna(ind["macd_hist"].iloc[i-1]) else False
            price_near_ema = row["close"] >= row["ema50"] * 0.98

            if macd_turning or price_near_ema:
                if not engine.position:
                    sl_dist = max(recent_high - row["close"], row["atr"] * 0.8)
                    tp_dist = sl_dist * 2.5
                    engine.open_trade("SHORT", row["close"], sl_dist, tp_dist, ts)
                    last_trade_ts = ts

    return engine


# ══════════════════════════════════════════════════════════════
# ESTRATEGIA 4: EMA PULLBACK TREND RIDER (patrón MT5 EA)
# ══════════════════════════════════════════════════════════════
# Basada en los EAs más rentables de MetaTrader 5:
# - Espera tendencia fuerte (ADX > 25, EMAs alineadas)
# - Entra cuando el precio hace pullback a la EMA21
# - Sale con trailing stop o TP fijo
# - Muy selectivo: solo opera en tendencias claras
# - Ratio R:R 1:2 con SL en EMA50

def run_ema_pullback(df_15m, long_only=False):
    name = "EMA Pullback" + (" LONG only" if long_only else "")
    engine = BacktestEngine(name)

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
    was_away_from_ema = {}  # Track si precio se alejó de EMA21

    print(f"Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 60 or pd.isna(row["adx"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        # ── CONDICIONES BASE ──
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

        # Detectar pullback: precio tocó o cruzó EMA21 desde arriba/abajo
        price = row["close"]
        ema21 = row["ema21"]
        ema50 = row["ema50"]
        touch_margin = row["atr"] * 0.3  # Margen de "toque"

        # Track si el precio se alejó de EMA21 (necesario para pullback)
        if ema_ribbon_bull and price > ema21 + row["atr"] * 0.5:
            was_away_from_ema["long"] = True
        if ema_ribbon_bear and price < ema21 - row["atr"] * 0.5:
            was_away_from_ema["short"] = True

        pullback_long = (abs(price - ema21) < touch_margin) and was_away_from_ema.get("long", False)
        pullback_short = (abs(price - ema21) < touch_margin) and was_away_from_ema.get("short", False)

        # RSI no extremo
        rsi_val = row["rsi"]

        # ── SEÑAL LONG ──
        if ema_ribbon_bull and tf_bull and pullback_long and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(price - ema50) + row["atr"] * 0.3  # SL debajo de EMA50
                sl_dist = max(sl_dist, row["atr"] * 1.0)  # Mínimo 1x ATR
                tp_dist = sl_dist * 2.0  # Ratio 1:2
                engine.open_trade("LONG", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts
                was_away_from_ema["long"] = False

        # ── SEÑAL SHORT ──
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
# INFORME
# ══════════════════════════════════════════════════════════════

def print_report(results):
    print("\n" + "=" * 70)
    print("  BACKTEST v2 — COMPARATIVA DE ESTRATEGIAS")
    print(f"  Periodo: {START_DATE} → hoy | Capital: ${INITIAL_BALANCE:,.0f}")
    print("=" * 70)

    for r in results:
        if r.get("trades", 0) == 0:
            print(f"\n{r['name']}: Sin trades")
            continue
        icon = "✅" if r["total_pnl"] > 0 else "❌"
        print(f"""
{'─'*70}
📊 {r['name']}
{'─'*70}
  Capital final:    ${r['final_balance']:>10,.2f}  ({r['return_pct']:+.1f}%)  {icon}
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

    pd.DataFrame(results).to_csv("backtest_v2_results.csv", index=False)
    print("Resultados guardados en backtest_v2_results.csv")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  BACKTEST v2 — 8 ESTRATEGIAS (4 tipos × L+S / LONG only)")
    print("=" * 70)

    df_15m = fetch_all_candles("15m")

    # 8 estrategias: 4 tipos × 2 modos
    strategies = []

    print("\n── Estrategia 1: Sniper v2 ──")
    strategies.append(run_sniper_v2(df_15m, long_only=False))
    strategies.append(run_sniper_v2(df_15m, long_only=True))

    print("\n── Estrategia 2: BB Squeeze + MACD ──")
    strategies.append(run_bollinger_momentum(df_15m, long_only=False))
    strategies.append(run_bollinger_momentum(df_15m, long_only=True))

    print("\n── Estrategia 3: RSI Mean Reversion ──")
    strategies.append(run_rsi_mean_reversion(df_15m, long_only=False))
    strategies.append(run_rsi_mean_reversion(df_15m, long_only=True))

    print("\n── Estrategia 4: EMA Pullback Trend Rider ──")
    strategies.append(run_ema_pullback(df_15m, long_only=False))
    strategies.append(run_ema_pullback(df_15m, long_only=True))

    results = [s.results() for s in strategies]
    print_report(results)
