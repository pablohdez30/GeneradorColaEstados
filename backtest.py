"""
backtest.py - Backtesting de 3 estrategias BTC/USDT

Estrategias:
1. Sniper (LONG + SHORT) - scoring 7 condiciones, multi-timeframe
2. Sniper LONG only      - igual pero solo compras
3. Trend Following       - EMA 9/21 cross + ADX + RSI

Datos: BTC/USDT 15m desde 2022 hasta hoy (Binance, gratis)
Uso:   python3 backtest.py
"""

import numpy as np
import pandas as pd
import ccxt
import time
from datetime import datetime, timezone

# ── Configuración ──────────────────────────────────────────────
SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 10_000.0
RISK_PCT = 0.01          # 1% del capital por trade
FEE_RATE = 0.0004        # 0.04% comisión Binance
STOP_ATR_MULT = 1.5
TP_ATR_MULT = 3.0

# Sniper
ADX_STRONG = 30
ADX_MIN = 20
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VOLUME_SPIKE = 2.0
FEAR_THRESHOLD = 25      # F&G no disponible en histórico, ignorar
MIN_SCORE = 4
SNIPER_COOLDOWN = 15     # minutos entre trades Sniper

# Trend
ADX_MIN_TREND = 20
TREND_COOLDOWN = 15      # minutos entre trades Trend

# Periodo de backtest
START_DATE = "2022-01-01"


# ══════════════════════════════════════════════════════════════════
# DESCARGA DE DATOS
# ══════════════════════════════════════════════════════════════════

def fetch_all_candles(timeframe: str = "15m") -> pd.DataFrame:
    """Descarga datos históricos de Binance desde START_DATE."""
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
            print(f"  {len(all_candles)} velas descargadas...", end="\r")
            time.sleep(0.3)

            # Si la última vela es reciente, parar
            last_ts = candles[-1][0] / 1000
            if last_ts > time.time() - 86400:
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


# ══════════════════════════════════════════════════════════════════
# INDICADORES
# ══════════════════════════════════════════════════════════════════

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

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

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores sobre un DataFrame."""
    d = df.copy()
    d["ema9"] = ema(d["close"], 9)
    d["ema21"] = ema(d["close"], 21)
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
    return d


# ══════════════════════════════════════════════════════════════════
# MOTOR DE BACKTEST
# ══════════════════════════════════════════════════════════════════

class BacktestEngine:
    def __init__(self, name: str):
        self.name = name
        self.balance = INITIAL_BALANCE
        self.peak_balance = INITIAL_BALANCE
        self.trades = []
        self.equity_curve = []
        self.position = None  # {direction, entry, sl, tp, qty}

    def open_trade(self, direction, price, atr_val, ts):
        if self.position:
            return

        sl_dist = atr_val * STOP_ATR_MULT
        tp_dist = atr_val * TP_ATR_MULT

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
            "direction": direction,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "qty": qty,
            "open_ts": ts,
        }

    def check_position(self, row, ts):
        if not self.position:
            return

        pos = self.position
        high = row["high"]
        low = row["low"]
        close = row["close"]

        hit_tp = hit_sl = False

        if pos["direction"] == "LONG":
            if low <= pos["sl"]:
                hit_sl = True
                exit_price = pos["sl"]
            elif high >= pos["tp"]:
                hit_tp = True
                exit_price = pos["tp"]
        else:
            if high >= pos["sl"]:
                hit_sl = True
                exit_price = pos["sl"]
            elif low <= pos["tp"]:
                hit_tp = True
                exit_price = pos["tp"]

        if hit_sl or hit_tp:
            reason = "TP" if hit_tp else "SL"
            if pos["direction"] == "LONG":
                pnl = (exit_price - pos["entry"]) * pos["qty"]
            else:
                pnl = (pos["entry"] - exit_price) * pos["qty"]

            fee = exit_price * pos["qty"] * FEE_RATE
            pnl -= fee
            self.balance += pnl
            self.peak_balance = max(self.peak_balance, self.balance)

            self.trades.append({
                "ts": ts,
                "direction": pos["direction"],
                "entry": pos["entry"],
                "exit": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl / (pos["entry"] * pos["qty"]),
                "reason": reason,
            })
            self.position = None

        self.equity_curve.append({"ts": ts, "balance": self.balance})

    def results(self) -> dict:
        if not self.trades:
            return {"name": self.name, "trades": 0}

        trades_df = pd.DataFrame(self.trades)
        wins = (trades_df["pnl"] > 0).sum()
        losses = len(trades_df) - wins
        total_pnl = trades_df["pnl"].sum()
        win_rate = wins / len(trades_df)

        # Max drawdown
        eq = pd.DataFrame(self.equity_curve)
        eq["peak"] = eq["balance"].cummax()
        eq["dd"] = (eq["peak"] - eq["balance"]) / eq["peak"]
        max_dd = eq["dd"].max()

        # Profit factor
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe
        pnl_arr = trades_df["pnl_pct"].values
        sharpe = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(252)) if pnl_arr.std() > 0 else 0

        return {
            "name": self.name,
            "trades": len(trades_df),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "final_balance": INITIAL_BALANCE + total_pnl,
            "return_pct": total_pnl / INITIAL_BALANCE * 100,
            "max_drawdown": max_dd,
            "profit_factor": round(profit_factor, 2),
            "sharpe": round(sharpe, 4),
            "avg_win": trades_df[trades_df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0,
            "avg_loss": trades_df[trades_df["pnl"] < 0]["pnl"].mean() if losses > 0 else 0,
            "best_trade": trades_df["pnl"].max(),
            "worst_trade": trades_df["pnl"].min(),
        }


# ══════════════════════════════════════════════════════════════════
# ESTRATEGIA 1 y 2: SNIPER (LONG+SHORT y LONG only)
# ══════════════════════════════════════════════════════════════════

def run_sniper(df_15m: pd.DataFrame, long_only: bool = False) -> BacktestEngine:
    name = "Sniper LONG only" if long_only else "Sniper (LONG+SHORT)"
    engine = BacktestEngine(name)

    # Resamplear a 1h y 4h para multi-timeframe
    df_1h = df_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    df_4h = df_15m.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    # Calcular indicadores en cada timeframe
    ind_15 = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)
    ind_4h = compute_indicators(df_4h)

    last_trade_ts = None
    last_signal_dir = None

    print(f"Ejecutando {name}...")

    for ts, row in ind_15.iterrows():
        if pd.isna(row["adx"]) or pd.isna(row["rsi"]):
            continue

        # Gestionar posición abierta
        engine.check_position(row, ts)

        # Cooldown
        if last_trade_ts and (ts - last_trade_ts).total_seconds() < SNIPER_COOLDOWN * 60:
            continue

        # Sin tendencia mínima
        if row["adx"] < ADX_MIN:
            last_signal_dir = None
            continue

        # Obtener datos 1h y 4h más cercanos
        try:
            r1h = ind_1h.asof(ts)
            r4h = ind_4h.asof(ts)
        except Exception:
            continue

        if r1h is None or r4h is None:
            continue

        # Scoring
        buy_score = sell_score = 0

        # 1. ADX fuerte
        if row["adx"] >= ADX_STRONG:
            buy_score += 1
            sell_score += 1

        # 2. EMA cross reciente (2 pts)
        if row["ema_cross_up"]:
            buy_score += 2
        if row["ema_cross_down"]:
            sell_score += 2

        # 3. EMA alineadas
        if row["ema_above"]:
            buy_score += 1
        else:
            sell_score += 1

        # 4. RSI favorable
        rsi_val = row["rsi"]
        if 35 <= rsi_val <= 55:
            buy_score += 1
        if 45 <= rsi_val <= 65:
            sell_score += 1

        # RSI bloqueante
        if rsi_val > RSI_OVERBOUGHT:
            buy_score = 0
        if rsi_val < RSI_OVERSOLD:
            sell_score = 0

        # 5. Volumen spike
        if row["vol_ratio"] >= VOLUME_SPIKE:
            buy_score += 1
            sell_score += 1

        # 6. F&G no disponible en histórico → omitir

        # 7. Multi-timeframe (2 pts)
        bullish_tfs = bearish_tfs = 0
        for r_tf in [r1h, r4h]:
            if not pd.isna(r_tf.get("adx", np.nan)) and r_tf.get("adx", 0) > 20:
                if r_tf.get("ema_above", False):
                    bullish_tfs += 1
                else:
                    bearish_tfs += 1
        if bullish_tfs >= 2:
            buy_score += 2
        if bearish_tfs >= 2:
            sell_score += 2

        # Decisión
        if buy_score >= MIN_SCORE and buy_score > sell_score:
            if last_signal_dir != "LONG" and not engine.position:
                engine.open_trade("LONG", row["close"], row["atr"], ts)
                last_trade_ts = ts
                last_signal_dir = "LONG"
        elif not long_only and sell_score >= MIN_SCORE and sell_score > buy_score:
            if last_signal_dir != "SHORT" and not engine.position:
                engine.open_trade("SHORT", row["close"], row["atr"], ts)
                last_trade_ts = ts
                last_signal_dir = "SHORT"
        else:
            last_signal_dir = None

    return engine


# ══════════════════════════════════════════════════════════════════
# ESTRATEGIA 3: TREND FOLLOWING (EMA cross + ADX + RSI)
# ══════════════════════════════════════════════════════════════════

def run_trend(df_15m: pd.DataFrame) -> BacktestEngine:
    engine = BacktestEngine("Trend Following (EMA 9/21)")
    ind = compute_indicators(df_15m)
    last_trade_ts = None

    print("Ejecutando Trend Following...")

    for ts, row in ind.iterrows():
        if pd.isna(row["adx"]) or pd.isna(row["rsi"]):
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < TREND_COOLDOWN * 60:
            continue

        if row["adx"] < ADX_MIN_TREND:
            continue

        rsi_val = row["rsi"]
        action = None
        confidence = 0

        if row["ema_cross_up"]:
            action = "LONG"
            confidence = 0.70
        elif row["ema_cross_down"]:
            action = "SHORT"
            confidence = 0.70
        elif row["ema_above"]:
            action = "LONG"
            confidence = 0.40
        else:
            action = "SHORT"
            confidence = 0.40

        # Filtro RSI
        if action == "LONG" and rsi_val > RSI_OVERBOUGHT:
            continue
        if action == "SHORT" and rsi_val < RSI_OVERSOLD:
            continue

        if confidence >= 0.35 and not engine.position:
            engine.open_trade(action, row["close"], row["atr"], ts)
            last_trade_ts = ts

    return engine


# ══════════════════════════════════════════════════════════════════
# INFORME COMPARATIVO
# ══════════════════════════════════════════════════════════════════

def print_report(results: list[dict]):
    print("\n" + "=" * 70)
    print("  BACKTEST BTC/USDT — COMPARATIVA DE ESTRATEGIAS")
    print(f"  Periodo: {START_DATE} → hoy | Capital inicial: ${INITIAL_BALANCE:,.0f}")
    print("=" * 70)

    for r in results:
        if r.get("trades", 0) == 0:
            print(f"\n{r['name']}: Sin trades")
            continue

        pnl_color = "✅" if r["total_pnl"] > 0 else "❌"
        print(f"""
{'─'*70}
📊 {r['name']}
{'─'*70}
  Capital final:    ${r['final_balance']:>10,.2f}  ({r['return_pct']:+.1f}%)  {pnl_color}
  PnL total:        ${r['total_pnl']:>+10,.2f}
  Total trades:     {r['trades']:>10}
  Wins / Losses:    {r['wins']:>5} / {r['losses']}   (Win Rate: {r['win_rate']:.1%})
  Profit Factor:    {r['profit_factor']:>10.2f}
  Sharpe Ratio:     {r['sharpe']:>10.4f}
  Max Drawdown:     {r['max_drawdown']:>9.2%}
  Avg Win:          ${r['avg_win']:>+10,.2f}
  Avg Loss:         ${r['avg_loss']:>+10,.2f}
  Mejor trade:      ${r['best_trade']:>+10,.2f}
  Peor trade:       ${r['worst_trade']:>+10,.2f}""")

    print("\n" + "=" * 70)
    print("  GANADOR: " + max(results, key=lambda x: x.get("total_pnl", float("-inf")))["name"])
    print("=" * 70 + "\n")

    # Guardar CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("backtest_results.csv", index=False)
    print("Resultados guardados en backtest_results.csv")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  BACKTEST BTC/USDT — 3 ESTRATEGIAS")
    print("=" * 70)

    # Descargar datos una sola vez
    df_15m = fetch_all_candles("15m")

    # Ejecutar las 3 estrategias
    sniper_both = run_sniper(df_15m, long_only=False)
    sniper_long = run_sniper(df_15m, long_only=True)
    trend = run_trend(df_15m)

    # Imprimir resultados
    results = [
        sniper_both.results(),
        sniper_long.results(),
        trend.results(),
    ]
    print_report(results)
