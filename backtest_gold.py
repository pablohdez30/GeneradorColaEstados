"""
backtest_gold.py - Backtesting de estrategias para XAUUSD (Oro)

Estrategias adaptadas al comportamiento del oro:
1. EMA Pullback Gold - Adaptacion de la ganadora de BTC (v3)
2. Golden Cross Trend Rider - EMA 50/200 (indicador rey del oro)
3. Session Breakout - Ruptura del rango asiatico en apertura Londres/NY

Cada una en L+S y LONG only = 6 variantes.

Datos: PAXG/USDT 15m desde 2023 (gold-backed token en Binance)
Uso:   python3 backtest_gold.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ccxt
import time

INITIAL_BALANCE = 10_000.0
RISK_PCT = 0.01
FEE_RATE = 0.0004
START_DATE = "2023-01-01"


# ══════════════════════════════════════════════════════════════
# DESCARGA DE DATOS
# ══════════════════════════════════════════════════════════════

def fetch_gold_candles(timeframe="15m"):
    """Intenta descargar datos de oro de varias fuentes."""
    sources = [
        ("bybit", "XAU/USDT", "swap"),
        ("binance", "PAXG/USDT", "swap"),
        ("binance", "PAXG/USDT", "spot"),
    ]
    for exch_name, symbol, dtype in sources:
        try:
            print(f"Probando {exch_name} {symbol} ({dtype})...")
            if exch_name == "binance":
                exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": dtype}})
            elif exch_name == "bybit":
                exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": dtype}})
            else:
                continue

            since = exchange.parse8601(f"{START_DATE}T00:00:00Z")
            test = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=10)
            if not test:
                print(f"  Sin datos en {exch_name} {symbol}")
                continue

            print(f"  Datos encontrados! Descargando {symbol} de {exch_name}...")
            all_candles = []
            while True:
                try:
                    candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                    if not candles:
                        break
                    all_candles.extend(candles)
                    since = candles[-1][0] + 1
                    print(f"  {len(all_candles)} velas...", end="\r")
                    time.sleep(0.3)
                    if candles[-1][0] / 1000 > time.time() - 86400:
                        break
                except Exception as e:
                    print(f"  Error: {e}. Reintentando...")
                    time.sleep(5)

            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            df = df[~df.index.duplicated(keep="last")]
            print(f"\nTotal: {len(df)} velas de {df.index[0].date()} a {df.index[-1].date()}")
            print(f"Precio rango: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            return df, symbol

        except Exception as e:
            print(f"  Fallo {exch_name} {symbol}: {e}")
            continue

    print("ERROR: No se encontraron datos de oro en ningun exchange.")
    return None, None


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
    d["ema10"] = ema(d["close"], 10)
    d["ema20"] = ema(d["close"], 20)
    d["ema21"] = ema(d["close"], 21)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["rsi"] = rsi(d["close"])
    d["adx"] = adx(d)
    d["atr"] = atr(d)
    d["vol_ma"] = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma"]
    d["ema_above"] = d["ema9"] > d["ema21"]
    d["golden_cross"] = d["ema50"] > d["ema200"]
    d["macd_line"], d["macd_signal"], d["macd_hist"] = macd(d["close"])
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
        self.max_duration = max_duration

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
# ESTRATEGIA 1: EMA PULLBACK GOLD
# ══════════════════════════════════════════════════════════════
# Adaptacion de la ganadora de BTC (v3 EMA Pullback Original)
# Cambios para oro:
# - ADX 20 (oro tiende mas suave)
# - Touch margin 0.4x ATR (mas ancho)
# - Multi-TF solo 15m + 1H (no 4H)
# - SIN trailing stop (arruino en v4)

def run_ema_pullback_gold(df_15m, long_only=False):
    name = "EMA Pullback Gold" + (" LONG" if long_only else " L+S")
    engine = BacktestEngine(name, max_duration=100)

    df_1h = df_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    ind = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)

    ema_above_1h = ind_1h["ema_above"].astype(float)
    adx_1h = ind_1h["adx"]

    last_trade_ts = None
    cooldown_secs = 3 * 3600
    was_away = {"long": False, "short": False}

    print(f"  Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 60 or pd.isna(row["adx"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        # ADX minimo 20 (mas bajo que BTC)
        if row["adx"] < 20:
            was_away["long"] = False
            was_away["short"] = False
            continue

        # EMA ribbon
        ribbon_bull = row["ema9"] > row["ema21"] > row["ema50"]
        ribbon_bear = row["ema9"] < row["ema21"] < row["ema50"]

        # Multi-TF 1H
        try:
            idx_1h = ema_above_1h.index.asof(ts)
            tf_bull = False
            tf_bear = False
            if idx_1h is not pd.NaT:
                a1h = adx_1h.get(idx_1h, 0)
                if a1h > 15:
                    bull_1h = ema_above_1h.get(idx_1h, 0) > 0.5
                    tf_bull = bull_1h
                    tf_bear = not bull_1h
        except Exception:
            continue

        price = row["close"]
        ema21 = row["ema21"]
        ema50 = row["ema50"]
        touch_margin = row["atr"] * 0.4  # mas ancho para oro
        away_margin = row["atr"] * 0.6

        if ribbon_bull and price > ema21 + away_margin:
            was_away["long"] = True
        if ribbon_bear and price < ema21 - away_margin:
            was_away["short"] = True

        pullback_long = (abs(price - ema21) < touch_margin) and was_away["long"]
        pullback_short = (abs(price - ema21) < touch_margin) and was_away["short"]

        rsi_val = row["rsi"]

        if ribbon_bull and tf_bull and pullback_long and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(price - ema50) + row["atr"] * 0.3
                sl_dist = max(sl_dist, row["atr"] * 1.0)
                tp_dist = sl_dist * 2.0
                engine.open_trade("LONG", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts
                was_away["long"] = False

        elif not long_only and ribbon_bear and tf_bear and pullback_short and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(ema50 - price) + row["atr"] * 0.3
                sl_dist = max(sl_dist, row["atr"] * 1.0)
                tp_dist = sl_dist * 2.0
                engine.open_trade("SHORT", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts
                was_away["short"] = False

    return engine


# ══════════════════════════════════════════════════════════════
# ESTRATEGIA 2: GOLDEN CROSS TREND RIDER
# ══════════════════════════════════════════════════════════════
# EMA 50/200 - el indicador mas respetado en oro.
# Entra en pullbacks a EMA50 cuando hay golden cross activo.
# R:R 1:3 porque las tendencias del oro duran mas.

def run_golden_cross(df_15m, long_only=False):
    name = "Golden Cross" + (" LONG" if long_only else " L+S")
    engine = BacktestEngine(name, max_duration=200)

    ind = compute_indicators(df_15m)

    last_trade_ts = None
    cooldown_secs = 6 * 3600  # 6h cooldown
    was_away = {"long": False, "short": False}

    print(f"  Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 210 or pd.isna(row["ema200"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        price = row["close"]
        ema50 = row["ema50"]
        ema200 = row["ema200"]
        golden = ema50 > ema200
        death = ema50 < ema200

        # ADX minimo
        if row["adx"] < 18:
            was_away["long"] = False
            was_away["short"] = False
            continue

        # Track si precio se alejo de EMA50
        if golden and price > ema50 + row["atr"] * 0.8:
            was_away["long"] = True
        if death and price < ema50 - row["atr"] * 0.8:
            was_away["short"] = True

        touch_margin = row["atr"] * 0.5
        pullback_long = golden and (abs(price - ema50) < touch_margin) and was_away["long"]
        pullback_short = death and (abs(price - ema50) < touch_margin) and was_away["short"]

        rsi_val = row["rsi"]

        # LONG: Golden cross activo + pullback a EMA50
        if pullback_long and 30 < rsi_val < 65:
            if not engine.position:
                sl_dist = row["atr"] * 1.5
                tp_dist = sl_dist * 3.0  # R:R 1:3
                engine.open_trade("LONG", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts
                was_away["long"] = False

        # SHORT: Death cross activo + pullback a EMA50
        elif not long_only and pullback_short and 35 < rsi_val < 70:
            if not engine.position:
                sl_dist = row["atr"] * 1.5
                tp_dist = sl_dist * 3.0
                engine.open_trade("SHORT", price, sl_dist, tp_dist, ts)
                last_trade_ts = ts
                was_away["short"] = False

    return engine


# ══════════════════════════════════════════════════════════════
# ESTRATEGIA 3: SESSION BREAKOUT
# ══════════════════════════════════════════════════════════════
# Detecta rango de sesion asiatica (00:00-07:00 UTC)
# Opera la ruptura cuando abre Londres/NY
# Solo 1 trade por dia maximo

def run_session_breakout(df_15m, long_only=False):
    name = "Session Breakout" + (" LONG" if long_only else " L+S")
    engine = BacktestEngine(name, max_duration=32)  # 8 horas max

    ind = compute_indicators(df_15m)

    last_trade_date = None
    asian_high = None
    asian_low = None

    print(f"  Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 60 or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        hour = ts.hour
        current_date = ts.date()

        # Construir rango asiatico (00:00-07:00 UTC)
        if hour == 0 and ts.minute == 0:
            asian_high = row["high"]
            asian_low = row["low"]
        elif 0 < hour < 7 and asian_high is not None:
            asian_high = max(asian_high, row["high"])
            asian_low = min(asian_low, row["low"])

        # Operar durante Londres-NY overlap (13:00-16:00 UTC)
        if hour < 7 or hour > 20:
            continue

        if asian_high is None or asian_low is None:
            continue

        # Solo 1 trade por dia
        if last_trade_date == current_date:
            continue

        price = row["close"]
        asian_range = asian_high - asian_low
        if asian_range < row["atr"] * 0.3:
            continue  # Rango asiatico demasiado pequeño

        # Confirmacion: EMA20 direccion
        ema20_bull = price > row["ema20"]
        ema20_bear = price < row["ema20"]

        # LONG: Precio rompe por encima del rango asiatico
        if price > asian_high and ema20_bull:
            if not engine.position:
                sl_dist = min(price - asian_low, row["atr"] * 2.0)
                sl_dist = max(sl_dist, row["atr"] * 0.8)
                tp_dist = sl_dist * 1.5
                engine.open_trade("LONG", price, sl_dist, tp_dist, ts)
                last_trade_date = current_date

        # SHORT: Precio rompe por debajo del rango asiatico
        elif not long_only and price < asian_low and ema20_bear:
            if not engine.position:
                sl_dist = min(asian_high - price, row["atr"] * 2.0)
                sl_dist = max(sl_dist, row["atr"] * 0.8)
                tp_dist = sl_dist * 1.5
                engine.open_trade("SHORT", price, sl_dist, tp_dist, ts)
                last_trade_date = current_date

    return engine


# ══════════════════════════════════════════════════════════════
# INFORME
# ══════════════════════════════════════════════════════════════

def print_report(results, symbol_name):
    print("\n" + "=" * 75)
    print("  BACKTEST GOLD — COMPARATIVA DE ESTRATEGIAS")
    print(f"  Simbolo: {symbol_name} | Periodo: {START_DATE} -> hoy | Capital: ${INITIAL_BALANCE:,.0f}")
    print("=" * 75)

    for r in results:
        if r.get("trades", 0) == 0:
            print(f"\n{r['name']}: Sin trades")
            continue
        icon = "+" if r["total_pnl"] > 0 else "-"
        print(f"""
{'─'*75}
 {r['name']}
{'─'*75}
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

    print("\n" + "=" * 75)
    profitable = [r for r in results if r.get("total_pnl", 0) > 0]
    if profitable:
        best = max(profitable, key=lambda x: x["total_pnl"])
        safest = min(profitable, key=lambda x: x["max_drawdown"])
        print(f"  MAYOR RETORNO:  {best['name']} ({best['return_pct']:+.1f}%, DD: {best['max_drawdown']:.1%})")
        if safest != best:
            print(f"  MENOR RIESGO:   {safest['name']} ({safest['return_pct']:+.1f}%, DD: {safest['max_drawdown']:.1%})")
    else:
        best = max(results, key=lambda x: x.get("total_pnl", float("-inf")))
        print(f"  MEJOR (menos malo): {best['name']} ({best.get('return_pct', 0):+.1f}%)")
    print("=" * 75 + "\n")

    pd.DataFrame(results).to_csv("backtest_gold_results.csv", index=False)
    print("Resultados guardados en backtest_gold_results.csv")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 75)
    print("  BACKTEST GOLD — 3 ESTRATEGIAS ADAPTADAS AL ORO")
    print("=" * 75)

    df_15m, symbol_name = fetch_gold_candles("15m")
    if df_15m is None:
        print("No se pudieron descargar datos. Abortando.")
        exit(1)

    strategies = []

    print("\n-- Estrategia 1: EMA Pullback Gold (adaptada de v3 BTC) --")
    strategies.append(run_ema_pullback_gold(df_15m, long_only=False))
    strategies.append(run_ema_pullback_gold(df_15m, long_only=True))

    print("\n-- Estrategia 2: Golden Cross Trend Rider (EMA 50/200) --")
    strategies.append(run_golden_cross(df_15m, long_only=False))
    strategies.append(run_golden_cross(df_15m, long_only=True))

    print("\n-- Estrategia 3: Session Breakout (rango asiatico) --")
    strategies.append(run_session_breakout(df_15m, long_only=False))
    strategies.append(run_session_breakout(df_15m, long_only=True))

    results = [s.results() for s in strategies]
    print_report(results, symbol_name)
