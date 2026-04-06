"""
backtest_v4.py - Backtest FINAL: EMA Pullback optimizado + variantes R:R

Estrategias (8 variantes):
1. EMA Pullback Hybrid (L+S / LONG) - Lo mejor de original + v2 (trailing + vol suave)
2. EMA Pullback R:R 1:1.5 (L+S / LONG) - Ratio apretado, mayor win rate
3. EMA Pullback R:R 1:2.5 (L+S / LONG) - Ratio amplio, menor win rate
4. EMA Pullback R:R 1:3 (L+S / LONG) - Swing, muy selectivo

Datos: BTC/USDT 15m desde 2022 hasta hoy
Uso:   python3 backtest_v4.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ccxt
import time

# ── Configuracion global ──────────────────────────────────────
SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 10_000.0
RISK_PCT = 0.01
FEE_RATE = 0.0004
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
    d["macd_line"], d["macd_signal"], d["macd_hist"] = macd(d["close"])
    return d


# ══════════════════════════════════════════════════════════════
# MOTOR DE BACKTEST (con trailing stop)
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

        # Trailing stop: si precio mueve X*ATR en profit, mover SL a breakeven
        if (pos.get("trailing_stop_atr") is not None
                and pos.get("atr_value") is not None
                and not pos.get("trailing_activated", False)):
            threshold = pos["trailing_stop_atr"] * pos["atr_value"]
            if pos["direction"] == "LONG":
                if high >= pos["entry"] + threshold:
                    pos["sl"] = pos["entry"]
                    pos["trailing_activated"] = True
            else:
                if low <= pos["entry"] - threshold:
                    pos["sl"] = pos["entry"]
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
            reason = "TP" if hit_tp else ("BE" if pos.get("trailing_activated") else "SL/MAX")
            self.trades.append({
                "ts": ts, "direction": pos["direction"],
                "entry": pos["entry"], "exit": exit_price,
                "pnl": pnl, "pnl_pct": pnl / (pos["entry"] * pos["qty"]),
                "reason": reason,
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
        # Contar breakeven exits
        be_count = (t["reason"] == "BE").sum() if "reason" in t.columns else 0
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
            "breakeven_exits": int(be_count),
        }


# ══════════════════════════════════════════════════════════════
# CORE: EMA PULLBACK STRATEGY (parametrizable)
# ══════════════════════════════════════════════════════════════
# Centraliza la logica para evitar duplicar codigo.
# Parametros ajustables:
#   - tp_ratio: multiplicador del SL para TP (1.5, 2.0, 2.5, 3.0)
#   - adx_min: umbral ADX minimo (25)
#   - use_trailing: activar trailing stop a breakeven
#   - trailing_threshold: ATR multiplier para activar trailing (1.5)
#   - vol_min: ratio minimo de volumen (0 = sin filtro)
#   - touch_margin_mult: multiplicador ATR para margen de pullback (0.3)
#   - away_margin_mult: multiplicador ATR para "alejado de EMA" (0.5)

def run_ema_pullback(df_15m, ind, ind_1h, ind_4h, long_only=False, name="EMA Pullback",
                     tp_ratio=2.0, adx_min=25, use_trailing=False,
                     trailing_threshold=1.5, vol_min=0.0,
                     touch_margin_mult=0.3, away_margin_mult=0.5,
                     cooldown_h=3, max_dur=100):

    engine = BacktestEngine(name, max_duration=max_dur)

    adx_1h = ind_1h["adx"]
    ema_above_1h = ind_1h["ema_above"].astype(float)
    ema_above_4h = ind_4h["ema_above"].astype(float)

    last_trade_ts = None
    cooldown_secs = cooldown_h * 3600
    was_away = {"long": False, "short": False}

    print(f"  Ejecutando {name}...")

    for i, (ts, row) in enumerate(ind.iterrows()):
        if i < 60 or pd.isna(row["adx"]) or pd.isna(row["atr"]) or row["atr"] == 0:
            continue

        engine.check_position(row, ts)

        if last_trade_ts and (ts - last_trade_ts).total_seconds() < cooldown_secs:
            continue

        # ADX minimo
        if row["adx"] < adx_min:
            was_away["long"] = False
            was_away["short"] = False
            continue

        # Volumen minimo (0 = sin filtro)
        if vol_min > 0 and (pd.isna(row["vol_ratio"]) or row["vol_ratio"] < vol_min):
            continue

        # EMAs alineadas en 15m (ribbon)
        ema_ribbon_bull = row["ema9"] > row["ema21"] > row["ema50"]
        ema_ribbon_bear = row["ema9"] < row["ema21"] < row["ema50"]

        # Multi-TF confirmation (1h + 4h)
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

        price = row["close"]
        ema21 = row["ema21"]
        ema50 = row["ema50"]
        current_atr = row["atr"]
        touch_margin = current_atr * touch_margin_mult
        away_margin = current_atr * away_margin_mult

        # Track si precio se alejo de EMA21
        if ema_ribbon_bull and price > ema21 + away_margin:
            was_away["long"] = True
        if ema_ribbon_bear and price < ema21 - away_margin:
            was_away["short"] = True

        pullback_long = (abs(price - ema21) < touch_margin) and was_away["long"]
        pullback_short = (abs(price - ema21) < touch_margin) and was_away["short"]

        rsi_val = row["rsi"]

        # LONG
        if ema_ribbon_bull and tf_bull and pullback_long and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(price - ema50) + current_atr * 0.3
                sl_dist = max(sl_dist, current_atr * 1.0)
                tp_dist = sl_dist * tp_ratio
                trail_atr = trailing_threshold if use_trailing else None
                trail_val = current_atr if use_trailing else None
                engine.open_trade("LONG", price, sl_dist, tp_dist, ts,
                                  trailing_stop_atr=trail_atr, atr_value=trail_val)
                last_trade_ts = ts
                was_away["long"] = False

        # SHORT
        elif not long_only and ema_ribbon_bear and tf_bear and pullback_short and 35 < rsi_val < 65:
            if not engine.position:
                sl_dist = abs(ema50 - price) + current_atr * 0.3
                sl_dist = max(sl_dist, current_atr * 1.0)
                tp_dist = sl_dist * tp_ratio
                trail_atr = trailing_threshold if use_trailing else None
                trail_val = current_atr if use_trailing else None
                engine.open_trade("SHORT", price, sl_dist, tp_dist, ts,
                                  trailing_stop_atr=trail_atr, atr_value=trail_val)
                last_trade_ts = ts
                was_away["short"] = False

    return engine


# ══════════════════════════════════════════════════════════════
# INFORME
# ══════════════════════════════════════════════════════════════

def print_report(results):
    print("\n" + "=" * 75)
    print("  BACKTEST v4 FINAL — EMA PULLBACK: HYBRID + OPTIMIZACION R:R")
    print(f"  Periodo: {START_DATE} -> hoy | Capital: ${INITIAL_BALANCE:,.0f}")
    print("=" * 75)

    for r in results:
        if r.get("trades", 0) == 0:
            print(f"\n{r['name']}: Sin trades")
            continue
        icon = "+" if r["total_pnl"] > 0 else "-"
        be = r.get("breakeven_exits", 0)
        be_str = f"  Breakeven exits: {be:>10}" if be > 0 else ""
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
  Peor trade:       ${r['worst_trade']:>+10,.2f}{be_str}""")

    print("\n" + "=" * 75)
    profitable = [r for r in results if r.get("total_pnl", 0) > 0]
    if profitable:
        best = max(profitable, key=lambda x: x["total_pnl"])
        safest = min(profitable, key=lambda x: x["max_drawdown"])
        print(f"  MAYOR RETORNO:  {best['name']} ({best['return_pct']:+.1f}%, DD: {best['max_drawdown']:.1%})")
        print(f"  MENOR RIESGO:   {safest['name']} ({safest['return_pct']:+.1f}%, DD: {safest['max_drawdown']:.1%})")
    else:
        best = max(results, key=lambda x: x.get("total_pnl", float("-inf")))
        print(f"  MEJOR (menos malo): {best['name']} ({best.get('return_pct', 0):+.1f}%)")
    print("=" * 75 + "\n")

    pd.DataFrame(results).to_csv("backtest_v4_results.csv", index=False)
    print("Resultados guardados en backtest_v4_results.csv")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 75)
    print("  BACKTEST v4 FINAL — EMA PULLBACK OPTIMIZADO")
    print("  8 variantes: Hybrid + R:R 1:1.5, 1:2, 1:2.5, 1:3")
    print("=" * 75)

    df_15m = fetch_all_candles("15m")

    # Pre-calcular indicadores (una sola vez)
    print("\nCalculando indicadores...")
    df_1h = df_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    df_4h = df_15m.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    ind_15 = compute_indicators(df_15m)
    ind_1h = compute_indicators(df_1h)
    ind_4h = compute_indicators(df_4h)

    strategies = []

    # ── 1. HYBRID: Original + trailing stop + vol suave ──
    # Lo mejor del original (ADX 25, touch 0.3) + trailing stop breakeven + vol > 1.0
    print("\n── 1. EMA Pullback Hybrid (original + trailing + vol suave) ──")
    for lo in [False, True]:
        s = run_ema_pullback(
            df_15m, ind_15, ind_1h, ind_4h, long_only=lo,
            name=f"Hybrid R:R 1:2{' LONG' if lo else ' L+S'}",
            tp_ratio=2.0, adx_min=25, use_trailing=True,
            trailing_threshold=1.5, vol_min=1.0,
            touch_margin_mult=0.3, away_margin_mult=0.5,
            cooldown_h=3, max_dur=100
        )
        strategies.append(s)

    # ── 2. R:R 1:1.5 (ratio apretado, busca mayor win rate) ──
    print("\n── 2. EMA Pullback R:R 1:1.5 (ratio apretado) ──")
    for lo in [False, True]:
        s = run_ema_pullback(
            df_15m, ind_15, ind_1h, ind_4h, long_only=lo,
            name=f"R:R 1:1.5{' LONG' if lo else ' L+S'}",
            tp_ratio=1.5, adx_min=25, use_trailing=False,
            vol_min=0.0,
            touch_margin_mult=0.3, away_margin_mult=0.5,
            cooldown_h=3, max_dur=80
        )
        strategies.append(s)

    # ── 3. R:R 1:2.5 (ratio medio-amplio) ──
    print("\n── 3. EMA Pullback R:R 1:2.5 ──")
    for lo in [False, True]:
        s = run_ema_pullback(
            df_15m, ind_15, ind_1h, ind_4h, long_only=lo,
            name=f"R:R 1:2.5{' LONG' if lo else ' L+S'}",
            tp_ratio=2.5, adx_min=25, use_trailing=True,
            trailing_threshold=1.0, vol_min=0.0,
            touch_margin_mult=0.3, away_margin_mult=0.5,
            cooldown_h=3, max_dur=120
        )
        strategies.append(s)

    # ── 4. R:R 1:3 (swing, TP grande) ──
    print("\n── 4. EMA Pullback R:R 1:3 (swing) ──")
    for lo in [False, True]:
        s = run_ema_pullback(
            df_15m, ind_15, ind_1h, ind_4h, long_only=lo,
            name=f"R:R 1:3{' LONG' if lo else ' L+S'}",
            tp_ratio=3.0, adx_min=25, use_trailing=True,
            trailing_threshold=1.0, vol_min=0.0,
            touch_margin_mult=0.3, away_margin_mult=0.5,
            cooldown_h=3, max_dur=150
        )
        strategies.append(s)

    results = [s.results() for s in strategies]
    print_report(results)
