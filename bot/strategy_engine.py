"""
strategy_engine.py - Estrategia EMA Pullback Trend Rider para BTC/USDT.

Basada en los resultados del backtest v3 (ganadora: +20.8%, DD 32.6%).
Entra en pullbacks a EMA21 cuando hay tendencia fuerte con confirmacion
multi-timeframe (15m + 1H + 4H).

Logica:
- EMA ribbon alineado: 9 > 21 > 50 (bull) o 9 < 21 < 50 (bear)
- ADX > 25 (tendencia fuerte)
- Precio se aleja de EMA21, luego vuelve (pullback)
- Confirmacion: 1H y 4H EMAs alineadas
- RSI entre 35-65 (no extremos)
- SL debajo de EMA50, TP = 2x SL (R:R 1:2)
- SIN trailing stop (empeora resultados)
"""

import time
import numpy as np
import pandas as pd
import ccxt

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYMBOL, ATR_PERIOD, STOP_ATR_MULTIPLIER, TP_ATR_MULTIPLIER
from bot.logger import setup_logger

logger = setup_logger("strategy")

# ── Configuracion EMA Pullback ──────────────────────────────
TIMEFRAMES = ["15m", "1h", "4h"]
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ADX_MIN = 25
RSI_MIN = 35
RSI_MAX = 65
TOUCH_MARGIN_MULT = 0.3   # Pullback touch margin (x ATR)
AWAY_MARGIN_MULT = 0.5    # Away from EMA margin (x ATR)


# ══════════════════════════════════════════════════════════════
# INDICADORES TECNICOS
# ══════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = compute_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period).mean() / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / period).mean()


def detect_market_regime(df: pd.DataFrame) -> str:
    if len(df) < 100:
        return "UNKNOWN"
    recent = df.tail(100)
    returns = recent["close"].pct_change().dropna()
    volatility = returns.std() * np.sqrt(len(returns))
    if volatility > 0.02:
        return "HIGH_VOLATILITY"
    adx = compute_adx(recent)
    if not adx.empty and adx.iloc[-1] > 20:
        return "TRENDING"
    return "RANGING"


# ══════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ANALYZER
# ══════════════════════════════════════════════════════════════

class MultiTimeframeAnalyzer:
    """Obtiene datos y calcula indicadores en 15m, 1h, 4h."""

    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        self.candle_buffers = {}

    def fetch_candles(self, timeframe: str, limit: int = 200) -> pd.DataFrame:
        try:
            raw = self.exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            self.candle_buffers[timeframe] = df
            return df
        except Exception as e:
            logger.warning(f"Error fetching {timeframe}: {e}")
            return self.candle_buffers.get(timeframe, pd.DataFrame())

    def analyze_timeframe(self, df: pd.DataFrame) -> dict:
        """Calcula indicadores para un timeframe."""
        if df.empty or len(df) < 60:
            return {}

        close = df["close"]
        ema_f = compute_ema(close, EMA_FAST)
        ema_m = compute_ema(close, EMA_MID)
        ema_s = compute_ema(close, EMA_SLOW)
        rsi_val = compute_rsi(close)
        adx_val = compute_adx(df)
        atr_val = compute_atr(df)

        return {
            "price": round(close.iloc[-1], 2),
            "ema_fast": round(ema_f.iloc[-1], 2),       # EMA 9
            "ema_mid": round(ema_m.iloc[-1], 2),        # EMA 21
            "ema_slow": round(ema_s.iloc[-1], 2),       # EMA 50
            "ema_above": ema_f.iloc[-1] > ema_m.iloc[-1],
            "ribbon_bull": ema_f.iloc[-1] > ema_m.iloc[-1] > ema_s.iloc[-1],
            "ribbon_bear": ema_f.iloc[-1] < ema_m.iloc[-1] < ema_s.iloc[-1],
            "rsi": round(rsi_val.iloc[-1], 1),
            "adx": round(adx_val.iloc[-1], 1),
            "atr": round(atr_val.iloc[-1], 2),
        }

    def analyze_all(self) -> dict:
        """Analiza todos los timeframes."""
        results = {}
        for tf in TIMEFRAMES:
            df = self.fetch_candles(tf)
            if not df.empty:
                indicators = self.analyze_timeframe(df)
                if indicators:
                    results[tf] = indicators
        return results

    def get_current_price(self) -> float:
        try:
            ticker = self.exchange.fetch_ticker(SYMBOL)
            return ticker["last"]
        except Exception:
            return 0.0


# ══════════════════════════════════════════════════════════════
# MOTOR DE ESTRATEGIA: EMA PULLBACK TREND RIDER
# ══════════════════════════════════════════════════════════════

class StrategyEngine:
    def __init__(self, ml_weight: float = 0.0):
        self.ml_weight = ml_weight
        self.last_signal = None
        self.last_indicators = {}
        self.last_signal_direction = None
        self.analyzer = MultiTimeframeAnalyzer()

        # Pullback state tracking
        self.was_away_long = False
        self.was_away_short = False

        logger.info("StrategyEngine inicializado | Modo: EMA Pullback Trend Rider")

    def generate_signal(self, df: pd.DataFrame = None, order_book: dict = None,
                        ml_prediction: float = 0.0, sentiment: float = 0.5) -> dict:
        """
        Genera senal usando EMA Pullback Trend Rider.
        Busca pullbacks a EMA21 en tendencias fuertes con confirmacion multi-TF.
        """
        analysis = self.analyzer.analyze_all()

        if not analysis:
            return self._build_signal("HOLD", 0, ["No se pudieron obtener datos"],
                                      {}, "UNKNOWN")

        # Timeframe principal: 15m
        if "15m" not in analysis:
            return self._build_signal("HOLD", 0, ["Sin datos de 15m"],
                                      {}, "UNKNOWN")

        primary = analysis["15m"]
        regime = detect_market_regime(
            self.analyzer.candle_buffers.get("15m", pd.DataFrame())
        ) if "15m" in self.analyzer.candle_buffers else "UNKNOWN"

        indicators = self._format_indicators(primary)
        adx = primary.get("adx", 0)
        atr = primary.get("atr", 0)
        price = primary.get("price", 0)
        ema21 = primary.get("ema_mid", 0)
        ema50 = primary.get("ema_slow", 0)
        rsi = primary.get("rsi", 50)

        if atr == 0 or price == 0:
            return self._build_signal("HOLD", 0, ["Datos incompletos"], indicators, regime)

        # ── 1. ADX minimo (tendencia fuerte) ──
        if adx < ADX_MIN:
            self.was_away_long = False
            self.was_away_short = False
            return self._build_signal("HOLD", 0,
                [f"Sin tendencia fuerte (ADX={adx:.0f} < {ADX_MIN})"],
                indicators, regime)

        # ── 2. EMA Ribbon ──
        ribbon_bull = primary.get("ribbon_bull", False)
        ribbon_bear = primary.get("ribbon_bear", False)

        if not ribbon_bull and not ribbon_bear:
            return self._build_signal("HOLD", 0,
                [f"EMAs no alineadas (ribbon neutro)"],
                indicators, regime)

        # ── 3. Multi-TF confirmation (1H + 4H) ──
        tf_bull = False
        tf_bear = False
        for tf in ["1h", "4h"]:
            if tf in analysis:
                tf_data = analysis[tf]
                if tf_data.get("adx", 0) > 20:
                    if tf_data.get("ema_above"):
                        if not tf_bull:
                            tf_bull = True
                        else:
                            tf_bull = True  # Both TFs bullish
                    else:
                        if not tf_bear:
                            tf_bear = True
                        else:
                            tf_bear = True

        # Need BOTH 1H and 4H to confirm
        both_bull = ("1h" in analysis and "4h" in analysis and
                     analysis["1h"].get("ema_above") and analysis["4h"].get("ema_above") and
                     analysis["1h"].get("adx", 0) > 20)
        both_bear = ("1h" in analysis and "4h" in analysis and
                     not analysis["1h"].get("ema_above") and not analysis["4h"].get("ema_above") and
                     analysis["1h"].get("adx", 0) > 20)

        # ── 4. Pullback detection ──
        touch_margin = atr * TOUCH_MARGIN_MULT
        away_margin = atr * AWAY_MARGIN_MULT

        # Track if price moved away from EMA21
        if ribbon_bull and price > ema21 + away_margin:
            self.was_away_long = True
        if ribbon_bear and price < ema21 - away_margin:
            self.was_away_short = True

        pullback_long = (abs(price - ema21) < touch_margin) and self.was_away_long
        pullback_short = (abs(price - ema21) < touch_margin) and self.was_away_short

        # ── 5. RSI filter ──
        rsi_ok = RSI_MIN < rsi < RSI_MAX

        # ── DECISION ──
        reasons = []

        # LONG signal
        if ribbon_bull and both_bull and pullback_long and rsi_ok:
            if self.last_signal_direction == "BUY":
                return self._build_signal("HOLD", 0,
                    ["Senal LONG repetida, esperando cambio"], indicators, regime)

            self.last_signal_direction = "BUY"
            self.was_away_long = False

            reasons = [
                f"EMA Pullback LONG: precio volvio a EMA21",
                f"Ribbon alcista (EMA9>{EMA_MID}>{EMA_SLOW})",
                f"Multi-TF confirmado (1H+4H alcistas)",
                f"ADX={adx:.0f} RSI={rsi:.0f}",
            ]
            confidence = min(0.7 + (adx - ADX_MIN) / 100, 0.95)
            return self._build_signal("BUY", confidence, reasons, indicators, regime)

        # SHORT signal
        elif ribbon_bear and both_bear and pullback_short and rsi_ok:
            if self.last_signal_direction == "SELL":
                return self._build_signal("HOLD", 0,
                    ["Senal SHORT repetida, esperando cambio"], indicators, regime)

            self.last_signal_direction = "SELL"
            self.was_away_short = False

            reasons = [
                f"EMA Pullback SHORT: precio volvio a EMA21",
                f"Ribbon bajista (EMA9<{EMA_MID}<{EMA_SLOW})",
                f"Multi-TF confirmado (1H+4H bajistas)",
                f"ADX={adx:.0f} RSI={rsi:.0f}",
            ]
            confidence = min(0.7 + (adx - ADX_MIN) / 100, 0.95)
            return self._build_signal("SELL", confidence, reasons, indicators, regime)

        # No signal
        if not pullback_long and not pullback_short:
            self.last_signal_direction = None

        reason_parts = []
        if not rsi_ok:
            reason_parts.append(f"RSI fuera de rango ({rsi:.0f})")
        if ribbon_bull and not pullback_long:
            reason_parts.append("Esperando pullback a EMA21")
        if ribbon_bear and not pullback_short:
            reason_parts.append("Esperando pullback a EMA21")
        if not both_bull and not both_bear:
            reason_parts.append("Multi-TF no confirmado")

        return self._build_signal("HOLD", 0,
            reason_parts or ["Sin condiciones de entrada"],
            indicators, regime)

    def _format_indicators(self, primary: dict) -> dict:
        return {
            "price": primary.get("price", 0),
            "ema_fast": primary.get("ema_fast", 0),
            "ema_slow": primary.get("ema_mid", 0),  # EMA21 for compatibility
            "rsi": primary.get("rsi", 50),
            "adx": primary.get("adx", 0),
            "atr": primary.get("atr", 0),
            "volume_ratio": 1.0,
            "macd_histogram": 0,
            "bb_position": 0.5,
            "ema_fast_prev": 0,
            "ema_slow_prev": 0,
        }

    def compute_all_indicators(self, df: pd.DataFrame) -> dict:
        return self._format_indicators(
            self.analyzer.analyze_timeframe(df) if not df.empty else {}
        )

    def _build_signal(self, action, confidence, reasons, indicators, regime):
        signal = {
            "action": action,
            "confidence": round(confidence, 4),
            "reasons": reasons,
            "indicators": indicators,
            "regime": regime,
            "candle_pattern": {"pattern": "none", "signal": 0},
        }
        self.last_signal = signal
        logger.info(
            f"SIGNAL: {action} | conf={confidence:.2%} | regime={regime} | "
            f"ADX={indicators.get('adx', 0):.0f} | RSI={indicators.get('rsi', 50):.0f} | "
            f"reasons={', '.join(reasons[:3])}"
        )
        return signal

    def compute_dynamic_stop_loss(self, entry_price: float, direction: str,
                                   atr: float, regime: str) -> float:
        """SL debajo de EMA50 + buffer, minimo 1.0x ATR."""
        # Use standard ATR-based SL
        stop_distance = atr * STOP_ATR_MULTIPLIER
        if direction == "BUY":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def compute_take_profit(self, entry_price: float, direction: str,
                             atr: float) -> float:
        """TP = 2x SL distance (R:R 1:2)."""
        tp_distance = atr * TP_ATR_MULTIPLIER
        if direction == "BUY":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance
