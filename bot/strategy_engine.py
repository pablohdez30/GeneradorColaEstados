"""
strategy_engine.py - Estrategia Trend Following simplificada.

Filosofía: MENOS ES MÁS
─────────────────────────
Solo 2 indicadores para decidir + 1 filtro:
1. EMA 9/21 cross → dirección del trade
2. ADX > 20 → confirma que hay tendencia (si no, NO operar)
3. RSI → filtro de seguridad (no comprar sobrecomprado, no vender sobrevendido)

Stop loss y take profit basados en ATR con ratio 1:2.
"""

import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EMA_FAST, EMA_SLOW,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    ADX_PERIOD, ADX_MIN_TREND,
    ATR_PERIOD, STOP_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
    REGIME_LOOKBACK, VOLATILITY_HIGH_THRESHOLD, TREND_ADX_THRESHOLD,
)
from bot.logger import setup_logger

logger = setup_logger("strategy")


# ══════════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS
# ══════════════════════════════════════════════════════════════════

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
    adx = dx.ewm(alpha=1 / period).mean()
    return adx


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN DE RÉGIMEN DE MERCADO
# ══════════════════════════════════════════════════════════════════

def detect_market_regime(df: pd.DataFrame) -> str:
    if len(df) < REGIME_LOOKBACK:
        return "UNKNOWN"

    recent = df.tail(REGIME_LOOKBACK)
    returns = recent["close"].pct_change().dropna()
    volatility = returns.std() * np.sqrt(len(returns))

    if volatility > VOLATILITY_HIGH_THRESHOLD:
        return "HIGH_VOLATILITY"

    adx = compute_adx(recent)
    if not adx.empty and adx.iloc[-1] > TREND_ADX_THRESHOLD:
        return "TRENDING"

    return "RANGING"


# ══════════════════════════════════════════════════════════════════
# MOTOR DE ESTRATEGIA
# ══════════════════════════════════════════════════════════════════

class StrategyEngine:
    def __init__(self, ml_weight: float = 0.0):
        self.ml_weight = ml_weight
        self.last_signal = None
        self.last_indicators = {}
        self.prev_ema_fast = None
        self.prev_ema_slow = None
        logger.info(f"StrategyEngine inicializado | ML weight: {ml_weight}")

    def compute_all_indicators(self, df: pd.DataFrame) -> dict:
        close = df["close"]

        ema_fast = compute_ema(close, EMA_FAST)
        ema_slow = compute_ema(close, EMA_SLOW)
        rsi = compute_rsi(close)
        atr = compute_atr(df)
        adx = compute_adx(df)

        indicators = {
            "ema_fast": round(ema_fast.iloc[-1], 2),
            "ema_slow": round(ema_slow.iloc[-1], 2),
            "ema_fast_prev": round(ema_fast.iloc[-2], 2) if len(ema_fast) > 1 else 0,
            "ema_slow_prev": round(ema_slow.iloc[-2], 2) if len(ema_slow) > 1 else 0,
            "rsi": round(rsi.iloc[-1], 2) if not rsi.empty else 50,
            "atr": round(atr.iloc[-1], 2) if not atr.empty else 0,
            "adx": round(adx.iloc[-1], 2) if not adx.empty else 0,
            "price": round(close.iloc[-1], 2),
            # Legacy fields for ML module compatibility
            "macd_histogram": 0,
            "bb_position": 0.5,
            "volume_ratio": 1.0,
        }
        self.last_indicators = indicators
        return indicators

    def generate_signal(self, df: pd.DataFrame, order_book: dict = None,
                        ml_prediction: float = 0.0, sentiment: float = 0.5) -> dict:
        """
        Genera señal basada en EMA cross + ADX filter + RSI safety.

        Reglas simples:
        1. ADX > 20 → hay tendencia → podemos operar
        2. EMA 9 cruza por encima de EMA 21 → BUY
        3. EMA 9 cruza por debajo de EMA 21 → SELL
        4. RSI < 30 → no SHORT (probable rebote)
        5. RSI > 70 → no LONG (probable caída)
        """
        min_data = max(EMA_SLOW, ADX_PERIOD, RSI_PERIOD) + 10
        if len(df) < min_data:
            return {"action": "HOLD", "confidence": 0, "reasons": ["Datos insuficientes"],
                    "indicators": {}, "regime": "UNKNOWN", "candle_pattern": {}}

        indicators = self.compute_all_indicators(df)
        regime = detect_market_regime(df)

        reasons = []
        action = "HOLD"
        confidence = 0.0

        adx = indicators["adx"]
        rsi = indicators["rsi"]
        ema_f = indicators["ema_fast"]
        ema_s = indicators["ema_slow"]
        ema_f_prev = indicators["ema_fast_prev"]
        ema_s_prev = indicators["ema_slow_prev"]

        # ── 1. Filtro de tendencia: ADX debe confirmar ────────
        if adx < ADX_MIN_TREND:
            reasons.append(f"Sin tendencia (ADX={adx:.0f} < {ADX_MIN_TREND})")
            return self._build_signal("HOLD", 0, reasons, indicators, regime)

        reasons.append(f"Tendencia confirmada (ADX={adx:.0f})")

        # ── 2. EMA Cross: detectar cruce ──────────────────────
        ema_cross_up = ema_f_prev <= ema_s_prev and ema_f > ema_s
        ema_cross_down = ema_f_prev >= ema_s_prev and ema_f < ema_s
        ema_above = ema_f > ema_s
        ema_below = ema_f < ema_s

        if ema_cross_up:
            action = "BUY"
            confidence = 0.70
            reasons.append("EMA 9 cruza ENCIMA de EMA 21 (cruce alcista)")
        elif ema_cross_down:
            action = "SELL"
            confidence = 0.70
            reasons.append("EMA 9 cruza DEBAJO de EMA 21 (cruce bajista)")
        elif ema_above:
            # No hay cruce nuevo pero EMA sigue alcista → señal más débil
            action = "BUY"
            confidence = 0.40
            reasons.append("EMA 9 > EMA 21 (tendencia alcista activa)")
        elif ema_below:
            action = "SELL"
            confidence = 0.40
            reasons.append("EMA 9 < EMA 21 (tendencia bajista activa)")

        # ── 3. Filtro RSI: no operar contra extremos ──────────
        if action == "BUY" and rsi > RSI_OVERBOUGHT:
            reasons.append(f"LONG bloqueado: RSI sobrecompra ({rsi:.0f})")
            return self._build_signal("HOLD", 0, reasons, indicators, regime)

        if action == "SELL" and rsi < RSI_OVERSOLD:
            reasons.append(f"SHORT bloqueado: RSI sobreventa ({rsi:.0f})")
            return self._build_signal("HOLD", 0, reasons, indicators, regime)

        # RSI en zona favorable → boost de confianza
        if action == "BUY" and rsi < 50:
            confidence += 0.10
            reasons.append(f"RSI favorable para LONG ({rsi:.0f})")
        elif action == "SELL" and rsi > 50:
            confidence += 0.10
            reasons.append(f"RSI favorable para SHORT ({rsi:.0f})")

        # ── 4. Régimen: no operar en alta volatilidad ─────────
        if regime == "HIGH_VOLATILITY":
            confidence *= 0.7
            reasons.append("Alta volatilidad: confianza reducida")

        # ── 5. ML (si está entrenado) ─────────────────────────
        if self.ml_weight > 0:
            if action == "BUY" and ml_prediction > 0.6:
                confidence += self.ml_weight * 0.5
            elif action == "SELL" and ml_prediction < 0.4:
                confidence += self.ml_weight * 0.5

        # ── 6. Sentimiento (muy bajo peso) ────────────────────
        from config import SENTIMENT_WEIGHT, ENABLE_SENTIMENT
        if ENABLE_SENTIMENT:
            if sentiment < 25 and action == "BUY":
                confidence += SENTIMENT_WEIGHT
            elif sentiment > 75 and action == "SELL":
                confidence += SENTIMENT_WEIGHT

        confidence = min(confidence, 1.0)

        # Mínimo 35% de confianza para operar
        if confidence < 0.35:
            reasons.append(f"Confianza insuficiente ({confidence:.0%})")
            action = "HOLD"
            confidence = 0.0

        return self._build_signal(action, confidence, reasons, indicators, regime)

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
        """Stop loss basado en ATR. Ratio 1:2 con take profit."""
        stop_distance = atr * STOP_ATR_MULTIPLIER

        if direction == "BUY":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def compute_take_profit(self, entry_price: float, direction: str,
                             atr: float) -> float:
        """Take profit = 2x el stop loss distance (ratio 1:2)."""
        tp_distance = atr * TP_ATR_MULTIPLIER

        if direction == "BUY":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance
