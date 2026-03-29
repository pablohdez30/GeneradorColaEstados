"""
strategy_engine.py - Lógica de señales y condiciones de entrada/salida.

Diseño de la estrategia combinada (Multi-Indicator Confluence):
───────────────────────────────────────────────────────────────
Se requiere CONFLUENCIA de al menos 2 de 4 indicadores para generar señal.
Esto reduce los falsos positivos típicos de depender de un solo indicador.

Indicadores utilizados:
1. RSI (14): Identifica zonas de sobreventa/sobrecompra
2. MACD (12,26,9): Detecta cambios de momentum
3. Bollinger Bands (20,2): Identifica reversiones desde extremos
4. EMA Cross (9/21): Confirma dirección de tendencia

Señales adicionales:
- Patrones de velas japonesas (hammer, engulfing, doji)
- Volumen relativo (spikes indican interés institucional)
- Order book imbalance (presión compradora/vendedora)

Régimen de mercado:
- TRENDING: Se priorizan señales de momentum (MACD, EMA cross)
- RANGING: Se priorizan señales de reversión (RSI extremos, BB bounce)
- HIGH_VOL: Se amplían stops y se reduce tamaño de posición
"""

import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD,
    EMA_FAST, EMA_SLOW,
    REGIME_LOOKBACK, VOLATILITY_HIGH_THRESHOLD, TREND_ADX_THRESHOLD,
    VOLUME_SPIKE_THRESHOLD, VOLUME_MA_PERIOD,
)
from bot.logger import setup_logger

logger = setup_logger("strategy")


# ══════════════════════════════════════════════════════════════════
# CÁLCULO DE INDICADORES TÉCNICOS (sin TA-Lib, implementación pura)
# Decisión: Implementación propia para evitar dependencia de compilación
# de TA-Lib que falla en muchos entornos. Misma lógica matemática.
# ══════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI - Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD - Moving Average Convergence Divergence. Retorna (macd, signal, histogram)."""
    ema_fast = series.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = series.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Retorna (upper, middle, lower)."""
    middle = series.rolling(BB_PERIOD).mean()
    std = series.rolling(BB_PERIOD).std()
    upper = middle + BB_STD * std
    lower = middle - BB_STD * std
    return upper, middle, lower


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - usado para stops dinámicos."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index - fuerza de la tendencia."""
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
# DETECCIÓN DE PATRONES DE VELAS JAPONESAS
# ══════════════════════════════════════════════════════════════════

def detect_candle_patterns(df: pd.DataFrame) -> dict:
    """
    Detecta patrones de velas en las últimas 3 velas.

    Retorna dict con nombre del patrón y señal (1=bullish, -1=bearish, 0=neutral).
    """
    if len(df) < 3:
        return {"pattern": "none", "signal": 0}

    c = df.iloc[-1]   # Vela actual
    p = df.iloc[-2]   # Vela anterior
    pp = df.iloc[-3]  # Dos velas atrás

    body = abs(c["close"] - c["open"])
    upper_shadow = c["high"] - max(c["close"], c["open"])
    lower_shadow = min(c["close"], c["open"]) - c["low"]
    total_range = c["high"] - c["low"]

    if total_range == 0:
        return {"pattern": "doji", "signal": 0}

    # Doji: cuerpo muy pequeño
    if body / total_range < 0.1:
        return {"pattern": "doji", "signal": 0}

    # Hammer (bullish): sombra inferior larga, cuerpo pequeño arriba
    if lower_shadow > body * 2 and upper_shadow < body * 0.5:
        if p["close"] < p["open"]:  # Tras vela bajista
            return {"pattern": "hammer", "signal": 1}

    # Shooting Star (bearish): sombra superior larga, cuerpo pequeño abajo
    if upper_shadow > body * 2 and lower_shadow < body * 0.5:
        if p["close"] > p["open"]:  # Tras vela alcista
            return {"pattern": "shooting_star", "signal": -1}

    # Bullish Engulfing
    if (p["close"] < p["open"] and  # Vela anterior bajista
            c["close"] > c["open"] and  # Vela actual alcista
            c["open"] <= p["close"] and
            c["close"] >= p["open"]):
        return {"pattern": "bullish_engulfing", "signal": 1}

    # Bearish Engulfing
    if (p["close"] > p["open"] and  # Vela anterior alcista
            c["close"] < c["open"] and  # Vela actual bajista
            c["open"] >= p["close"] and
            c["close"] <= p["open"]):
        return {"pattern": "bearish_engulfing", "signal": -1}

    # Morning Star (bullish reversal - 3 velas)
    if (pp["close"] < pp["open"] and  # Primera bajista
            abs(p["close"] - p["open"]) / (p["high"] - p["low"] + 1e-10) < 0.3 and  # Segunda: cuerpo pequeño
            c["close"] > c["open"] and  # Tercera alcista
            c["close"] > (pp["open"] + pp["close"]) / 2):
        return {"pattern": "morning_star", "signal": 1}

    # Evening Star (bearish reversal - 3 velas)
    if (pp["close"] > pp["open"] and
            abs(p["close"] - p["open"]) / (p["high"] - p["low"] + 1e-10) < 0.3 and
            c["close"] < c["open"] and
            c["close"] < (pp["open"] + pp["close"]) / 2):
        return {"pattern": "evening_star", "signal": -1}

    return {"pattern": "none", "signal": 0}


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN DE RÉGIMEN DE MERCADO
# ══════════════════════════════════════════════════════════════════

def detect_market_regime(df: pd.DataFrame) -> str:
    """
    Clasifica el mercado en: TRENDING, RANGING, HIGH_VOLATILITY.

    Lógica:
    - ADX > 25 → TRENDING (hay tendencia fuerte)
    - Volatilidad > 2% → HIGH_VOLATILITY (ajustar stops)
    - Else → RANGING (operar reversiones)
    """
    if len(df) < REGIME_LOOKBACK:
        return "UNKNOWN"

    recent = df.tail(REGIME_LOOKBACK)

    # Volatilidad: desviación estándar de retornos
    returns = recent["close"].pct_change().dropna()
    volatility = returns.std() * np.sqrt(len(returns))

    if volatility > VOLATILITY_HIGH_THRESHOLD:
        return "HIGH_VOLATILITY"

    # ADX para detectar tendencia
    adx = compute_adx(recent)
    if not adx.empty and adx.iloc[-1] > TREND_ADX_THRESHOLD:
        return "TRENDING"

    return "RANGING"


# ══════════════════════════════════════════════════════════════════
# MOTOR DE ESTRATEGIA PRINCIPAL
# ══════════════════════════════════════════════════════════════════

class StrategyEngine:
    """
    Motor de señales multi-indicador con confluencia.

    Genera señales de trading basadas en la confluencia de múltiples
    indicadores técnicos, patrones de velas y análisis de volumen.

    Cada señal tiene un score de confianza (0-1) basado en cuántos
    indicadores confirman la dirección.
    """

    def __init__(self, ml_weight: float = 0.0):
        """
        ml_weight: peso de la predicción del modelo ML en la decisión final.
        Empieza en 0 (sin ML) y se incrementa conforme el modelo mejora.
        """
        self.ml_weight = ml_weight
        self.last_signal = None
        self.last_indicators = {}
        logger.info(f"StrategyEngine inicializado | ML weight: {ml_weight}")

    def compute_all_indicators(self, df: pd.DataFrame) -> dict:
        """Calcula todos los indicadores técnicos sobre el DataFrame de velas."""
        close = df["close"]

        # RSI
        rsi = compute_rsi(close)
        rsi_value = rsi.iloc[-1] if not rsi.empty else 50

        # MACD
        macd_line, signal_line, histogram = compute_macd(close)
        macd_val = macd_line.iloc[-1] if not macd_line.empty else 0
        signal_val = signal_line.iloc[-1] if not signal_line.empty else 0
        hist_val = histogram.iloc[-1] if not histogram.empty else 0

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close)
        price = close.iloc[-1]
        bb_position = 0
        if not bb_upper.empty and not bb_lower.empty:
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            if bb_range > 0:
                bb_position = (price - bb_lower.iloc[-1]) / bb_range

        # EMAs
        ema_fast = compute_ema(close, EMA_FAST)
        ema_slow = compute_ema(close, EMA_SLOW)
        ema_fast_val = ema_fast.iloc[-1] if not ema_fast.empty else price
        ema_slow_val = ema_slow.iloc[-1] if not ema_slow.empty else price

        # ATR (para stops dinámicos)
        atr = compute_atr(df)
        atr_val = atr.iloc[-1] if not atr.empty else 0

        # Volumen relativo
        vol_ma = df["volume"].rolling(VOLUME_MA_PERIOD).mean()
        vol_ratio = df["volume"].iloc[-1] / vol_ma.iloc[-1] if not vol_ma.empty and vol_ma.iloc[-1] > 0 else 1.0

        indicators = {
            "rsi": round(rsi_value, 2),
            "macd": round(macd_val, 4),
            "macd_signal": round(signal_val, 4),
            "macd_histogram": round(hist_val, 4),
            "bb_upper": round(bb_upper.iloc[-1], 2) if not bb_upper.empty else 0,
            "bb_middle": round(bb_middle.iloc[-1], 2) if not bb_middle.empty else 0,
            "bb_lower": round(bb_lower.iloc[-1], 2) if not bb_lower.empty else 0,
            "bb_position": round(bb_position, 4),
            "ema_fast": round(ema_fast_val, 2),
            "ema_slow": round(ema_slow_val, 2),
            "atr": round(atr_val, 2),
            "volume_ratio": round(vol_ratio, 2),
            "price": round(price, 2),
        }
        self.last_indicators = indicators
        return indicators

    def generate_signal(self, df: pd.DataFrame, order_book: dict = None,
                        ml_prediction: float = 0.0, sentiment: float = 0.5,
                        df_5m: pd.DataFrame = None) -> dict:
        """
        Genera señal de trading basada en confluencia de indicadores.

        Retorna:
        {
            "action": "BUY" | "SELL" | "HOLD",
            "confidence": float (0-1),
            "reasons": list[str],
            "indicators": dict,
            "regime": str,
            "candle_pattern": dict,
        }
        """
        if len(df) < max(BB_PERIOD, MACD_SLOW, REGIME_LOOKBACK) + 10:
            return {"action": "HOLD", "confidence": 0, "reasons": ["Datos insuficientes"],
                    "indicators": {}, "regime": "UNKNOWN", "candle_pattern": {}}

        indicators = self.compute_all_indicators(df)
        regime = detect_market_regime(df)
        candle = detect_candle_patterns(df)

        buy_signals = []
        sell_signals = []
        reasons = []

        # ── 1. RSI ─────────────────────────────────────────────
        if indicators["rsi"] < RSI_OVERSOLD:
            buy_signals.append(("RSI", 0.25))
            reasons.append(f"RSI sobreventa ({indicators['rsi']})")
        elif indicators["rsi"] > RSI_OVERBOUGHT:
            sell_signals.append(("RSI", 0.25))
            reasons.append(f"RSI sobrecompra ({indicators['rsi']})")

        # ── 2. MACD Crossover ──────────────────────────────────
        if indicators["macd"] > indicators["macd_signal"] and indicators["macd_histogram"] > 0:
            buy_signals.append(("MACD", 0.25))
            reasons.append("MACD cruce alcista")
        elif indicators["macd"] < indicators["macd_signal"] and indicators["macd_histogram"] < 0:
            sell_signals.append(("MACD", 0.25))
            reasons.append("MACD cruce bajista")

        # ── 3. Bollinger Bands ─────────────────────────────────
        if indicators["bb_position"] < 0.15:  # Precio cerca de banda inferior
            buy_signals.append(("BB", 0.20))
            reasons.append("Precio en banda inferior BB")
        elif indicators["bb_position"] > 0.85:  # Precio cerca de banda superior
            sell_signals.append(("BB", 0.20))
            reasons.append("Precio en banda superior BB")

        # ── 4. EMA Cross ───────────────────────────────────────
        if indicators["ema_fast"] > indicators["ema_slow"]:
            buy_signals.append(("EMA", 0.15))
            reasons.append("EMA rápida > EMA lenta")
        elif indicators["ema_fast"] < indicators["ema_slow"]:
            sell_signals.append(("EMA", 0.15))
            reasons.append("EMA rápida < EMA lenta")

        # ── 5. Patrones de Velas ───────────────────────────────
        if candle["signal"] == 1:
            buy_signals.append(("CANDLE", 0.10))
            reasons.append(f"Patrón alcista: {candle['pattern']}")
        elif candle["signal"] == -1:
            sell_signals.append(("CANDLE", 0.10))
            reasons.append(f"Patrón bajista: {candle['pattern']}")

        # ── 6. Volumen ─────────────────────────────────────────
        if indicators["volume_ratio"] > VOLUME_SPIKE_THRESHOLD:
            reasons.append(f"Spike volumen ({indicators['volume_ratio']:.1f}x)")
            # El volumen confirma la señal existente, no genera por sí solo

        # ── 7. Order Book Imbalance ────────────────────────────
        if order_book and order_book.get("imbalance", 1.0) > 1.5:
            buy_signals.append(("OB", 0.05))
            reasons.append(f"Order book bullish (imb={order_book['imbalance']:.2f})")
        elif order_book and order_book.get("imbalance", 1.0) < 0.67:
            sell_signals.append(("OB", 0.05))
            reasons.append(f"Order book bearish (imb={order_book['imbalance']:.2f})")

        # ── Calcular score de confluencia ──────────────────────
        buy_score = sum(weight for _, weight in buy_signals)
        sell_score = sum(weight for _, weight in sell_signals)

        # Ajustar según régimen
        if regime == "TRENDING":
            # En tendencia, dar más peso a MACD y EMA
            for name, weight in buy_signals:
                if name in ("MACD", "EMA"):
                    buy_score += weight * 0.2
            for name, weight in sell_signals:
                if name in ("MACD", "EMA"):
                    sell_score += weight * 0.2
        elif regime == "RANGING":
            # En rango, dar más peso a RSI y BB
            for name, weight in buy_signals:
                if name in ("RSI", "BB"):
                    buy_score += weight * 0.2
            for name, weight in sell_signals:
                if name in ("RSI", "BB"):
                    sell_score += weight * 0.2

        # ── Incorporar ML y sentimiento ────────────────────────
        if self.ml_weight > 0:
            if ml_prediction > 0.5:
                buy_score += self.ml_weight * (ml_prediction - 0.5) * 2
            elif ml_prediction < 0.5:
                sell_score += self.ml_weight * (0.5 - ml_prediction) * 2

        from config import SENTIMENT_WEIGHT, ENABLE_SENTIMENT
        if ENABLE_SENTIMENT:
            # sentiment: 0=extreme fear, 100=extreme greed
            if sentiment < 25:  # Fear = oportunidad de compra contrarian
                buy_score += SENTIMENT_WEIGHT
            elif sentiment > 75:  # Greed = oportunidad de venta
                sell_score += SENTIMENT_WEIGHT

        # ── 8. Confirmación Multi-Timeframe (5m) ──────────────
        htf = self.check_higher_timeframe(df_5m) if df_5m is not None else {"trend": "NEUTRAL", "strength": 0.0}

        if htf["trend"] == "BULLISH":
            buy_score += 0.05  # Bonus reducido (era 0.10, demasiado dominante)
            reasons.append(f"5m confirma alcista ({htf['strength']:.0%})")
        elif htf["trend"] == "BEARISH":
            sell_score += 0.05
            reasons.append(f"5m confirma bajista ({htf['strength']:.0%})")

        # Penalizar señales contra la tendencia de 5m (reducido de 30% a 15%)
        if htf["trend"] == "BULLISH" and sell_score > buy_score:
            sell_score *= 0.85
            reasons.append("SHORT penalizado: contra tendencia 5m")
        elif htf["trend"] == "BEARISH" and buy_score > sell_score:
            buy_score *= 0.85
            reasons.append("LONG penalizado: contra tendencia 5m")

        # ── 9. Protección RSI: no operar contra extremos ─────
        # Si RSI está en sobreventa (<30), NO shortear (probable rebote)
        # Si RSI está en sobrecompra (>70), NO comprar (probable caída)
        rsi_val = indicators["rsi"]
        if rsi_val < 30:
            sell_score *= 0.3  # Penalizar shorts un 70% en sobreventa extrema
            reasons.append(f"SHORT bloqueado: RSI sobreventa extrema ({rsi_val:.0f})")
        elif rsi_val < RSI_OVERSOLD:
            sell_score *= 0.6  # Penalizar shorts un 40% en sobreventa
            reasons.append(f"SHORT penalizado: RSI sobreventa ({rsi_val:.0f})")
        elif rsi_val > 70:
            buy_score *= 0.3  # Penalizar longs un 70% en sobrecompra extrema
            reasons.append(f"LONG bloqueado: RSI sobrecompra extrema ({rsi_val:.0f})")
        elif rsi_val > RSI_OVERBOUGHT:
            buy_score *= 0.6  # Penalizar longs un 40% en sobrecompra
            reasons.append(f"LONG penalizado: RSI sobrecompra ({rsi_val:.0f})")

        # ── Decisión final ─────────────────────────────────────
        min_confidence = 0.30  # Mínimo 30% de confluencia para operar
        min_margin = 1.3  # La señal ganadora debe ser 30% más fuerte que la contraria

        # En RANGING exigir más confluencia (mercado lateral = más ruido)
        if regime == "RANGING":
            min_confidence = 0.40
            min_margin = 1.5  # 50% más fuerte que la contraria

        # Comparar BUY vs SELL: solo operar si hay dirección clara
        if buy_score >= min_confidence and buy_score > sell_score * min_margin:
            action = "BUY"
            confidence = min(buy_score, 1.0)
        elif sell_score >= min_confidence and sell_score > buy_score * min_margin:
            action = "SELL"
            confidence = min(sell_score, 1.0)
        else:
            action = "HOLD"
            confidence = 0.0
            if buy_score > 0 and sell_score > 0:
                reasons.append("Señales contradictorias, esperando claridad")
            else:
                reasons.append("Confluencia insuficiente")

        # Volumen confirma la señal (boost confidence)
        if action != "HOLD" and indicators["volume_ratio"] > VOLUME_SPIKE_THRESHOLD:
            confidence = min(confidence * 1.15, 1.0)

        signal = {
            "action": action,
            "confidence": round(confidence, 4),
            "reasons": reasons,
            "indicators": indicators,
            "regime": regime,
            "candle_pattern": candle,
        }

        self.last_signal = signal
        logger.info(
            f"SIGNAL: {action} | conf={confidence:.2%} | regime={regime} | "
            f"reasons={', '.join(reasons[:3])}"
        )
        return signal

    def check_higher_timeframe(self, df_5m: pd.DataFrame) -> dict:
        """
        Analiza timeframe superior (5m) para confirmar tendencia general.

        Retorna:
        {
            "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
            "strength": float (0-1),
        }
        """
        if df_5m.empty or len(df_5m) < MACD_SLOW + 5:
            return {"trend": "NEUTRAL", "strength": 0.0}

        close = df_5m["close"]

        # EMA trend en 5m
        ema_f = compute_ema(close, EMA_FAST)
        ema_s = compute_ema(close, EMA_SLOW)
        ema_bullish = ema_f.iloc[-1] > ema_s.iloc[-1]

        # MACD direction en 5m
        macd_line, signal_line, hist = compute_macd(close)
        macd_bullish = hist.iloc[-1] > 0

        # RSI en 5m
        rsi = compute_rsi(close)
        rsi_val = rsi.iloc[-1] if not rsi.empty else 50

        if ema_bullish and macd_bullish:
            strength = 0.8 + (0.2 if rsi_val > 50 else 0.0)
            return {"trend": "BULLISH", "strength": round(strength, 2)}
        elif not ema_bullish and not macd_bullish:
            strength = 0.8 + (0.2 if rsi_val < 50 else 0.0)
            return {"trend": "BEARISH", "strength": round(strength, 2)}
        else:
            return {"trend": "NEUTRAL", "strength": 0.3}

    def compute_dynamic_stop_loss(self, entry_price: float, direction: str,
                                   atr: float, regime: str) -> float:
        """
        Calcula stop-loss dinámico basado en ATR y régimen.

        En alta volatilidad se amplía el stop para evitar que
        el ruido normal del mercado lo active prematuramente.
        """
        multiplier = 1.0  # Base: 1x ATR
        if regime == "HIGH_VOLATILITY":
            multiplier = 1.5
        elif regime == "TRENDING":
            multiplier = 1.2

        stop_distance = atr * multiplier

        if direction == "BUY":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def compute_take_profit(self, entry_price: float, direction: str,
                             atr: float, regime: str = "UNKNOWN") -> float:
        """
        Take profit basado en ATR con ratio riesgo:beneficio POSITIVO.
        TP siempre >= SL para que las ganancias compensen las pérdidas.
        """
        # TP = 1.5x a 2x el stop distance (R:R de 1:1.5 a 1:2)
        if regime == "HIGH_VOLATILITY":
            tp_distance = atr * 2.5  # Más margen en alta vol
        elif regime == "TRENDING":
            tp_distance = atr * 2.0  # Dejar correr en tendencia
        else:
            tp_distance = atr * 1.5  # RANGING: TP más cercano pero > SL

        if direction == "BUY":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    @staticmethod
    def compute_adaptive_tp_levels(atr: float, entry_price: float, regime: str) -> list[tuple[float, float]]:
        """
        Take-profit: cerrar 100% en un solo nivel.
        Sin escalonado — simplifica y asegura capturar el beneficio completo.
        """
        atr_pct = atr / entry_price if entry_price > 0 else 0.003

        if regime == "HIGH_VOLATILITY":
            return [(atr_pct * 2.5, 1.0)]  # Cierre total a 2.5x ATR
        elif regime == "TRENDING":
            return [(atr_pct * 2.0, 1.0)]  # Cierre total a 2x ATR
        else:
            return [(atr_pct * 1.5, 1.0)]  # Cierre total a 1.5x ATR
