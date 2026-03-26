"""
strategy_engine.py — Motor de señales y estrategia de scalping
================================================================
Implementa la lógica de generación de señales combinando:

  1. RSI (14) — sobrecompra/sobreventa
  2. MACD (12/26/9) — momentum y cruce de señal
  3. Bollinger Bands (20, 2σ) — volatilidad y reversión a media
  4. EMA 9/21/50 — tendencia a corto/medio plazo
  5. ATR (14) — volatilidad real para stops dinámicos
  6. Stochastic RSI — confirmación adicional en scalping
  7. Volume analysis — confirmación de movimiento con volumen
  8. Order Book imbalance — presión compradora/vendedora
  9. Detección de patrones de velas japonesas

Estrategia principal:
  LONG: RSI < 35, MACD cruzando al alza, precio cerca de BB inferior,
        EMA rápida sobre EMA lenta, volumen en spike, imbalance > 0.1
  SHORT: Inverso de las condiciones anteriores

Régimen de mercado: ajusta umbrales automáticamente según si el mercado
  está en tendencia, rango lateral o alta volatilidad.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum

from config import CONFIG
from logger import LOG, EventType

# Intentar importar TA-Lib nativo (más rápido), fallback a pandas_ta
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    try:
        import pandas_ta as ta
        PANDAS_TA_AVAILABLE = True
    except ImportError:
        PANDAS_TA_AVAILABLE = False


# ─────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────
class Signal(str, Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────
#  RESULTADO DE ANÁLISIS
# ─────────────────────────────────────────────
@dataclass
class AnalysisResult:
    signal: Signal = Signal.HOLD
    confidence: float = 0.0        # [0, 1] — qué tan fuerte es la señal
    reason: str = ""
    regime: MarketRegime = MarketRegime.UNKNOWN

    # Valores de indicadores para logging/ML
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5       # 0=en lower, 1=en upper
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_trend: float = 0.0
    atr: float = 0.0
    atr_normalized: float = 0.0   # ATR / precio
    volume_ratio: float = 1.0     # volumen actual / media
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    adx: float = 0.0              # fuerza de tendencia
    ob_imbalance: float = 0.0     # order book imbalance
    candle_pattern: str = "none"  # patrón de vela detectado
    price: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "rsi": self.rsi,
            "macd_hist": self.macd_hist,
            "bb_position": self.bb_position,
            "ema_cross": "bullish" if self.ema_fast > self.ema_slow else "bearish",
            "volume_ratio": self.volume_ratio,
            "atr_normalized": self.atr_normalized,
            "market_regime": self.regime.value,
            "adx": self.adx,
            "stoch_k": self.stoch_k,
            "ob_imbalance": self.ob_imbalance,
            "candle_pattern": self.candle_pattern,
        }


# ─────────────────────────────────────────────
#  CÁLCULO DE INDICADORES
# ─────────────────────────────────────────────
class IndicatorCalculator:
    """Calcula indicadores técnicos usando TA-Lib o pandas_ta."""

    def __init__(self):
        self.cfg = CONFIG.indicators

    def compute(self, df: pd.DataFrame) -> Optional[AnalysisResult]:
        """
        Calcula todos los indicadores sobre el DataFrame de velas.
        Devuelve None si no hay suficientes datos.
        """
        if df is None or len(df) < self.cfg.min_candles:
            return None

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        result = AnalysisResult()
        result.price = float(close[-1])

        if TALIB_AVAILABLE:
            self._compute_talib(result, close, high, low, volume)
        elif PANDAS_TA_AVAILABLE:
            self._compute_pandas_ta(result, df)
        else:
            self._compute_manual(result, close, high, low, volume)

        return result

    def _compute_talib(self, r: AnalysisResult, close, high, low, volume):
        """Cálculo usando TA-Lib (más rápido, C nativo)."""
        cfg = self.cfg

        # RSI
        rsi = talib.RSI(close, timeperiod=cfg.rsi_period)
        r.rsi = float(rsi[-1])

        # MACD
        macd, signal, hist = talib.MACD(
            close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )
        r.macd = float(macd[-1])
        r.macd_signal = float(signal[-1])
        r.macd_hist = float(hist[-1])

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close, timeperiod=cfg.bb_period, nbdevup=cfg.bb_std, nbdevdn=cfg.bb_std
        )
        r.bb_upper = float(upper[-1])
        r.bb_middle = float(middle[-1])
        r.bb_lower = float(lower[-1])
        band_width = r.bb_upper - r.bb_lower
        if band_width > 0:
            r.bb_position = (r.price - r.bb_lower) / band_width
        else:
            r.bb_position = 0.5

        # EMAs
        r.ema_fast = float(talib.EMA(close, cfg.ema_fast)[-1])
        r.ema_slow = float(talib.EMA(close, cfg.ema_slow)[-1])
        r.ema_trend = float(talib.EMA(close, cfg.ema_trend)[-1])

        # ATR
        atr = talib.ATR(high, low, close, timeperiod=cfg.atr_period)
        r.atr = float(atr[-1])
        r.atr_normalized = r.atr / r.price if r.price > 0 else 0

        # Stochastic RSI
        fastk, fastd = talib.STOCHRSI(
            close, cfg.stoch_rsi_period, cfg.stoch_rsi_k, cfg.stoch_rsi_d
        )
        r.stoch_k = float(fastk[-1]) if not np.isnan(fastk[-1]) else 50.0
        r.stoch_d = float(fastd[-1]) if not np.isnan(fastd[-1]) else 50.0

        # ADX (fuerza de tendencia)
        adx = talib.ADX(high, low, close, timeperiod=14)
        r.adx = float(adx[-1]) if not np.isnan(adx[-1]) else 0.0

        # Volume ratio
        vol_ma = talib.SMA(volume, cfg.volume_ma_period)
        r.volume_ratio = (float(volume[-1]) / float(vol_ma[-1])
                          if vol_ma[-1] > 0 else 1.0)

        # Detección de patrones de velas
        r.candle_pattern = self._detect_candle_pattern_talib(
            high, low, close, volume
        )

    def _compute_pandas_ta(self, r: AnalysisResult, df: pd.DataFrame):
        """Cálculo usando pandas_ta (más portable)."""
        cfg = self.cfg
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        r.rsi = float(ta.rsi(close, length=cfg.rsi_period).iloc[-1])

        # MACD
        macd_df = ta.macd(close, fast=cfg.macd_fast, slow=cfg.macd_slow,
                          signal=cfg.macd_signal)
        if macd_df is not None:
            r.macd = float(macd_df.iloc[-1, 0])
            r.macd_signal = float(macd_df.iloc[-1, 2])
            r.macd_hist = float(macd_df.iloc[-1, 1])

        # Bollinger Bands
        bb_df = ta.bbands(close, length=cfg.bb_period, std=cfg.bb_std)
        if bb_df is not None:
            r.bb_upper = float(bb_df.iloc[-1]["BBU_20_2.0"])
            r.bb_middle = float(bb_df.iloc[-1]["BBM_20_2.0"])
            r.bb_lower = float(bb_df.iloc[-1]["BBL_20_2.0"])
            band_width = r.bb_upper - r.bb_lower
            r.bb_position = ((r.price - r.bb_lower) / band_width
                             if band_width > 0 else 0.5)

        # EMAs
        r.ema_fast = float(ta.ema(close, length=cfg.ema_fast).iloc[-1])
        r.ema_slow = float(ta.ema(close, length=cfg.ema_slow).iloc[-1])
        r.ema_trend = float(ta.ema(close, length=cfg.ema_trend).iloc[-1])

        # ATR
        atr_series = ta.atr(high, low, close, length=cfg.atr_period)
        if atr_series is not None:
            r.atr = float(atr_series.iloc[-1])
            r.atr_normalized = r.atr / r.price if r.price > 0 else 0

        # ADX
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None:
            r.adx = float(adx_df["ADX_14"].iloc[-1])

        # Volume ratio
        vol_ma = float(ta.sma(volume, length=cfg.volume_ma_period).iloc[-1])
        r.volume_ratio = float(volume.iloc[-1]) / vol_ma if vol_ma > 0 else 1.0

        # Stochastic RSI
        stoch_df = ta.stochrsi(close, length=cfg.stoch_rsi_period,
                                rsi_length=cfg.stoch_rsi_period,
                                k=cfg.stoch_rsi_k, d=cfg.stoch_rsi_d)
        if stoch_df is not None:
            r.stoch_k = float(stoch_df.iloc[-1, 0])
            r.stoch_d = float(stoch_df.iloc[-1, 1])

        # Patrones de velas (manual, sin talib)
        r.candle_pattern = self._detect_candle_pattern_manual(
            df["open"].values, df["high"].values,
            df["low"].values, df["close"].values
        )

    def _compute_manual(self, r: AnalysisResult, close, high, low, volume):
        """Implementación manual para cuando no hay TA-Lib ni pandas_ta."""
        cfg = self.cfg

        # RSI manual
        r.rsi = self._rsi_manual(close, cfg.rsi_period)

        # MACD manual
        ema_fast = self._ema_manual(close, cfg.macd_fast)
        ema_slow = self._ema_manual(close, cfg.macd_slow)
        macd_line = ema_fast - ema_slow
        r.macd = float(macd_line[-1])

        macd_signal = self._ema_manual(macd_line, cfg.macd_signal)
        r.macd_signal = float(macd_signal[-1])
        r.macd_hist = r.macd - r.macd_signal

        # Bollinger Bands manual
        sma = np.mean(close[-cfg.bb_period:])
        std = np.std(close[-cfg.bb_period:])
        r.bb_upper = sma + cfg.bb_std * std
        r.bb_middle = sma
        r.bb_lower = sma - cfg.bb_std * std
        bw = r.bb_upper - r.bb_lower
        r.bb_position = (r.price - r.bb_lower) / bw if bw > 0 else 0.5

        # EMAs
        r.ema_fast = float(self._ema_manual(close, cfg.ema_fast)[-1])
        r.ema_slow = float(self._ema_manual(close, cfg.ema_slow)[-1])
        r.ema_trend = float(self._ema_manual(close, cfg.ema_trend)[-1])

        # ATR manual
        r.atr = self._atr_manual(high, low, close, cfg.atr_period)
        r.atr_normalized = r.atr / r.price if r.price > 0 else 0

        # Volume ratio
        vol_ma = np.mean(volume[-cfg.volume_ma_period:])
        r.volume_ratio = float(volume[-1]) / vol_ma if vol_ma > 0 else 1.0

        r.candle_pattern = self._detect_candle_pattern_manual(
            close - 0.01, high, low, close  # aproximación sin open
        )

    @staticmethod
    def _ema_manual(data: np.ndarray, period: int) -> np.ndarray:
        k = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = data[i] * k + ema[i - 1] * (1 - k)
        return ema

    @staticmethod
    def _rsi_manual(close: np.ndarray, period: int) -> float:
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    @staticmethod
    def _atr_manual(high, low, close, period: int) -> float:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _detect_candle_pattern_talib(high, low, close, volume) -> str:
        """Detecta patrones de velas usando TA-Lib."""
        patterns = {
            "hammer": talib.CDLHAMMER(
                close - abs(close * 0.001), high, low, close
            )[-1],
            "doji": talib.CDLDOJI(
                close - abs(close * 0.001), high, low, close
            )[-1],
            "engulfing": talib.CDLENGULFING(
                close - abs(close * 0.001), high, low, close
            )[-1],
            "morning_star": talib.CDLMORNINGSTAR(
                close - abs(close * 0.001), high, low, close
            )[-1],
            "shooting_star": talib.CDLSHOOTINGSTAR(
                close - abs(close * 0.001), high, low, close
            )[-1],
            "pin_bar_bull": talib.CDLHIGHWAVE(
                close - abs(close * 0.001), high, low, close
            )[-1],
        }
        for name, val in patterns.items():
            if val != 0:
                return name
        return "none"

    @staticmethod
    def _detect_candle_pattern_manual(open_, high, low, close) -> str:
        """Detección manual de patrones para los últimas 3 velas."""
        if len(close) < 3:
            return "none"

        o, h, l, c = open_[-1], high[-1], low[-1], close[-1]
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        full_range = h - l if h != l else 1

        # Doji: cuerpo muy pequeño
        if body / full_range < 0.1:
            return "doji"

        # Hammer (señal alcista): wick inferior largo, cuerpo arriba
        if (lower_wick > body * 2 and upper_wick < body * 0.5
                and c > o):
            return "hammer"

        # Shooting star (señal bajista): wick superior largo
        if (upper_wick > body * 2 and lower_wick < body * 0.5
                and c < o):
            return "shooting_star"

        # Engulfing alcista: vela bajista anterior, vela alcista mayor
        o_prev, c_prev = open_[-2], close[-2]
        if c_prev < o_prev and c > o and c > o_prev and o < c_prev:
            return "bullish_engulfing"

        # Engulfing bajista
        if c_prev > o_prev and c < o and c < o_prev and o > c_prev:
            return "bearish_engulfing"

        return "none"


# ─────────────────────────────────────────────
#  DETECTOR DE RÉGIMEN DE MERCADO
# ─────────────────────────────────────────────
class RegimeDetector:
    """
    Clasifica el régimen de mercado actual en 4 categorías:
      - trending_up   : ADX > 25 y precio sobre EMA50
      - trending_down : ADX > 25 y precio bajo EMA50
      - ranging       : ADX < 20 y BB estrechas
      - volatile      : ATR normalizado > umbral
    """

    def detect(self, r: AnalysisResult) -> MarketRegime:
        # Alta volatilidad supera a todo lo demás
        if r.atr_normalized > 0.008:  # >0.8% de rango por vela
            return MarketRegime.VOLATILE

        # Mercado en tendencia fuerte
        if r.adx > 25:
            if r.price > r.ema_trend:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN

        # Mercado lateral
        if r.adx < 20:
            return MarketRegime.RANGING

        return MarketRegime.UNKNOWN


# ─────────────────────────────────────────────
#  MOTOR DE SEÑALES PRINCIPAL
# ─────────────────────────────────────────────
class StrategyEngine:
    """
    Motor central de señales. Combina todos los indicadores y el
    contexto de mercado para generar señales con puntuación de confianza.

    Sistema de puntuación:
      Cada condición alcista/bajista suma puntos ponderados (0-1).
      La señal se emite si la puntuación supera el umbral del régimen.
    """

    def __init__(self):
        self.cfg = CONFIG.indicators
        self.risk_cfg = CONFIG.risk
        self.calculator = IndicatorCalculator()
        self.regime_detector = RegimeDetector()
        self._prev_macd_hist: float = 0.0   # para detectar cruces
        self._prev_ema_fast: float = 0.0
        self._confirmation_count: int = 0   # velas consecutivas confirmando señal

    def analyze(self, df: pd.DataFrame,
                ob_imbalance: float = 0.0,
                fear_greed: int = 50,
                btc_dominance: float = 50.0) -> AnalysisResult:
        """
        Análisis completo de una vela cerrada.
        Devuelve AnalysisResult con la señal y todos los indicadores.
        """
        result = self.calculator.compute(df)
        if result is None:
            return AnalysisResult()

        # Enriquecer con datos externos
        result.ob_imbalance = ob_imbalance

        # Detectar régimen
        result.regime = self.regime_detector.detect(result)

        # Seleccionar umbrales según régimen
        thresholds = self._get_regime_thresholds(result.regime)

        # Calcular puntuación de señal
        long_score, short_score, reasons_long, reasons_short = \
            self._score_signals(result, thresholds, fear_greed, btc_dominance)

        # Seleccionar señal ganadora
        min_confidence = thresholds["min_confidence"]

        if long_score >= min_confidence and long_score > short_score:
            result.signal = Signal.LONG
            result.confidence = long_score
            result.reason = " | ".join(reasons_long)
        elif short_score >= min_confidence and short_score > long_score:
            result.signal = Signal.SHORT
            result.confidence = short_score
            result.reason = " | ".join(reasons_short)
        else:
            result.signal = Signal.HOLD
            result.confidence = max(long_score, short_score)
            result.reason = "Sin señal clara"

        # Loggear cambio de régimen
        LOG.debug(
            EventType.SIGNAL,
            f"Análisis: RSI={result.rsi:.1f} MACD_H={result.macd_hist:.2f} "
            f"BB_pos={result.bb_position:.2f} EMA_cross={'↑' if result.ema_fast > result.ema_slow else '↓'} "
            f"Vol={result.volume_ratio:.2f}x | {result.regime.value} "
            f"→ Long:{long_score:.2f} Short:{short_score:.2f}",
            price=result.price,
        )

        return result

    def _get_regime_thresholds(self, regime: MarketRegime) -> Dict:
        """
        Devuelve umbrales adaptados al régimen de mercado.
        En tendencia: señales más agresivas.
        En rango: señales más conservadoras (evitar whipsaws).
        En alta volatilidad: umbrales altos para evitar ruido.
        """
        base = {
            "rsi_oversold": self.cfg.rsi_oversold,
            "rsi_overbought": self.cfg.rsi_overbought,
            "min_confidence": 0.60,
            "volume_required": True,
            "ob_imbalance_threshold": 0.10,
        }

        if regime == MarketRegime.TRENDING_UP:
            # En tendencia alcista: solo LONG, menos restricciones
            base["rsi_oversold"] = 45      # comprar en pullbacks más amplios
            base["min_confidence"] = 0.55
        elif regime == MarketRegime.TRENDING_DOWN:
            base["rsi_overbought"] = 55
            base["min_confidence"] = 0.55
        elif regime == MarketRegime.RANGING:
            # Mercado lateral: solo reversiones a los extremos de Bollinger
            base["rsi_oversold"] = 30
            base["rsi_overbought"] = 70
            base["min_confidence"] = 0.65
        elif regime == MarketRegime.VOLATILE:
            # Alta volatilidad: exigir mucha confirmación
            base["min_confidence"] = 0.75
            base["ob_imbalance_threshold"] = 0.20

        return base

    def _score_signals(self, r: AnalysisResult, thresholds: Dict,
                       fear_greed: int, btc_dominance: float
                       ) -> Tuple[float, float, List[str], List[str]]:
        """
        Sistema de puntuación ponderada para señales LONG y SHORT.
        Cada condición aporta una fracción de la puntuación total (1.0).
        """
        long_score = 0.0
        short_score = 0.0
        long_reasons = []
        short_reasons = []

        # ── 1. RSI (peso: 0.20) ───────────────────
        if r.rsi < thresholds["rsi_oversold"]:
            long_score += 0.20
            long_reasons.append(f"RSI={r.rsi:.1f} (sobreventa)")
        elif r.rsi > thresholds["rsi_overbought"]:
            short_score += 0.20
            short_reasons.append(f"RSI={r.rsi:.1f} (sobrecompra)")
        elif r.rsi < 45:
            long_score += 0.08   # RSI bajando pero no en extremo
        elif r.rsi > 55:
            short_score += 0.08

        # ── 2. MACD cruce (peso: 0.20) ─────────────
        macd_cross_up = (self._prev_macd_hist < 0 and r.macd_hist >= 0)
        macd_cross_down = (self._prev_macd_hist > 0 and r.macd_hist <= 0)

        if macd_cross_up:
            long_score += 0.20
            long_reasons.append("MACD cruce alcista")
        elif r.macd_hist > 0 and r.macd > r.macd_signal:
            long_score += 0.10
            long_reasons.append(f"MACD hist={r.macd_hist:.2f} positivo")

        if macd_cross_down:
            short_score += 0.20
            short_reasons.append("MACD cruce bajista")
        elif r.macd_hist < 0 and r.macd < r.macd_signal:
            short_score += 0.10
            short_reasons.append(f"MACD hist={r.macd_hist:.2f} negativo")

        self._prev_macd_hist = r.macd_hist

        # ── 3. Bollinger Bands (peso: 0.15) ────────
        if r.bb_position < 0.10:
            long_score += 0.15
            long_reasons.append(f"Precio en BB inferior ({r.bb_position:.2f})")
        elif r.bb_position < 0.25:
            long_score += 0.08

        if r.bb_position > 0.90:
            short_score += 0.15
            short_reasons.append(f"Precio en BB superior ({r.bb_position:.2f})")
        elif r.bb_position > 0.75:
            short_score += 0.08

        # ── 4. EMA cruce (peso: 0.15) ───────────────
        ema_bull = r.ema_fast > r.ema_slow
        ema_bear = r.ema_fast < r.ema_slow
        ema_cross_up = (self._prev_ema_fast < r.ema_slow and r.ema_fast >= r.ema_slow)
        ema_cross_down = (self._prev_ema_fast > r.ema_slow and r.ema_fast <= r.ema_slow)

        if ema_cross_up:
            long_score += 0.15
            long_reasons.append("EMA cruce dorado 9/21")
        elif ema_bull and r.price > r.ema_trend:
            long_score += 0.07  # alineación alcista
            long_reasons.append("EMA alineadas alcistas")

        if ema_cross_down:
            short_score += 0.15
            short_reasons.append("EMA cruce muerto 9/21")
        elif ema_bear and r.price < r.ema_trend:
            short_score += 0.07
            short_reasons.append("EMA alineadas bajistas")

        self._prev_ema_fast = r.ema_fast

        # ── 5. Volumen (peso: 0.10) ─────────────────
        volume_spike = r.volume_ratio >= self.cfg.volume_spike_multiplier
        if volume_spike:
            # El volumen confirma la dirección dominante
            if long_score > short_score:
                long_score += 0.10
                long_reasons.append(f"Spike de volumen {r.volume_ratio:.1f}x")
            else:
                short_score += 0.10
                short_reasons.append(f"Spike de volumen {r.volume_ratio:.1f}x")

        # ── 6. Stochastic RSI (peso: 0.08) ─────────
        stoch_oversold = r.stoch_k < 20 and r.stoch_d < 20
        stoch_overbought = r.stoch_k > 80 and r.stoch_d > 80
        stoch_cross_up = r.stoch_k > r.stoch_d and r.stoch_k < 50

        if stoch_oversold or stoch_cross_up:
            long_score += 0.08
            long_reasons.append(f"StochRSI K={r.stoch_k:.0f}")
        if stoch_overbought:
            short_score += 0.08
            short_reasons.append(f"StochRSI sobrecompra K={r.stoch_k:.0f}")

        # ── 7. Order Book Imbalance (peso: 0.07) ────
        ob_thresh = thresholds["ob_imbalance_threshold"]
        if r.ob_imbalance > ob_thresh:
            long_score += 0.07
            long_reasons.append(f"OB imbalance={r.ob_imbalance:.2f} (presión compradora)")
        elif r.ob_imbalance < -ob_thresh:
            short_score += 0.07
            short_reasons.append(f"OB imbalance={r.ob_imbalance:.2f} (presión vendedora)")

        # ── 8. Patrones de velas (peso: 0.05) ───────
        bullish_patterns = {"hammer", "bullish_engulfing", "morning_star", "doji"}
        bearish_patterns = {"shooting_star", "bearish_engulfing", "hanging_man", "doji"}

        if r.candle_pattern in bullish_patterns:
            long_score += 0.05
            long_reasons.append(f"Patrón: {r.candle_pattern}")
        if r.candle_pattern in bearish_patterns:
            short_score += 0.05
            short_reasons.append(f"Patrón: {r.candle_pattern}")

        # ── 9. Fear & Greed (ajuste de sesgo) ───────
        # Miedo extremo → sesgo alcista leve (posibles suelos)
        if fear_greed < CONFIG.external.extreme_fear_threshold:
            long_score = min(1.0, long_score + 0.05)
        # Codicia extrema → sesgo bajista leve
        elif fear_greed > CONFIG.external.extreme_greed_threshold:
            short_score = min(1.0, short_score + 0.05)

        # ── Normalizar (cap en 1.0) ──────────────────
        long_score = min(1.0, long_score)
        short_score = min(1.0, short_score)

        return long_score, short_score, long_reasons, short_reasons

    def get_stop_loss_take_profits(self, direction: Signal, entry_price: float,
                                   atr: float) -> Tuple[float, float, float]:
        """
        Calcula stop-loss y take-profits dinámicos basados en ATR.

        Decisión de diseño: el stop-loss basado en ATR se adapta a la
        volatilidad actual. En mercados tranquilos el SL es más ajustado;
        en volátiles más holgado, evitando stops prematuros.
        """
        rc = CONFIG.risk
        sl_dist = atr * rc.atr_multiplier_sl
        tp1_dist = atr * rc.atr_multiplier_tp1
        tp2_dist = atr * rc.atr_multiplier_tp2

        if direction == Signal.LONG:
            stop_loss = entry_price - sl_dist
            take_profit_1 = entry_price + tp1_dist
            take_profit_2 = entry_price + tp2_dist
        else:  # SHORT
            stop_loss = entry_price + sl_dist
            take_profit_1 = entry_price - tp1_dist
            take_profit_2 = entry_price - tp2_dist

        return stop_loss, take_profit_1, take_profit_2
