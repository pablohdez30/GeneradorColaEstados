"""
strategy_engine.py - Estrategia basada en el Sniper Bot.

Usa la misma lógica multi-timeframe con scoring de 7 condiciones.
Genera señales LONG y SHORT para paper trading.
"""

import json
import urllib.request
import time
import numpy as np
import pandas as pd
import ccxt

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SYMBOL, ATR_PERIOD, STOP_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
)
from bot.logger import setup_logger

logger = setup_logger("strategy")

# ── Configuración del scoring (misma que Sniper) ──────────────
TIMEFRAMES = ["15m", "1h", "4h"]
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ADX_PERIOD = 14

ADX_STRONG_TREND = 30
ADX_MIN_TREND = 20
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VOLUME_SPIKE_THRESHOLD = 2.0

MIN_SCORE_TO_TRADE = 4  # Mismo que Sniper
FEAR_GREED_API = "https://api.alternative.me/fng/"
FEAR_THRESHOLD = 25
GREED_THRESHOLD = 75


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


# ══════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ANALYZER
# ══════════════════════════════════════════════════════════════════

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
        if df.empty or len(df) < 50:
            return {}

        close = df["close"]
        ema_f = compute_ema(close, EMA_FAST)
        ema_s = compute_ema(close, EMA_SLOW)
        rsi_val = compute_rsi(close)
        adx_val = compute_adx(df)
        atr_val = compute_atr(df)

        # Volumen vs media
        vol_ma = df["volume"].rolling(20).mean()
        vol_ratio = df["volume"].iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1.0

        # Detectar cruce EMA reciente (últimas 3 velas)
        ema_cross_up = False
        ema_cross_down = False
        for i in range(-3, 0):
            if i - 1 >= -len(ema_f):
                prev_diff = ema_f.iloc[i - 1] - ema_s.iloc[i - 1]
                curr_diff = ema_f.iloc[i] - ema_s.iloc[i]
                if prev_diff <= 0 and curr_diff > 0:
                    ema_cross_up = True
                elif prev_diff >= 0 and curr_diff < 0:
                    ema_cross_down = True

        return {
            "price": round(close.iloc[-1], 2),
            "ema_fast": round(ema_f.iloc[-1], 2),
            "ema_slow": round(ema_s.iloc[-1], 2),
            "ema_above": ema_f.iloc[-1] > ema_s.iloc[-1],
            "ema_cross_up": ema_cross_up,
            "ema_cross_down": ema_cross_down,
            "rsi": round(rsi_val.iloc[-1], 1),
            "adx": round(adx_val.iloc[-1], 1),
            "atr": round(atr_val.iloc[-1], 2),
            "volume_ratio": round(vol_ratio, 2),
        }

    def analyze_all(self) -> dict[str, dict]:
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


# ══════════════════════════════════════════════════════════════════
# MOTOR DE ESTRATEGIA (scoring del Sniper)
# ══════════════════════════════════════════════════════════════════

class StrategyEngine:
    def __init__(self, ml_weight: float = 0.0):
        self.ml_weight = ml_weight
        self.last_signal = None
        self.last_indicators = {}
        self.last_fear_greed = 50
        self.last_fg_fetch = 0
        self.last_signal_direction = None
        self.analyzer = MultiTimeframeAnalyzer()
        logger.info("StrategyEngine inicializado | Modo: Sniper multi-timeframe scoring")

    def get_fear_greed(self) -> int:
        """Obtiene Fear & Greed Index. Cachea por 15 minutos."""
        if time.time() - self.last_fg_fetch < 900:
            return self.last_fear_greed
        try:
            req = urllib.request.Request(FEAR_GREED_API)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                self.last_fear_greed = int(data["data"][0]["value"])
                self.last_fg_fetch = time.time()
        except Exception:
            pass
        return self.last_fear_greed

    def generate_signal(self, df: pd.DataFrame = None, order_book: dict = None,
                        ml_prediction: float = 0.0, sentiment: float = 0.5) -> dict:
        """
        Genera señal usando scoring multi-timeframe (misma lógica que Sniper).
        Evalúa 7 condiciones. Score >= 4 para operar.
        """
        # Obtener análisis multi-timeframe
        analysis = self.analyzer.analyze_all()

        if not analysis:
            return self._build_signal("HOLD", 0, ["No se pudieron obtener datos"],
                                      {}, "UNKNOWN")

        # Timeframe principal: 1h
        primary_tf = "1h" if "1h" in analysis else TIMEFRAMES[0]
        if primary_tf not in analysis:
            return self._build_signal("HOLD", 0, ["Sin datos del timeframe principal"],
                                      {}, "UNKNOWN")

        primary = analysis[primary_tf]
        regime = detect_market_regime(
            self.analyzer.candle_buffers.get("1h", pd.DataFrame())
        ) if "1h" in self.analyzer.candle_buffers else "UNKNOWN"

        # Necesita tendencia mínima
        adx = primary.get("adx", 0)
        if adx < ADX_MIN_TREND:
            indicators = self._format_indicators(primary)
            return self._build_signal(
                "HOLD", 0,
                [f"Sin tendencia (ADX={adx:.0f} < {ADX_MIN_TREND})"],
                indicators, regime
            )

        # ── Scoring (7 condiciones) ──────────────────────────
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        # 1. ADX tendencia fuerte
        if adx >= ADX_STRONG_TREND:
            buy_score += 1
            sell_score += 1
            buy_reasons.append(f"Tendencia fuerte (ADX={adx:.0f})")
            sell_reasons.append(f"Tendencia fuerte (ADX={adx:.0f})")

        # 2. EMA Cross reciente (2 puntos)
        if primary.get("ema_cross_up"):
            buy_score += 2
            buy_reasons.append("Cruce EMA alcista reciente")
        if primary.get("ema_cross_down"):
            sell_score += 2
            sell_reasons.append("Cruce EMA bajista reciente")

        # 3. EMA alineadas (1 punto)
        if primary.get("ema_above"):
            buy_score += 1
            buy_reasons.append("EMA 9 > EMA 21 (alcista)")
        else:
            sell_score += 1
            sell_reasons.append("EMA 9 < EMA 21 (bajista)")

        # 4. RSI favorable (1 punto)
        rsi = primary.get("rsi", 50)
        if 35 <= rsi <= 55:
            buy_score += 1
            buy_reasons.append(f"RSI favorable LONG ({rsi:.0f})")
        if 45 <= rsi <= 65:
            sell_score += 1
            sell_reasons.append(f"RSI favorable SHORT ({rsi:.0f})")

        # RSI bloqueante
        if rsi > RSI_OVERBOUGHT:
            buy_score = 0
            buy_reasons = [f"BLOQUEADO: RSI sobrecompra ({rsi:.0f})"]
        if rsi < RSI_OVERSOLD:
            sell_score = 0
            sell_reasons = [f"BLOQUEADO: RSI sobreventa ({rsi:.0f})"]

        # 5. Volumen spike (1 punto)
        vol_ratio = primary.get("volume_ratio", 1.0)
        if vol_ratio >= VOLUME_SPIKE_THRESHOLD:
            buy_score += 1
            sell_score += 1
            buy_reasons.append(f"Volumen inusual ({vol_ratio:.1f}x)")
            sell_reasons.append(f"Volumen inusual ({vol_ratio:.1f}x)")

        # 6. Fear & Greed extremo (1 punto)
        fg = self.get_fear_greed()
        if fg <= FEAR_THRESHOLD:
            buy_score += 1
            buy_reasons.append(f"Miedo extremo (F&G={fg})")
        elif fg >= GREED_THRESHOLD:
            sell_score += 1
            sell_reasons.append(f"Avaricia extrema (F&G={fg})")

        # 7. Multi-timeframe alineados (2 puntos)
        bullish_tfs = 0
        bearish_tfs = 0
        for tf in TIMEFRAMES:
            if tf in analysis:
                tf_data = analysis[tf]
                if tf_data.get("ema_above") and tf_data.get("adx", 0) > 20:
                    bullish_tfs += 1
                elif not tf_data.get("ema_above") and tf_data.get("adx", 0) > 20:
                    bearish_tfs += 1

        if bullish_tfs >= 2:
            buy_score += 2
            buy_reasons.append(f"{bullish_tfs} timeframes confirman alcista")
        if bearish_tfs >= 2:
            sell_score += 2
            sell_reasons.append(f"{bearish_tfs} timeframes confirman bajista")

        # ── Decisión final ────────────────────────────────────
        indicators = self._format_indicators(primary)

        if buy_score >= MIN_SCORE_TO_TRADE and buy_score > sell_score:
            if self.last_signal_direction == "BUY":
                return self._build_signal("HOLD", 0,
                    ["Señal LONG repetida, esperando cambio"], indicators, regime)
            self.last_signal_direction = "BUY"
            confidence = min(buy_score / 7, 1.0)
            return self._build_signal("BUY", confidence, buy_reasons, indicators, regime)

        elif sell_score >= MIN_SCORE_TO_TRADE and sell_score > buy_score:
            if self.last_signal_direction == "SELL":
                return self._build_signal("HOLD", 0,
                    ["Señal SHORT repetida, esperando cambio"], indicators, regime)
            self.last_signal_direction = "SELL"
            confidence = min(sell_score / 7, 1.0)
            return self._build_signal("SELL", confidence, sell_reasons, indicators, regime)

        # Sin señal suficiente → reset para permitir futuras señales
        self.last_signal_direction = None
        best_score = max(buy_score, sell_score)
        return self._build_signal("HOLD", 0,
            [f"Score insuficiente ({best_score}/7, necesita {MIN_SCORE_TO_TRADE})"],
            indicators, regime)

    def _format_indicators(self, primary: dict) -> dict:
        """Formatea indicadores para compatibilidad con paper_trader."""
        return {
            "price": primary.get("price", 0),
            "ema_fast": primary.get("ema_fast", 0),
            "ema_slow": primary.get("ema_slow", 0),
            "rsi": primary.get("rsi", 50),
            "adx": primary.get("adx", 0),
            "atr": primary.get("atr", 0),
            "volume_ratio": primary.get("volume_ratio", 1.0),
            # Legacy fields for ML/dashboard compatibility
            "macd_histogram": 0,
            "bb_position": 0.5,
            "ema_fast_prev": 0,
            "ema_slow_prev": 0,
        }

    def compute_all_indicators(self, df: pd.DataFrame) -> dict:
        """Compatibilidad con ML module."""
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
        stop_distance = atr * STOP_ATR_MULTIPLIER
        if direction == "BUY":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def compute_take_profit(self, entry_price: float, direction: str,
                             atr: float) -> float:
        tp_distance = atr * TP_ATR_MULTIPLIER
        if direction == "BUY":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance
