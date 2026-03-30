"""
scorer.py — Sistema de puntuación para señales de alta confianza.

Evalúa múltiples condiciones y solo genera alerta si el score >= 3.
Cada condición suma puntos. Más puntos = más confianza = más apalancamiento.
"""

import json
import urllib.request
from datetime import datetime, timezone

from sniper_bot.config import (
    ADX_STRONG_TREND, RSI_OVERSOLD, RSI_OVERBOUGHT,
    VOLUME_SPIKE_THRESHOLD, FEAR_THRESHOLD, GREED_THRESHOLD,
    MIN_SCORE_TO_ALERT, SCORES, LEVERAGE_BY_SCORE,
    STOP_ATR_MULTIPLIER, TP_ATR_MULTIPLIER, RISK_PCT,
    FEAR_GREED_API, TIMEFRAMES,
)


class SignalScorer:
    """Evalúa condiciones del mercado y genera señales con puntuación."""

    def __init__(self):
        self.last_fear_greed = 50  # Neutral por defecto
        self.last_fg_fetch = 0
        self.last_signal_direction = None  # Evitar repetir misma señal

    def get_fear_greed(self) -> int:
        """Obtiene Fear & Greed Index. Cachea por 15 minutos."""
        import time
        if time.time() - self.last_fg_fetch < 900:
            return self.last_fear_greed

        try:
            req = urllib.request.Request(FEAR_GREED_API)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                self.last_fear_greed = int(data["data"][0]["value"])
                self.last_fg_fetch = time.time()
        except Exception:
            pass  # Usar último valor
        return self.last_fear_greed

    def evaluate(self, analysis: dict[str, dict]) -> dict | None:
        """
        Evalúa los indicadores de todos los timeframes y genera una señal
        si el score es suficiente.

        Args:
            analysis: {timeframe: {indicators}} de MarketAnalyzer.analyze_all()

        Returns:
            Signal dict o None si no hay señal suficiente.
        """
        if not analysis:
            return None

        # Usar el timeframe principal (1h) para la señal base
        primary_tf = "1h" if "1h" in analysis else TIMEFRAMES[0]
        if primary_tf not in analysis:
            return None

        primary = analysis[primary_tf]
        if not primary:
            return None

        # Sin tendencia mínima, no hay señal posible
        adx = primary.get("adx", 0)
        if adx < 20:
            return None  # No hay tendencia, no alertar

        # Evaluar dirección
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        # ── 1. ADX: tendencia fuerte ──────────────────────────
        if adx >= ADX_STRONG_TREND:
            # ADX solo confirma fuerza, no dirección
            buy_score += SCORES["adx_strong"]
            sell_score += SCORES["adx_strong"]
            buy_reasons.append(f"Tendencia fuerte (ADX={adx:.0f})")
            sell_reasons.append(f"Tendencia fuerte (ADX={adx:.0f})")

        # ── 2. EMA Cross reciente (vale doble) ────────────────
        if primary.get("ema_cross_up"):
            buy_score += SCORES["ema_cross_recent"]
            buy_reasons.append("Cruce EMA alcista reciente")
        if primary.get("ema_cross_down"):
            sell_score += SCORES["ema_cross_recent"]
            sell_reasons.append("Cruce EMA bajista reciente")

        # ── 3. EMA alineadas ──────────────────────────────────
        if primary.get("ema_above"):
            buy_score += SCORES["ema_aligned"]
            buy_reasons.append("EMA 9 > EMA 21 (tendencia alcista)")
        else:
            sell_score += SCORES["ema_aligned"]
            sell_reasons.append("EMA 9 < EMA 21 (tendencia bajista)")

        # ── 4. RSI favorable ──────────────────────────────────
        rsi = primary.get("rsi", 50)
        if 35 <= rsi <= 55:
            buy_score += SCORES["rsi_favorable"]
            buy_reasons.append(f"RSI favorable para LONG ({rsi:.0f})")
        if 45 <= rsi <= 65:
            sell_score += SCORES["rsi_favorable"]
            sell_reasons.append(f"RSI favorable para SHORT ({rsi:.0f})")

        # RSI bloqueante: si está en extremo, BLOQUEAR la señal contraria
        if rsi > RSI_OVERBOUGHT:
            buy_score = 0  # No comprar en sobrecompra
            buy_reasons = [f"BLOQUEADO: RSI sobrecompra ({rsi:.0f})"]
        if rsi < RSI_OVERSOLD:
            sell_score = 0  # No vender en sobreventa
            sell_reasons = [f"BLOQUEADO: RSI sobreventa ({rsi:.0f})"]

        # ── 5. Volumen spike ──────────────────────────────────
        vol_ratio = primary.get("volume_ratio", 1.0)
        if vol_ratio >= VOLUME_SPIKE_THRESHOLD:
            buy_score += SCORES["volume_spike"]
            sell_score += SCORES["volume_spike"]
            buy_reasons.append(f"Volumen inusual ({vol_ratio:.1f}x)")
            sell_reasons.append(f"Volumen inusual ({vol_ratio:.1f}x)")

        # ── 6. Fear & Greed extremo ──────────────────────────
        fg = self.get_fear_greed()
        if fg <= FEAR_THRESHOLD:
            buy_score += SCORES["fear_greed_extreme"]
            buy_reasons.append(f"Miedo extremo (F&G={fg}) = oportunidad de compra")
        elif fg >= GREED_THRESHOLD:
            sell_score += SCORES["fear_greed_extreme"]
            sell_reasons.append(f"Avaricia extrema (F&G={fg}) = probable caída")

        # ── 7. Confirmación multi-timeframe (vale doble) ──────
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
            buy_score += SCORES["multi_tf_aligned"]
            buy_reasons.append(f"{bullish_tfs} timeframes confirman alcista")
        if bearish_tfs >= 2:
            sell_score += SCORES["multi_tf_aligned"]
            sell_reasons.append(f"{bearish_tfs} timeframes confirman bajista")

        # ── Decisión final (solo LONG - usuario opera en Spot) ──
        if buy_score >= MIN_SCORE_TO_ALERT and buy_score > sell_score:
            # No repetir la misma señal consecutivamente
            if self.last_signal_direction == "LONG":
                return None
            self.last_signal_direction = "LONG"
            return self._build_signal("LONG", buy_score, buy_reasons, primary, fg)

        # Reset si no hay señal (permite alertar cuando vuelva a haber)
        self.last_signal_direction = None
        return None

    def _build_signal(self, direction: str, score: int, reasons: list,
                      indicators: dict, fear_greed: int) -> dict:
        """Construye la señal completa con entrada, SL, TP y apalancamiento."""
        price = indicators["price"]
        atr = indicators["atr"]

        # Stop loss y take profit basados en ATR
        sl_distance = atr * STOP_ATR_MULTIPLIER
        tp_distance = atr * TP_ATR_MULTIPLIER

        if direction == "LONG":
            entry = price
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:
            entry = price
            stop_loss = price + sl_distance
            take_profit = price - tp_distance

        # Apalancamiento según score
        leverage = LEVERAGE_BY_SCORE.get(min(score, 6), 2)

        # Risk % como texto
        sl_pct = (sl_distance / price) * 100

        return {
            "direction": direction,
            "score": score,
            "confidence": min(score / 7 * 100, 100),  # 7 = score máximo teórico
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "sl_pct": round(sl_pct, 2),
            "tp_pct": round((tp_distance / price) * 100, 2),
            "leverage": leverage,
            "risk_reward": f"1:{TP_ATR_MULTIPLIER / STOP_ATR_MULTIPLIER:.1f}",
            "reasons": reasons,
            "indicators": indicators,
            "fear_greed": fear_greed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
