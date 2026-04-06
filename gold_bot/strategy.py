"""
strategy.py - Golden Cross Trend Rider for Gold.

Adapted from MT5 EA optimized parameters.
Uses 15m candles, EMA crossovers, ADX filter, RSI filter,
and ATR-based stops with pullback entry logic.

Key concept:
- Golden Cross: EMA50 > EMA200 (bullish regime)
- Death Cross: EMA50 < EMA200 (bearish regime)
- Entry on pullback to EMA50 after price moves away and returns
- Confirmation: EMA9 > EMA21 aligned with regime, ADX > 40
- Session filter: 07:00-20:00 UTC only
"""

import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

from gold_bot.config import (
    SYMBOL, FALLBACK_SYMBOL, EXCHANGE, FALLBACK_EXCHANGE,
    EMA_FAST, EMA_MID, EMA_SLOW, EMA_TREND,
    RSI_PERIOD, ADX_PERIOD, ATR_PERIOD,
    ADX_MIN, SL_ATR_MULT, TP_RATIO,
    TOUCH_MARGIN_MULT, AWAY_MARGIN_MULT,
    SESSION_START_HOUR, SESSION_END_HOUR,
)
from gold_bot.logger import setup_logger

logger = setup_logger("gold_strategy")


class GoldStrategy:
    """Golden Cross Trend Rider strategy for gold."""

    def __init__(self):
        self.exchange = None
        self.symbol = None
        self._price_was_away = False  # Track if price moved away from EMA50
        self._last_away_direction = None
        self._connect_exchange()

    def _connect_exchange(self):
        """Try to connect to exchange and find a valid gold symbol."""
        # Try primary exchange + symbol first
        for exch_id, sym in [
            (EXCHANGE, SYMBOL),
            (EXCHANGE, FALLBACK_SYMBOL),
            (FALLBACK_EXCHANGE, SYMBOL),
            (FALLBACK_EXCHANGE, FALLBACK_SYMBOL),
        ]:
            try:
                logger.info(f"Trying {exch_id} with {sym}...")
                exchange_class = getattr(ccxt, exch_id)
                exchange = exchange_class({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
                exchange.load_markets()

                if sym in exchange.markets:
                    self.exchange = exchange
                    self.symbol = sym
                    logger.info(f"Connected to {exch_id} | Symbol: {sym}")
                    return
                else:
                    logger.info(f"{sym} not found on {exch_id}, trying next...")
            except Exception as e:
                logger.warning(f"Failed to connect to {exch_id}: {e}")

        raise RuntimeError(
            "Could not connect to any exchange with a valid gold symbol. "
            "Tried: bybit (XAU/USDT, PAXG/USDT), binance (XAU/USDT, PAXG/USDT)"
        )

    def get_current_price(self) -> float | None:
        """Return current price from exchange."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker["last"]
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return None

    def _fetch_candles(self, timeframe: str = "15m", limit: int = 300) -> pd.DataFrame | None:
        """Fetch OHLCV candles and return as DataFrame."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 210:
                logger.warning(f"Not enough candles: got {len(ohlcv) if ohlcv else 0}")
                return None

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return None

    @staticmethod
    def _calc_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr)

        dx = (100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)))
        adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()
        return adx

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
        return atr

    def _compute_indicators(self, df: pd.DataFrame) -> dict | None:
        """Compute all indicators on the DataFrame."""
        try:
            close = df["close"]

            ema_fast = self._calc_ema(close, EMA_FAST)
            ema_mid = self._calc_ema(close, EMA_MID)
            ema_slow = self._calc_ema(close, EMA_SLOW)
            ema_trend = self._calc_ema(close, EMA_TREND)
            rsi = self._calc_rsi(close, RSI_PERIOD)
            adx = self._calc_adx(df, ADX_PERIOD)
            atr = self._calc_atr(df, ATR_PERIOD)

            idx = len(df) - 1
            return {
                "price": close.iloc[idx],
                "ema_fast": ema_fast.iloc[idx],
                "ema_mid": ema_mid.iloc[idx],
                "ema_slow": ema_slow.iloc[idx],
                "ema_trend": ema_trend.iloc[idx],
                "rsi": rsi.iloc[idx],
                "adx": adx.iloc[idx],
                "atr": atr.iloc[idx],
                "ema_fast_prev": ema_fast.iloc[idx - 1],
                "ema_mid_prev": ema_mid.iloc[idx - 1],
            }
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            return None

    def generate_signal(self) -> dict:
        """
        Generate trading signal based on Golden Cross strategy.

        Returns dict with keys:
            action: BUY / SELL / HOLD
            confidence: float 0-1
            entry_price: float
            stop_loss: float
            take_profit: float
            indicators: dict of current indicator values
            reasons: list of strings explaining the decision
            regime: str describing market regime
        """
        result = {
            "action": "HOLD",
            "confidence": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "indicators": {},
            "reasons": [],
            "regime": "UNKNOWN",
        }

        # Fetch candles
        df = self._fetch_candles()
        if df is None:
            result["reasons"].append("No candle data available")
            return result

        # Compute indicators
        indicators = self._compute_indicators(df)
        if indicators is None:
            result["reasons"].append("Indicator computation failed")
            return result

        result["indicators"] = indicators
        price = indicators["price"]
        ema_fast = indicators["ema_fast"]
        ema_mid = indicators["ema_mid"]
        ema_slow = indicators["ema_slow"]
        ema_trend = indicators["ema_trend"]
        rsi = indicators["rsi"]
        adx = indicators["adx"]
        atr = indicators["atr"]

        # ── 1. Determine regime (Golden Cross / Death Cross) ────
        if ema_slow > ema_trend:
            regime = "GOLDEN_CROSS"
            result["regime"] = "GOLDEN_CROSS"
            result["reasons"].append(f"Golden Cross: EMA50({ema_slow:.2f}) > EMA200({ema_trend:.2f})")
        elif ema_slow < ema_trend:
            regime = "DEATH_CROSS"
            result["regime"] = "DEATH_CROSS"
            result["reasons"].append(f"Death Cross: EMA50({ema_slow:.2f}) < EMA200({ema_trend:.2f})")
        else:
            regime = "NEUTRAL"
            result["regime"] = "NEUTRAL"
            result["reasons"].append("No clear cross regime")
            return result

        # ── 2. Session filter ──────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour
        if not (SESSION_START_HOUR <= current_hour < SESSION_END_HOUR):
            result["reasons"].append(
                f"Outside trading session ({SESSION_START_HOUR}:00-{SESSION_END_HOUR}:00 UTC), "
                f"current hour: {current_hour}"
            )
            return result

        # ── 3. ADX filter (strong trend only) ──────────────────
        if adx < ADX_MIN:
            result["reasons"].append(f"ADX too low: {adx:.1f} < {ADX_MIN}")
            return result
        result["reasons"].append(f"ADX OK: {adx:.1f} >= {ADX_MIN}")

        # ── 4. Pullback logic ──────────────────────────────────
        away_margin = atr * AWAY_MARGIN_MULT
        touch_margin = atr * TOUCH_MARGIN_MULT
        dist_to_ema50 = price - ema_slow

        # Check if price is currently away from EMA50
        if abs(dist_to_ema50) > away_margin:
            if not self._price_was_away:
                self._price_was_away = True
                self._last_away_direction = "ABOVE" if dist_to_ema50 > 0 else "BELOW"
                result["reasons"].append(
                    f"Price moved away from EMA50 ({self._last_away_direction}): "
                    f"dist={dist_to_ema50:.2f}, margin={away_margin:.2f}"
                )

        # Check if price returned close to EMA50 (pullback complete)
        pullback_ready = False
        if self._price_was_away and abs(dist_to_ema50) <= touch_margin:
            pullback_ready = True
            result["reasons"].append(
                f"Pullback to EMA50: dist={dist_to_ema50:.2f}, "
                f"touch_margin={touch_margin:.2f}"
            )
            # Reset for next cycle
            self._price_was_away = False
            self._last_away_direction = None

        if not pullback_ready:
            if not self._price_was_away:
                result["reasons"].append(
                    f"Waiting for price to move away from EMA50 (dist={dist_to_ema50:.2f}, "
                    f"need>{away_margin:.2f})"
                )
            else:
                result["reasons"].append(
                    f"Price away but no pullback yet (dist={dist_to_ema50:.2f}, "
                    f"need<{touch_margin:.2f})"
                )
            return result

        # ── 5. Short-term trend alignment ──────────────────────
        if regime == "GOLDEN_CROSS":
            if ema_fast <= ema_mid:
                result["reasons"].append(
                    f"Short-term trend not aligned for LONG: "
                    f"EMA9({ema_fast:.2f}) <= EMA21({ema_mid:.2f})"
                )
                return result
            result["reasons"].append(f"Short-term bullish: EMA9({ema_fast:.2f}) > EMA21({ema_mid:.2f})")
        else:
            if ema_fast >= ema_mid:
                result["reasons"].append(
                    f"Short-term trend not aligned for SHORT: "
                    f"EMA9({ema_fast:.2f}) >= EMA21({ema_mid:.2f})"
                )
                return result
            result["reasons"].append(f"Short-term bearish: EMA9({ema_fast:.2f}) < EMA21({ema_mid:.2f})")

        # ── 6. RSI filter ──────────────────────────────────────
        if regime == "GOLDEN_CROSS":
            if not (30 <= rsi <= 65):
                result["reasons"].append(f"RSI out of LONG range: {rsi:.1f} (need 30-65)")
                return result
            result["reasons"].append(f"RSI OK for LONG: {rsi:.1f}")
        else:
            if not (35 <= rsi <= 70):
                result["reasons"].append(f"RSI out of SHORT range: {rsi:.1f} (need 35-70)")
                return result
            result["reasons"].append(f"RSI OK for SHORT: {rsi:.1f}")

        # ── 7. Generate signal ─────────────────────────────────
        sl_distance = atr * SL_ATR_MULT
        tp_distance = sl_distance * TP_RATIO

        if regime == "GOLDEN_CROSS":
            action = "BUY"
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:
            action = "SELL"
            stop_loss = price + sl_distance
            take_profit = price - tp_distance

        # Confidence based on how many factors align
        confidence = 0.6  # Base: passed all filters
        if adx > 50:
            confidence += 0.1
        if regime == "GOLDEN_CROSS" and rsi < 50:
            confidence += 0.1
        elif regime == "DEATH_CROSS" and rsi > 50:
            confidence += 0.1
        if abs(ema_slow - ema_trend) / ema_trend > 0.005:
            confidence += 0.1
        confidence = min(confidence, 1.0)

        result["action"] = action
        result["confidence"] = confidence
        result["entry_price"] = price
        result["stop_loss"] = stop_loss
        result["take_profit"] = take_profit
        result["reasons"].append(
            f"SIGNAL: {action} @ {price:.2f} | SL={stop_loss:.2f} | "
            f"TP={take_profit:.2f} | conf={confidence:.0%}"
        )

        logger.info(
            f"Signal: {action} | price={price:.2f} | SL={stop_loss:.2f} | "
            f"TP={take_profit:.2f} | ADX={adx:.1f} | RSI={rsi:.1f} | "
            f"regime={regime} | conf={confidence:.0%}"
        )

        return result
