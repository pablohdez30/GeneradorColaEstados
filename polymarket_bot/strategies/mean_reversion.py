"""
Mean Reversion Strategy: Bets that prices will revert to their historical mean.
Identifies overbought/oversold conditions using Bollinger Bands and RSI.
"""

import logging
import math
from typing import Optional

from .base import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using statistical indicators.
    Assumes that large price movements tend to revert toward the mean.
    """

    def __init__(self, bb_window: int = 20, bb_std: float = 2.0,
                 rsi_window: int = 14):
        super().__init__("mean_reversion")
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window

    def analyze(self, market, price_history: list[dict],
                order_book: dict) -> Optional[TradeSignal]:
        """Analyze mean reversion indicators."""
        prices = self._extract_prices(price_history)

        if len(prices) < self.bb_window:
            return None

        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(prices)

        # Calculate RSI
        rsi = self._calculate_rsi(prices)

        current_price = prices[-1]

        # Oversold: price below lower band and RSI < 30
        if current_price < bb_lower and rsi < 35:
            distance = (bb_middle - current_price) / bb_middle if bb_middle > 0 else 0
            confidence = min(0.85, 0.5 + distance * 2 + (35 - rsi) / 100)

            if market.outcome_yes_price < 0.85:
                return TradeSignal(
                    signal=Signal.STRONG_BUY if rsi < 25 else Signal.BUY,
                    confidence=confidence,
                    target_outcome="yes",
                    suggested_size=min(1.0, confidence * 0.9),
                    reason=(f"Oversold: price {current_price:.3f} below BB lower "
                            f"{bb_lower:.3f}, RSI={rsi:.0f}"),
                    strategy_name=self.name,
                )

        # Overbought: price above upper band and RSI > 70
        elif current_price > bb_upper and rsi > 65:
            distance = (current_price - bb_middle) / bb_middle if bb_middle > 0 else 0
            confidence = min(0.85, 0.5 + distance * 2 + (rsi - 65) / 100)

            if market.outcome_no_price < 0.85:
                return TradeSignal(
                    signal=Signal.STRONG_BUY if rsi > 75 else Signal.BUY,
                    confidence=confidence,
                    target_outcome="no",
                    suggested_size=min(1.0, confidence * 0.9),
                    reason=(f"Overbought: price {current_price:.3f} above BB upper "
                            f"{bb_upper:.3f}, RSI={rsi:.0f}"),
                    strategy_name=self.name,
                )

        return None

    def _bollinger_bands(self, prices: list[float]) -> tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        window = prices[-self.bb_window:]
        mean = sum(window) / len(window)
        variance = sum((p - mean) ** 2 for p in window) / len(window)
        std = math.sqrt(variance)

        upper = mean + self.bb_std * std
        lower = mean - self.bb_std * std

        return upper, mean, lower

    def _calculate_rsi(self, prices: list[float]) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < self.rsi_window + 1:
            return 50.0  # Neutral

        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        recent = changes[-self.rsi_window:]

        gains = [c for c in recent if c > 0]
        losses = [-c for c in recent if c < 0]

        avg_gain = sum(gains) / self.rsi_window if gains else 0
        avg_loss = sum(losses) / self.rsi_window if losses else 0

        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _extract_prices(self, price_history: list[dict]) -> list[float]:
        """Extract price values from history data."""
        prices = []
        for point in price_history:
            price = point.get("p") or point.get("price") or point.get("t")
            if price is not None:
                try:
                    prices.append(float(price))
                except (ValueError, TypeError):
                    continue
        return prices
