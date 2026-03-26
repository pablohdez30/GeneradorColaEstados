"""
Momentum Strategy: Trades in the direction of recent price trends.
If a market is trending up, buy. If trending down, sell.
Uses simple moving averages and rate of change.
"""

import logging
from typing import Optional

from .base import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    Looks for markets with strong directional movement and trades
    in the direction of the trend.
    """

    def __init__(self, short_window: int = 6, long_window: int = 20,
                 roc_threshold: float = 0.05):
        super().__init__("momentum")
        self.short_window = short_window
        self.long_window = long_window
        self.roc_threshold = roc_threshold  # 5% rate of change threshold

    def analyze(self, market, price_history: list[dict],
                order_book: dict) -> Optional[TradeSignal]:
        """Analyze momentum indicators to generate a signal."""
        prices = self._extract_prices(price_history)

        if len(prices) < self.long_window:
            return None

        # Calculate moving averages
        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices[-self.long_window:]) / self.long_window

        # Rate of change (recent vs older)
        recent_avg = sum(prices[-5:]) / min(5, len(prices[-5:]))
        older_avg = sum(prices[-15:-5]) / min(10, len(prices[-15:-5])) if len(prices) > 5 else recent_avg
        roc = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        # Trend strength
        ma_diff = (short_ma - long_ma) / long_ma if long_ma > 0 else 0

        # Determine signal
        current_price = market.outcome_yes_price

        if ma_diff > 0.02 and roc > self.roc_threshold:
            # Strong upward momentum on YES
            confidence = min(0.9, 0.5 + abs(ma_diff) * 5 + abs(roc) * 3)
            if current_price < 0.85:  # Don't chase near certainty
                return TradeSignal(
                    signal=Signal.STRONG_BUY if roc > self.roc_threshold * 2 else Signal.BUY,
                    confidence=confidence,
                    target_outcome="yes",
                    suggested_size=min(1.0, confidence),
                    reason=f"Upward momentum: SMA crossover +{ma_diff:.1%}, ROC +{roc:.1%}",
                    strategy_name=self.name,
                )

        elif ma_diff < -0.02 and roc < -self.roc_threshold:
            # Strong downward momentum on YES → Buy NO
            confidence = min(0.9, 0.5 + abs(ma_diff) * 5 + abs(roc) * 3)
            if current_price > 0.15:  # Don't chase near zero
                return TradeSignal(
                    signal=Signal.STRONG_BUY if roc < -self.roc_threshold * 2 else Signal.BUY,
                    confidence=confidence,
                    target_outcome="no",
                    suggested_size=min(1.0, confidence),
                    reason=f"Downward momentum: SMA crossover {ma_diff:.1%}, ROC {roc:.1%}",
                    strategy_name=self.name,
                )

        return None

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
