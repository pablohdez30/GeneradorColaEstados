"""
Value Strategy: Looks for mispricings in the market.
Identifies markets where prices don't add up correctly or where
order book imbalances suggest the price is wrong.
"""

import logging
from typing import Optional

from .base import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger(__name__)


class ValueStrategy(BaseStrategy):
    """
    Value-based trading strategy.
    Looks for markets that appear mispriced based on:
    1. Price spread (YES + NO should equal ~1.0)
    2. Order book imbalance
    3. Extreme prices that may revert
    """

    def __init__(self, spread_threshold: float = 0.05,
                 imbalance_threshold: float = 0.3):
        super().__init__("value")
        self.spread_threshold = spread_threshold
        self.imbalance_threshold = imbalance_threshold

    def analyze(self, market, price_history: list[dict],
                order_book: dict) -> Optional[TradeSignal]:
        """Analyze market value indicators."""
        signals = []

        # Check 1: Price spread arbitrage
        spread_signal = self._check_spread(market)
        if spread_signal:
            signals.append(spread_signal)

        # Check 2: Order book imbalance
        ob_signal = self._check_order_book(market, order_book)
        if ob_signal:
            signals.append(ob_signal)

        # Check 3: Extreme value detection
        extreme_signal = self._check_extreme_prices(market, price_history)
        if extreme_signal:
            signals.append(extreme_signal)

        if not signals:
            return None

        # Return the highest confidence signal
        return max(signals, key=lambda s: s.confidence)

    def _check_spread(self, market) -> Optional[TradeSignal]:
        """Check if the YES/NO spread indicates mispricing."""
        total = market.outcome_yes_price + market.outcome_no_price

        if total < 1.0 - self.spread_threshold:
            # Prices sum to less than 1 — both outcomes are cheap
            # Buy the more likely outcome
            if market.outcome_yes_price > market.outcome_no_price:
                target = "yes"
                price = market.outcome_yes_price
            else:
                target = "no"
                price = market.outcome_no_price

            discount = 1.0 - total
            confidence = min(0.85, 0.5 + discount * 3)

            return TradeSignal(
                signal=Signal.BUY,
                confidence=confidence,
                target_outcome=target,
                suggested_size=min(1.0, confidence * 0.8),
                reason=f"Spread discount: prices sum to {total:.3f} (discount {discount:.1%})",
                strategy_name=self.name,
            )

        return None

    def _check_order_book(self, market, order_book: dict) -> Optional[TradeSignal]:
        """Check order book for significant imbalances."""
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks:
            return None

        # Calculate total bid/ask volume
        bid_volume = sum(float(b.get("size", 0)) for b in bids[:10])
        ask_volume = sum(float(a.get("size", 0)) for a in asks[:10])
        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return None

        imbalance = (bid_volume - ask_volume) / total_volume

        if abs(imbalance) > self.imbalance_threshold:
            if imbalance > 0:
                # More buyers → price likely to go up
                return TradeSignal(
                    signal=Signal.BUY,
                    confidence=min(0.8, 0.5 + abs(imbalance) * 0.5),
                    target_outcome="yes",
                    suggested_size=0.6,
                    reason=f"Order book imbalance: {imbalance:.1%} buy pressure",
                    strategy_name=self.name,
                )
            else:
                # More sellers → price likely to go down
                return TradeSignal(
                    signal=Signal.BUY,
                    confidence=min(0.8, 0.5 + abs(imbalance) * 0.5),
                    target_outcome="no",
                    suggested_size=0.6,
                    reason=f"Order book imbalance: {imbalance:.1%} sell pressure",
                    strategy_name=self.name,
                )

        return None

    def _check_extreme_prices(self, market, price_history: list[dict]) -> Optional[TradeSignal]:
        """Look for markets at extreme prices that may revert."""
        prices = [float(p.get("p", p.get("price", 0.5)))
                  for p in price_history if p.get("p") or p.get("price")]

        if len(prices) < 10:
            return None

        avg_price = sum(prices) / len(prices)
        current = market.outcome_yes_price

        # If current price is significantly different from average
        deviation = current - avg_price

        if current < 0.20 and avg_price > 0.30:
            # YES has crashed — potential value buy
            confidence = min(0.75, 0.4 + abs(deviation) * 1.5)
            return TradeSignal(
                signal=Signal.BUY,
                confidence=confidence,
                target_outcome="yes",
                suggested_size=0.5,
                reason=f"Potential undervalue: current {current:.2f} vs avg {avg_price:.2f}",
                strategy_name=self.name,
            )

        elif current > 0.80 and avg_price < 0.70:
            # YES has spiked — potential value in NO
            confidence = min(0.75, 0.4 + abs(deviation) * 1.5)
            return TradeSignal(
                signal=Signal.BUY,
                confidence=confidence,
                target_outcome="no",
                suggested_size=0.5,
                reason=f"Potential overvalue: current {current:.2f} vs avg {avg_price:.2f}",
                strategy_name=self.name,
            )

        return None
