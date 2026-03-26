"""
Slippage Simulator: Models realistic execution costs when trading against an order book.
Instead of assuming you get the midpoint price, it walks the real order book
to calculate the actual fill price you'd get for a given trade size.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FillResult:
    """Result of simulating an order fill against the order book."""
    avg_fill_price: float      # Volume-weighted average fill price
    total_filled: float        # Total shares filled
    total_cost: float          # Total cost in USDC
    slippage_bps: float        # Slippage in basis points vs midpoint
    levels_consumed: int       # How many price levels were consumed
    fully_filled: bool         # Whether the full order was filled


class SlippageSimulator:
    """
    Simulates order execution against a live order book.
    Walks ask levels (for buys) or bid levels (for sells) to compute
    the realistic fill price including market impact.
    """

    # Extra slippage added on top of book-walk (models hidden costs)
    BASE_SLIPPAGE_BPS = 5  # 0.05% base slippage for market microstructure

    def simulate_buy(self, order_book: dict, amount_usdc: float,
                     midpoint: float) -> FillResult:
        """
        Simulate buying shares by walking ask levels.

        Args:
            order_book: Order book with 'asks' list of {price, size}
            amount_usdc: How much USDC to spend
            midpoint: Current midpoint price for slippage calculation

        Returns:
            FillResult with realistic fill details
        """
        asks = order_book.get("asks", [])
        return self._walk_book(asks, amount_usdc, midpoint, side="buy")

    def simulate_sell(self, order_book: dict, shares: float,
                      midpoint: float) -> FillResult:
        """
        Simulate selling shares by walking bid levels.

        Args:
            order_book: Order book with 'bids' list of {price, size}
            shares: How many shares to sell
            midpoint: Current midpoint price for slippage calculation

        Returns:
            FillResult with realistic fill details
        """
        bids = order_book.get("bids", [])
        return self._walk_book(bids, shares, midpoint, side="sell")

    def _walk_book(self, levels: list, amount: float, midpoint: float,
                   side: str) -> FillResult:
        """
        Walk through order book levels to simulate a fill.

        For buys: amount is USDC to spend, levels are asks (lowest first)
        For sells: amount is shares to sell, levels are bids (highest first)
        """
        if not levels or amount <= 0:
            # No order book — use midpoint with base slippage
            slippage_mult = 1 + (self.BASE_SLIPPAGE_BPS / 10000)
            if side == "buy":
                fill_price = midpoint * slippage_mult if midpoint > 0 else 0.5
            else:
                fill_price = midpoint / slippage_mult if midpoint > 0 else 0.5
            return FillResult(
                avg_fill_price=fill_price,
                total_filled=amount / fill_price if side == "buy" and fill_price > 0 else amount * fill_price if side == "sell" else 0,
                total_cost=amount if side == "buy" else amount * fill_price,
                slippage_bps=self.BASE_SLIPPAGE_BPS,
                levels_consumed=0,
                fully_filled=True,
            )

        remaining = amount
        total_cost = 0.0
        total_shares = 0.0
        levels_used = 0

        for level in levels:
            price = float(level.get("price", level.get("p", 0)))
            size = float(level.get("size", level.get("s", 0)))

            if price <= 0 or size <= 0:
                continue

            levels_used += 1

            if side == "buy":
                # Spending USDC: can buy `affordable` shares at this level
                affordable = remaining / price
                fill_shares = min(affordable, size)
                fill_cost = fill_shares * price

                total_shares += fill_shares
                total_cost += fill_cost
                remaining -= fill_cost

            else:
                # Selling shares: fill as many shares as this level can absorb
                fill_shares = min(remaining, size)
                fill_cost = fill_shares * price

                total_shares += fill_shares
                total_cost += fill_cost
                remaining -= fill_shares

            if remaining <= 0.001:  # Close enough to fully filled
                break

        if total_shares <= 0:
            # Fallback
            return FillResult(
                avg_fill_price=midpoint,
                total_filled=0,
                total_cost=0,
                slippage_bps=0,
                levels_consumed=0,
                fully_filled=False,
            )

        avg_price = total_cost / total_shares

        # Add base slippage for hidden market microstructure costs
        if side == "buy":
            avg_price *= (1 + self.BASE_SLIPPAGE_BPS / 10000)
        else:
            avg_price *= (1 - self.BASE_SLIPPAGE_BPS / 10000)

        # Calculate slippage vs midpoint
        if midpoint > 0:
            if side == "buy":
                slippage_bps = ((avg_price - midpoint) / midpoint) * 10000
            else:
                slippage_bps = ((midpoint - avg_price) / midpoint) * 10000
        else:
            slippage_bps = self.BASE_SLIPPAGE_BPS

        fully_filled = remaining <= 0.001

        if slippage_bps > 50:  # More than 0.5% slippage
            logger.warning(
                f"High slippage detected: {slippage_bps:.0f}bps "
                f"({side}, amount={amount:.2f}, levels={levels_used})")

        return FillResult(
            avg_fill_price=avg_price,
            total_filled=total_shares if side == "buy" else total_cost,
            total_cost=total_cost if side == "buy" else total_shares,
            slippage_bps=max(0, slippage_bps),
            levels_consumed=levels_used,
            fully_filled=fully_filled,
        )
