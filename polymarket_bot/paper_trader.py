"""
Paper Trading Engine: Simulates buying and selling with fake money.
Tracks portfolio, positions, P&L, and risk management.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .config import BotConfig
from .database import Database
from .slippage import SlippageSimulator
from .strategies.base import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open paper trading position."""
    id: int
    market_id: str
    market_question: str
    outcome: str
    entry_price: float
    shares: float
    amount_invested: float
    strategy: str
    stop_loss: float
    take_profit: float
    opened_at: str

    @property
    def current_value(self):
        """Must be calculated externally with current price."""
        return self.shares  # Placeholder — updated by engine

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        current_value = self.shares * current_price
        return current_value - self.amount_invested

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.amount_invested == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.amount_invested


class PaperTradingEngine:
    """
    Simulated trading engine.
    Manages a paper money portfolio with realistic trading mechanics.
    """

    def __init__(self, config: BotConfig, db: Database):
        self.config = config
        self.db = db
        self.balance = config.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_value = config.initial_balance
        self.circuit_breaker_active = False
        self.circuit_breaker_until = 0.0  # timestamp when cooldown ends
        self.slippage_sim = SlippageSimulator()
        self._load_state()

    def _load_state(self):
        """Load existing state from database."""
        history = self.db.get_portfolio_history(limit=1)
        if history:
            latest = history[0]
            self.balance = latest["balance"]
            self.total_trades = latest["total_trades"]
            self.winning_trades = latest["winning_trades"]
            self.losing_trades = latest["losing_trades"]
            logger.info(f"Loaded state: balance=${self.balance:.2f}, "
                        f"trades={self.total_trades}")
        else:
            logger.info(f"Starting fresh with ${self.balance:.2f}")

    def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        rows = self.db.get_open_positions()
        return [Position(
            id=r["id"],
            market_id=r["market_id"],
            market_question=r["market_question"] or "",
            outcome=r["outcome"],
            entry_price=r["entry_price"],
            shares=r["shares"],
            amount_invested=r["amount_invested"],
            strategy=r["strategy"],
            stop_loss=r["stop_loss"] or 0,
            take_profit=r["take_profit"] or 1,
            opened_at=r["opened_at"],
        ) for r in rows]

    def can_trade(self, amount: float) -> bool:
        """Check if we can make a trade of the given amount."""
        # Circuit breaker check
        if self._is_circuit_breaker_active():
            return False

        positions = self.get_open_positions()

        if len(positions) >= self.config.max_open_positions:
            logger.debug("Max open positions reached")
            return False

        if amount > self.balance:
            logger.debug(f"Insufficient balance: ${self.balance:.2f} < ${amount:.2f}")
            return False

        if amount < self.config.min_trade_size:
            logger.debug(f"Trade too small: ${amount:.2f}")
            return False

        if amount > self.config.max_position_size:
            logger.debug(f"Trade too large: ${amount:.2f}")
            return False

        # Check portfolio risk
        total_invested = sum(p.amount_invested for p in positions)
        total_value = self.balance + total_invested
        risk_ratio = (total_invested + amount) / total_value if total_value > 0 else 1

        if risk_ratio > self.config.max_portfolio_risk:
            logger.debug(f"Portfolio risk too high: {risk_ratio:.1%}")
            return False

        return True

    def _is_circuit_breaker_active(self) -> bool:
        """Check if the circuit breaker is preventing trades."""
        if not self.config.circuit_breaker_enabled:
            return False

        # Check cooldown
        if self.circuit_breaker_active:
            if time.time() < self.circuit_breaker_until:
                remaining = int(self.circuit_breaker_until - time.time())
                logger.debug(f"Circuit breaker active, {remaining}s remaining")
                return True
            else:
                logger.info("Circuit breaker cooldown ended, resuming trading")
                self.circuit_breaker_active = False
                return False

        # Check drawdown from peak
        current_value = self.portfolio_value
        if current_value > self.peak_value:
            self.peak_value = current_value

        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown >= self.config.circuit_breaker_drawdown_pct:
                self.circuit_breaker_active = True
                self.circuit_breaker_until = (
                    time.time() + self.config.circuit_breaker_cooldown_seconds)
                logger.warning(
                    f"CIRCUIT BREAKER TRIGGERED: drawdown {drawdown:.1%} "
                    f"(peak=${self.peak_value:.2f}, current=${current_value:.2f}). "
                    f"Trading paused for {self.config.circuit_breaker_cooldown_seconds}s")
                return True

        return False

    def execute_buy(self, signal: TradeSignal, market,
                    order_book: Optional[dict] = None) -> Optional[int]:
        """
        Execute a paper buy order with realistic slippage simulation.
        Returns position ID if successful, None otherwise.
        """
        # Calculate position size
        size_fraction = signal.suggested_size * signal.confidence
        amount = min(
            self.config.max_position_size * size_fraction,
            self.balance * 0.1,  # Max 10% of balance per trade
        )
        amount = max(amount, self.config.min_trade_size)

        if not self.can_trade(amount):
            return None

        # Check if we already have a position in this market/outcome
        existing = self.db.get_position_for_market(
            market.condition_id, signal.target_outcome)
        if existing:
            logger.debug(f"Already have position in {market.condition_id}/{signal.target_outcome}")
            return None

        # Determine midpoint price
        if signal.target_outcome == "yes":
            midpoint = market.outcome_yes_price
        else:
            midpoint = market.outcome_no_price

        if midpoint <= 0 or midpoint >= 1:
            logger.debug(f"Invalid price: {midpoint}")
            return None

        # Simulate slippage against real order book
        if self.config.slippage_enabled and order_book:
            fill = self.slippage_sim.simulate_buy(order_book, amount, midpoint)

            if not fill.fully_filled:
                logger.debug(f"Order book too thin for ${amount:.2f}")
                return None

            if fill.slippage_bps > self.config.max_slippage_bps:
                logger.info(
                    f"Rejected: slippage {fill.slippage_bps:.0f}bps > "
                    f"max {self.config.max_slippage_bps:.0f}bps")
                return None

            price = fill.avg_fill_price
            shares = fill.total_filled
            actual_cost = fill.total_cost
            slippage_info = f" | Slippage: {fill.slippage_bps:.0f}bps"
        else:
            # No order book — use midpoint (legacy behavior)
            price = midpoint
            shares = amount / price
            actual_cost = amount
            slippage_info = ""

        # Calculate stop loss and take profit prices
        stop_loss = price * (1 - self.config.stop_loss_pct)
        take_profit = price * (1 + self.config.take_profit_pct)

        # Deduct from balance
        self.balance -= actual_cost

        # Record in database
        trade_id = self.db.record_trade(
            market_id=market.condition_id,
            market_question=market.question,
            outcome=signal.target_outcome,
            side="buy",
            price=price,
            amount=actual_cost,
            shares=shares,
            strategy=signal.strategy_name,
            confidence=signal.confidence,
            reason=signal.reason,
        )

        position_id = self.db.open_position(
            market_id=market.condition_id,
            market_question=market.question,
            outcome=signal.target_outcome,
            entry_price=price,
            shares=shares,
            amount=actual_cost,
            strategy=signal.strategy_name,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.total_trades += 1

        logger.info(
            f"BUY {signal.target_outcome.upper()} | {market.question[:60]} | "
            f"Price: {price:.3f} | Amount: ${actual_cost:.2f} | "
            f"Shares: {shares:.2f} | Strategy: {signal.strategy_name} | "
            f"Confidence: {signal.confidence:.1%}{slippage_info}"
        )

        self._save_snapshot()
        return position_id

    def check_positions(self, get_current_price,
                        get_market_active=None) -> list[dict]:
        """
        Check all open positions for stop loss / take profit / market resolution.

        Args:
            get_current_price: callable(market_id, outcome) -> float
            get_market_active: callable(market_id) -> bool or None
                               Returns whether a market is still active.
                               If it returns False, the market has resolved.

        Returns list of closed position results.
        """
        closed = []
        positions = self.get_open_positions()

        for pos in positions:
            current_price = get_current_price(pos.market_id, pos.outcome)
            if current_price is None:
                continue

            self.db.update_position_price(pos.id, current_price)

            pnl_pct = pos.unrealized_pnl_pct(current_price)
            should_close = False
            reason = ""

            # Check 1: Market resolution detection
            if get_market_active is not None:
                is_active = get_market_active(pos.market_id)
                if is_active is False:
                    should_close = True
                    # Determine resolution: price near 1.0 means our outcome won
                    if current_price >= 0.95:
                        close_price = 1.0  # Full payout
                        reason = "Market resolved: outcome WON"
                    elif current_price <= 0.05:
                        close_price = 0.0  # Total loss
                        reason = "Market resolved: outcome LOST"
                    else:
                        close_price = current_price
                        reason = f"Market resolved (price={current_price:.3f})"

                    result = self._close_position(pos, close_price, reason)
                    closed.append(result)
                    continue

            # Check 2: Price snapped to extreme (likely resolved even if API says active)
            if current_price >= 0.99 or current_price <= 0.01:
                should_close = True
                close_price = 1.0 if current_price >= 0.99 else 0.0
                reason = f"Market resolved (price snapped to {current_price:.3f})"

            # Check 3: Stop loss
            elif pnl_pct <= -self.config.stop_loss_pct:
                should_close = True
                close_price = current_price
                reason = f"Stop loss triggered ({pnl_pct:.1%})"

            # Check 4: Take profit
            elif pnl_pct >= self.config.take_profit_pct:
                should_close = True
                close_price = current_price
                reason = f"Take profit triggered ({pnl_pct:.1%})"

            # Check 5: Near resolution (high probability of resolution)
            elif current_price >= 0.95 or current_price <= 0.05:
                should_close = True
                close_price = current_price
                reason = f"Market near resolution (price={current_price:.3f})"

            if should_close:
                result = self._close_position(pos, close_price, reason)
                closed.append(result)

        # Update peak value for circuit breaker
        current_value = self.portfolio_value
        if current_value > self.peak_value:
            self.peak_value = current_value

        return closed

    def _close_position(self, pos: Position, close_price: float,
                        reason: str) -> dict:
        """Close a position and record the result."""
        pnl = pos.unrealized_pnl(close_price)
        pnl_pct = pos.unrealized_pnl_pct(close_price)

        # Return capital + P&L to balance
        current_value = pos.shares * close_price
        self.balance += current_value

        # Update records
        self.db.close_position(pos.id)

        # Find and close the corresponding trade
        trades = self.db.get_recent_trades(limit=100)
        for trade in trades:
            if (trade["market_id"] == pos.market_id and
                    trade["outcome"] == pos.outcome and
                    trade["status"] == "open"):
                self.db.close_trade(trade["id"], close_price, pnl, pnl_pct)
                break

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        logger.info(
            f"CLOSE {pos.outcome.upper()} | {pos.market_question[:60]} | "
            f"Entry: {pos.entry_price:.3f} → Exit: {close_price:.3f} | "
            f"P&L: ${pnl:.2f} ({pnl_pct:.1%}) | Reason: {reason}"
        )

        self._save_snapshot()

        return {
            "market_id": pos.market_id,
            "outcome": pos.outcome,
            "entry_price": pos.entry_price,
            "close_price": close_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "strategy": pos.strategy,
            "reason": reason,
        }

    def force_close_all(self, get_current_price) -> list[dict]:
        """Force close all positions (for shutdown)."""
        closed = []
        for pos in self.get_open_positions():
            price = get_current_price(pos.market_id, pos.outcome)
            if price is not None:
                result = self._close_position(pos, price, "Force close")
                closed.append(result)
        return closed

    def _save_snapshot(self):
        """Save current portfolio state."""
        positions = self.get_open_positions()
        total_invested = sum(p.amount_invested for p in positions)
        total_value = self.balance + total_invested

        self.db.record_portfolio_snapshot(
            balance=self.balance,
            total_value=total_value,
            open_positions=len(positions),
            total_trades=self.total_trades,
            winning=self.winning_trades,
            losing=self.losing_trades,
        )

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (cash + invested)."""
        positions = self.get_open_positions()
        total_invested = sum(p.amount_invested for p in positions)
        return self.balance + total_invested

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        closed = self.winning_trades + self.losing_trades
        if closed == 0:
            return 0.0
        return self.winning_trades / closed

    def get_summary(self) -> dict:
        """Get portfolio summary."""
        positions = self.get_open_positions()
        current_value = self.portfolio_value
        drawdown = ((self.peak_value - current_value) / self.peak_value
                     if self.peak_value > 0 else 0)
        return {
            "balance": self.balance,
            "portfolio_value": current_value,
            "open_positions": len(positions),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "initial_balance": self.config.initial_balance,
            "total_return": (current_value - self.config.initial_balance)
                            / self.config.initial_balance if self.config.initial_balance > 0 else 0,
            "peak_value": self.peak_value,
            "drawdown": drawdown,
            "circuit_breaker_active": self.circuit_breaker_active,
        }
