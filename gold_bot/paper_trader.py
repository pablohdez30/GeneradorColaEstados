"""
paper_trader.py - Paper trading simulator for the Gold Bot.

- Virtual balance
- Simulated fees (0.04%)
- Simulated slippage (+/-0.01%)
- Full position close (no scaling)
- ATR-based SL/TP (no trailing stop)
- Max trade duration enforcement
"""

import random
from datetime import datetime, timezone

import numpy as np

from gold_bot.config import (
    INITIAL_BALANCE, FEE_RATE, RISK_PCT, LEVERAGE,
    MAX_OPEN_POSITIONS, MAX_DRAWDOWN, MAX_TRADE_DURATION_MINUTES,
)
from gold_bot.logger import setup_logger, GoldTradeLogger

logger = setup_logger("gold_paper_trader")


class Position:
    """Represents an open position."""

    def __init__(self, trade_id: int, direction: str, entry_price: float,
                 quantity: float, stop_loss: float, take_profit: float):
        self.trade_id = trade_id
        self.direction = direction
        self.entry_price = entry_price
        self.quantity = quantity
        self.remaining_quantity = quantity
        self.stop_loss = stop_loss
        self.initial_stop_loss = stop_loss
        self.take_profit = take_profit
        self.open_time = datetime.now(timezone.utc)
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.partial_pnl = 0.0

    @property
    def duration_minutes(self) -> float:
        delta = datetime.now(timezone.utc) - self.open_time
        return delta.total_seconds() / 60

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "BUY":
            return (current_price - self.entry_price) * self.remaining_quantity
        else:
            return (self.entry_price - current_price) * self.remaining_quantity


class GoldPaperTrader:
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.trade_logger = GoldTradeLogger()
        self.open_positions: list[Position] = []
        self.closed_trades: list[dict] = []
        self.total_fees_paid = 0.0
        self.consecutive_losses = 0
        self.is_paused = False
        self.pause_reason = ""

        logger.info(
            f"GoldPaperTrader initialized | Balance: {initial_balance} USDT | "
            f"PAPER MODE | No leverage"
        )

    def _can_open_trade(self) -> tuple[bool, str]:
        """Check if we can open a new trade."""
        if self.is_paused:
            return False, f"Trading paused: {self.pause_reason}"

        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"Max positions reached ({MAX_OPEN_POSITIONS})"

        current_dd = self.current_drawdown()
        if current_dd >= MAX_DRAWDOWN:
            self.is_paused = True
            self.pause_reason = f"Max drawdown reached: {current_dd:.2%}"
            return False, self.pause_reason

        if self.balance < self.initial_balance * 0.05:
            self.is_paused = True
            self.pause_reason = "Critical balance"
            return False, self.pause_reason

        return True, "OK"

    def current_drawdown(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return max(0, (self.peak_balance - self.balance) / self.peak_balance)

    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Position sizing: risk RISK_PCT of capital per trade."""
        risk_amount = self.balance * RISK_PCT
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0.0

        quantity = risk_amount / price_risk

        # Limit: don't use more than 95% of balance
        max_quantity = (self.balance * 0.95) / entry_price
        quantity = min(quantity, max_quantity)

        # Minimum notional
        if quantity * entry_price < 10:
            return 0.0

        return quantity

    def execute_open(self, signal: dict, indicators: dict, regime: str) -> Position | None:
        """Open a position if validation passes."""
        can_trade, reason = self._can_open_trade()
        if not can_trade:
            logger.info(f"Trade rejected: {reason}")
            self.trade_logger.log_decision(
                "REJECTED", reason, indicators, signal.get("confidence", 0)
            )
            return None

        direction = signal["action"]
        entry_price = signal.get("entry_price", indicators.get("price", 0))
        stop_loss = signal.get("stop_loss", 0)
        take_profit = signal.get("take_profit", 0)

        if entry_price == 0 or stop_loss == 0 or take_profit == 0:
            logger.warning("Invalid signal prices, skipping trade")
            return None

        # Slippage simulation
        slippage = entry_price * random.uniform(-0.0001, 0.0001)
        entry_price += slippage

        # Position sizing
        quantity = self._calculate_position_size(entry_price, stop_loss)
        if quantity == 0:
            logger.info("Position size = 0, trade cancelled")
            return None

        # Opening fee
        notional_value = entry_price * quantity
        fee = notional_value * FEE_RATE
        self.balance -= fee
        self.total_fees_paid += fee

        # Log
        dir_label = "LONG" if direction == "BUY" else "SHORT"
        reasons_text = " | ".join(signal.get("reasons", []))
        justification = (
            f"Signal {dir_label} with {signal.get('confidence', 0):.0%} confidence. "
            f"{reasons_text}"
        )

        trade_id = self.trade_logger.log_trade_open(
            direction=direction, entry_price=entry_price, quantity=quantity,
            stop_loss=stop_loss, take_profit=take_profit, indicators=indicators,
            regime=regime, justification=justification,
        )

        position = Position(
            trade_id=trade_id, direction=direction, entry_price=entry_price,
            quantity=quantity, stop_loss=stop_loss, take_profit=take_profit,
        )

        self.open_positions.append(position)

        logger.info(
            f"OPEN #{trade_id} | {dir_label} @ {entry_price:.2f} | "
            f"qty={quantity:.6f} | notional={notional_value:.2f} | "
            f"SL={stop_loss:.2f} | TP={take_profit:.2f} | fee={fee:.4f}"
        )
        return position

    def execute_close(self, position: Position, price: float, quantity: float,
                      reason: str) -> float:
        """Close position (full or partial)."""
        slippage = price * random.uniform(-0.0001, 0.0001)
        exit_price = price + slippage

        # PnL calculation
        if position.direction == "BUY":
            pnl = (exit_price - position.entry_price) * quantity
        else:
            pnl = (position.entry_price - exit_price) * quantity

        # Closing fee
        fee = exit_price * quantity * FEE_RATE
        pnl -= fee
        self.total_fees_paid += fee

        # Update balance
        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)

        position.remaining_quantity -= quantity

        pnl_pct = pnl / (position.entry_price * quantity) if position.entry_price > 0 else 0

        # If position fully closed
        if position.remaining_quantity <= 0:
            total_pnl = pnl + position.partial_pnl
            total_pnl_pct = (
                total_pnl / (position.entry_price * position.quantity)
                if position.entry_price > 0 else 0
            )

            self.trade_logger.log_trade_close(
                position.trade_id, exit_price, total_pnl, total_pnl_pct
            )

            # Track consecutive losses for circuit breaker
            if total_pnl < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= 5:
                    self.is_paused = True
                    self.pause_reason = "5 consecutive losses"
                    logger.warning("Circuit breaker activated: 5 consecutive losses")
            else:
                self.consecutive_losses = 0

            self.closed_trades.append({
                "trade_id": position.trade_id,
                "direction": position.direction,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "pnl": total_pnl,
                "pnl_pct": total_pnl_pct,
            })

            if position in self.open_positions:
                self.open_positions.remove(position)
        else:
            position.partial_pnl += pnl

        logger.info(
            f"CLOSE #{position.trade_id} | {reason} | exit={exit_price:.2f} | "
            f"PnL={pnl:+.2f} ({pnl_pct:+.2%}) | fee={fee:.4f}"
        )
        return pnl

    def check_and_manage_positions(self, current_price: float):
        """Manage open positions: check SL, TP, max duration. No trailing stop."""
        for pos in list(self.open_positions):
            exit_signal = None

            # 1. Stop-loss
            if pos.direction == "BUY" and current_price <= pos.stop_loss:
                exit_signal = {
                    "price": current_price,
                    "quantity": pos.remaining_quantity,
                    "reason": f"Stop-loss hit at {pos.stop_loss:.2f}",
                }
            elif pos.direction == "SELL" and current_price >= pos.stop_loss:
                exit_signal = {
                    "price": current_price,
                    "quantity": pos.remaining_quantity,
                    "reason": f"Stop-loss hit at {pos.stop_loss:.2f}",
                }

            # 2. Take-profit
            if exit_signal is None:
                if pos.direction == "BUY" and current_price >= pos.take_profit:
                    exit_signal = {
                        "price": current_price,
                        "quantity": pos.remaining_quantity,
                        "reason": f"Take-profit hit at {pos.take_profit:.2f}",
                    }
                elif pos.direction == "SELL" and current_price <= pos.take_profit:
                    exit_signal = {
                        "price": current_price,
                        "quantity": pos.remaining_quantity,
                        "reason": f"Take-profit hit at {pos.take_profit:.2f}",
                    }

            # 3. Max trade duration (NO trailing stop for this strategy)
            if exit_signal is None:
                if pos.duration_minutes >= MAX_TRADE_DURATION_MINUTES:
                    exit_signal = {
                        "price": current_price,
                        "quantity": pos.remaining_quantity,
                        "reason": f"Max duration ({MAX_TRADE_DURATION_MINUTES}min) reached",
                    }

            if exit_signal:
                self.execute_close(
                    pos, exit_signal["price"],
                    exit_signal["quantity"], exit_signal["reason"]
                )

        # Log equity curve
        unrealized = sum(p.unrealized_pnl(current_price) for p in self.open_positions)
        drawdown = self.current_drawdown()
        self.trade_logger.log_equity(self.balance, unrealized, drawdown)

    def generate_report(self) -> str:
        """Generate performance report."""
        total_trades = len(self.closed_trades)
        if total_trades == 0:
            report = "No closed trades yet."
            logger.info(report)
            return report

        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        losses = total_trades - wins
        total_pnl = sum(t["pnl"] for t in self.closed_trades)
        win_rate = wins / total_trades * 100

        pnl_list = [t["pnl_pct"] for t in self.closed_trades]
        pnl_arr = np.array(pnl_list)
        sharpe = (
            (pnl_arr.mean() / pnl_arr.std() * np.sqrt(252))
            if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0
        )

        report = f"""
+==============================================================+
|           GOLD BOT REPORT - Golden Cross Strategy            |
+==============================================================+
|  Current Balance:    {self.balance:>10,.2f} USDT
|  Initial Balance:    {self.initial_balance:>10,.2f} USDT
|  Total PnL:          {total_pnl:>+10,.2f} USDT ({total_pnl / self.initial_balance:+.2%})
|--------------------------------------------------------------+
|  Total Trades:       {total_trades:>10}
|  Wins / Losses:      {wins:>5} / {losses}
|  Win Rate:           {win_rate:>9.1f}%
|  Sharpe Ratio:       {sharpe:>10.4f}
|  Max Drawdown:       {self.current_drawdown():>9.2%}
|--------------------------------------------------------------+
|  Fees Paid:          {self.total_fees_paid:>10.4f} USDT
|  Open Positions:     {len(self.open_positions):>10}
+==============================================================+
"""
        logger.info(report)
        return report
