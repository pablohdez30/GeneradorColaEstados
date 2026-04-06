"""Paper trader for Gold bot - simulates trades without real money."""

import time
from gold_bot.config import (
    INITIAL_BALANCE, FEE_RATE, RISK_PCT, MAX_TRADE_DURATION_MINUTES,
    SL_ATR_MULT, TP_RATIO,
)
from gold_bot.logger import setup_logger, GoldTradeLogger

logger = setup_logger("gold_paper_trader")


class GoldPaperTrader:
    def __init__(self, initial_balance=INITIAL_BALANCE):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.position = None
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_logger = GoldTradeLogger()

    def execute_open(self, direction, price, stop_loss, take_profit, indicators=None, reasons=None):
        if self.position is not None:
            return False

        sl_dist = abs(price - stop_loss)
        if sl_dist == 0:
            return False

        qty = (self.balance * RISK_PCT) / sl_dist
        if qty * price < 10:
            return False

        fee = price * qty * FEE_RATE
        slippage = price * 0.0001
        entry_price = price + slippage if direction == "BUY" else price - slippage

        self.position = {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "quantity": qty,
            "open_time": time.time(),
            "fee_paid": fee,
        }
        self.balance -= fee

        logger.info(
            f"OPEN {direction} @ {entry_price:.2f} | "
            f"SL={stop_loss:.2f} TP={take_profit:.2f} | "
            f"Qty={qty:.6f} | Fee={fee:.2f}"
        )

        self.trade_logger.log_trade_open(
            direction=direction,
            entry_price=entry_price,
            quantity=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicators=indicators or {},
            regime="GOLD",
            justification=", ".join(reasons) if reasons else "",
        )
        return True

    def execute_close(self, price, reason="manual"):
        if self.position is None:
            return False

        pos = self.position
        slippage = price * 0.0001
        exit_price = price - slippage if pos["direction"] == "BUY" else price + slippage

        if pos["direction"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]

        fee = exit_price * pos["quantity"] * FEE_RATE
        pnl -= fee
        pnl_pct = pnl / (pos["entry_price"] * pos["quantity"]) * 100

        self.balance += pnl
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        self.peak_balance = max(self.peak_balance, self.balance)

        logger.info(
            f"CLOSE {pos['direction']} @ {exit_price:.2f} | "
            f"PnL={pnl:+.2f} ({pnl_pct:+.1f}%) | "
            f"Reason={reason} | Balance={self.balance:.2f}"
        )

        self.trade_logger.log_trade_close(
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

        self.position = None
        return True

    def check_and_manage_positions(self, current_price):
        if self.position is None:
            return

        pos = self.position
        duration_min = (time.time() - pos["open_time"]) / 60

        if pos["direction"] == "BUY":
            if current_price <= pos["stop_loss"]:
                self.execute_close(pos["stop_loss"], "stop_loss")
            elif current_price >= pos["take_profit"]:
                self.execute_close(pos["take_profit"], "take_profit")
            elif duration_min >= MAX_TRADE_DURATION_MINUTES:
                self.execute_close(current_price, "max_duration")
        else:
            if current_price >= pos["stop_loss"]:
                self.execute_close(pos["stop_loss"], "stop_loss")
            elif current_price <= pos["take_profit"]:
                self.execute_close(pos["take_profit"], "take_profit")
            elif duration_min >= MAX_TRADE_DURATION_MINUTES:
                self.execute_close(current_price, "max_duration")

        # Log equity
        unrealized = 0
        if self.position:
            if self.position["direction"] == "BUY":
                unrealized = (current_price - self.position["entry_price"]) * self.position["quantity"]
            else:
                unrealized = (self.position["entry_price"] - current_price) * self.position["quantity"]

        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.trade_logger.log_equity(self.balance, unrealized, drawdown)

    def generate_report(self):
        wr = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        dd = (self.peak_balance - self.balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
        return {
            "balance": round(self.balance, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_trades": self.total_trades,
            "win_rate": round(wr, 1),
            "drawdown": round(dd, 1),
        }
