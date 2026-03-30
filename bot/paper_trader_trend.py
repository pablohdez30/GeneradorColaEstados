"""
paper_trader_trend.py - Paper trader para el bot Trend Following.

Usa base de datos separada (trading_bot_trend.db) para no mezclar
resultados con el bot Sniper.
"""

import random
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INITIAL_BALANCE, FEE_RATE, PAPER_MODE, LEVERAGE, MARKET_TYPE
from bot.risk_manager import RiskManager, Position
from bot.logger import setup_logger, TradeLoggerTrend

logger = setup_logger("paper_trader_trend")


class PaperTrader:
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        if not PAPER_MODE:
            raise RuntimeError("PAPER_MODE está desactivado!")

        self.balance = initial_balance
        self.risk_manager = RiskManager(initial_balance)
        self.trade_logger = TradeLoggerTrend()
        self.open_positions: list[Position] = []
        self.closed_trades: list[dict] = []
        self.total_fees_paid = 0.0
        self.leverage = LEVERAGE
        self.market_type = MARKET_TYPE

        logger.info(f"PaperTrader Trend iniciado | Balance: {initial_balance} USDT | PAPER MODE")

    def execute_open(self, signal: dict, indicators: dict, regime: str) -> Position | None:
        can_trade, reason = self.risk_manager.can_open_trade()
        if not can_trade:
            logger.info(f"Trade rechazado: {reason}")
            return None

        direction = signal["action"]
        base_price = indicators["price"]
        atr = indicators.get("atr", base_price * 0.005)

        slippage = base_price * random.uniform(-0.0001, 0.0001)
        entry_price = base_price + slippage

        from bot.strategy_engine_trend import StrategyEngine
        engine = StrategyEngine()
        stop_loss = engine.compute_dynamic_stop_loss(entry_price, direction, atr, regime)
        take_profit = engine.compute_take_profit(entry_price, direction, atr)

        quantity = self.risk_manager.calculate_position_size(entry_price, stop_loss)
        if quantity == 0:
            return None

        notional_value = entry_price * quantity
        fee = notional_value * FEE_RATE
        self.balance -= fee
        self.total_fees_paid += fee

        dir_label = "LONG" if direction == "BUY" else "SHORT"
        reasons_text = " | ".join(signal.get("reasons", []))
        justification = f"Señal {dir_label} con {signal.get('confidence', 0):.0%} confianza. {reasons_text}"

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
        self.risk_manager.positions = self.open_positions

        logger.info(
            f"[TREND] OPEN #{trade_id} | {dir_label} @ {entry_price:.2f} | "
            f"qty={quantity:.6f} | SL={stop_loss:.2f} | TP={take_profit:.2f}"
        )
        return position

    def execute_close(self, position: Position, price: float, quantity: float,
                      reason: str) -> float:
        slippage = price * random.uniform(-0.0001, 0.0001)
        exit_price = price + slippage

        if position.direction == "BUY":
            pnl = (exit_price - position.entry_price) * quantity
        else:
            pnl = (position.entry_price - exit_price) * quantity

        fee = exit_price * quantity * FEE_RATE
        pnl -= fee
        self.total_fees_paid += fee

        self.balance += pnl
        self.risk_manager.update_balance(pnl)

        position.remaining_quantity -= quantity

        pnl_pct = pnl / (position.entry_price * quantity) if position.entry_price > 0 else 0

        if position.remaining_quantity <= 0:
            total_pnl = pnl + position.partial_pnl
            total_pnl_pct = total_pnl / (position.entry_price * position.quantity) if position.entry_price > 0 else 0

            self.trade_logger.log_trade_close(position.trade_id, exit_price, total_pnl, total_pnl_pct)
            self.risk_manager.record_trade_result(total_pnl)

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
            self.risk_manager.positions = self.open_positions
        else:
            position.partial_pnl += pnl

        logger.info(
            f"[TREND] CLOSE #{position.trade_id} | {reason} | exit={exit_price:.2f} | "
            f"PnL={pnl:+.2f} ({pnl_pct:+.2%})"
        )
        return pnl

    def check_and_manage_positions(self, current_price: float):
        for pos in list(self.open_positions):
            exit_signal = self.risk_manager.check_exit_conditions(pos, current_price)
            if exit_signal:
                self.execute_close(pos, exit_signal["price"],
                                   exit_signal["quantity"], exit_signal["reason"])

        unrealized = sum(p.unrealized_pnl(current_price) for p in self.open_positions)
        drawdown = self.risk_manager.current_drawdown()
        self.trade_logger.log_equity(self.balance, unrealized, drawdown)

    def generate_report(self) -> str:
        total_trades = len(self.closed_trades)
        if total_trades == 0:
            return "Sin trades cerrados aún."

        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        losses = total_trades - wins
        total_pnl = sum(t["pnl"] for t in self.closed_trades)
        win_rate = wins / total_trades * 100

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              REPORTE - TREND FOLLOWING BOT                  ║
╠══════════════════════════════════════════════════════════════╣
║  Balance actual:     {self.balance:>10,.2f} USDT
║  Balance inicial:    {INITIAL_BALANCE:>10,.2f} USDT
║  PnL total:          {total_pnl:>+10,.2f} USDT ({total_pnl/INITIAL_BALANCE:+.2%})
║  Total trades:       {total_trades:>10}
║  Wins / Losses:      {wins:>5} / {losses}
║  Win Rate:           {win_rate:>9.1f}%
║  Comisiones pagadas: {self.total_fees_paid:>10.4f} USDT
╚══════════════════════════════════════════════════════════════╝
"""
        logger.info(report)
        return report
