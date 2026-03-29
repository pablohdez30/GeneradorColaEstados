"""
risk_manager.py - Gestión de riesgo simplificada.

Reglas:
- 1% de riesgo por trade
- Sin apalancamiento (x1)
- 1 posición a la vez
- Stop loss + take profit basados en ATR
- Trailing stop una vez en beneficio
- Circuit breaker tras 5 pérdidas consecutivas
"""

import time
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MAX_RISK_PER_TRADE, MAX_OPEN_POSITIONS, MAX_DRAWDOWN,
    TRAILING_STOP_PCT, MAX_TRADE_DURATION_MINUTES,
    FEE_RATE, LEVERAGE,
)
from bot.logger import setup_logger

logger = setup_logger("risk_manager")


class Position:
    """Representa una posición abierta."""

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
        self.tp_levels_hit = []
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


class RiskManager:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.positions: list[Position] = []
        self.is_paused = False
        self.pause_reason = ""
        self.total_trades = 0
        self.consecutive_losses = 0
        logger.info(f"RiskManager inicializado | Balance: {initial_balance} USDT")

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Position sizing: arriesgar 1% del capital.
        Quantity = (capital * risk%) / |entry - stop|
        """
        risk_amount = self.current_balance * MAX_RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0.0

        quantity = risk_amount / price_risk

        # Límite: no usar más del 95% del balance como margen
        max_quantity = (self.current_balance * 0.95 * LEVERAGE) / entry_price
        quantity = min(quantity, max_quantity)

        if quantity * entry_price < 10:
            return 0.0

        margin_used = (quantity * entry_price) / LEVERAGE
        logger.debug(
            f"Position size: qty={quantity:.6f} | risk={risk_amount:.2f} USDT | "
            f"price_risk={price_risk:.2f} | leverage={LEVERAGE}x | margin={margin_used:.2f}"
        )
        return quantity

    def can_open_trade(self) -> tuple[bool, str]:
        """Verifica si se puede abrir trade."""
        if self.is_paused:
            return False, f"Trading pausado: {self.pause_reason}"

        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False, f"Máximo de posiciones alcanzado ({MAX_OPEN_POSITIONS})"

        current_dd = self.current_drawdown()
        if current_dd >= MAX_DRAWDOWN:
            self.is_paused = True
            self.pause_reason = f"Max drawdown alcanzado: {current_dd:.2%}"
            return False, self.pause_reason

        if self.current_balance < self.initial_balance * 0.05:
            self.is_paused = True
            self.pause_reason = "Balance crítico"
            return False, self.pause_reason

        return True, "OK"

    def current_drawdown(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return max(0, (self.peak_balance - self.current_balance) / self.peak_balance)

    def update_balance(self, pnl: float):
        self.current_balance += pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)

    def check_exit_conditions(self, pos: Position, current_price: float) -> dict | None:
        """
        Verifica condiciones de salida en orden de prioridad:
        1. Stop-loss
        2. Take-profit (cierre 100%)
        3. Trailing stop
        4. Duración máxima
        """
        # 1. Stop-loss
        if pos.direction == "BUY" and current_price <= pos.stop_loss:
            return {"type": "STOP_LOSS", "quantity": pos.remaining_quantity,
                    "price": current_price, "reason": f"Stop-loss activado en {pos.stop_loss:.2f}"}
        elif pos.direction == "SELL" and current_price >= pos.stop_loss:
            return {"type": "STOP_LOSS", "quantity": pos.remaining_quantity,
                    "price": current_price, "reason": f"Stop-loss activado en {pos.stop_loss:.2f}"}

        # 2. Take-profit (cierre completo)
        if pos.direction == "BUY" and current_price >= pos.take_profit:
            return {"type": "TAKE_PROFIT", "quantity": pos.remaining_quantity,
                    "price": current_price, "reason": f"Take-profit alcanzado en {pos.take_profit:.2f}"}
        elif pos.direction == "SELL" and current_price <= pos.take_profit:
            return {"type": "TAKE_PROFIT", "quantity": pos.remaining_quantity,
                    "price": current_price, "reason": f"Take-profit alcanzado en {pos.take_profit:.2f}"}

        # 3. Trailing stop: mover SL hacia arriba/abajo si hay beneficio
        self._update_trailing_stop(pos, current_price)

        # 4. Duración máxima
        if pos.duration_minutes >= MAX_TRADE_DURATION_MINUTES:
            return {"type": "MAX_DURATION", "quantity": pos.remaining_quantity,
                    "price": current_price,
                    "reason": f"Duración máxima ({MAX_TRADE_DURATION_MINUTES}min) alcanzada"}

        return None

    def _update_trailing_stop(self, pos: Position, price: float):
        if pos.direction == "BUY":
            if price > pos.highest_price:
                pos.highest_price = price
                new_stop = price * (1 - TRAILING_STOP_PCT)
                if new_stop > pos.stop_loss:
                    old_stop = pos.stop_loss
                    pos.stop_loss = new_stop
                    logger.debug(f"Trailing stop #{pos.trade_id}: {old_stop:.2f} → {new_stop:.2f}")
        else:
            if price < pos.lowest_price:
                pos.lowest_price = price
                new_stop = price * (1 + TRAILING_STOP_PCT)
                if new_stop < pos.stop_loss:
                    old_stop = pos.stop_loss
                    pos.stop_loss = new_stop
                    logger.debug(f"Trailing stop #{pos.trade_id}: {old_stop:.2f} → {new_stop:.2f}")

    def record_trade_result(self, pnl: float):
        """Registra resultado para circuit breaker."""
        self.total_trades += 1
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 5:
                self.is_paused = True
                self.pause_reason = f"5 pérdidas consecutivas"
                logger.warning(f"Circuit breaker activado: {self.consecutive_losses} pérdidas seguidas")
        else:
            self.consecutive_losses = 0

        # Auto-reset circuit breaker
        if self.is_paused and self.pause_reason.startswith("5 pérdidas"):
            # Reset after 30 minutes
            self.is_paused = False
            self.pause_reason = ""
            logger.info("Circuit breaker reseteado")

    def reset_circuit_breaker(self):
        self.is_paused = False
        self.pause_reason = ""
        self.consecutive_losses = 0
        logger.info("Circuit breaker reseteado manualmente")
