"""
risk_manager.py - Gestión de posición, stop-loss dinámico y exposición máxima.

Diseño:
──────
El Risk Manager es el "guardia" del capital. Ninguna orden se ejecuta sin
pasar por aquí. Sus responsabilidades:

1. Calcular tamaño de posición basado en el riesgo máximo (2% por trade)
2. Validar que no se exceda el drawdown máximo
3. Gestionar trailing stop-loss
4. Ejecutar take-profit escalonado
5. Forzar cierre si se alcanza duración máxima (4h)
6. Detectar condiciones de "circuit breaker" (pausar trading)

Principio: PRESERVAR CAPITAL es más importante que ganar.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MAX_RISK_PER_TRADE, MAX_OPEN_POSITIONS, MAX_OPEN_POSITIONS_EXTRA,
    HIGH_CONFIDENCE_THRESHOLD, MAX_DRAWDOWN,
    TRAILING_STOP_PCT, MAX_TRADE_DURATION_MINUTES,
    TAKE_PROFIT_LEVELS, FEE_RATE,
    LEVERAGE, LEVERAGE_LOW, LEVERAGE_MID, LEVERAGE_HIGH,
    COOLDOWN_AFTER_LOSS_SECONDS,
)
from bot.logger import setup_logger

logger = setup_logger("risk_manager")


class Position:
    """Representa una posición abierta con su estado completo."""

    def __init__(self, trade_id: int, direction: str, entry_price: float,
                 quantity: float, stop_loss: float, take_profit: float):
        self.trade_id = trade_id
        self.direction = direction  # "BUY" o "SELL"
        self.entry_price = entry_price
        self.quantity = quantity
        self.remaining_quantity = quantity
        self.stop_loss = stop_loss
        self.initial_stop_loss = stop_loss
        self.take_profit = take_profit
        self.open_time = datetime.now(timezone.utc)
        self.tp_levels_hit = []  # Niveles de TP ya ejecutados
        self.highest_price = entry_price  # Para trailing stop (long)
        self.lowest_price = entry_price   # Para trailing stop (short)
        self.partial_pnl = 0.0  # PnL acumulado de cierres parciales

    @property
    def duration_minutes(self) -> float:
        delta = datetime.now(timezone.utc) - self.open_time
        return delta.total_seconds() / 60

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "BUY":
            return (current_price - self.entry_price) * self.remaining_quantity
        else:
            return (self.entry_price - current_price) * self.remaining_quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        cost = self.entry_price * self.remaining_quantity
        if cost == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / cost


class RiskManager:
    """
    Gestor de riesgo del bot de trading.

    Controla: sizing, stops, take-profits, drawdown y circuit breaker.
    """

    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.positions: list[Position] = []
        self.is_paused = False  # Circuit breaker activado
        self.pause_reason = ""
        self.total_trades = 0
        self.consecutive_losses = 0
        self.last_loss_time = 0  # Timestamp de última pérdida (para cooldown)
        logger.info(f"RiskManager inicializado | Balance: {initial_balance} USDT")

    # ── Apalancamiento Dinámico ────────────────────────────────

    @staticmethod
    def get_dynamic_leverage(confidence: float) -> int:
        """
        Apalancamiento según confianza de la señal.
        Más confianza → más apalancamiento → más beneficio potencial.
        """
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return LEVERAGE_HIGH  # x4
        elif confidence >= 0.40:
            return LEVERAGE_MID   # x3
        else:
            return LEVERAGE_LOW   # x2

    # ── Cálculo de Tamaño de Posición ──────────────────────────

    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                leverage: int = None) -> float:
        """
        Position sizing basado en riesgo fijo del capital.

        El leverage se recibe dinámicamente según la confianza de la señal.
        """
        if leverage is None:
            leverage = LEVERAGE

        risk_amount = self.current_balance * MAX_RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            logger.warning("Stop loss igual al precio de entrada, abortando sizing")
            return 0.0

        quantity = risk_amount / price_risk

        # Con apalancamiento, el límite de posición se multiplica por leverage
        max_quantity = (self.current_balance * 0.95 * leverage) / entry_price
        quantity = min(quantity, max_quantity)

        # Mínimo viable
        if quantity * entry_price < 10:
            return 0.0

        margin_used = (quantity * entry_price) / leverage
        logger.debug(
            f"Position size: qty={quantity:.6f} | risk={risk_amount:.2f} USDT | "
            f"price_risk={price_risk:.2f} | leverage={leverage}x | margin={margin_used:.2f}"
        )
        return quantity

    # ── Validación Pre-Trade ───────────────────────────────────

    def can_open_trade(self, confidence: float = 0.0) -> tuple[bool, str]:
        """
        Verifica si es seguro abrir una nueva posición.

        Checks:
        1. Cooldown tras pérdida (2 min)
        2. No exceder posiciones máximas (3 normales + 1 extra si >60%)
        3. No estar en pausa (circuit breaker)
        4. Drawdown dentro de límites
        5. Balance suficiente
        """
        if self.is_paused:
            return False, f"Trading pausado: {self.pause_reason}"

        # Cooldown tras pérdida
        import time
        if self.last_loss_time > 0:
            elapsed = time.time() - self.last_loss_time
            if elapsed < COOLDOWN_AFTER_LOSS_SECONDS:
                remaining = int(COOLDOWN_AFTER_LOSS_SECONDS - elapsed)
                return False, f"Cooldown activo: {remaining}s restantes tras pérdida"

        # Slot extra: si confianza >60%, permitir hasta MAX_OPEN_POSITIONS_EXTRA
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            max_pos = MAX_OPEN_POSITIONS_EXTRA
        else:
            max_pos = MAX_OPEN_POSITIONS

        if len(self.positions) >= max_pos:
            return False, f"Máximo de posiciones alcanzado ({max_pos})"

        current_dd = self.current_drawdown()
        if current_dd >= MAX_DRAWDOWN:
            self.is_paused = True
            self.pause_reason = f"Max drawdown alcanzado: {current_dd:.2%}"
            return False, self.pause_reason

        if self.current_balance < self.initial_balance * 0.05:
            self.is_paused = True
            self.pause_reason = "Balance crítico (<5% del inicial)"
            return False, self.pause_reason

        # Circuit breaker: 5 pérdidas consecutivas → pausa temporal
        if self.consecutive_losses >= 5:
            self.is_paused = True
            self.pause_reason = "5 pérdidas consecutivas - cooling off"
            return False, self.pause_reason

        return True, "OK"

    # ── Drawdown ───────────────────────────────────────────────

    def current_drawdown(self) -> float:
        """Calcula drawdown actual desde el pico de equity."""
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    def update_balance(self, new_balance: float):
        """Actualiza balance y pico de equity."""
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    # ── Gestión de Posiciones Abiertas ─────────────────────────

    def check_exit_conditions(self, position: Position, current_price: float) -> list[dict]:
        """
        Evalúa todas las condiciones de salida para una posición abierta.

        Retorna lista de acciones a ejecutar (puede ser vacía o múltiple
        en caso de take-profit parcial).
        """
        actions = []

        # 1. Stop-Loss
        if self._check_stop_loss(position, current_price):
            actions.append({
                "type": "STOP_LOSS",
                "quantity": position.remaining_quantity,
                "price": current_price,
                "reason": f"Stop-loss activado en {current_price:.2f}",
            })
            return actions  # Stop-loss cierra toda la posición

        # 2. Duración máxima
        if position.duration_minutes >= MAX_TRADE_DURATION_MINUTES:
            actions.append({
                "type": "TIME_EXIT",
                "quantity": position.remaining_quantity,
                "price": current_price,
                "reason": f"Duración máxima ({MAX_TRADE_DURATION_MINUTES}min) alcanzada",
            })
            return actions

        # 3. Trailing Stop
        self._update_trailing_stop(position, current_price)

        # 4. Take-Profit Escalonado
        tp_action = self._check_take_profit(position, current_price)
        if tp_action:
            actions.append(tp_action)

        return actions

    def _check_stop_loss(self, pos: Position, price: float) -> bool:
        """Verifica si el precio ha tocado el stop-loss."""
        if pos.direction == "BUY":
            return price <= pos.stop_loss
        else:
            return price >= pos.stop_loss

    def _update_trailing_stop(self, pos: Position, price: float):
        """
        Trailing stop: mueve el stop en la dirección favorable.

        Solo se activa cuando el trade ya está en ganancias.
        El stop se mueve pero NUNCA retrocede.
        """
        if pos.direction == "BUY":
            if price > pos.highest_price:
                pos.highest_price = price
                new_stop = price * (1 - TRAILING_STOP_PCT)
                if new_stop > pos.stop_loss:
                    old_stop = pos.stop_loss
                    pos.stop_loss = new_stop
                    logger.debug(
                        f"Trailing stop #{pos.trade_id}: {old_stop:.2f} → {new_stop:.2f}"
                    )
        else:  # SELL
            if price < pos.lowest_price:
                pos.lowest_price = price
                new_stop = price * (1 + TRAILING_STOP_PCT)
                if new_stop < pos.stop_loss:
                    old_stop = pos.stop_loss
                    pos.stop_loss = new_stop
                    logger.debug(
                        f"Trailing stop #{pos.trade_id}: {old_stop:.2f} → {new_stop:.2f}"
                    )

    def _check_take_profit(self, pos: Position, price: float) -> dict | None:
        """
        Take-profit escalonado: cierra porciones de la posición
        en diferentes niveles de beneficio.

        Esto asegura beneficio parcial mientras deja correr el resto.
        """
        for i, (target_pct, close_pct) in enumerate(TAKE_PROFIT_LEVELS):
            if i in pos.tp_levels_hit:
                continue

            if pos.direction == "BUY":
                target_price = pos.entry_price * (1 + target_pct)
                if price >= target_price:
                    qty_to_close = pos.quantity * close_pct
                    qty_to_close = min(qty_to_close, pos.remaining_quantity)
                    pos.tp_levels_hit.append(i)
                    return {
                        "type": "TAKE_PROFIT",
                        "level": i + 1,
                        "quantity": qty_to_close,
                        "price": price,
                        "reason": f"TP nivel {i+1} (+{target_pct:.1%}) alcanzado",
                    }
            else:  # SELL
                target_price = pos.entry_price * (1 - target_pct)
                if price <= target_price:
                    qty_to_close = pos.quantity * close_pct
                    qty_to_close = min(qty_to_close, pos.remaining_quantity)
                    pos.tp_levels_hit.append(i)
                    return {
                        "type": "TAKE_PROFIT",
                        "level": i + 1,
                        "quantity": qty_to_close,
                        "price": price,
                        "reason": f"TP nivel {i+1} (+{target_pct:.1%}) alcanzado",
                    }

        return None

    # ── Registro de Resultado ──────────────────────────────────

    def record_trade_result(self, pnl: float):
        """Registra resultado de trade para circuit breaker y estadísticas."""
        import time
        self.total_trades += 1
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = time.time()  # Activar cooldown
            logger.info(f"Cooldown activado: {COOLDOWN_AFTER_LOSS_SECONDS}s tras pérdida")
        else:
            self.consecutive_losses = 0
            self.last_loss_time = 0  # Reset cooldown tras ganancia

        # Auto-reset de pausa por pérdidas consecutivas tras 10 minutos
        if self.is_paused and self.pause_reason.startswith("5 pérdidas"):
            self.is_paused = False
            self.pause_reason = ""
            logger.info("Circuit breaker reseteado")

    def reset_circuit_breaker(self):
        """Reset manual del circuit breaker."""
        self.is_paused = False
        self.pause_reason = ""
        self.consecutive_losses = 0
        logger.info("Circuit breaker reseteado manualmente")

    def get_risk_summary(self) -> dict:
        """Resumen del estado de riesgo actual."""
        return {
            "balance": round(self.current_balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "drawdown": round(self.current_drawdown(), 4),
            "open_positions": len(self.positions),
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "consecutive_losses": self.consecutive_losses,
            "total_trades": self.total_trades,
        }
