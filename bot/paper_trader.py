"""
paper_trader.py - Simulador de órdenes con balance virtual y registro de trades.

Diseño:
──────
Este módulo simula la ejecución de órdenes como si fuera un exchange real,
pero sin tocar fondos reales. Implementa:

- Balance virtual configurable
- Comisiones simuladas (0.1% taker fee de Binance)
- Slippage simulado (±0.01% aleatorio para realismo)
- Registro completo de cada operación en SQLite
- Generación de reportes semanales automáticos

El paper trader es el ÚNICO módulo que gestiona el balance.
Ni strategy_engine ni risk_manager modifican el balance directamente.
"""

import random
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INITIAL_BALANCE, FEE_RATE, PAPER_MODE, LEVERAGE, MARKET_TYPE
from bot.risk_manager import RiskManager, Position
from bot.logger import setup_logger, TradeLogger

logger = setup_logger("paper_trader")


class PaperTrader:
    """
    Simulador de trading en paper mode.

    Gestiona el ciclo de vida completo de las operaciones:
    señal → validación → ejecución → gestión → cierre → registro.
    """

    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        if not PAPER_MODE:
            raise RuntimeError(
                "¡PAPER_MODE está desactivado! Este bot SOLO opera en simulación. "
                "Para operar en real, se requiere implementación separada con "
                "autenticación y confirmación explícita."
            )

        self.balance = initial_balance
        self.risk_manager = RiskManager(initial_balance)
        self.trade_logger = TradeLogger()
        self.open_positions: list[Position] = []
        self.closed_trades: list[dict] = []
        self.total_fees_paid = 0.0
        self.leverage = LEVERAGE
        self.market_type = MARKET_TYPE

        mode_str = f"FUTUROS x{LEVERAGE}" if MARKET_TYPE == "futures" else "SPOT"
        logger.info(f"PaperTrader iniciado | Balance: {initial_balance} USDT | {mode_str} | PAPER MODE")

    def execute_open(self, signal: dict, indicators: dict, regime: str) -> Position | None:
        """
        Ejecuta apertura de posición si pasa validación de riesgo.

        Flujo:
        1. Verificar que se puede abrir trade (risk_manager)
        2. Calcular stop-loss dinámico
        3. Calcular tamaño de posición
        4. Simular ejecución con slippage
        5. Registrar en log y DB
        """
        can_trade, reason = self.risk_manager.can_open_trade()
        if not can_trade:
            logger.info(f"Trade rechazado: {reason}")
            self.trade_logger.log_decision(
                "REJECTED", reason, indicators, signal.get("confidence", 0)
            )
            return None

        direction = signal["action"]  # "BUY" o "SELL"
        base_price = indicators["price"]
        atr = indicators.get("atr", base_price * 0.005)  # Fallback: 0.5% del precio

        # Simular slippage (±0.01%)
        slippage = base_price * random.uniform(-0.0001, 0.0001)
        entry_price = base_price + slippage

        # Stop-loss dinámico basado en ATR y régimen
        from bot.strategy_engine import StrategyEngine
        engine = StrategyEngine()
        stop_loss = engine.compute_dynamic_stop_loss(entry_price, direction, atr, regime)
        take_profit = engine.compute_take_profit(entry_price, direction, atr)

        # Calcular cantidad
        quantity = self.risk_manager.calculate_position_size(entry_price, stop_loss)
        if quantity == 0:
            logger.info("Position size = 0, trade cancelado")
            return None

        # Simular comisión de apertura (sobre el valor nocional completo)
        notional_value = entry_price * quantity
        fee = notional_value * FEE_RATE
        self.balance -= fee
        self.total_fees_paid += fee

        # Calcular margen requerido (con apalancamiento)
        margin_required = notional_value / self.leverage if self.market_type == "futures" else notional_value

        # Justificación para el log
        dir_label = f"LONG" if direction == "BUY" else "SHORT"
        reasons_text = " | ".join(signal.get("reasons", []))
        justification = (
            f"Señal {dir_label} x{self.leverage} con {signal.get('confidence', 0):.0%} confluencia. "
            f"Razones: {reasons_text}"
        )

        # Registrar en DB
        trade_id = self.trade_logger.log_trade_open(
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicators=indicators,
            regime=regime,
            justification=justification,
        )

        # Crear posición
        position = Position(
            trade_id=trade_id,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.open_positions.append(position)
        self.risk_manager.positions = self.open_positions

        dir_log = "LONG" if direction == "BUY" else "SHORT"
        logger.info(
            f"OPEN #{trade_id} | {dir_log} x{self.leverage} @ {entry_price:.2f} | "
            f"qty={quantity:.6f} | margin={margin_required:.2f} | "
            f"SL={stop_loss:.2f} | TP={take_profit:.2f} | fee={fee:.4f}"
        )
        return position

    def execute_close(self, position: Position, price: float, quantity: float,
                      reason: str) -> float:
        """
        Ejecuta cierre (total o parcial) de posición.

        Retorna el PnL realizado.
        """
        # Simular slippage en salida
        slippage = price * random.uniform(-0.0001, 0.0001)
        exit_price = price + slippage

        # Calcular PnL (con apalancamiento, las ganancias/pérdidas se multiplican)
        if position.direction == "BUY":  # LONG
            pnl = (exit_price - position.entry_price) * quantity
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * quantity

        # Comisión de cierre (sobre valor nocional)
        fee = exit_price * quantity * FEE_RATE
        pnl -= fee
        self.total_fees_paid += fee

        # Actualizar cantidad restante
        position.remaining_quantity -= quantity
        position.partial_pnl += pnl

        # Actualizar balance
        self.balance += pnl
        self.risk_manager.update_balance(self.balance)

        is_full_close = position.remaining_quantity <= 0.000001

        if is_full_close:
            # Cierre completo: registrar y eliminar posición
            total_pnl = position.partial_pnl
            cost = position.entry_price * position.quantity
            pnl_pct = total_pnl / cost if cost > 0 else 0

            self.trade_logger.log_trade_close(
                position.trade_id, exit_price, total_pnl, pnl_pct
            )
            self.risk_manager.record_trade_result(total_pnl)

            self.closed_trades.append({
                "trade_id": position.trade_id,
                "direction": position.direction,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "quantity": position.quantity,
                "pnl": total_pnl,
                "pnl_pct": pnl_pct,
                "duration_min": position.duration_minutes,
                "reason": reason,
            })

            self.open_positions = [p for p in self.open_positions if p.trade_id != position.trade_id]
            self.risk_manager.positions = self.open_positions

            logger.info(
                f"CLOSE #{position.trade_id} | {reason} | exit={exit_price:.2f} | "
                f"PnL={total_pnl:.2f} ({pnl_pct:+.2%}) | fee={fee:.4f}"
            )
        else:
            logger.info(
                f"PARTIAL CLOSE #{position.trade_id} | {reason} | "
                f"qty_closed={quantity:.6f} | remaining={position.remaining_quantity:.6f} | "
                f"partial_pnl={pnl:.2f}"
            )

        # Registrar punto de equity
        unrealized = sum(
            p.unrealized_pnl(exit_price) for p in self.open_positions
        )
        self.trade_logger.log_equity(
            self.balance, unrealized, self.risk_manager.current_drawdown()
        )

        return pnl

    def check_and_manage_positions(self, current_price: float):
        """
        Loop de gestión: revisa todas las posiciones abiertas y ejecuta
        salidas si las condiciones de riesgo lo requieren.

        Se llama en cada tick del bot principal.
        """
        for position in list(self.open_positions):
            exit_actions = self.risk_manager.check_exit_conditions(position, current_price)

            for action in exit_actions:
                self.execute_close(
                    position,
                    action["price"],
                    action["quantity"],
                    action["reason"],
                )

    def get_portfolio_summary(self) -> dict:
        """Resumen completo del portafolio para el dashboard."""
        total_pnl = self.balance - INITIAL_BALANCE
        total_trades = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        losses = sum(1 for t in self.closed_trades if t["pnl"] <= 0)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Sharpe Ratio (simplificado)
        if total_trades > 1:
            returns = [t["pnl_pct"] for t in self.closed_trades]
            import numpy as np
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe = 0

        return {
            "balance": round(self.balance, 2),
            "initial_balance": INITIAL_BALANCE,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / INITIAL_BALANCE, 4),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(self.risk_manager.current_drawdown(), 4),
            "open_positions": len(self.open_positions),
            "total_fees": round(self.total_fees_paid, 4),
            "risk": self.risk_manager.get_risk_summary(),
        }

    def generate_weekly_report(self) -> str:
        """Genera reporte semanal textual con estadísticas."""
        summary = self.get_portfolio_summary()
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                 REPORTE SEMANAL - SCALPING BOT              ║
╠══════════════════════════════════════════════════════════════╣
║  Balance actual:     {summary['balance']:>12,.2f} USDT
║  Balance inicial:    {summary['initial_balance']:>12,.2f} USDT
║  PnL total:          {summary['total_pnl']:>+12,.2f} USDT ({summary['total_pnl_pct']:+.2%})
║──────────────────────────────────────────────────────────────║
║  Total trades:       {summary['total_trades']:>12}
║  Wins / Losses:      {summary['wins']:>5} / {summary['losses']:<5}
║  Win Rate:           {summary['win_rate']:>12.1%}
║  Sharpe Ratio:       {summary['sharpe_ratio']:>12.4f}
║  Max Drawdown:       {summary['max_drawdown']:>12.2%}
║──────────────────────────────────────────────────────────────║
║  Comisiones pagadas: {summary['total_fees']:>12,.4f} USDT
║  Posiciones abiertas:{summary['open_positions']:>11}
╚══════════════════════════════════════════════════════════════╝
"""
        logger.info(report)
        return report
