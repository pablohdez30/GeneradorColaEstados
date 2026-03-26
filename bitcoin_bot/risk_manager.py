"""
risk_manager.py — Gestión de riesgo y control de posición
===========================================================
Responsabilidades:
  1. Calcular el tamaño óptimo de posición (position sizing)
  2. Verificar si una operación está permitida (límites de riesgo)
  3. Gestionar trailing stop dinámico
  4. Detectar niveles de drawdown y activar parada de emergencia
  5. Calcular métricas de riesgo: Sharpe, Sortino, Max Drawdown

Regla de oro: el riesgo máximo por operación es el 2% del capital.
  Si el stop-loss está a X distancia, el tamaño se ajusta para que
  la pérdida máxima sea el 2% del balance, no más.

Decisión de diseño: el risk manager es BLOQUEANTE — si no aprueba
una operación, esta no se ejecuta bajo ninguna circunstancia.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone
import numpy as np

from config import CONFIG
from logger import LOG, EventType
from strategy_engine import Signal


# ─────────────────────────────────────────────
#  POSICIÓN ACTIVA
# ─────────────────────────────────────────────
@dataclass
class Position:
    """Representa una posición abierta."""
    trade_id: str
    direction: Signal
    entry_price: float
    size_btc: float
    size_usdt: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    tp1_hit: bool = False           # ¿Se alcanzó el primer TP?
    entry_ts: float = field(default_factory=time.time)
    trailing_stop: Optional[float] = None   # precio de trailing stop activo
    highest_price: float = 0.0      # máximo alcanzado (para trailing)
    lowest_price: float = float("inf")  # mínimo alcanzado

    @property
    def duration_minutes(self) -> float:
        return (time.time() - self.entry_ts) / 60

    @property
    def is_long(self) -> bool:
        return self.direction == Signal.LONG

    def unrealized_pnl(self, current_price: float) -> float:
        """P&L no realizado en USDT."""
        if self.is_long:
            return (current_price - self.entry_price) * self.size_btc
        else:
            return (self.entry_price - current_price) * self.size_btc

    def unrealized_pnl_pct(self, current_price: float) -> float:
        return self.unrealized_pnl(current_price) / self.size_usdt


# ─────────────────────────────────────────────
#  RISK MANAGER
# ─────────────────────────────────────────────
class RiskManager:
    """
    Controlador de riesgo centralizado. Toda apertura y gestión de
    posición pasa por este módulo.
    """

    def __init__(self):
        self.cfg = CONFIG.risk
        self._position: Optional[Position] = None
        self._daily_pnl: float = 0.0
        self._session_start_balance: float = CONFIG.paper.initial_balance_usdt
        self._peak_balance: float = CONFIG.paper.initial_balance_usdt
        self._max_drawdown: float = 0.0
        self._last_loss_ts: float = 0.0      # para cooldown tras pérdida
        self._trade_count: int = 0

        LOG.log(EventType.SYSTEM, "RiskManager inicializado",
                max_risk=f"{self.cfg.max_risk_per_trade:.0%}",
                max_drawdown=f"{self.cfg.max_total_drawdown:.0%}")

    # ── VALIDACIÓN DE ENTRADA ───────────────────

    def can_open_trade(self, balance: float, signal: Signal,
                       confidence: float) -> Tuple[bool, str]:
        """
        Verifica si se puede abrir una nueva posición.
        Devuelve (permitido, motivo_de_bloqueo).
        """
        # 1. Posición ya abierta (solo 1 posición simultánea)
        if self._position is not None:
            return False, "Ya hay una posición abierta"

        # 2. Drawdown diario excesivo
        if abs(self._daily_pnl) / self._session_start_balance > self.cfg.max_daily_drawdown:
            return False, (
                f"Drawdown diario {self._daily_pnl:.2f} USDT supera el límite "
                f"({self.cfg.max_daily_drawdown:.0%})"
            )

        # 3. Drawdown total excesivo — PARADA DE EMERGENCIA
        current_drawdown = (self._peak_balance - balance) / self._peak_balance
        if current_drawdown > self.cfg.max_total_drawdown:
            LOG.error(EventType.RISK_BLOCK,
                      f"PARADA DE EMERGENCIA: Drawdown total {current_drawdown:.1%} "
                      f"supera límite {self.cfg.max_total_drawdown:.0%}")
            return False, f"Parada de emergencia: drawdown {current_drawdown:.1%}"

        # 4. Cooldown tras pérdida reciente
        cooldown = CONFIG.trading.cooldown_after_loss_seconds
        if time.time() - self._last_loss_ts < cooldown:
            remaining = cooldown - (time.time() - self._last_loss_ts)
            return False, f"Cooldown tras stop-loss: {remaining:.0f}s restantes"

        # 5. Confianza mínima de la señal
        if confidence < 0.55:
            return False, f"Confianza de señal insuficiente ({confidence:.1%} < 55%)"

        # 6. Balance insuficiente
        if balance < 50:
            return False, f"Balance insuficiente: {balance:.2f} USDT"

        return True, "OK"

    def calculate_position_size(self, balance: float, entry_price: float,
                                 stop_loss: float) -> Tuple[float, float]:
        """
        Calcula el tamaño de posición usando el método de riesgo fijo.

        Fórmula: size_usdt = (balance × max_risk) / (distancia_sl_%)
        Esto garantiza que la pérdida máxima sea exactamente el 2% del balance.

        Devuelve: (size_btc, size_usdt)
        """
        sl_distance_pct = abs(entry_price - stop_loss) / entry_price
        if sl_distance_pct <= 0:
            LOG.warning(EventType.RISK_BLOCK, "SL distance = 0, abortando")
            return 0.0, 0.0

        max_loss_usdt = balance * self.cfg.max_risk_per_trade
        size_usdt = max_loss_usdt / sl_distance_pct

        # No usar más del 95% del balance total
        size_usdt = min(size_usdt, balance * 0.95)

        # Redondear al paso mínimo de Binance (0.00001 BTC)
        size_btc = size_usdt / entry_price
        size_btc = math.floor(size_btc * 100000) / 100000  # 5 decimales

        size_usdt = size_btc * entry_price  # recalcular tras redondeo

        LOG.debug(EventType.SYSTEM,
                  f"Position sizing: {size_btc:.5f} BTC ({size_usdt:.2f} USDT) | "
                  f"Riesgo: {max_loss_usdt:.2f} USDT ({self.cfg.max_risk_per_trade:.0%})",
                  price=entry_price)

        return size_btc, size_usdt

    # ── GESTIÓN DE POSICIÓN ABIERTA ─────────────

    def register_open(self, position: Position):
        """Registra una nueva posición abierta."""
        self._position = position
        self._trade_count += 1
        position.highest_price = position.entry_price
        position.lowest_price = position.entry_price

    def update_position(self, current_price: float) -> Optional[str]:
        """
        Actualiza la posición con el precio actual.
        Devuelve la razón de cierre si se debe cerrar, None si continúa.
        """
        if self._position is None:
            return None

        pos = self._position

        # Actualizar extremos de precio
        pos.highest_price = max(pos.highest_price, current_price)
        pos.lowest_price = min(pos.lowest_price, current_price)

        # ── Check Stop Loss ──────────────────────
        if pos.is_long and current_price <= pos.stop_loss:
            return "stop_loss"
        if not pos.is_long and current_price >= pos.stop_loss:
            return "stop_loss"

        # ── Check Take Profit 1 (cierre parcial 50%) ──
        if not pos.tp1_hit:
            if pos.is_long and current_price >= pos.take_profit_1:
                pos.tp1_hit = True
                pos.size_btc *= 0.5       # cerrar la mitad
                pos.size_usdt *= 0.5
                LOG.log(EventType.TAKE_PROFIT,
                        f"TP1 alcanzado — precio: ${current_price:,.2f} | "
                        f"50% de posición cerrada. Ajustando trailing stop.",
                        price=current_price)
                # Mover stop a breakeven tras TP1
                pos.stop_loss = pos.entry_price
                self._activate_trailing_stop(pos, current_price)

            elif not pos.is_long and current_price <= pos.take_profit_1:
                pos.tp1_hit = True
                pos.size_btc *= 0.5
                pos.size_usdt *= 0.5
                pos.stop_loss = pos.entry_price
                self._activate_trailing_stop(pos, current_price)

        # ── Check Take Profit 2 (cierre total) ───
        if pos.tp1_hit:
            if pos.is_long and current_price >= pos.take_profit_2:
                return "take_profit_2"
            if not pos.is_long and current_price <= pos.take_profit_2:
                return "take_profit_2"

        # ── Check Trailing Stop ───────────────────
        if pos.trailing_stop is not None:
            if pos.is_long and current_price <= pos.trailing_stop:
                return "trailing_stop"
            if not pos.is_long and current_price >= pos.trailing_stop:
                return "trailing_stop"
            else:
                self._update_trailing_stop(pos, current_price)

        # ── Check Timeout (máximo 4 horas) ────────
        if pos.duration_minutes > CONFIG.trading.max_trade_duration_minutes:
            return "timeout"

        return None  # posición continúa

    def _activate_trailing_stop(self, pos: Position, current_price: float):
        """Activa el trailing stop tras alcanzar TP1."""
        dist = CONFIG.risk.trailing_stop_distance
        if pos.is_long:
            pos.trailing_stop = current_price * (1 - dist)
        else:
            pos.trailing_stop = current_price * (1 + dist)
        LOG.log(EventType.SYSTEM,
                f"Trailing stop activado @ ${pos.trailing_stop:,.2f}",
                price=current_price)

    def _update_trailing_stop(self, pos: Position, current_price: float):
        """Mueve el trailing stop si el precio avanza favorablemente."""
        dist = CONFIG.risk.trailing_stop_distance
        if pos.is_long:
            new_ts = current_price * (1 - dist)
            if new_ts > pos.trailing_stop:
                pos.trailing_stop = new_ts
        else:
            new_ts = current_price * (1 + dist)
            if new_ts < pos.trailing_stop:
                pos.trailing_stop = new_ts

    def register_close(self, pnl_usdt: float, balance: float):
        """
        Registra el cierre de posición y actualiza métricas de riesgo.
        """
        if pnl_usdt < 0:
            self._last_loss_ts = time.time()
            self._daily_pnl += pnl_usdt
        else:
            self._daily_pnl += pnl_usdt

        # Actualizar peak y drawdown
        self._peak_balance = max(self._peak_balance, balance)
        current_drawdown = (self._peak_balance - balance) / self._peak_balance
        self._max_drawdown = max(self._max_drawdown, current_drawdown)

        self._position = None

    def reset_daily_pnl(self):
        """Resetear P&L diario (llamar a medianoche UTC)."""
        self._daily_pnl = 0.0

    # ── MÉTRICAS ────────────────────────────────

    def calculate_metrics(self, trade_history: List[Dict],
                          balance: float) -> Dict:
        """
        Calcula métricas avanzadas de rendimiento.
        """
        if not trade_history:
            return self._empty_metrics(balance)

        pnls = [t["pnl_usdt"] for t in trade_history if t.get("pnl_usdt") is not None]
        if not pnls:
            return self._empty_metrics(balance)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(pnls)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_pnl = sum(pnls)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor = suma ganancias / suma pérdidas
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe Ratio (anualizado, asumiendo ~1440 trades/día potenciales en 1m)
        if len(pnls) > 1:
            pnl_arr = np.array(pnls)
            avg_ret = np.mean(pnl_arr)
            std_ret = np.std(pnl_arr)
            sharpe = (avg_ret / std_ret * np.sqrt(252 * 1440 / max(total_trades, 1))
                      if std_ret > 0 else 0.0)
        else:
            sharpe = 0.0

        # Sortino (solo volatilidad bajista)
        downside = [p for p in pnls if p < 0]
        if downside and len(pnls) > 1:
            downside_std = np.std(downside)
            avg_ret = np.mean(pnls)
            sortino = avg_ret / downside_std * np.sqrt(252 * 1440 / max(total_trades, 1)) \
                      if downside_std > 0 else 0.0
        else:
            sortino = 0.0

        # Max Drawdown calculado sobre curva de equity
        equity_curve = [CONFIG.paper.initial_balance_usdt]
        for p in pnls:
            equity_curve.append(equity_curve[-1] + p)

        max_dd = self._calc_max_drawdown(equity_curve)
        current_drawdown = (self._peak_balance - balance) / self._peak_balance

        return {
            "balance_usdt": balance,
            "equity_usdt": balance + (
                self._position.unrealized_pnl(balance) if self._position else 0
            ),
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "current_drawdown": current_drawdown,
            "open_position": self._position,
        }

    @staticmethod
    def _calc_max_drawdown(equity_curve: List[float]) -> float:
        """Calcula el max drawdown de una curva de equity."""
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _empty_metrics(balance: float) -> Dict:
        return {
            "balance_usdt": balance,
            "equity_usdt": balance,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "open_position": None,
        }

    @property
    def current_position(self) -> Optional[Position]:
        return self._position

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl
