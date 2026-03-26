"""
paper_trader.py — Simulador de órdenes con balance virtual
============================================================
Simula un exchange completo en modo paper trading:
  1. Balance virtual configurable (10.000 USDT por defecto)
  2. Simulación realista de fees (maker/taker)
  3. Slippage simulado (bid/ask spread)
  4. Registro de trades con todos los metadatos para ML
  5. Generación de reportes semanales automáticos

Decisión de diseño: el paper trader es el único módulo que "ejecuta"
órdenes. El risk manager valida, la estrategia genera señales, pero es
paper_trader quien modifica el balance y registra en la base de datos.

IMPORTANTE: NUNCA conecta a fondos reales. Está diseñado para ser
reemplazado por un LiveTrader cuando el usuario active el modo real,
manteniendo la misma interfaz pública.
"""

import uuid
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import asdict

from config import CONFIG
from logger import LOG, EventType
from database import DB
from risk_manager import RiskManager, Position
from strategy_engine import Signal, AnalysisResult


# ─────────────────────────────────────────────
#  RESULTADO DE TRADE
# ─────────────────────────────────────────────
class TradeResult:
    """Encapsula el resultado de una operación cerrada."""

    def __init__(self, trade_id: str, direction: str, entry: float,
                 exit_price: float, size_btc: float, size_usdt: float,
                 fee: float, reason: str, entry_ts: float):
        self.trade_id = trade_id
        self.direction = direction
        self.entry = entry
        self.exit_price = exit_price
        self.size_btc = size_btc
        self.size_usdt = size_usdt
        self.fee_usdt = fee
        self.close_reason = reason
        self.entry_ts = entry_ts
        self.exit_ts = time.time()
        self.duration_min = (self.exit_ts - entry_ts) / 60

        # P&L bruto
        if direction == "long":
            gross_pnl = (exit_price - entry) * size_btc
        else:
            gross_pnl = (entry - exit_price) * size_btc

        # P&L neto (descontando fees)
        self.pnl_usdt = gross_pnl - fee
        self.pnl_pct = self.pnl_usdt / size_usdt if size_usdt > 0 else 0

    @property
    def is_winner(self) -> bool:
        return self.pnl_usdt > 0

    def __repr__(self):
        result = "WIN" if self.is_winner else "LOSS"
        return (f"Trade[{self.trade_id[:8]}] {result} "
                f"{self.direction.upper()} | "
                f"P&L: {self.pnl_usdt:+.2f} USDT ({self.pnl_pct:+.2%})")


# ─────────────────────────────────────────────
#  PAPER TRADER PRINCIPAL
# ─────────────────────────────────────────────
class PaperTrader:
    """
    Simulador completo de trading en paper mode.
    Expone la misma interfaz que un LiveTrader futuro.
    """

    def __init__(self, risk_manager: RiskManager):
        if not CONFIG.paper.enabled:
            raise RuntimeError(
                "SEGURIDAD: PaperTrader instanciado con paper.enabled=False. "
                "Para trading real, usa LiveTrader explícitamente."
            )

        self.cfg = CONFIG.paper
        self.risk_cfg = CONFIG.risk
        self.rm = risk_manager

        # Balance virtual
        self._balance: float = self.cfg.initial_balance_usdt
        self._initial_balance: float = self.cfg.initial_balance_usdt

        # Historial de trades (también en DB)
        self._trades: List[TradeResult] = []
        self._current_analysis: Optional[AnalysisResult] = None

        # Stats de sesión
        self._session_start = datetime.now(timezone.utc)
        self._last_report_ts = time.time()
        self._report_interval = 7 * 24 * 3600  # 1 semana en segundos

        LOG.log(EventType.SYSTEM,
                f"PaperTrader iniciado | Balance virtual: "
                f"{self._balance:,.2f} USDT | MODO SIMULACIÓN",
                balance=self._balance)

    # ── APERTURA DE POSICIÓN ────────────────────

    def open_position(self, analysis: AnalysisResult,
                      current_price: float,
                      ml_action: str = "strategy",
                      ml_confidence: float = 0.0,
                      ml_state: Optional[List] = None) -> Optional[str]:
        """
        Abre una posición simulada.
        Devuelve el trade_id si se ejecutó, None si fue bloqueada.
        """
        self._current_analysis = analysis
        direction = analysis.signal

        # ── Validación de riesgo ─────────────────
        allowed, reason = self.rm.can_open_trade(
            self._balance, direction, analysis.confidence
        )
        if not allowed:
            LOG.risk_block(reason, {"price": current_price})
            DB.save_signal({
                "direction": direction.value,
                "confidence": analysis.confidence,
                "price": current_price,
                "executed": 0,
                "block_reason": reason,
                "indicators": analysis.to_dict(),
            })
            return None

        # ── Calcular stop-loss y take-profits ────
        stop_loss, tp1, tp2 = StrategyEngineRef.get_stops(
            analysis, current_price
        )

        # ── Calcular tamaño de posición ──────────
        size_btc, size_usdt = self.rm.calculate_position_size(
            self._balance, current_price, stop_loss
        )
        if size_btc <= 0:
            LOG.risk_block("Tamaño de posición calculado = 0", {})
            return None

        # ── Simular precio de ejecución (slippage) ──
        exec_price = self._apply_slippage(current_price, direction)

        # ── Fee de apertura (taker) ──────────────
        fee_open = size_usdt * self.cfg.taker_fee

        # ── Crear posición ───────────────────────
        trade_id = str(uuid.uuid4())[:12].upper()
        position = Position(
            trade_id=trade_id,
            direction=direction,
            entry_price=exec_price,
            size_btc=size_btc,
            size_usdt=size_usdt,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
        )
        self.rm.register_open(position)

        # ── Reservar capital (deducir del balance) ──
        self._balance -= (size_usdt + fee_open)

        # ── Persistir en BD ──────────────────────
        DB.save_trade_open({
            "id": trade_id,
            "symbol": CONFIG.exchange.symbol,
            "direction": direction.value,
            "entry_price": exec_price,
            "size_btc": size_btc,
            "size_usdt": size_usdt,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "entry_ts": datetime.now(timezone.utc).isoformat(),
            "rsi_entry": analysis.rsi,
            "macd_hist_entry": analysis.macd_hist,
            "bb_position_entry": analysis.bb_position,
            "ema_cross_entry": "bull" if analysis.ema_fast > analysis.ema_slow else "bear",
            "volume_ratio_entry": analysis.volume_ratio,
            "atr_entry": analysis.atr,
            "market_regime": analysis.regime.value,
            "fear_greed": None,
            "btc_dominance": None,
            "ml_action": ml_action,
            "ml_confidence": ml_confidence,
            "ml_state": json.dumps(ml_state) if ml_state else None,
        })
        DB.save_signal({
            "direction": direction.value,
            "confidence": analysis.confidence,
            "price": exec_price,
            "executed": 1,
            "block_reason": None,
            "indicators": analysis.to_dict(),
        })

        LOG.order_open(
            trade_id=trade_id,
            direction=direction.value,
            price=exec_price,
            size=size_btc,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            balance=self._balance,
        )

        return trade_id

    # ── CIERRE DE POSICIÓN ──────────────────────

    def close_position(self, current_price: float,
                       reason: str = "manual") -> Optional[TradeResult]:
        """
        Cierra la posición abierta actual.
        Devuelve TradeResult con el resultado o None si no había posición.
        """
        position = self.rm.current_position
        if position is None:
            return None

        # Simular precio de ejecución con slippage
        # En stop-loss la ejecución es menos favorable
        exec_price = self._apply_slippage_close(current_price, position.direction, reason)

        # Fee de cierre (taker en market orders)
        fee_close = position.size_usdt * self.cfg.taker_fee

        # Calcular P&L
        result = TradeResult(
            trade_id=position.trade_id,
            direction=position.direction.value,
            entry=position.entry_price,
            exit_price=exec_price,
            size_btc=position.size_btc,
            size_usdt=position.size_usdt,
            fee=fee_close,
            reason=reason,
            entry_ts=position.entry_ts,
        )

        # Devolver capital + P&L al balance
        self._balance += position.size_usdt + result.pnl_usdt - fee_close

        # Actualizar risk manager
        self.rm.register_close(result.pnl_usdt, self._balance)

        # Persistir cierre en BD
        DB.save_trade_close(position.trade_id, {
            "exit_price": exec_price,
            "exit_ts": datetime.now(timezone.utc).isoformat(),
            "duration_min": result.duration_min,
            "pnl_usdt": result.pnl_usdt,
            "pnl_pct": result.pnl_pct,
            "fee_usdt": fee_close,
            "close_reason": reason,
        })

        self._trades.append(result)

        LOG.order_close(
            trade_id=position.trade_id,
            direction=position.direction.value,
            entry=position.entry_price,
            exit_price=exec_price,
            size=position.size_btc,
            pnl=result.pnl_usdt,
            pnl_pct=result.pnl_pct,
            duration_min=result.duration_min,
            reason=reason,
            balance=self._balance,
        )

        # Check informe semanal
        self._check_weekly_report()

        return result

    # ── CICLO DE ACTUALIZACIÓN ──────────────────

    def tick(self, current_price: float) -> Optional[TradeResult]:
        """
        Llamado en cada tick de precio para gestionar la posición abierta.
        Devuelve TradeResult si la posición se cerró en este tick.
        """
        close_reason = self.rm.update_position(current_price)
        if close_reason:
            return self.close_position(current_price, close_reason)
        return None

    # ── SLIPPAGE ─────────────────────────────────

    def _apply_slippage(self, price: float, direction: Signal) -> float:
        """Aplica slippage en la apertura: compramos más caro, vendemos más barato."""
        slip = self.cfg.slippage_pct
        if direction == Signal.LONG:
            return price * (1 + slip)
        else:
            return price * (1 - slip)

    def _apply_slippage_close(self, price: float, direction: Signal,
                               reason: str) -> float:
        """
        Slippage en cierre. Los stop-loss suelen ejecutarse con más slippage
        porque son órdenes de mercado activadas en momentos de movimiento rápido.
        """
        slip = self.cfg.slippage_pct
        if reason == "stop_loss":
            slip *= 2  # doble slippage en stop-loss
        if direction == Signal.LONG:
            return price * (1 - slip)
        else:
            return price * (1 + slip)

    # ── INFORME SEMANAL ─────────────────────────

    def _check_weekly_report(self):
        """Genera informe semanal si pasó 1 semana desde el último."""
        if time.time() - self._last_report_ts >= self._report_interval:
            self.generate_weekly_report()
            self._last_report_ts = time.time()

    def generate_weekly_report(self) -> Dict:
        """
        Genera y persiste el informe semanal de rendimiento.
        Incluye evolución del modelo ML y estadísticas de trades.
        """
        report_data = DB.generate_weekly_report()
        trades = self._trades

        # Métricas de la semana
        week_trades = [t for t in trades
                       if t.exit_ts >= time.time() - 7 * 24 * 3600]

        report = {
            "period": "SEMANA",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "session_start": self._session_start.isoformat(),
            "initial_balance": self._initial_balance,
            "current_balance": self._balance,
            "total_return_pct": (self._balance - self._initial_balance) / self._initial_balance,
            "week_trades": len(week_trades),
            "week_winners": sum(1 for t in week_trades if t.is_winner),
            "week_pnl": sum(t.pnl_usdt for t in week_trades),
            "week_win_rate": (sum(1 for t in week_trades if t.is_winner) /
                              max(len(week_trades), 1)),
            "best_trade": max((t.pnl_usdt for t in week_trades), default=0),
            "worst_trade": min((t.pnl_usdt for t in week_trades), default=0),
            "regime_breakdown": report_data.get("regime_distribution", []),
        }

        # Guardar en archivo
        report_path = f"bitcoin_bot/logs/report_{datetime.now().strftime('%Y%m%d')}.json"
        import json
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        LOG.log(EventType.SYSTEM,
                f"INFORME SEMANAL generado: {len(week_trades)} trades | "
                f"Win rate: {report['week_win_rate']:.1%} | "
                f"P&L: {report['week_pnl']:+.2f} USDT | "
                f"Balance: {self._balance:,.2f} USDT")

        return report

    # ── ACCESO A DATOS ──────────────────────────

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        """Balance + valor no realizado de posición abierta."""
        pos = self.rm.current_position
        if pos and self._current_analysis:
            return self._balance + pos.unrealized_pnl(
                self._current_analysis.price
            )
        return self._balance

    @property
    def all_trades(self) -> List[TradeResult]:
        return self._trades.copy()

    @property
    def total_pnl(self) -> float:
        return self._balance - self._initial_balance

    @property
    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        return sum(1 for t in self._trades if t.is_winner) / len(self._trades)


# ─────────────────────────────────────────────
#  REFERENCIA A STRATEGY ENGINE (evita import circular)
# ─────────────────────────────────────────────
class StrategyEngineRef:
    """Wrapper para obtener stops sin importar directamente strategy_engine."""

    @staticmethod
    def get_stops(analysis: AnalysisResult, price: float):
        from strategy_engine import StrategyEngine
        engine = StrategyEngine()
        return engine.get_stop_loss_take_profits(
            analysis.signal, price, analysis.atr
        )
