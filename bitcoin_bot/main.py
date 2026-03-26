"""
main.py — Orquestador principal del bot de scalping BTC/USDT
=============================================================
Responsabilidad: coordinar todos los módulos en un ciclo de vida coherente.

Flujo de una iteración:
  1. DataFeed notifica vela cerrada → llama on_candle_close()
  2. StrategyEngine analiza los indicadores
  3. MLAgent ajusta la confianza de la señal
  4. RiskManager valida si se puede operar
  5. PaperTrader ejecuta la orden (simulada)
  6. En cada tick de precio: RiskManager revisa SL/TP/trailing
  7. Cada N trades: MLAgent se reentrena con el historial
  8. Cada hora: guardar métricas en BD para el dashboard

Diseño: patrón Observer — el DataFeed emite eventos a los que el
orquestador reacciona. No hay polling activo en el hilo principal.
"""

import sys
import os
import time
import signal
import threading
import argparse
from datetime import datetime, timezone
from typing import Optional

# Añadir directorio padre al path para imports relativos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from logger import LOG, EventType
from database import DB
from data_feed import DataFeed, Candle
from strategy_engine import StrategyEngine, Signal, AnalysisResult
from risk_manager import RiskManager
from paper_trader import PaperTrader
from ml_module import QLearningAgent, StateEncoder, MLRegimeAdvisor


# ─────────────────────────────────────────────
#  ORQUESTADOR
# ─────────────────────────────────────────────
class TradingBot:
    """
    Orquestador central del sistema de trading.
    Coordina todos los módulos y gestiona el ciclo de vida del bot.
    """

    def __init__(self):
        LOG.log(EventType.SYSTEM, "=" * 60)
        LOG.log(EventType.SYSTEM, "  BTC/USDT SCALPING BOT — Iniciando...")
        LOG.log(EventType.SYSTEM, f"  Modo: {'PAPER TRADING (SIMULACIÓN)' if CONFIG.paper.enabled else '🚨 LIVE TRADING 🚨'}")
        LOG.log(EventType.SYSTEM, f"  Capital virtual: ${CONFIG.paper.initial_balance_usdt:,.0f} USDT")
        LOG.log(EventType.SYSTEM, f"  Riesgo/operación: {CONFIG.risk.max_risk_per_trade:.0%}")
        LOG.log(EventType.SYSTEM, "=" * 60)

        # Módulos del sistema
        self.data_feed = DataFeed()
        self.strategy = StrategyEngine()
        self.risk_manager = RiskManager()
        self.paper_trader = PaperTrader(self.risk_manager)
        self.ml_agent = QLearningAgent()
        self.state_encoder = StateEncoder(n_bins=CONFIG.ml.state_bins)
        self.ml_advisor = MLRegimeAdvisor(self.ml_agent)

        # Estado del bot
        self._running = False
        self._current_analysis: Optional[AnalysisResult] = None
        self._last_analysis_indicators: Optional[dict] = None
        self._last_open_analysis: Optional[dict] = None  # indicadores al abrir
        self._trades_since_retrain: int = 0
        self._last_metrics_save: float = 0.0
        self._metrics_interval: float = 60.0   # guardar métricas cada 60s
        self._last_regime: str = "unknown"
        self._confirmation_count: int = 0
        self._pending_signal: Optional[AnalysisResult] = None

        # Lock para thread safety entre callbacks y loop principal
        self._lock = threading.Lock()

        # Handler de señales del SO para apagado limpio
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ── CICLO PRINCIPAL ─────────────────────────

    def run(self):
        """Arrancar el bot. Bloquea hasta que se detenga."""
        self._running = True

        # Arrancar DataFeed en hilo separado
        self.data_feed.start(on_candle_close=self._on_candle_close)

        LOG.log(EventType.SYSTEM, "Bot arrancado. Esperando datos suficientes...")

        # Esperar a que haya suficientes velas históricas
        self._wait_for_data()

        # Reentrenar con historial previo si existe
        self._initial_retrain()

        LOG.log(EventType.SYSTEM,
                f"Sistema listo. Analizando {CONFIG.exchange.symbol} en "
                f"tiempo real. Precio actual: ${self.data_feed.current_price:,.2f}")

        # Loop de mantenimiento (métricas, reentrenamiento, etc.)
        self._maintenance_loop()

    def _wait_for_data(self):
        """Espera bloqueante hasta tener datos suficientes."""
        while self._running and not self.data_feed.is_ready:
            time.sleep(2)
        if self._running:
            LOG.log(EventType.SYSTEM, "Datos históricos listos. Bot activo.")

    def _initial_retrain(self):
        """Reentrenar el ML con trades históricos al arrancar."""
        trades = DB.get_closed_trades(limit=500)
        if len(trades) >= CONFIG.ml.min_trades_for_training:
            LOG.log(EventType.ML_RETRAIN,
                    f"Reentrenando con {len(trades)} trades históricos...")
            self.ml_agent.retrain_from_history(
                trades,
                fear_greed=self.data_feed.fear_greed_index,
                btc_dominance=self.data_feed.btc_dominance,
            )

    # ── CALLBACK DE VELA CERRADA ────────────────

    def _on_candle_close(self, candle: Candle):
        """
        Llamado por DataFeed en cada vela de 1 minuto cerrada.
        Es el corazón del bot: aquí se toman todas las decisiones.
        """
        with self._lock:
            # 1. Obtener DataFrame actualizado
            df = self.data_feed.get_candles_df()
            if df is None or df.empty:
                return

            # 2. Obtener contexto externo
            ob = self.data_feed.orderbook
            ob_imbalance = ob.imbalance if ob else 0.0
            fear_greed = self.data_feed.fear_greed_index
            btc_dom = self.data_feed.btc_dominance

            # 3. Análisis técnico completo
            analysis = self.strategy.analyze(
                df, ob_imbalance, fear_greed, btc_dom
            )
            self._current_analysis = analysis
            self._last_analysis_indicators = analysis.to_dict()

            current_price = analysis.price

            # 4. Detectar cambio de régimen
            if analysis.regime.value != self._last_regime:
                LOG.market_regime(
                    regime=analysis.regime.value,
                    adx=analysis.adx,
                    volatility=analysis.atr_normalized,
                    trend_strength=analysis.ema_fast - analysis.ema_slow,
                )
                self._last_regime = analysis.regime.value
                CONFIG.market_regime = analysis.regime.value

            # 5. Tick de gestión de posición activa
            trade_result = self.paper_trader.tick(current_price)
            if trade_result:
                self._on_trade_closed(trade_result, analysis)

            # 6. Si no hay posición, evaluar nueva entrada
            if self.risk_manager.current_position is None:
                self._evaluate_entry(analysis, fear_greed, btc_dom)

    def _evaluate_entry(self, analysis: AnalysisResult,
                        fear_greed: int, btc_dom: float):
        """
        Evalúa si abrir una nueva posición basándose en la señal
        de la estrategia, ajustada por el agente ML.
        """
        if analysis.signal == Signal.HOLD:
            self._confirmation_count = 0
            return

        # Codificar estado para ML
        state = self.state_encoder.encode(
            analysis.to_dict(), fear_greed, btc_dom
        )

        # Ajustar confianza con Q-Learning
        adjusted_confidence = self.ml_agent.adjust_signal_confidence(
            state, analysis.confidence, analysis.signal.value
        )

        # Seleccionar acción ML
        ml_action = self.ml_agent.select_action(state)

        # Verificar que ML y estrategia están de acuerdo
        if not self._signals_agree(analysis.signal, ml_action):
            LOG.debug(EventType.ML_ACTION,
                      f"Desacuerdo ML/Estrategia: estrategia={analysis.signal.value} "
                      f"ml={ml_action} — esperando confirmación")
            self._confirmation_count = 0
            return

        # Sistema de confirmación: esperar N velas seguidas con la misma señal
        required = CONFIG.trading.signal_confirmation_candles
        self._confirmation_count += 1

        if self._confirmation_count < required:
            LOG.debug(EventType.SIGNAL,
                      f"Confirmación {self._confirmation_count}/{required} — "
                      f"{analysis.signal.value} ({adjusted_confidence:.1%})",
                      price=analysis.price)
            return

        # Señal confirmada: ejecutar
        self._confirmation_count = 0
        analysis.confidence = adjusted_confidence  # usar confianza ajustada

        if analysis.signal in (Signal.LONG, Signal.SHORT):
            # Guardar indicadores de entrada para aprendizaje posterior
            self._last_open_analysis = analysis.to_dict()

            trade_id = self.paper_trader.open_position(
                analysis=analysis,
                current_price=analysis.price,
                ml_action=ml_action,
                ml_confidence=adjusted_confidence,
                ml_state=list(state),
            )

            if trade_id:
                LOG.signal(
                    direction=analysis.signal.value,
                    price=analysis.price,
                    confidence=adjusted_confidence,
                    indicators=analysis.to_dict(),
                    reason=f"[ML:{ml_action}] {analysis.reason}",
                )

    @staticmethod
    def _signals_agree(strategy_signal: Signal, ml_action: str) -> bool:
        """Verifica que ML y estrategia apuntan en la misma dirección."""
        if ml_action == "hold":
            return False  # ML dice esperar, no abrir
        if strategy_signal == Signal.LONG and ml_action in ("long", "hold"):
            return ml_action == "long"
        if strategy_signal == Signal.SHORT and ml_action in ("short", "hold"):
            return ml_action == "short"
        return False

    # ── CIERRE DE TRADE ─────────────────────────

    def _on_trade_closed(self, trade_result, current_analysis: AnalysisResult):
        """
        Procesamiento post-cierre: aprendizaje ML, métricas, reentrenamiento.
        """
        self._trades_since_retrain += 1

        # Enseñar al agente ML sobre este trade
        if self._last_open_analysis:
            self.ml_agent.on_trade_closed(
                entry_analysis=self._last_open_analysis,
                exit_analysis=current_analysis.to_dict(),
                pnl_usdt=trade_result.pnl_usdt,
                pnl_pct=trade_result.pnl_pct,
                duration_min=trade_result.duration_min,
                drawdown=self.risk_manager.max_drawdown,
                size_usdt=trade_result.size_usdt,
                action_taken=trade_result.direction,
                fear_greed=self.data_feed.fear_greed_index,
                btc_dominance=self.data_feed.btc_dominance,
            )
            self._last_open_analysis = None

        # Reentrenar periódicamente
        if self._trades_since_retrain >= CONFIG.ml.retrain_interval_trades:
            self._retrain_ml()
            self._trades_since_retrain = 0

        # Guardar métricas
        self._save_metrics()

    def _retrain_ml(self):
        """Reentrenar el modelo ML con el historial reciente."""
        trades = DB.get_closed_trades(limit=500)
        self.ml_agent.retrain_from_history(
            trades,
            fear_greed=self.data_feed.fear_greed_index,
            btc_dominance=self.data_feed.btc_dominance,
        )

    # ── MÉTRICAS ────────────────────────────────

    def _save_metrics(self):
        """Guarda snapshot de métricas en la BD."""
        now = time.time()
        if now - self._last_metrics_save < self._metrics_interval:
            return
        self._last_metrics_save = now

        trades = DB.get_closed_trades(limit=200)
        metrics = self.risk_manager.calculate_metrics(
            trades, self.paper_trader.balance
        )
        metrics["balance_usdt"] = self.paper_trader.balance
        metrics["equity_usdt"] = self.paper_trader.equity
        metrics["open_position"] = None

        DB.save_metrics(metrics)

    # ── LOOP DE MANTENIMIENTO ───────────────────

    def _maintenance_loop(self):
        """
        Loop principal del hilo principal: métricas, informes, health checks.
        Las decisiones de trading ocurren en el callback de DataFeed.
        """
        while self._running:
            try:
                self._save_metrics()

                # Resetear P&L diario a medianoche UTC
                now = datetime.now(timezone.utc)
                if now.hour == 0 and now.minute == 0:
                    self.risk_manager.reset_daily_pnl()
                    LOG.log(EventType.SYSTEM, "P&L diario reseteado (medianoche UTC)")

                # Log de estado cada 5 minutos
                if int(time.time()) % 300 < 5:
                    self._log_status()

                time.sleep(5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                LOG.error(EventType.SYSTEM, f"Error en maintenance loop: {e}")
                time.sleep(10)

    def _log_status(self):
        """Log periódico del estado del bot."""
        pos = self.risk_manager.current_position
        pos_str = (
            f"Posición: {pos.direction.value.upper()} @ ${pos.entry_price:,.2f} "
            f"({pos.duration_minutes:.0f}min)"
            if pos else "Sin posición"
        )

        ml_stats = self.ml_agent.stats
        LOG.log(EventType.SYSTEM,
                f"STATUS | Balance: ${self.paper_trader.balance:,.2f} | "
                f"P&L: ${self.paper_trader.total_pnl:+.2f} | "
                f"Win rate: {self.paper_trader.win_rate:.1%} | "
                f"Trades: {len(self.paper_trader.all_trades)} | "
                f"ML ε={ml_stats['epsilon']:.3f} | {pos_str}",
                price=self.data_feed.current_price)

    # ── APAGADO ─────────────────────────────────

    def _handle_shutdown(self, signum, frame):
        """Apagado limpio al recibir SIGINT/SIGTERM."""
        LOG.log(EventType.SYSTEM, "Señal de apagado recibida. Cerrando bot...")
        self.stop()

    def stop(self):
        """Detener el bot limpiamente."""
        self._running = False

        # Cerrar posición abierta si existe
        if self.risk_manager.current_position:
            LOG.log(EventType.SYSTEM,
                    "Cerrando posición abierta antes de apagar...")
            self.paper_trader.close_position(
                self.data_feed.current_price,
                reason="shutdown"
            )

        # Guardar estado ML
        self.ml_agent._save_model()

        # Generar informe final
        self.paper_trader.generate_weekly_report()

        # Detener feed
        self.data_feed.stop()

        # Stats finales
        LOG.log(EventType.SYSTEM, "=" * 60)
        LOG.log(EventType.SYSTEM, "  BOT DETENIDO — Estadísticas Finales")
        LOG.log(EventType.SYSTEM, f"  Balance Final: ${self.paper_trader.balance:,.2f} USDT")
        LOG.log(EventType.SYSTEM, f"  P&L Total: ${self.paper_trader.total_pnl:+.2f} USDT")
        LOG.log(EventType.SYSTEM, f"  Trades Totales: {len(self.paper_trader.all_trades)}")
        LOG.log(EventType.SYSTEM, f"  Win Rate: {self.paper_trader.win_rate:.1%}")
        LOG.log(EventType.SYSTEM, f"  Max Drawdown: {self.risk_manager.max_drawdown:.1%}")
        LOG.log(EventType.SYSTEM, "=" * 60)

        sys.exit(0)


# ─────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="BTC/USDT Scalping Bot — Paper Trading"
    )
    parser.add_argument(
        "--balance", type=float, default=10_000.0,
        help="Balance virtual inicial en USDT (default: 10000)"
    )
    parser.add_argument(
        "--risk", type=float, default=0.02,
        help="Riesgo máximo por operación 0-0.05 (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging (default: INFO)"
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Arrancar sin dashboard (solo CLI)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Aplicar argumentos CLI sobre la config
    CONFIG.paper.initial_balance_usdt = args.balance
    CONFIG.risk.max_risk_per_trade = args.risk
    CONFIG.log_level = args.log_level

    # Validar config
    CONFIG.validate()

    # Arrancar dashboard en proceso separado (opcional)
    if not args.no_dashboard:
        import subprocess
        dashboard_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             "bitcoin_bot/dashboard.py",
             "--server.port", str(CONFIG.dashboard.port),
             "--server.headless", "true",
             "--logger.level", "error"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        LOG.log(EventType.SYSTEM,
                f"Dashboard disponible en: http://localhost:{CONFIG.dashboard.port}")

    # Arrancar bot
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
