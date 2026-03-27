"""
main.py - Orquestador principal del bot de scalping BTC/USDT.

Arquitectura del loop principal:
────────────────────────────────
1. Inicializar todos los módulos
2. Cargar datos históricos iniciales (500 velas)
3. Loop infinito cada ~10 segundos:
   a. Actualizar datos de mercado (velas + order book)
   b. Calcular indicadores y generar señal
   c. Consultar modelo ML para predicción
   d. Si hay señal fuerte → ejecutar trade (paper)
   e. Gestionar posiciones abiertas (stops, TPs)
   f. Registrar equity y decisiones
   g. Reentrenar ML si es momento
   h. Mostrar resumen periódico

El bot arranca EXCLUSIVAMENTE en modo paper trading.
No hay forma de operar en real sin modificar PAPER_MODE en config.py.
"""

import time
import signal as sig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    SYMBOL, TIMEFRAME, PAPER_MODE, INITIAL_BALANCE,
    ENABLE_SENTIMENT, ML_MIN_TRADES_FOR_TRAINING,
    LEVERAGE, MARKET_TYPE, MAX_OPEN_POSITIONS,
    MAX_OPEN_POSITIONS_EXTRA, HIGH_CONFIDENCE_THRESHOLD,
)
from bot.data_feed import DataFeed
from bot.strategy_engine import StrategyEngine
from bot.risk_manager import RiskManager
from bot.paper_trader import PaperTrader
from bot.ml_module import QLearningAgent, MarketState, SentimentAnalyzer
from bot.logger import setup_logger, TradeLogger

logger = setup_logger("main")

# ── Control de shutdown graceful ──────────────────────────────
running = True

def shutdown_handler(signum, frame):
    global running
    logger.info("Señal de shutdown recibida. Cerrando bot...")
    running = False

sig.signal(sig.SIGINT, shutdown_handler)
sig.signal(sig.SIGTERM, shutdown_handler)


def print_banner():
    mode_str = f"FUTUROS x{LEVERAGE}" if MARKET_TYPE == "futures" else "SPOT"
    banner = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║       BTC/USDT SCALPING BOT - PAPER TRADING                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Par:          BTC/USDT Perpetual                           ║
    ║  Mercado:      {mode_str:<45}║
    ║  Timeframe:    1m (scalping)                                ║
    ║  Modo:         PAPER TRADING (simulacion)                   ║
    ║  Balance:      {INITIAL_BALANCE:>10,.2f} USDT (ficticio)                  ║
    ║  Riesgo/trade: 1% maximo                                    ║
    ║  Max Drawdown: 15%                                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Direccion:    LONG + SHORT (ambas direcciones)             ║
    ║  Indicadores:  RSI + MACD + BB + EMA + Volume + OrderBook   ║
    ║  ML:           Q-Learning adaptativo                        ║
    ║  Dashboard:    streamlit run bot/dashboard.py               ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Loop principal del bot."""
    if not PAPER_MODE:
        logger.critical("PAPER_MODE está desactivado. Abortando por seguridad.")
        sys.exit(1)

    print_banner()

    # ── Inicializar módulos ────────────────────────────────────
    logger.info("Inicializando módulos...")

    data_feed = DataFeed()
    strategy = StrategyEngine(ml_weight=0.0)  # Sin ML hasta tener datos
    paper_trader = PaperTrader(INITIAL_BALANCE)
    ml_agent = QLearningAgent()
    sentiment_analyzer = SentimentAnalyzer()
    trade_logger = TradeLogger()

    # ── Cargar datos históricos ────────────────────────────────
    logger.info("Cargando datos históricos...")
    try:
        df = data_feed.fetch_ohlcv(limit=500)
        logger.info(f"Datos cargados: {len(df)} velas desde {df.index[0]}")
    except Exception as e:
        logger.error(f"Error cargando datos iniciales: {e}")
        logger.info("Reintentando en 30 segundos...")
        time.sleep(30)
        df = data_feed.fetch_ohlcv(limit=500)

    # ── Variables de estado ────────────────────────────────────
    loop_count = 0
    last_sentiment_check = 0
    sentiment_value = 50  # Neutral
    last_report_time = time.time()
    report_interval = 3600  # Reporte cada hora
    last_5m_fetch = 0
    df_5m = None  # Velas de 5m para confirmación multi-timeframe

    logger.info("Bot iniciado. Entrando en loop principal...")
    logger.info("Presiona Ctrl+C para detener.")

    # ── Loop Principal ─────────────────────────────────────────
    while running:
        try:
            loop_start = time.time()
            loop_count += 1

            # 1. Actualizar datos de mercado
            df = data_feed.update_buffer()
            if df.empty:
                logger.warning("Buffer vacío, esperando datos...")
                time.sleep(10)
                continue

            current_price = df["close"].iloc[-1]

            # 2. Obtener order book
            order_book = data_feed.fetch_order_book(depth=20)

            # 2.5. Velas 5m para confirmación multi-timeframe (cada 60s)
            if time.time() - last_5m_fetch > 60:
                try:
                    df_5m = data_feed.fetch_ohlcv_higher_tf("5m", limit=100)
                    last_5m_fetch = time.time()
                except Exception:
                    pass  # Seguir con df_5m anterior si falla

            # 3. Sentimiento (cada 15 minutos para no saturar API)
            if ENABLE_SENTIMENT and (time.time() - last_sentiment_check > 900):
                sentiment_value = sentiment_analyzer.get_fear_greed_index()
                last_sentiment_check = time.time()

            # 4. Predicción ML
            ml_prediction = 0.5  # Neutral por defecto
            if ml_agent.training_count > 0:
                indicators_for_ml = strategy.compute_all_indicators(df)
                regime = strategy.last_signal["regime"] if strategy.last_signal else "UNKNOWN"
                state = MarketState.discretize(indicators_for_ml, regime, sentiment_value)
                ml_action, ml_confidence = ml_agent.predict(state)
                ml_prediction = ml_confidence if ml_action == "BUY" else (1 - ml_confidence)

                # Ajustar peso del ML según confianza del modelo
                if ml_agent.training_count >= 3:
                    strategy.ml_weight = min(0.15, ml_agent.training_count * 0.02)

            # 5. Generar señal de trading (con confirmación multi-timeframe)
            signal = strategy.generate_signal(
                df, order_book, ml_prediction, sentiment_value, df_5m
            )

            # 6. Gestionar posiciones abiertas PRIMERO
            paper_trader.check_and_manage_positions(current_price)

            # 7. Si hay señal → comprobar si hay espacio (el risk_manager decide según confianza)
            confidence = signal.get("confidence", 0)
            max_pos = MAX_OPEN_POSITIONS_EXTRA if confidence >= HIGH_CONFIDENCE_THRESHOLD else MAX_OPEN_POSITIONS
            if signal["action"] in ("BUY", "SELL") and len(paper_trader.open_positions) < max_pos:
                position = paper_trader.execute_open(
                    signal, signal["indicators"], signal["regime"]
                )

                if position and ml_agent.training_count > 0:
                    # Registrar experiencia para ML
                    state = MarketState.discretize(
                        signal["indicators"], signal["regime"], sentiment_value
                    )
                    # La recompensa se asignará al cerrar el trade
            else:
                # Registrar decisión de HOLD
                trade_logger.log_decision(
                    signal["action"],
                    signal["reasons"][0] if signal["reasons"] else "Sin señal",
                    signal["indicators"],
                    signal["confidence"],
                )

            # 8. Registrar equity
            unrealized = sum(
                p.unrealized_pnl(current_price) for p in paper_trader.open_positions
            )
            trade_logger.log_equity(
                paper_trader.balance, unrealized,
                paper_trader.risk_manager.current_drawdown()
            )

            # 9. Reentrenar ML si es momento
            if ml_agent.should_retrain():
                logger.info("Iniciando reentrenamiento del modelo ML...")
                ml_agent.train_from_history()

            # 10. Resumen periódico
            if time.time() - last_report_time > report_interval:
                report = paper_trader.generate_weekly_report()
                ml_stats = ml_agent.get_model_stats()
                logger.info(f"ML Stats: {ml_stats}")
                last_report_time = time.time()

            # 11. Log de estado cada 10 loops
            if loop_count % 10 == 0:
                summary = paper_trader.get_portfolio_summary()
                logger.info(
                    f"Loop #{loop_count} | Price: {current_price:.2f} | "
                    f"Balance: {summary['balance']:.2f} | "
                    f"PnL: {summary['total_pnl']:+.2f} | "
                    f"Trades: {summary['total_trades']} | "
                    f"WR: {summary['win_rate']:.1%} | "
                    f"Open: {summary['open_positions']}"
                )

            # Esperar hasta próximo tick (10 segundos entre iteraciones)
            elapsed = time.time() - loop_start
            sleep_time = max(0, 5 - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error en loop principal: {e}", exc_info=True)
            time.sleep(30)  # Esperar más ante errores

    # ── Shutdown ───────────────────────────────────────────────
    logger.info("Cerrando bot...")

    # Cerrar posiciones abiertas
    if paper_trader.open_positions:
        logger.info("Cerrando posiciones abiertas...")
        current_price = data_feed.get_current_price()
        for pos in list(paper_trader.open_positions):
            paper_trader.execute_close(pos, current_price, pos.remaining_quantity, "SHUTDOWN")

    # Reporte final
    paper_trader.generate_weekly_report()
    ml_agent._save_model()
    logger.info("Bot detenido correctamente.")


if __name__ == "__main__":
    main()
