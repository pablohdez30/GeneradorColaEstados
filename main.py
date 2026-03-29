"""
main.py - Orquestador del bot de Trend Following BTC/USDT.

Loop principal:
1. Actualizar datos de mercado (cada 30s)
2. Generar señal SOLO cuando cierra nueva vela de 15m
3. Gestionar posiciones abiertas
4. Registrar métricas

Timeframe: 15 minutos (menos ruido, mejores señales)
"""

import time
import signal as sig
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    SYMBOL, TIMEFRAME, PAPER_MODE, INITIAL_BALANCE,
    ENABLE_SENTIMENT, ML_MIN_TRADES_FOR_TRAINING,
    LEVERAGE, MARKET_TYPE,
)
from bot.data_feed import DataFeed
from bot.strategy_engine import StrategyEngine
from bot.risk_manager import RiskManager
from bot.paper_trader import PaperTrader
from bot.ml_module import QLearningAgent, MarketState, SentimentAnalyzer
from bot.logger import setup_logger, TradeLogger

logger = setup_logger("main")

# Bandera para shutdown limpio
running = True


def signal_handler(signum, frame):
    global running
    running = False
    logger.info("Cerrando bot...")


def main():
    global running

    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║       BTC/USDT TREND FOLLOWING BOT                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Par:          {SYMBOL} Perpetual{' ' * 27}║
    ║  Mercado:      FUTUROS x{LEVERAGE}{' ' * 39}║
    ║  Timeframe:    {TIMEFRAME} (trend following){' ' * 24}║
    ║  Modo:         PAPER TRADING (simulacion){' ' * 17}║
    ║  Balance:       {INITIAL_BALANCE:,.2f} USDT (ficticio){' ' * 18}║
    ║  Riesgo/trade: 1% maximo{' ' * 33}║
    ║  Max Drawdown: 10%{' ' * 39}║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Estrategia:   EMA 9/21 Cross + ADX Filter + RSI Safety    ║
    ║  Stops:        ATR-based (SL=1.5x, TP=3x, ratio 1:2)      ║
    ║  ML:           Q-Learning adaptativo{' ' * 21}║
    ║  Dashboard:    streamlit run bot/dashboard.py{' ' * 14}║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Inicializar módulos
    logger.info("Inicializando módulos...")
    data_feed = DataFeed()
    strategy = StrategyEngine()
    paper_trader = PaperTrader(INITIAL_BALANCE)
    ml_agent = QLearningAgent()
    sentiment_analyzer = SentimentAnalyzer()

    # Cargar datos históricos
    logger.info("Cargando datos históricos...")
    df = data_feed.fetch_ohlcv(limit=500)
    if df.empty:
        logger.error("No se pudieron cargar datos. Reintentando en 30s...")
        time.sleep(30)
        df = data_feed.fetch_ohlcv(limit=500)

    # Variables de estado
    loop_count = 0
    last_sentiment_check = 0
    sentiment_value = 50
    last_report_time = time.time()
    report_interval = 3600  # Reporte cada hora
    last_candle_time = df.index[-1] if not df.empty else None

    logger.info(f"Datos cargados: {len(df)} velas desde {df.index[0] if not df.empty else 'N/A'}")
    logger.info("Bot iniciado. Entrando en loop principal...")
    logger.info("Presiona Ctrl+C para detener.")

    # ── Loop Principal ─────────────────────────────────────────
    while running:
        try:
            loop_start = time.time()
            loop_count += 1

            # 1. Actualizar datos
            df = data_feed.update_buffer()
            if df.empty or len(df) < 50:
                logger.warning("Buffer insuficiente, esperando...")
                time.sleep(30)
                continue

            current_price = df["close"].iloc[-1]
            current_candle_time = df.index[-1]

            # 2. Gestionar posiciones abiertas (cada iteración)
            paper_trader.check_and_manage_positions(current_price)

            # 3. Solo generar señales cuando cierra una nueva vela de 15m
            new_candle = current_candle_time != last_candle_time
            if new_candle:
                last_candle_time = current_candle_time

                # Sentimiento (cada 15 minutos)
                if ENABLE_SENTIMENT and (time.time() - last_sentiment_check > 900):
                    sentiment_value = sentiment_analyzer.get_fear_greed_index()
                    last_sentiment_check = time.time()

                # ML prediction
                ml_prediction = 0.5
                if ml_agent.training_count > 0:
                    indicators_for_ml = strategy.compute_all_indicators(df)
                    regime = strategy.last_signal.get("regime", "UNKNOWN") if strategy.last_signal else "UNKNOWN"
                    state = MarketState.discretize(indicators_for_ml, regime, sentiment_value)
                    ml_action, ml_confidence = ml_agent.predict(state)
                    ml_prediction = ml_confidence if ml_action == "BUY" else (1 - ml_confidence)

                    if ml_agent.training_count >= 3:
                        strategy.ml_weight = min(0.15, ml_agent.training_count * 0.02)

                # Order book
                order_book = data_feed.fetch_order_book(depth=20)

                # Generar señal
                signal = strategy.generate_signal(df, order_book, ml_prediction, sentiment_value)

                # Registrar cada señal en la DB (para el dashboard)
                paper_trader.trade_logger.log_decision(
                    signal["action"],
                    " | ".join(signal.get("reasons", [])[:3]),
                    signal.get("indicators"),
                    signal.get("confidence", 0),
                )

                # Si hay señal y no hay posición abierta → abrir
                if signal["action"] in ("BUY", "SELL") and not paper_trader.open_positions:
                    position = paper_trader.execute_open(
                        signal, signal["indicators"], signal["regime"]
                    )

                    if position and ml_agent.training_count > 0:
                        state = MarketState.discretize(
                            signal["indicators"], signal["regime"], sentiment_value
                        )

                # ML retraining
                if ml_agent.should_retrain():
                    ml_agent.train_from_history()

            # 4. Status cada 10 loops
            if loop_count % 10 == 0:
                open_count = len(paper_trader.open_positions)
                closed_count = len(paper_trader.closed_trades)
                wins = sum(1 for t in paper_trader.closed_trades if t["pnl"] > 0)
                wr = wins / closed_count * 100 if closed_count > 0 else 0
                pnl = sum(t["pnl"] for t in paper_trader.closed_trades)

                logger.info(
                    f"Loop #{loop_count} | Price: {current_price:.2f} | "
                    f"Balance: {paper_trader.balance:.2f} | PnL: {pnl:+.2f} | "
                    f"Trades: {closed_count} | WR: {wr:.1f}% | Open: {open_count}"
                )

            # 5. Reporte periódico
            if time.time() - last_report_time > report_interval:
                paper_trader.generate_report()
                last_report_time = time.time()

            # Esperar 30 segundos (suficiente para velas de 15m)
            elapsed = time.time() - loop_start
            sleep_time = max(1, 30 - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            running = False
        except Exception as e:
            logger.error(f"Error en loop: {e}", exc_info=True)
            time.sleep(30)

    # ── Shutdown limpio ────────────────────────────────────────
    logger.info("Cerrando posiciones abiertas...")
    for pos in list(paper_trader.open_positions):
        try:
            current = data_feed.get_current_price()
            if current:
                paper_trader.execute_close(pos, current, pos.remaining_quantity, "SHUTDOWN")
        except Exception:
            pass

    paper_trader.generate_report()
    logger.info("Bot detenido correctamente.")


if __name__ == "__main__":
    main()
