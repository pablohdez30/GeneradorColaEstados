"""
main_trend.py - Bot BTC/USDT Trend Following (estrategia original).

Estrategia: EMA 9/21 Cross + ADX Filter + RSI Safety.
Timeframe: 15 minutos.
Paper trading con balance ficticio separado.
"""

import time
import signal as sig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    SYMBOL, TIMEFRAME, PAPER_MODE, INITIAL_BALANCE,
    ENABLE_SENTIMENT, LEVERAGE, MARKET_TYPE,
)
from bot.data_feed import DataFeed
from bot.strategy_engine_trend import StrategyEngine
from bot.risk_manager import RiskManager, Position
from bot.paper_trader_trend import PaperTrader
from bot.ml_module import QLearningAgent, MarketState, SentimentAnalyzer
from bot.logger import setup_logger, TradeLoggerTrend

logger = setup_logger("main_trend")

running = True


def signal_handler(signum, frame):
    global running
    running = False
    logger.info("Cerrando bot trend...")


def main():
    global running

    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║       BTC/USDT TREND FOLLOWING BOT (Original)              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Par:          {SYMBOL} Perpetual                          ║
    ║  Mercado:      FUTUROS x{LEVERAGE}                                  ║
    ║  Timeframe:    {TIMEFRAME} (trend following)                       ║
    ║  Modo:         PAPER TRADING (simulacion)                   ║
    ║  Balance:      {INITIAL_BALANCE:,.2f} USDT (ficticio)                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Estrategia:   EMA 9/21 Cross + ADX Filter + RSI Safety    ║
    ║  Stops:        ATR-based (SL=1.5x, TP=3x, ratio 1:2)      ║
    ║  Dashboard:    streamlit run bot/dashboard_trend.py         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Inicializar
    logger.info("Inicializando módulos (Trend Following)...")
    data_feed = DataFeed()
    strategy = StrategyEngine()
    paper_trader = PaperTrader(INITIAL_BALANCE)
    sentiment_analyzer = SentimentAnalyzer()

    # Cargar datos históricos
    logger.info("Cargando datos históricos...")
    df = data_feed.fetch_ohlcv(limit=500)
    if df.empty:
        logger.error("No se pudieron cargar datos. Reintentando en 30s...")
        time.sleep(30)
        df = data_feed.fetch_ohlcv(limit=500)

    loop_count = 0
    last_sentiment_check = 0
    sentiment_value = 50
    last_report_time = time.time()
    report_interval = 3600
    last_candle_time = df.index[-1] if not df.empty else None

    logger.info(f"Bot Trend iniciado | {SYMBOL} | {TIMEFRAME}")
    logger.info("Esperando señales EMA cross...")

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

            # 2. Gestionar posiciones abiertas
            paper_trader.check_and_manage_positions(current_price)

            # 3. Solo generar señales cuando cierra nueva vela de 15m
            new_candle = current_candle_time != last_candle_time
            if new_candle:
                last_candle_time = current_candle_time

                # Sentimiento
                if ENABLE_SENTIMENT and (time.time() - last_sentiment_check > 900):
                    sentiment_value = sentiment_analyzer.get_fear_greed_index()
                    last_sentiment_check = time.time()

                # Generar señal
                signal = strategy.generate_signal(df, None, 0.0, sentiment_value)

                # Registrar en DB
                paper_trader.trade_logger.log_decision(
                    signal["action"],
                    " | ".join(signal.get("reasons", [])[:3]),
                    signal.get("indicators"),
                    signal.get("confidence", 0),
                )

                # Si hay señal y no hay posición → abrir
                if signal["action"] in ("BUY", "SELL") and not paper_trader.open_positions:
                    paper_trader.execute_open(
                        signal, signal["indicators"], signal["regime"]
                    )

            # 4. Status cada 10 loops
            if loop_count % 10 == 0:
                open_count = len(paper_trader.open_positions)
                closed_count = len(paper_trader.closed_trades)
                wins = sum(1 for t in paper_trader.closed_trades if t["pnl"] > 0)
                wr = wins / closed_count * 100 if closed_count > 0 else 0
                pnl = sum(t["pnl"] for t in paper_trader.closed_trades)

                logger.info(
                    f"[TREND] Loop #{loop_count} | Price: {current_price:.2f} | "
                    f"Balance: {paper_trader.balance:.2f} | PnL: {pnl:+.2f} | "
                    f"Trades: {closed_count} | WR: {wr:.1f}% | Open: {open_count}"
                )

            # 5. Reporte periódico
            if time.time() - last_report_time > report_interval:
                paper_trader.generate_report()
                last_report_time = time.time()

            # Esperar 30 segundos
            elapsed = time.time() - loop_start
            sleep_time = max(1, 30 - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            running = False
        except Exception as e:
            logger.error(f"Error en loop trend: {e}", exc_info=True)
            time.sleep(30)

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Cerrando posiciones trend...")
    for pos in list(paper_trader.open_positions):
        try:
            current = data_feed.get_current_price()
            if current:
                paper_trader.execute_close(pos, current, pos.remaining_quantity, "SHUTDOWN")
        except Exception:
            pass

    paper_trader.generate_report()
    logger.info("Bot Trend detenido correctamente.")


if __name__ == "__main__":
    main()
