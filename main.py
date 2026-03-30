"""
main.py - Bot BTC/USDT Paper Trading con lógica Sniper.

Usa la misma estrategia multi-timeframe y scoring de 7 condiciones
que el Sniper Bot, pero ejecuta paper trades automáticamente.

Loop: analiza cada 5 min, cooldown 15 min entre trades.
"""

import time
import signal as sig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    SYMBOL, PAPER_MODE, INITIAL_BALANCE, LEVERAGE, MARKET_TYPE,
)
from bot.strategy_engine import StrategyEngine
from bot.risk_manager import RiskManager
from bot.paper_trader import PaperTrader
from bot.logger import setup_logger, TradeLogger

logger = setup_logger("main")

running = True
TRADE_COOLDOWN_SECONDS = 900  # 15 minutos entre trades
CHECK_INTERVAL_SECONDS = 300  # Analizar cada 5 minutos


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
    ║       BTC/USDT PAPER TRADING BOT (Sniper Logic)            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Par:          {SYMBOL} Perpetual                          ║
    ║  Mercado:      FUTUROS x{LEVERAGE}                                  ║
    ║  Timeframes:   15m, 1h, 4h (multi-timeframe)               ║
    ║  Modo:         PAPER TRADING (simulacion)                   ║
    ║  Balance:      {INITIAL_BALANCE:,.2f} USDT (ficticio)                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Estrategia:   Sniper Scoring (7 condiciones, min 4/7)     ║
    ║  Operaciones:  LONG + SHORT (paper)                        ║
    ║  Cooldown:     15 min entre trades                          ║
    ║  Análisis:     Cada 5 minutos                               ║
    ║  Stops:        ATR-based (SL=1.5x, TP=3x, ratio 1:2)      ║
    ║  Dashboard:    streamlit run bot/dashboard.py               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Inicializar
    logger.info("Inicializando módulos...")
    strategy = StrategyEngine()
    paper_trader = PaperTrader(INITIAL_BALANCE)

    loop_count = 0
    last_trade_time = 0
    last_report_time = time.time()
    report_interval = 3600

    logger.info(f"Bot iniciado | Vigilando {SYMBOL} en 15m, 1h, 4h")
    logger.info("Esperando oportunidades...")

    # ── Loop Principal ─────────────────────────────────────────
    while running:
        try:
            loop_start = time.time()
            loop_count += 1

            # 1. Generar señal multi-timeframe
            signal = strategy.generate_signal()

            if not signal or not signal.get("indicators"):
                logger.warning("Sin datos, reintentando...")
                time.sleep(30)
                continue

            current_price = signal["indicators"].get("price", 0)
            if current_price == 0:
                time.sleep(30)
                continue

            # 2. Gestionar posiciones abiertas
            paper_trader.check_and_manage_positions(current_price)

            # 3. Registrar señal en DB (para dashboard)
            paper_trader.trade_logger.log_decision(
                signal["action"],
                " | ".join(signal.get("reasons", [])[:3]),
                signal.get("indicators"),
                signal.get("confidence", 0),
            )

            # 4. Si hay señal de trade y no hay posición abierta → verificar cooldown
            if signal["action"] in ("BUY", "SELL") and not paper_trader.open_positions:
                elapsed = time.time() - last_trade_time
                if elapsed >= TRADE_COOLDOWN_SECONDS:
                    position = paper_trader.execute_open(
                        signal, signal["indicators"], signal["regime"]
                    )
                    if position:
                        last_trade_time = time.time()
                else:
                    remaining = int((TRADE_COOLDOWN_SECONDS - elapsed) / 60)
                    logger.info(
                        f"Señal {signal['action']} detectada pero en cooldown "
                        f"({remaining}min restantes)"
                    )

            # 5. Status cada 10 loops (~50 min)
            if loop_count % 10 == 0:
                open_count = len(paper_trader.open_positions)
                closed_count = len(paper_trader.closed_trades)
                wins = sum(1 for t in paper_trader.closed_trades if t["pnl"] > 0)
                wr = wins / closed_count * 100 if closed_count > 0 else 0
                pnl = sum(t["pnl"] for t in paper_trader.closed_trades)

                logger.info(
                    f"Status #{loop_count} | Price: {current_price:.2f} | "
                    f"Balance: {paper_trader.balance:.2f} | PnL: {pnl:+.2f} | "
                    f"Trades: {closed_count} | WR: {wr:.1f}% | Open: {open_count}"
                )

            # 6. Reporte periódico
            if time.time() - last_report_time > report_interval:
                paper_trader.generate_report()
                last_report_time = time.time()

            # Esperar 5 minutos
            elapsed = time.time() - loop_start
            sleep_time = max(1, CHECK_INTERVAL_SECONDS - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            running = False
        except Exception as e:
            logger.error(f"Error en loop: {e}", exc_info=True)
            time.sleep(60)

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Cerrando posiciones abiertas...")
    for pos in list(paper_trader.open_positions):
        try:
            current = strategy.analyzer.get_current_price()
            if current:
                paper_trader.execute_close(pos, current, pos.remaining_quantity, "SHUTDOWN")
        except Exception:
            pass

    paper_trader.generate_report()
    logger.info("Bot detenido correctamente.")


if __name__ == "__main__":
    main()
