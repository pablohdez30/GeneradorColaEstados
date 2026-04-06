"""
main_gold.py - Gold Trading Bot (Golden Cross Trend Rider)

Strategy optimized in MetaTrader 5:
- Golden Cross: EMA50 > EMA200 (bullish regime)
- Entry: Pullback to EMA50 during golden/death cross
- ADX > 40 (only VERY strong trends)
- SL: 3.0x ATR, TP: 4.0x SL (R:R 1:4)
- Session filter: 07:00-20:00 UTC (London + New York)
- Cooldown: 4 hours between trades

Backtest MT5: $10,000 -> $64,334 (+543%), PF 1.30, Sharpe 1.64
"""

import time
import signal as sig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gold_bot.config import (
    INITIAL_BALANCE, CHECK_INTERVAL_SECONDS, TRADE_COOLDOWN_SECONDS,
)
from gold_bot.strategy import GoldStrategy
from gold_bot.paper_trader import GoldPaperTrader
from gold_bot.logger import setup_logger

logger = setup_logger("main_gold")

running = True


def signal_handler(signum, frame):
    global running
    running = False
    logger.info("Cerrando gold bot...")


def main():
    global running

    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       GOLD TRADING BOT (Golden Cross Trend Rider)          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Activo:       XAU/USDT (Oro)                              ║
    ║  Timeframe:    15m + confirmacion 1H/4H                    ║
    ║  Modo:         PAPER TRADING (simulacion)                   ║
    ║  Balance:      10,000.00 USDT (ficticio)                   ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Estrategia:   Golden Cross (MT5: +543%, PF 1.30)          ║
    ║  Entrada:      Pullback a EMA50 en Golden/Death Cross      ║
    ║  Filtros:      ADX>40, Session 07-20 UTC, EMA9>21 conf     ║
    ║  Stops:        SL=3.0 ATR, TP=4.0x SL (R:R 1:4)          ║
    ║  Dashboard:    streamlit run gold_bot/dashboard.py :8052    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    logger.info("Inicializando Gold Bot...")
    strategy = GoldStrategy()
    paper_trader = GoldPaperTrader(INITIAL_BALANCE)

    loop_count = 0
    last_trade_time = 0
    last_report_time = time.time()
    report_interval = 3600

    logger.info("Gold Bot iniciado | Vigilando XAU/USDT")
    logger.info("Esperando oportunidades...")

    while running:
        try:
            loop_start = time.time()
            loop_count += 1

            # 1. Generate signal
            signal = strategy.generate_signal()

            if not signal or not signal.get("indicators"):
                logger.warning("Sin datos, reintentando...")
                time.sleep(30)
                continue

            current_price = signal["indicators"].get("price", 0)
            if current_price == 0:
                time.sleep(30)
                continue

            # 2. Manage open positions
            paper_trader.check_and_manage_positions(current_price)

            # 3. Log decision
            paper_trader.trade_logger.log_decision(
                signal["action"],
                ", ".join(signal["reasons"][:3]),
                signal["indicators"],
                signal["confidence"],
            )

            # 4. Execute trade if signal + cooldown
            now = time.time()
            cooldown_ok = (now - last_trade_time) >= TRADE_COOLDOWN_SECONDS

            if signal["action"] in ("BUY", "SELL") and cooldown_ok:
                if paper_trader.position is None:
                    atr = signal["indicators"].get("atr", 0)
                    if atr > 0:
                        sl = signal.get("stop_loss", 0)
                        tp = signal.get("take_profit", 0)

                        if sl == 0 or tp == 0:
                            sl = strategy.compute_stop_loss(current_price, signal["action"], atr)
                            tp = strategy.compute_take_profit(current_price, signal["action"], atr)

                        success = paper_trader.execute_open(
                            direction=signal["action"],
                            price=current_price,
                            stop_loss=sl,
                            take_profit=tp,
                            indicators=signal["indicators"],
                            reasons=signal["reasons"],
                        )
                        if success:
                            last_trade_time = now

            # 5. Periodic report
            if now - last_report_time >= report_interval:
                report = paper_trader.generate_report()
                logger.info(
                    f"REPORT | Balance: ${report['balance']:,.2f} | "
                    f"PnL: ${report['total_pnl']:+,.2f} | "
                    f"Trades: {report['total_trades']} | "
                    f"WR: {report['win_rate']}% | DD: {report['drawdown']}%"
                )
                last_report_time = now

            # Wait for next cycle
            elapsed = time.time() - loop_start
            sleep_time = max(0, CHECK_INTERVAL_SECONDS - elapsed)
            if sleep_time > 0 and running:
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error en loop: {e}", exc_info=True)
            time.sleep(60)

    logger.info("Gold Bot detenido.")
    report = paper_trader.generate_report()
    logger.info(f"FINAL | {report}")


if __name__ == "__main__":
    main()
