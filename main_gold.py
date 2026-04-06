"""
main_gold.py - Gold Trading Bot with Golden Cross Strategy.

Uses the same architecture as the BTC bot but with parameters
optimized in MetaTrader 5 for gold (XAU/USDT or PAXG/USDT).

Loop: analyze every 5 min, cooldown 4 hours between trades.
"""

import time
import signal as sig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gold_bot.config import (
    SYMBOL, FALLBACK_SYMBOL, INITIAL_BALANCE,
    CHECK_INTERVAL_SECONDS, TRADE_COOLDOWN_SECONDS,
    ADX_MIN, SL_ATR_MULT, TP_RATIO,
    SESSION_START_HOUR, SESSION_END_HOUR,
    DASHBOARD_PORT,
)
from gold_bot.strategy import GoldStrategy
from gold_bot.paper_trader import GoldPaperTrader
from gold_bot.logger import setup_logger

logger = setup_logger("main_gold")

running = True


def signal_handler(signum, frame):
    global running
    running = False
    logger.info("Shutting down bot...")


def main():
    global running

    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)

    print(f"""
    +==============================================================+
    |       GOLD TRADING BOT (Golden Cross Strategy)               |
    +==============================================================+
    |  Symbol:       {SYMBOL} / {FALLBACK_SYMBOL}                  |
    |  Timeframe:    15m                                           |
    |  Mode:         PAPER TRADING (simulation)                    |
    |  Balance:      {INITIAL_BALANCE:,.2f} USDT (virtual)                 |
    +--------------------------------------------------------------+
    |  Strategy:     Golden Cross Trend Rider (MT5 optimized)      |
    |  Operations:   LONG + SHORT (paper)                          |
    |  Cooldown:     4 hours between trades                        |
    |  Analysis:     Every 5 minutes                               |
    |  Stops:        ATR-based (SL={SL_ATR_MULT}x, TP=SL*{TP_RATIO}x, R:R 1:{TP_RATIO:.0f})       |
    |  ADX filter:   >= {ADX_MIN}                                          |
    |  Session:      {SESSION_START_HOUR:02d}:00 - {SESSION_END_HOUR:02d}:00 UTC                            |
    |  Dashboard:    streamlit run gold_bot/dashboard.py           |
    |                --server.port {DASHBOARD_PORT}                        |
    +==============================================================+
    """)

    # Initialize
    logger.info("Initializing modules...")
    try:
        strategy = GoldStrategy()
    except RuntimeError as e:
        logger.error(f"Failed to initialize strategy: {e}")
        sys.exit(1)

    paper_trader = GoldPaperTrader(INITIAL_BALANCE)

    loop_count = 0
    last_trade_time = 0
    last_report_time = time.time()
    report_interval = 3600

    logger.info(f"Bot started | Watching {strategy.symbol} on 15m")
    logger.info("Waiting for opportunities...")

    # ── Main Loop ─────────────────────────────────────────────
    while running:
        try:
            loop_start = time.time()
            loop_count += 1

            # 1. Generate signal
            signal = strategy.generate_signal()

            if not signal or not signal.get("indicators"):
                logger.warning("No data, retrying...")
                time.sleep(30)
                continue

            current_price = signal["indicators"].get("price", 0)
            if current_price == 0:
                time.sleep(30)
                continue

            # 2. Manage open positions
            paper_trader.check_and_manage_positions(current_price)

            # 3. Log decision to DB (for dashboard)
            paper_trader.trade_logger.log_decision(
                signal["action"],
                " | ".join(signal.get("reasons", [])[:3]),
                signal.get("indicators"),
                signal.get("confidence", 0),
            )

            # 4. If trade signal and no open position -> check cooldown
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
                        f"Signal {signal['action']} detected but in cooldown "
                        f"({remaining}min remaining)"
                    )

            # 5. Status every 10 loops (~50 min)
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

            # 6. Periodic report
            if time.time() - last_report_time > report_interval:
                paper_trader.generate_report()
                last_report_time = time.time()

            # Wait 5 minutes
            elapsed = time.time() - loop_start
            sleep_time = max(1, CHECK_INTERVAL_SECONDS - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            running = False
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Closing open positions...")
    for pos in list(paper_trader.open_positions):
        try:
            current = strategy.get_current_price()
            if current:
                paper_trader.execute_close(
                    pos, current, pos.remaining_quantity, "SHUTDOWN"
                )
        except Exception:
            pass

    paper_trader.generate_report()
    logger.info("Bot stopped correctly.")


if __name__ == "__main__":
    main()
