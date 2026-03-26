"""
CLI entry point for the Polymarket Trading Bot.

Usage:
    python -m polymarket_bot              # Run the bot continuously
    python -m polymarket_bot --scan       # Run a single scan
    python -m polymarket_bot --status     # Show current status
    python -m polymarket_bot --reset      # Reset paper money balance
    python -m polymarket_bot --balance N  # Set initial balance to N
"""

import argparse
import os
import sys

from .bot import PolymarketBot
from .config import BotConfig


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Autonomous Trading Bot (Paper Money)")

    parser.add_argument(
        "--scan", action="store_true",
        help="Run a single market scan and exit")
    parser.add_argument(
        "--status", action="store_true",
        help="Show current portfolio status and exit")
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset the database and start fresh")
    parser.add_argument(
        "--balance", type=float, default=10000.0,
        help="Initial paper money balance (default: $10,000)")
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Scan interval in seconds (default: 300)")
    parser.add_argument(
        "--max-positions", type=int, default=20,
        help="Maximum open positions (default: 20)")
    parser.add_argument(
        "--max-position-size", type=float, default=500.0,
        help="Maximum position size in USDC (default: $500)")
    parser.add_argument(
        "--min-confidence", type=float, default=0.6,
        help="Minimum confidence to trade, 0-1 (default: 0.6)")
    parser.add_argument(
        "--db", type=str, default="polymarket_bot.db",
        help="Database file path (default: polymarket_bot.db)")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Handle reset
    if args.reset:
        if os.path.exists(args.db):
            confirm = input(f"Delete {args.db} and start fresh? (y/N): ")
            if confirm.lower() == "y":
                os.remove(args.db)
                print("Database reset. Starting fresh.")
            else:
                print("Cancelled.")
                return
        else:
            print("No database found. Will create a new one on first run.")
        return

    # Build config
    config = BotConfig(
        initial_balance=args.balance,
        max_position_size=args.max_position_size,
        max_open_positions=args.max_positions,
        min_confidence=args.min_confidence,
        scan_interval_seconds=args.interval,
        db_path=args.db,
        log_level=args.log_level,
    )

    # Create and run bot
    bot = PolymarketBot(config)

    if args.status:
        bot.show_status()
    elif args.scan:
        bot.run_single_scan()
    else:
        bot.run()


if __name__ == "__main__":
    main()
