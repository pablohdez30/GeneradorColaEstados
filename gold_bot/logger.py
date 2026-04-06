"""
logger.py - Logging and trade recording for the Gold Bot.

Dual output (file + console). SQLite database for trade history,
decisions, and equity curve.
"""

import logging
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from gold_bot.config import LOG_FILE, LOG_LEVEL, DB_PATH


def setup_logger(name: str = "gold_bot") -> logging.Logger:
    """Configure logger with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if logger.handlers:
        return logger

    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


class GoldTradeLogger:
    """
    Records every trade operation in SQLite with full context.

    Stores: timestamp, direction, entry/exit price, PnL,
    active indicators, market regime, and text justification.
    """

    def __init__(self):
        self.db_path = DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.logger = setup_logger("gold_trade_logger")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_open TEXT NOT NULL,
                    timestamp_close TEXT,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl REAL,
                    pnl_pct REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    indicators_snapshot TEXT,
                    market_regime TEXT,
                    justification TEXT,
                    status TEXT DEFAULT 'OPEN'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    indicators TEXT,
                    confidence REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    balance REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0,
                    drawdown REAL DEFAULT 0
                )
            """)

    def log_trade_open(self, direction, entry_price, quantity, stop_loss,
                       take_profit, indicators, regime, justification) -> int:
        """Record trade opening. Returns trade_id."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                   (timestamp_open, direction, entry_price, quantity, stop_loss,
                    take_profit, indicators_snapshot, market_regime, justification, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
                (now, direction, entry_price, quantity, stop_loss, take_profit,
                 json.dumps(indicators), regime, justification),
            )
            trade_id = cursor.lastrowid

        self.logger.info(
            f"OPEN #{trade_id} | {direction} | price={entry_price:.2f} | "
            f"qty={quantity:.6f} | SL={stop_loss:.2f} | TP={take_profit:.2f} | "
            f"regime={regime} | {justification}"
        )
        return trade_id

    def log_trade_close(self, trade_id, exit_price, pnl, pnl_pct):
        """Record trade closing."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE trades SET timestamp_close=?, exit_price=?, pnl=?,
                   pnl_pct=?, status='CLOSED' WHERE id=?""",
                (now, exit_price, pnl, pnl_pct, trade_id),
            )
        result = "WIN" if pnl > 0 else "LOSS"
        self.logger.info(
            f"CLOSE #{trade_id} [{result}] | exit={exit_price:.2f} | "
            f"PnL={pnl:.2f} USDT ({pnl_pct:+.2%})"
        )

    def log_decision(self, action, reason, indicators=None, confidence=0.0):
        """Record each decision (buy, sell, hold) with justification."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO decisions (timestamp, action, reason, indicators, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (now, action, reason, json.dumps(indicators or {}), confidence),
            )
        self.logger.debug(f"DECISION: {action} | conf={confidence:.2f} | {reason}")

    def log_equity(self, balance, unrealized_pnl=0.0, drawdown=0.0):
        """Record a point on the equity curve."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO equity_curve (timestamp, balance, unrealized_pnl, drawdown)
                   VALUES (?, ?, ?, ?)""",
                (now, balance, unrealized_pnl, drawdown),
            )

    def get_closed_trades(self, limit=None):
        """Get closed trades for analysis."""
        query = "SELECT * FROM trades WHERE status='CLOSED' ORDER BY id DESC"
        if limit:
            query += f" LIMIT {limit}"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(query).fetchall()]

    def get_equity_curve(self):
        """Get full equity curve."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(
                "SELECT * FROM equity_curve ORDER BY id"
            ).fetchall()]

    def get_trade_count(self):
        """Return total number of closed trades."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status='CLOSED'"
            ).fetchone()
            return row[0]
