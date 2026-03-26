"""
SQLite database for persisting trades, portfolio state, and learning data.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager for the trading bot."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_question TEXT,
                outcome TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                shares REAL NOT NULL,
                strategy TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'open',
                close_price REAL,
                close_timestamp TEXT,
                pnl REAL,
                pnl_pct REAL
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL NOT NULL,
                total_value REAL NOT NULL,
                open_positions INTEGER NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                market_question TEXT,
                outcome TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                shares REAL NOT NULL,
                amount_invested REAL NOT NULL,
                strategy TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open'
            );

            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                weight REAL DEFAULT 0.33
            );

            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                question TEXT,
                yes_price REAL,
                no_price REAL,
                volume_24h REAL,
                liquidity REAL
            );

            CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market_id);
        """)

        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def record_trade(self, market_id: str, market_question: str, outcome: str,
                     side: str, price: float, amount: float, shares: float,
                     strategy: str, confidence: float, reason: str) -> int:
        """Record a new trade."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (timestamp, market_id, market_question, outcome,
                                side, price, amount, shares, strategy, confidence, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), market_id, market_question,
              outcome, side, price, amount, shares, strategy, confidence, reason))
        self.conn.commit()
        return cursor.lastrowid

    def close_trade(self, trade_id: int, close_price: float, pnl: float, pnl_pct: float):
        """Close a trade with its result."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE trades SET status = 'closed', close_price = ?,
                             close_timestamp = ?, pnl = ?, pnl_pct = ?
            WHERE id = ?
        """, (close_price, datetime.utcnow().isoformat(), pnl, pnl_pct, trade_id))
        self.conn.commit()

    def open_position(self, market_id: str, market_question: str, outcome: str,
                      entry_price: float, shares: float, amount: float,
                      strategy: str, stop_loss: float, take_profit: float) -> int:
        """Open a new position."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO positions (market_id, market_question, outcome, entry_price,
                                   current_price, shares, amount_invested, strategy,
                                   opened_at, stop_loss, take_profit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (market_id, market_question, outcome, entry_price, entry_price,
              shares, amount, strategy, datetime.utcnow().isoformat(),
              stop_loss, take_profit))
        self.conn.commit()
        return cursor.lastrowid

    def close_position(self, position_id: int):
        """Mark a position as closed."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE positions SET status = 'closed' WHERE id = ?",
            (position_id,))
        self.conn.commit()

    def update_position_price(self, position_id: int, current_price: float):
        """Update the current price of a position."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE positions SET current_price = ? WHERE id = ?",
            (current_price, position_id))
        self.conn.commit()

    def get_open_positions(self) -> list[dict]:
        """Get all open positions."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE status = 'open'")
        return [dict(row) for row in cursor.fetchall()]

    def get_position_for_market(self, market_id: str, outcome: str) -> Optional[dict]:
        """Get open position for a specific market/outcome."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM positions WHERE market_id = ? AND outcome = ? AND status = 'open'",
            (market_id, outcome))
        row = cursor.fetchone()
        return dict(row) if row else None

    def record_portfolio_snapshot(self, balance: float, total_value: float,
                                  open_positions: int, total_trades: int,
                                  winning: int, losing: int):
        """Record a portfolio snapshot."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO portfolio (timestamp, balance, total_value, open_positions,
                                   total_trades, winning_trades, losing_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), balance, total_value,
              open_positions, total_trades, winning, losing))
        self.conn.commit()

    def get_strategy_stats(self, strategy_name: str) -> dict:
        """Get performance stats for a strategy."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                   COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(AVG(pnl_pct), 0) as avg_pnl_pct,
                   COALESCE(AVG(confidence), 0) as avg_confidence
            FROM trades WHERE strategy = ? AND status = 'closed'
        """, (strategy_name,))
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_recent_trades(self, limit: int = 20) -> list[dict]:
        """Get recent trades."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_portfolio_history(self, limit: int = 100) -> list[dict]:
        """Get portfolio value history."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM portfolio ORDER BY timestamp DESC LIMIT ?",
            (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def save_strategy_weights(self, weights: dict):
        """Save current strategy weights."""
        for name, weight in weights.items():
            stats = self.get_strategy_stats(name)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO strategy_performance
                    (timestamp, strategy_name, total_trades, winning_trades,
                     total_pnl, avg_confidence, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (datetime.utcnow().isoformat(), name,
                  stats.get("total", 0), stats.get("wins", 0),
                  stats.get("total_pnl", 0), stats.get("avg_confidence", 0),
                  weight))
        self.conn.commit()

    def record_market_snapshot(self, market_id: str, question: str,
                                yes_price: float, no_price: float,
                                volume: float, liquidity: float):
        """Record a market price snapshot for historical tracking."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_snapshots (timestamp, market_id, question,
                                          yes_price, no_price, volume_24h, liquidity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), market_id, question,
              yes_price, no_price, volume, liquidity))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
