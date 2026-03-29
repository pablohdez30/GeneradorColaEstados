"""
logger.py — Logging y persistencia de alertas del Sniper Bot.
"""

import logging
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

from sniper_bot.config import LOG_FILE, LOG_LEVEL, DB_PATH


def setup_logger(name: str = "sniper") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if logger.handlers:
        return logger

    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


class AlertLogger:
    """Registra alertas enviadas en SQLite."""

    def __init__(self):
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    score INTEGER,
                    confidence REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    leverage INTEGER,
                    reasons TEXT,
                    indicators TEXT,
                    fear_greed INTEGER,
                    telegram_sent INTEGER DEFAULT 0
                )
            """)

    def log_alert(self, signal: dict, telegram_sent: bool):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO alerts
                (timestamp, direction, score, confidence, entry_price, stop_loss,
                 take_profit, leverage, reasons, indicators, fear_greed, telegram_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, signal["direction"], signal["score"], signal["confidence"],
                signal["entry"], signal["stop_loss"], signal["take_profit"],
                signal["leverage"], json.dumps(signal["reasons"]),
                json.dumps(signal["indicators"]), signal["fear_greed"],
                1 if telegram_sent else 0,
            ))

    def get_alerts_today(self) -> int:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?",
                (f"{today}%",)
            ).fetchone()
            return row[0] if row else 0
