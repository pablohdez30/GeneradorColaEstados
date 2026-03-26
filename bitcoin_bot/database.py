"""
database.py — Capa de persistencia SQLite
==========================================
Almacena todo el historial del bot para:
  1. Auditoría y análisis posterior
  2. Reentrenamiento del módulo ML
  3. Generación de informes semanales
  4. Reproducción de sesiones

Diseño: esquema normalizado con tablas separadas por dominio.
Se usa sqlite3 nativo (sin ORM) para máximo control y mínimas dependencias.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from config import CONFIG


# ─────────────────────────────────────────────
#  ESQUEMA DDL
# ─────────────────────────────────────────────
SCHEMA_SQL = """
-- Operaciones cerradas (fuente de verdad del bot)
CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL DEFAULT 'BTC/USDT',
    direction       TEXT NOT NULL CHECK(direction IN ('long', 'short')),
    entry_price     REAL NOT NULL,
    exit_price      REAL,
    size_btc        REAL NOT NULL,       -- tamaño en BTC
    size_usdt       REAL NOT NULL,       -- valor nominal en USDT
    stop_loss       REAL NOT NULL,
    take_profit_1   REAL NOT NULL,
    take_profit_2   REAL NOT NULL,
    entry_ts        TEXT NOT NULL,
    exit_ts         TEXT,
    duration_min    REAL,
    pnl_usdt        REAL,               -- P&L en USDT (neto de fees)
    pnl_pct         REAL,               -- P&L en % del capital utilizado
    fee_usdt        REAL,
    close_reason    TEXT,               -- 'stop_loss' | 'take_profit_1' | 'take_profit_2' | 'trailing' | 'timeout' | 'ml_close' | 'manual'
    status          TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed', 'cancelled')),

    -- Indicadores en el momento de la entrada
    rsi_entry       REAL,
    macd_hist_entry REAL,
    bb_position_entry REAL,
    ema_cross_entry TEXT,
    volume_ratio_entry REAL,
    atr_entry       REAL,
    market_regime   TEXT,
    fear_greed      INTEGER,
    btc_dominance   REAL,

    -- Metadatos ML
    ml_action       TEXT,
    ml_confidence   REAL,
    ml_state        TEXT                -- JSON del vector de estado
);

-- Velas OHLCV almacenadas localmente para backtesting/reentrenamiento
CREATE TABLE IF NOT EXISTS candles (
    ts          TEXT NOT NULL,
    timeframe   TEXT NOT NULL DEFAULT '1m',
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    PRIMARY KEY (ts, timeframe)
);

-- Snapshot de métricas del bot cada N minutos
CREATE TABLE IF NOT EXISTS metrics (
    ts              TEXT PRIMARY KEY,
    balance_usdt    REAL NOT NULL,
    equity_usdt     REAL NOT NULL,      -- balance + valor de posición abierta
    total_trades    INTEGER,
    winning_trades  INTEGER,
    losing_trades   INTEGER,
    win_rate        REAL,
    total_pnl       REAL,
    sharpe_ratio    REAL,
    max_drawdown    REAL,
    current_drawdown REAL,
    open_position   TEXT                -- JSON de la posición abierta si existe
);

-- Señales generadas (incluidas las no ejecutadas)
CREATE TABLE IF NOT EXISTS signals (
    ts          TEXT NOT NULL,
    direction   TEXT NOT NULL,
    confidence  REAL NOT NULL,
    price       REAL NOT NULL,
    executed    INTEGER DEFAULT 0,      -- 1 si generó trade, 0 si fue bloqueada
    block_reason TEXT,
    indicators  TEXT,                   -- JSON con todos los indicadores
    PRIMARY KEY (ts, direction)
);

-- Historial del modelo ML (para seguir evolución del aprendizaje)
CREATE TABLE IF NOT EXISTS ml_history (
    ts              TEXT PRIMARY KEY,
    episode         INTEGER,
    avg_reward      REAL,
    win_rate        REAL,
    trades_used     INTEGER,
    epsilon         REAL,
    q_table_size    INTEGER
);

-- Señales externas históricas
CREATE TABLE IF NOT EXISTS external_signals (
    ts              TEXT PRIMARY KEY,
    fear_greed      INTEGER,
    fear_greed_label TEXT,
    btc_dominance   REAL,
    market_cap_usdt REAL
);

-- Índices para queries frecuentes
CREATE INDEX IF NOT EXISTS idx_trades_entry_ts ON trades(entry_ts);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles(ts);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts);
"""


# ─────────────────────────────────────────────
#  CLASE DATABASE
# ─────────────────────────────────────────────
class Database:
    """
    Interfaz de acceso a la base de datos SQLite del bot.
    Thread-safe: cada llamada abre y cierra su propia conexión.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or CONFIG.database.path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        """Context manager que garantiza commit/rollback automático."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row      # acceso por nombre de columna
        conn.execute("PRAGMA journal_mode=WAL")  # mejor concurrencia
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Crea tablas e índices si no existen."""
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    # ── TRADES ─────────────────────────────────

    def save_trade_open(self, trade: Dict[str, Any]):
        """Inserta un trade recién abierto."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO trades (
                    id, symbol, direction, entry_price, size_btc, size_usdt,
                    stop_loss, take_profit_1, take_profit_2, entry_ts, status,
                    rsi_entry, macd_hist_entry, bb_position_entry,
                    ema_cross_entry, volume_ratio_entry, atr_entry,
                    market_regime, fear_greed, btc_dominance,
                    ml_action, ml_confidence, ml_state
                ) VALUES (
                    :id, :symbol, :direction, :entry_price, :size_btc, :size_usdt,
                    :stop_loss, :take_profit_1, :take_profit_2, :entry_ts, 'open',
                    :rsi_entry, :macd_hist_entry, :bb_position_entry,
                    :ema_cross_entry, :volume_ratio_entry, :atr_entry,
                    :market_regime, :fear_greed, :btc_dominance,
                    :ml_action, :ml_confidence, :ml_state
                )
            """, trade)

    def save_trade_close(self, trade_id: str, close_data: Dict[str, Any]):
        """Actualiza trade con datos de cierre."""
        close_data["id"] = trade_id
        with self._conn() as conn:
            conn.execute("""
                UPDATE trades SET
                    exit_price   = :exit_price,
                    exit_ts      = :exit_ts,
                    duration_min = :duration_min,
                    pnl_usdt     = :pnl_usdt,
                    pnl_pct      = :pnl_pct,
                    fee_usdt     = :fee_usdt,
                    close_reason = :close_reason,
                    status       = 'closed'
                WHERE id = :id
            """, close_data)

    def get_open_trade(self) -> Optional[Dict]:
        """Devuelve la posición abierta actual (si existe)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE status='open' LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_closed_trades(self, limit: int = 500) -> List[Dict]:
        """Devuelve los últimos N trades cerrados para análisis/ML."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM trades
                WHERE status='closed'
                ORDER BY exit_ts DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_trade_stats(self) -> Dict[str, Any]:
        """Estadísticas agregadas de todos los trades cerrados."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                    AS total,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl_usdt < 0 THEN 1 ELSE 0 END) AS losses,
                    SUM(pnl_usdt)               AS total_pnl,
                    AVG(pnl_usdt)               AS avg_pnl,
                    MAX(pnl_usdt)               AS best_trade,
                    MIN(pnl_usdt)               AS worst_trade,
                    AVG(duration_min)           AS avg_duration_min
                FROM trades WHERE status='closed'
            """).fetchone()
            return dict(row) if row else {}

    # ── CANDLES ────────────────────────────────

    def save_candles(self, candles: List[Dict[str, Any]], timeframe: str = "1m"):
        """Inserta o reemplaza velas OHLCV."""
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO candles (ts, timeframe, open, high, low, close, volume)
                VALUES (:ts, :timeframe, :open, :high, :low, :close, :volume)
            """, [{**c, "timeframe": timeframe} for c in candles])

    def get_candles(self, limit: int = 500, timeframe: str = "1m") -> List[Dict]:
        """Recupera las últimas N velas."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM candles
                WHERE timeframe=?
                ORDER BY ts DESC
                LIMIT ?
            """, (timeframe, limit)).fetchall()
            return [dict(r) for r in reversed(rows)]  # orden cronológico

    # ── MÉTRICAS ───────────────────────────────

    def save_metrics(self, metrics: Dict[str, Any]):
        """Guarda snapshot de métricas del bot."""
        metrics["ts"] = datetime.now(timezone.utc).isoformat()
        if "open_position" in metrics and metrics["open_position"]:
            metrics["open_position"] = json.dumps(metrics["open_position"])
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metrics (
                    ts, balance_usdt, equity_usdt, total_trades,
                    winning_trades, losing_trades, win_rate, total_pnl,
                    sharpe_ratio, max_drawdown, current_drawdown, open_position
                ) VALUES (
                    :ts, :balance_usdt, :equity_usdt, :total_trades,
                    :winning_trades, :losing_trades, :win_rate, :total_pnl,
                    :sharpe_ratio, :max_drawdown, :current_drawdown, :open_position
                )
            """, metrics)

    def get_equity_curve(self, limit: int = 1000) -> List[Dict]:
        """Devuelve la curva de equity para el dashboard."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT ts, equity_usdt, balance_usdt, total_pnl
                FROM metrics ORDER BY ts DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in reversed(rows)]

    # ── SEÑALES ────────────────────────────────

    def save_signal(self, signal: Dict[str, Any]):
        """Persiste una señal generada (ejecutada o no)."""
        if "indicators" in signal:
            signal["indicators"] = json.dumps(signal["indicators"], default=str)
        signal.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO signals
                (ts, direction, confidence, price, executed, block_reason, indicators)
                VALUES (:ts, :direction, :confidence, :price, :executed, :block_reason, :indicators)
            """, signal)

    # ── ML HISTORY ─────────────────────────────

    def save_ml_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["ts"] = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO ml_history
                (ts, episode, avg_reward, win_rate, trades_used, epsilon, q_table_size)
                VALUES (:ts, :episode, :avg_reward, :win_rate, :trades_used, :epsilon, :q_table_size)
            """, checkpoint)

    # ── SEÑALES EXTERNAS ───────────────────────

    def save_external_signal(self, data: Dict[str, Any]):
        data.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO external_signals
                (ts, fear_greed, fear_greed_label, btc_dominance, market_cap_usdt)
                VALUES (:ts, :fear_greed, :fear_greed_label, :btc_dominance, :market_cap_usdt)
            """, data)

    def get_latest_external_signal(self) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT * FROM external_signals ORDER BY ts DESC LIMIT 1
            """).fetchone()
            return dict(row) if row else None

    # ── INFORME SEMANAL ────────────────────────

    def generate_weekly_report(self) -> Dict[str, Any]:
        """Genera datos para el informe semanal automático."""
        with self._conn() as conn:
            # Trades de los últimos 7 días
            week_trades = conn.execute("""
                SELECT * FROM trades
                WHERE status='closed'
                  AND exit_ts >= datetime('now', '-7 days')
                ORDER BY exit_ts
            """).fetchall()

            week_stats = conn.execute("""
                SELECT
                    COUNT(*)    AS total,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(pnl_usdt)   AS total_pnl,
                    AVG(pnl_usdt)   AS avg_pnl,
                    MAX(pnl_usdt)   AS best,
                    MIN(pnl_usdt)   AS worst
                FROM trades
                WHERE status='closed'
                  AND exit_ts >= datetime('now', '-7 days')
            """).fetchone()

            regime_dist = conn.execute("""
                SELECT market_regime, COUNT(*) AS count
                FROM trades
                WHERE status='closed'
                  AND exit_ts >= datetime('now', '-7 days')
                GROUP BY market_regime
            """).fetchall()

        return {
            "week_trades": [dict(t) for t in week_trades],
            "stats": dict(week_stats) if week_stats else {},
            "regime_distribution": [dict(r) for r in regime_dist],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


# ─────────────────────────────────────────────
#  INSTANCIA GLOBAL
# ─────────────────────────────────────────────
DB = Database()
