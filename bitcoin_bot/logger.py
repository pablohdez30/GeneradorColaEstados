"""
logger.py — Sistema de logging estructurado del bot
=====================================================
Diseño: cada decisión del bot (señal, orden, ML, riesgo) se registra con
contexto completo: precio actual, indicadores activos, justificación.
Esto permite auditar por qué el bot tomó cada acción y alimentar el ML.

Usa structlog para logs estructurados en JSON + Rich para consola legible.
Fallback a logging estándar si structlog no está disponible.
"""

import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from enum import Enum


# ─────────────────────────────────────────────
#  CATEGORÍAS DE EVENTOS
# ─────────────────────────────────────────────
class EventType(str, Enum):
    SIGNAL = "SIGNAL"           # señal generada por strategy_engine
    ORDER_OPEN = "ORDER_OPEN"   # apertura de posición
    ORDER_CLOSE = "ORDER_CLOSE" # cierre de posición (con resultado)
    STOP_LOSS = "STOP_LOSS"     # activación de stop-loss
    TAKE_PROFIT = "TAKE_PROFIT" # activación de take-profit
    ML_RETRAIN = "ML_RETRAIN"   # reentrenamiento del modelo
    ML_ACTION = "ML_ACTION"     # acción tomada por el agente ML
    RISK_BLOCK = "RISK_BLOCK"   # operación bloqueada por risk manager
    MARKET_REGIME = "MARKET_REGIME"  # cambio de régimen detectado
    EXTERNAL_SIGNAL = "EXTERNAL"     # Fear&Greed, noticias
    SYSTEM = "SYSTEM"           # arranque, parada, errores


# ─────────────────────────────────────────────
#  FORMATEADOR JSON (para archivos de log)
# ─────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Serializa cada registro como una línea JSON para análisis posterior."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "event_type": getattr(record, "event_type", "SYSTEM"),
            "message": record.getMessage(),
        }
        # Agregar contexto extra si lo hay
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in (
                "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "name",
                "message", "event_type"
            )
        }
        if extra_fields:
            log_entry["context"] = extra_fields
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ─────────────────────────────────────────────
#  FORMATEADOR CONSOLA (legible por humanos)
# ─────────────────────────────────────────────
class ConsoleFormatter(logging.Formatter):
    """Formato colorizado para la terminal."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    EVENT_ICONS = {
        EventType.SIGNAL: "📡",
        EventType.ORDER_OPEN: "🟢",
        EventType.ORDER_CLOSE: "🔴",
        EventType.STOP_LOSS: "🛑",
        EventType.TAKE_PROFIT: "💰",
        EventType.ML_RETRAIN: "🧠",
        EventType.ML_ACTION: "🤖",
        EventType.RISK_BLOCK: "⛔",
        EventType.MARKET_REGIME: "📊",
        EventType.EXTERNAL_SIGNAL: "🌐",
        EventType.SYSTEM: "⚙️",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        event_type = getattr(record, "event_type", EventType.SYSTEM)
        icon = self.EVENT_ICONS.get(event_type, "•")

        # Extraer precio si está disponible
        price = getattr(record, "price", None)
        price_str = f" @ ${price:,.2f}" if price else ""

        pnl = getattr(record, "pnl", None)
        pnl_str = ""
        if pnl is not None:
            pnl_color = "\033[32m" if pnl >= 0 else "\033[31m"
            pnl_str = f" {pnl_color}P&L: {pnl:+.4f} USDT{self.RESET}"

        line = (
            f"{color}[{ts}] {record.levelname:8}{self.RESET} "
            f"{icon} {record.getMessage()}{price_str}{pnl_str}"
        )
        return line


# ─────────────────────────────────────────────
#  CLASE PRINCIPAL DE LOGGER
# ─────────────────────────────────────────────
class BotLogger:
    """
    Logger centralizado del bot. Emite a tres destinos:
      1. Consola: formato legible con colores
      2. Archivo JSON rotativo: para análisis y ML
      3. Archivo de trades: solo operaciones cerradas (CSV-like)
    """

    def __init__(self, name: str = "bitcoin_bot", log_level: str = "INFO"):
        self.name = name
        self._setup_log_directory()

        # Logger raíz del bot
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.handlers.clear()

        # ── Handler 1: consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ConsoleFormatter())
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

        # ── Handler 2: log JSON completo (rotativo por día)
        log_date = datetime.now().strftime("%Y%m%d")
        json_path = Path("bitcoin_bot/logs") / f"bot_{log_date}.jsonl"
        json_handler = logging.FileHandler(json_path, encoding="utf-8")
        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(json_handler)

        # ── Handler 3: solo errores en archivo separado
        error_path = Path("bitcoin_bot/logs") / "errors.log"
        error_handler = logging.FileHandler(error_path, encoding="utf-8")
        error_handler.setFormatter(JSONFormatter())
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)

        self.log(EventType.SYSTEM, "Logger inicializado", log_level=log_level)

    def _setup_log_directory(self):
        """Crea los directorios de datos si no existen."""
        for d in ["bitcoin_bot/logs", "bitcoin_bot/data", "bitcoin_bot/models"]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _log(self, level: int, event_type: EventType, message: str, **kwargs):
        """Método interno que agrega contexto estructurado al registro."""
        extra = {"event_type": event_type, **kwargs}
        self.logger.log(level, message, extra=extra)

    # ── API pública ─────────────────────────────

    def log(self, event_type: EventType, message: str, **kwargs):
        self._log(logging.INFO, event_type, message, **kwargs)

    def debug(self, event_type: EventType, message: str, **kwargs):
        self._log(logging.DEBUG, event_type, message, **kwargs)

    def warning(self, event_type: EventType, message: str, **kwargs):
        self._log(logging.WARNING, event_type, message, **kwargs)

    def error(self, event_type: EventType, message: str, **kwargs):
        self._log(logging.ERROR, event_type, message, **kwargs)

    # ── Helpers semánticos ───────────────────────

    def signal(self, direction: str, price: float, confidence: float,
               indicators: dict, reason: str):
        """Registra una señal generada por la estrategia."""
        self._log(
            logging.INFO, EventType.SIGNAL,
            f"Señal {direction.upper()} detectada — Confianza: {confidence:.1%} — {reason}",
            price=price,
            direction=direction,
            confidence=confidence,
            indicators=indicators,
            reason=reason,
        )

    def order_open(self, trade_id: str, direction: str, price: float,
                   size: float, stop_loss: float, take_profit_1: float,
                   take_profit_2: float, balance: float):
        """Registra apertura de posición."""
        self._log(
            logging.INFO, EventType.ORDER_OPEN,
            f"[{trade_id}] OPEN {direction.upper()} {size:.6f} BTC @ ${price:,.2f}",
            price=price,
            trade_id=trade_id,
            direction=direction,
            size=size,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            balance_after=balance,
        )

    def order_close(self, trade_id: str, direction: str, entry: float,
                    exit_price: float, size: float, pnl: float,
                    pnl_pct: float, duration_min: float, reason: str,
                    balance: float):
        """Registra cierre de posición con resultado completo."""
        result = "WIN" if pnl > 0 else "LOSS"
        self._log(
            logging.INFO, EventType.ORDER_CLOSE,
            f"[{trade_id}] CLOSE {result} — {reason} | "
            f"Entrada: ${entry:,.2f} → Salida: ${exit_price:,.2f} | "
            f"Duración: {duration_min:.1f}min",
            price=exit_price,
            pnl=pnl,
            trade_id=trade_id,
            direction=direction,
            entry=entry,
            exit_price=exit_price,
            size=size,
            pnl_pct=pnl_pct,
            duration_minutes=duration_min,
            reason=reason,
            balance_after=balance,
        )

    def risk_block(self, reason: str, details: dict):
        """Registra cuando el risk manager bloquea una operación."""
        self._log(
            logging.WARNING, EventType.RISK_BLOCK,
            f"Operación bloqueada: {reason}",
            reason=reason,
            **details,
        )

    def ml_action(self, state: list, action: str, q_values: dict,
                  epsilon: float):
        """Registra la decisión del agente Q-Learning."""
        self._log(
            logging.DEBUG, EventType.ML_ACTION,
            f"ML → Acción: {action} (ε={epsilon:.3f})",
            action=action,
            q_values=q_values,
            epsilon=epsilon,
        )

    def ml_retrain(self, episodes: int, avg_reward: float,
                   win_rate: float, trades_used: int):
        """Registra resultado de reentrenamiento."""
        self._log(
            logging.INFO, EventType.ML_RETRAIN,
            f"Reentrenamiento completado — {episodes} episodios | "
            f"Recompensa media: {avg_reward:.4f} | Win rate: {win_rate:.1%}",
            episodes=episodes,
            avg_reward=avg_reward,
            win_rate=win_rate,
            trades_used=trades_used,
        )

    def market_regime(self, regime: str, adx: float, volatility: float,
                      trend_strength: float):
        """Registra cambio de régimen de mercado detectado."""
        self._log(
            logging.INFO, EventType.MARKET_REGIME,
            f"Régimen detectado: {regime.upper()} | ADX={adx:.1f} | "
            f"Vol={volatility:.4f} | Tendencia={trend_strength:.2f}",
            regime=regime,
            adx=adx,
            volatility=volatility,
            trend_strength=trend_strength,
        )

    def external_signal(self, source: str, value: Any, interpretation: str):
        """Registra señales externas (Fear&Greed, dominancia BTC)."""
        self._log(
            logging.INFO, EventType.EXTERNAL_SIGNAL,
            f"[{source}] Valor: {value} → {interpretation}",
            source=source,
            value=value,
            interpretation=interpretation,
        )


# ─────────────────────────────────────────────
#  INSTANCIA GLOBAL
# ─────────────────────────────────────────────
# Se importa desde todos los módulos: `from logger import LOG`
LOG = BotLogger(log_level=os.getenv("LOG_LEVEL", "INFO"))
