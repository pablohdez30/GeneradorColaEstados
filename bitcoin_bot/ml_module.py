"""
ml_module.py — Sistema de aprendizaje adaptativo (Q-Learning tabular)
======================================================================
Implementa un agente Q-Learning que aprende de cada operación cerrada
para ajustar la toma de decisiones del bot.

Arquitectura elegida: Q-Learning tabular (no redes neuronales)
Justificación:
  - Los datos de trading son escasos: 50-200 trades/día en scalping
  - Las redes neuronales requieren miles de ejemplos antes de converger
  - Q-Learning tabular converge rápido con pocos datos
  - Es interpretable: se puede inspeccionar la Q-table
  - Cuando haya suficientes datos (>1000 trades), migrar a DQN/PPO

El agente observa el ESTADO (indicadores discretizados) y aprende
qué ACCIONES maximizan la recompensa acumulada.

Estado: vector de 9 features discretizadas en bins
Acciones: hold, long, short, close_long, close_short
Recompensa: función de P&L normalizado + Sharpe ratio + penalización drawdown
"""

import os
import pickle
import time
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np

from config import CONFIG
from logger import LOG, EventType
from database import DB


# ─────────────────────────────────────────────
#  DISCRETIZACIÓN DE ESTADO
# ─────────────────────────────────────────────
class StateEncoder:
    """
    Convierte el vector continuo de indicadores en un estado discreto
    (tupla de enteros) que puede indexar la Q-table.

    Diseño: cada feature se mapea a un número de bin [0, N-1].
    La Q-table tiene size = N^features pero en la práctica usa
    un diccionario sparse (la mayoría de estados no se visitan).
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        # Rangos esperados de cada feature (definidos por conocimiento de dominio)
        self.feature_ranges = {
            "rsi":            (0.0, 100.0),
            "macd_hist":      (-500.0, 500.0),
            "bb_position":    (0.0, 1.0),
            "ema_crossover":  (-1.0, 1.0),   # -1=bajista, 0=neutro, 1=alcista
            "volume_ratio":   (0.0, 5.0),
            "atr_normalized": (0.0, 0.02),
            "market_regime":  (0.0, 4.0),    # enum discreto
            "fear_greed":     (0.0, 100.0),
            "btc_dominance":  (30.0, 70.0),
        }

        self.regime_map = {
            "trending_up": 0, "trending_down": 1,
            "ranging": 2, "volatile": 3, "unknown": 4,
        }

    def encode(self, analysis_dict: Dict, fear_greed: int,
               btc_dominance: float) -> Tuple[int, ...]:
        """
        Convierte un AnalysisResult.to_dict() en estado discreto.
        """
        raw_features = {
            "rsi": analysis_dict.get("rsi", 50.0),
            "macd_hist": analysis_dict.get("macd_hist", 0.0),
            "bb_position": analysis_dict.get("bb_position", 0.5),
            "ema_crossover": 1.0 if analysis_dict.get("ema_cross") == "bullish" else -1.0,
            "volume_ratio": min(analysis_dict.get("volume_ratio", 1.0), 5.0),
            "atr_normalized": min(analysis_dict.get("atr_normalized", 0.001), 0.02),
            "market_regime": float(self.regime_map.get(
                analysis_dict.get("market_regime", "unknown"), 4
            )),
            "fear_greed": float(fear_greed),
            "btc_dominance": float(btc_dominance),
        }

        state = []
        for feat, (low, high) in self.feature_ranges.items():
            val = raw_features.get(feat, (low + high) / 2)
            val = max(low, min(high, val))  # clip
            bin_idx = int((val - low) / (high - low) * (self.n_bins - 1))
            bin_idx = max(0, min(self.n_bins - 1, bin_idx))
            state.append(bin_idx)

        return tuple(state)


# ─────────────────────────────────────────────
#  FUNCIÓN DE RECOMPENSA
# ─────────────────────────────────────────────
class RewardFunction:
    """
    Calcula la recompensa para el agente Q-Learning basándose en el
    resultado de una operación cerrada.

    Componentes:
      1. P&L normalizado (principal): incentiva ganancias
      2. Bonus Sharpe: incentiva consistencia
      3. Penalización drawdown: desincentiva pérdidas grandes
      4. Bonus por duración: cortas y rentables > largas y rentables
    """

    def __init__(self):
        self.ml_cfg = CONFIG.ml
        self._recent_pnls: List[float] = []   # últimos 20 P&Ls para Sharpe

    def calculate(self, pnl_usdt: float, pnl_pct: float,
                  duration_min: float, drawdown: float,
                  size_usdt: float) -> float:
        """Recompensa en rango aproximado [-1, +1]."""

        # ── Componente 1: P&L normalizado ───────
        # Normalizar por el capital en riesgo (2% máx → pnl_pct típico 0.01-0.05)
        pnl_reward = np.clip(pnl_pct * 10, -1.0, 1.0)

        # ── Componente 2: Sharpe reciente ────────
        self._recent_pnls.append(pnl_usdt)
        if len(self._recent_pnls) > 20:
            self._recent_pnls.pop(0)

        if len(self._recent_pnls) > 3:
            arr = np.array(self._recent_pnls)
            sharpe = np.mean(arr) / (np.std(arr) + 1e-8)
            sharpe_reward = np.clip(sharpe * 0.1, -0.3, 0.3)
        else:
            sharpe_reward = 0.0

        # ── Componente 3: Penalización drawdown ──
        # Drawdown > 5% es un evento negativo incluso en trade ganador
        dd_penalty = -drawdown * self.ml_cfg.reward_drawdown_penalty * 5

        # ── Componente 4: Bonus por duración ─────
        # Scalping: penalizar trades que duran más de 2 horas
        duration_factor = max(0.5, 1.0 - duration_min / 240)

        # ── Combinar ─────────────────────────────
        reward = (
            pnl_reward * self.ml_cfg.reward_pnl_weight
            + sharpe_reward * self.ml_cfg.reward_sharpe_weight
            + dd_penalty
        ) * duration_factor

        return float(np.clip(reward, -2.0, 2.0))


# ─────────────────────────────────────────────
#  AGENTE Q-LEARNING
# ─────────────────────────────────────────────
class QLearningAgent:
    """
    Agente Q-Learning tabular con exploración ε-greedy.

    La Q-table mapea (estado, acción) → valor esperado de recompensa.
    Se actualiza mediante la regla de Bellman en cada trade cerrado.
    """

    def __init__(self):
        self.ml_cfg = CONFIG.ml
        self.encoder = StateEncoder(n_bins=self.ml_cfg.state_bins)
        self.reward_fn = RewardFunction()

        # Q-table: defaultdict con valor inicial 0
        self.q_table: Dict[Tuple, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.ml_cfg.actions}
        )

        # Parámetros de aprendizaje
        self.epsilon = self.ml_cfg.epsilon_start
        self.lr = self.ml_cfg.learning_rate
        self.gamma = self.ml_cfg.discount_factor

        # Buffer de experiencias (estado, acción, recompensa, estado_siguiente)
        self._experience_buffer: List[Tuple] = []
        self._total_episodes: int = 0

        # Historial de rendimiento del agente
        self._episode_rewards: List[float] = []

        # Cargar modelo guardado si existe
        self._load_model()

        LOG.log(EventType.SYSTEM,
                f"QLearningAgent inicializado | "
                f"Q-table: {len(self.q_table)} estados | "
                f"ε={self.epsilon:.3f}")

    # ── SELECCIÓN DE ACCIÓN ─────────────────────

    def select_action(self, state: Tuple,
                      force_close: bool = False) -> str:
        """
        Selecciona acción usando política ε-greedy.
        - Con probabilidad ε: acción aleatoria (exploración)
        - Con probabilidad 1-ε: acción con mayor Q-valor (explotación)

        force_close: sobreescribe con close si hay posición abierta y
                     el agente decide holdear indefinidamente.
        """
        if force_close:
            return "close_long"  # fallback de timeout

        # Exploración
        if random.random() < self.epsilon:
            action = random.choice(self.ml_cfg.actions)
            exploration = True
        else:
            # Explotación: mejor acción según Q-table
            q_values = self.q_table[state]
            action = max(q_values, key=q_values.get)
            exploration = False

        LOG.ml_action(
            state=list(state),
            action=action,
            q_values=self.q_table[state],
            epsilon=self.epsilon,
        )

        return action

    # ── ACTUALIZACIÓN Q-TABLE (APRENDIZAJE) ──────

    def update(self, state: Tuple, action: str, reward: float,
               next_state: Tuple, done: bool = True):
        """
        Actualiza la Q-table con la regla de Bellman:
        Q(s,a) ← Q(s,a) + α × [r + γ × max_a'(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            next_max_q = max(self.q_table[next_state].values())
            target = reward + self.gamma * next_max_q

        # Actualización gradual
        self.q_table[state][action] = current_q + self.lr * (target - current_q)
        self._episode_rewards.append(reward)

    # ── PROCESAMIENTO DE TRADES CERRADOS ─────────

    def on_trade_closed(self, entry_analysis: Dict, exit_analysis: Dict,
                        pnl_usdt: float, pnl_pct: float,
                        duration_min: float, drawdown: float,
                        size_usdt: float, action_taken: str,
                        fear_greed: int, btc_dominance: float):
        """
        Aprende de un trade cerrado.
        Llamado por main.py después de cada cierre de posición.
        """
        # Codificar estados
        state = self.encoder.encode(entry_analysis, fear_greed, btc_dominance)
        next_state = self.encoder.encode(exit_analysis, fear_greed, btc_dominance)

        # Calcular recompensa
        reward = self.reward_fn.calculate(
            pnl_usdt, pnl_pct, duration_min, drawdown, size_usdt
        )

        # Guardar experiencia
        self._experience_buffer.append(
            (state, action_taken, reward, next_state)
        )

        # Actualizar Q-table
        self.update(state, action_taken, reward, next_state, done=True)

        # Decaer epsilon (reducir exploración con el tiempo)
        self.epsilon = max(
            self.ml_cfg.epsilon_end,
            self.epsilon * self.ml_cfg.epsilon_decay
        )

        self._total_episodes += 1

        LOG.debug(EventType.ML_ACTION,
                  f"Aprendizaje: acción={action_taken} | "
                  f"recompensa={reward:.4f} | ε={self.epsilon:.4f}")

    # ── REENTRENAMIENTO CON HISTORIAL ────────────

    def retrain_from_history(self, trades: List[Dict],
                              fear_greed: int = 50,
                              btc_dominance: float = 50.0) -> Dict:
        """
        Reentrenamiento por lotes usando el historial de trades de la BD.
        Itera varias veces sobre los datos (epochs) para reforzar aprendizaje.
        """
        if len(trades) < self.ml_cfg.min_trades_for_training:
            LOG.warning(EventType.ML_RETRAIN,
                        f"Insuficientes trades para reentrenar: {len(trades)} "
                        f"< {self.ml_cfg.min_trades_for_training}")
            return {}

        LOG.log(EventType.ML_RETRAIN,
                f"Iniciando reentrenamiento con {len(trades)} trades...")

        total_reward = 0.0
        wins = 0
        n_epochs = 3  # pasar 3 veces por los datos

        for epoch in range(n_epochs):
            random.shuffle(trades)  # shuffle para evitar overfitting a orden temporal
            for trade in trades:
                if not trade.get("pnl_usdt"):
                    continue

                # Reconstruir indicadores del estado de entrada
                entry_indicators = {
                    "rsi": trade.get("rsi_entry", 50),
                    "macd_hist": trade.get("macd_hist_entry", 0),
                    "bb_position": trade.get("bb_position_entry", 0.5),
                    "ema_cross": trade.get("ema_cross_entry", "neutral"),
                    "volume_ratio": trade.get("volume_ratio_entry", 1.0),
                    "atr_normalized": trade.get("atr_entry", 0.001) / max(trade.get("entry_price", 50000), 1),
                    "market_regime": trade.get("market_regime", "unknown"),
                }

                state = self.encoder.encode(entry_indicators, fear_greed, btc_dominance)

                pnl_usdt = trade["pnl_usdt"]
                pnl_pct = trade.get("pnl_pct", pnl_usdt / max(trade.get("size_usdt", 1000), 1))

                reward = self.reward_fn.calculate(
                    pnl_usdt=pnl_usdt,
                    pnl_pct=pnl_pct,
                    duration_min=trade.get("duration_min", 30),
                    drawdown=0.0,  # no disponible en historial
                    size_usdt=trade.get("size_usdt", 1000),
                )

                action = trade.get("ml_action") or (
                    "long" if trade["direction"] == "long" else "short"
                )
                self.q_table[state][action] = (
                    self.q_table[state][action] * 0.9 + reward * 0.1
                )

                total_reward += reward
                if pnl_usdt > 0:
                    wins += 1

        avg_reward = total_reward / (len(trades) * n_epochs + 1e-8)
        win_rate = wins / (len(trades) * n_epochs + 1e-8)

        LOG.ml_retrain(
            episodes=len(trades) * n_epochs,
            avg_reward=avg_reward,
            win_rate=win_rate,
            trades_used=len(trades),
        )

        # Guardar checkpoint
        checkpoint = {
            "episode": self._total_episodes,
            "avg_reward": avg_reward,
            "win_rate": win_rate,
            "trades_used": len(trades),
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
        }
        DB.save_ml_checkpoint(checkpoint)
        self._save_model()

        return checkpoint

    # ── INFLUENCIA SOBRE LA ESTRATEGIA ───────────

    def adjust_signal_confidence(self, state: Tuple,
                                  base_confidence: float,
                                  direction: str) -> float:
        """
        Ajusta la confianza de la señal de la estrategia usando los
        Q-valores aprendidos.

        Si el agente ha aprendido que en este estado 'long' funciona bien,
        aumenta la confianza. Si aprendió que es malo, la reduce.
        """
        action = direction.lower()
        q_long = self.q_table[state].get("long", 0.0)
        q_short = self.q_table[state].get("short", 0.0)
        q_hold = self.q_table[state].get("hold", 0.0)

        q_range = max(abs(q_long), abs(q_short), abs(q_hold), 1e-8)
        q_normalized = self.q_table[state].get(action, 0.0) / q_range

        # Ajuste suave: ±20% sobre la confianza base
        adjustment = q_normalized * 0.20
        adjusted = base_confidence + adjustment

        return float(np.clip(adjusted, 0.0, 1.0))

    def get_market_regime_adjustment(self, state: Tuple) -> Dict[str, float]:
        """
        Devuelve los Q-valores del estado actual para informar al
        strategy_engine sobre qué acciones son más prometedoras.
        """
        return dict(self.q_table[state])

    # ── PERSISTENCIA ─────────────────────────────

    def _save_model(self):
        """Serializa la Q-table en disco."""
        model_path = "bitcoin_bot/models/q_table.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump({
                    "q_table": dict(self.q_table),
                    "epsilon": self.epsilon,
                    "total_episodes": self._total_episodes,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }, f)
            LOG.debug(EventType.ML_RETRAIN,
                      f"Modelo guardado: {len(self.q_table)} estados")
        except Exception as e:
            LOG.error(EventType.ML_RETRAIN, f"Error guardando modelo: {e}")

    def _load_model(self):
        """Carga Q-table existente si hay un modelo guardado."""
        model_path = "bitcoin_bot/models/q_table.pkl"
        if not os.path.exists(model_path):
            return

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            # Restaurar defaultdict desde dict normal
            for state, q_vals in data["q_table"].items():
                self.q_table[state] = q_vals
            self.epsilon = data.get("epsilon", self.ml_cfg.epsilon_start)
            self._total_episodes = data.get("total_episodes", 0)
            LOG.log(EventType.SYSTEM,
                    f"Modelo ML cargado: {len(self.q_table)} estados | "
                    f"ε={self.epsilon:.3f} | "
                    f"episodios={self._total_episodes}")
        except Exception as e:
            LOG.warning(EventType.ML_RETRAIN,
                        f"No se pudo cargar modelo previo: {e}")

    # ── PROPIEDADES ──────────────────────────────

    @property
    def stats(self) -> Dict:
        recent = self._episode_rewards[-50:] if self._episode_rewards else [0]
        return {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "total_episodes": self._total_episodes,
            "avg_recent_reward": np.mean(recent),
            "experience_buffer_size": len(self._experience_buffer),
        }


# ─────────────────────────────────────────────
#  DETECTOR DE RÉGIMEN CON ML
# ─────────────────────────────────────────────
class MLRegimeAdvisor:
    """
    Usa el agente Q-Learning para sugerir ajustes de parámetros
    según el régimen de mercado aprendido.

    Con el tiempo, el agente aprende que ciertos estados (ej: ADX alto,
    RSI medio, MACD positivo) correlacionan con buenos trades long.
    """

    def __init__(self, agent: QLearningAgent):
        self.agent = agent

    def get_parameter_adjustments(self, state: Tuple) -> Dict[str, float]:
        """
        Devuelve ajustes sugeridos para los parámetros de la estrategia.
        Basado en el aprendizaje histórico del agente.
        """
        q_vals = self.agent.get_market_regime_adjustment(state)

        # Si el agente ve valor en 'hold', sugiere reducir agresividad
        hold_preference = q_vals.get("hold", 0)
        long_preference = q_vals.get("long", 0)
        short_preference = q_vals.get("short", 0)

        # Umbral de confianza ajustado según preferencias del agente
        max_action = max(q_vals.values()) if q_vals else 0
        if max_action <= 0:
            # El agente no ha aprendido nada útil aún
            confidence_boost = 0.0
        else:
            confidence_boost = max_action / (abs(max_action) + abs(hold_preference) + 1e-8) * 0.1

        return {
            "confidence_adjustment": confidence_boost,
            "preferred_direction": "long" if long_preference > short_preference else "short",
            "hold_preference": hold_preference,
        }
