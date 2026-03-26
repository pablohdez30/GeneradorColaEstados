"""
ml_module.py - Módulo de aprendizaje adaptativo (Q-Learning tabular).

Diseño:
──────
Se eligió Q-Learning tabular en lugar de PPO/DRL por las siguientes razones:
1. Más interpretable (podemos ver qué estados aprende)
2. Converge más rápido con pocos datos (primeras 30-50 trades)
3. Menor complejidad computacional para paper trading
4. Se puede migrar a PPO cuando haya suficientes datos (>1000 trades)

Estado del agente (features discretizadas):
- RSI zone: oversold / neutral / overbought
- MACD direction: bullish / bearish
- BB position: lower / middle / upper
- EMA trend: up / down
- Volume: normal / spike
- Market regime: trending / ranging / high_vol
- Sentiment: fear / neutral / greed

Acciones: BUY, SELL, HOLD

Recompensa: PnL normalizado del trade cerrado.

El modelo se reentrena cada 50 trades cerrados con todo el histórico.
"""

import json
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ML_RETRAIN_INTERVAL, ML_MIN_TRADES_FOR_TRAINING,
    ML_LEARNING_RATE, ML_DISCOUNT_FACTOR,
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    DB_PATH,
)
from bot.logger import setup_logger, TradeLogger

logger = setup_logger("ml_module")

MODEL_PATH = "bot/data/q_table.pkl"


class MarketState:
    """
    Discretiza el estado del mercado en categorías para Q-Learning.

    Decisión: Se discretiza porque Q-Learning tabular necesita estados
    finitos. Cada combinación de categorías es un estado único.
    Total de estados posibles: 3*2*3*2*2*3*3 = 648 (manejable).
    """

    @staticmethod
    def discretize(indicators: dict, regime: str, sentiment: float = 50) -> str:
        """
        Convierte indicadores continuos en un estado discreto.
        Retorna string que sirve como key en la Q-table.
        """
        # RSI → 3 zonas
        rsi = indicators.get("rsi", 50)
        if rsi < RSI_OVERSOLD:
            rsi_zone = "oversold"
        elif rsi > RSI_OVERBOUGHT:
            rsi_zone = "overbought"
        else:
            rsi_zone = "neutral"

        # MACD → 2 direcciones
        macd_hist = indicators.get("macd_histogram", 0)
        macd_dir = "bullish" if macd_hist > 0 else "bearish"

        # BB position → 3 zonas
        bb_pos = indicators.get("bb_position", 0.5)
        if bb_pos < 0.2:
            bb_zone = "lower"
        elif bb_pos > 0.8:
            bb_zone = "upper"
        else:
            bb_zone = "middle"

        # EMA trend → 2 direcciones
        ema_fast = indicators.get("ema_fast", 0)
        ema_slow = indicators.get("ema_slow", 0)
        ema_trend = "up" if ema_fast > ema_slow else "down"

        # Volumen → 2 estados
        vol_ratio = indicators.get("volume_ratio", 1.0)
        vol_state = "spike" if vol_ratio > 1.5 else "normal"

        # Régimen → 3 estados
        regime_state = regime.lower() if regime in ("TRENDING", "RANGING", "HIGH_VOLATILITY") else "unknown"

        # Sentimiento → 3 zonas
        if sentiment < 30:
            sent_zone = "fear"
        elif sentiment > 70:
            sent_zone = "greed"
        else:
            sent_zone = "neutral"

        state = f"{rsi_zone}|{macd_dir}|{bb_zone}|{ema_trend}|{vol_state}|{regime_state}|{sent_zone}"
        return state


class QLearningAgent:
    """
    Agente Q-Learning para optimizar decisiones de trading.

    La Q-table mapea (estado, acción) → valor esperado.
    El agente aprende qué acciones son más rentables en cada
    combinación de condiciones de mercado.
    """

    ACTIONS = ["BUY", "SELL", "HOLD"]

    def __init__(self):
        self.q_table: dict[str, dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.ACTIONS}
        )
        self.learning_rate = ML_LEARNING_RATE
        self.discount_factor = ML_DISCOUNT_FACTOR
        self.epsilon = 1.0  # Exploración inicial al 100%
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.training_count = 0
        self.trade_logger = TradeLogger()

        # Intentar cargar modelo existente
        self._load_model()
        logger.info(
            f"Q-Learning Agent inicializado | "
            f"States: {len(self.q_table)} | epsilon: {self.epsilon:.3f}"
        )

    def predict(self, state: str) -> tuple[str, float]:
        """
        Predice la mejor acción para un estado dado.

        Retorna (acción, confianza).
        Confianza = Q-value normalizado [0, 1].
        """
        q_values = self.q_table[state]

        # Epsilon-greedy: con probabilidad epsilon, explorar
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.ACTIONS)
            confidence = 0.5  # Exploración = confianza media
        else:
            action = max(q_values, key=q_values.get)
            # Normalizar confianza
            values = list(q_values.values())
            max_val = max(values)
            min_val = min(values)
            if max_val - min_val > 0:
                confidence = (q_values[action] - min_val) / (max_val - min_val)
            else:
                confidence = 0.5

        return action, confidence

    def update(self, state: str, action: str, reward: float, next_state: str):
        """
        Actualiza la Q-table con la experiencia (s, a, r, s').

        Fórmula Q-Learning:
        Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values())

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state][action] = new_q

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_from_history(self):
        """
        Reentrena el modelo con el histórico completo de trades.

        Se ejecuta cada ML_RETRAIN_INTERVAL trades cerrados.
        Usa los trades almacenados en SQLite para crear experiencias
        de entrenamiento (estado_entrada → acción → recompensa → estado_siguiente).
        """
        trades = self.trade_logger.get_closed_trades()

        if len(trades) < ML_MIN_TRADES_FOR_TRAINING:
            logger.info(
                f"Trades insuficientes para entrenar: {len(trades)}/{ML_MIN_TRADES_FOR_TRAINING}"
            )
            return

        logger.info(f"Reentrenando con {len(trades)} trades históricos...")

        for i, trade in enumerate(trades):
            # Reconstruir estado desde los indicadores guardados
            indicators = json.loads(trade.get("indicators_snapshot", "{}"))
            if not indicators:
                continue

            regime = trade.get("market_regime", "UNKNOWN")
            state = MarketState.discretize(indicators, regime)

            # Acción ejecutada
            action = trade["direction"]

            # Recompensa = PnL porcentual normalizado
            pnl_pct = trade.get("pnl_pct", 0) or 0
            reward = np.clip(pnl_pct * 100, -5, 5)  # Normalizar entre -5 y 5

            # Estado siguiente (aproximación: usar siguiente trade si existe)
            if i + 1 < len(trades):
                next_indicators = json.loads(trades[i + 1].get("indicators_snapshot", "{}"))
                next_regime = trades[i + 1].get("market_regime", "UNKNOWN")
                if next_indicators:
                    next_state = MarketState.discretize(next_indicators, next_regime)
                else:
                    next_state = state
            else:
                next_state = state

            self.update(state, action, reward, next_state)

        self.training_count += 1
        self._save_model()
        logger.info(
            f"Entrenamiento #{self.training_count} completado | "
            f"States: {len(self.q_table)} | epsilon: {self.epsilon:.3f}"
        )

    def should_retrain(self) -> bool:
        """Verifica si es momento de reentrenar."""
        trade_count = self.trade_logger.get_trade_count()
        return (
            trade_count >= ML_MIN_TRADES_FOR_TRAINING
            and trade_count % ML_RETRAIN_INTERVAL == 0
        )

    def _save_model(self):
        """Persiste la Q-table en disco."""
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "training_count": self.training_count,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Modelo guardado en {MODEL_PATH}")

    def _load_model(self):
        """Carga Q-table de disco si existe."""
        path = Path(MODEL_PATH)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                # Reconstruir como defaultdict
                for state, actions in data["q_table"].items():
                    self.q_table[state] = actions
                self.epsilon = data.get("epsilon", self.epsilon)
                self.training_count = data.get("training_count", 0)
                logger.info(
                    f"Modelo cargado | States: {len(self.q_table)} | "
                    f"epsilon: {self.epsilon:.3f}"
                )
            except Exception as e:
                logger.warning(f"Error cargando modelo: {e}. Iniciando desde cero.")

    def get_model_stats(self) -> dict:
        """Estadísticas del modelo para el dashboard."""
        return {
            "states_learned": len(self.q_table),
            "epsilon": round(self.epsilon, 4),
            "training_count": self.training_count,
            "total_trades_seen": self.trade_logger.get_trade_count(),
        }


class SentimentAnalyzer:
    """
    Obtiene señales externas de sentimiento del mercado.

    Fuentes:
    - Fear & Greed Index (API gratuita)
    - Se puede extender con noticias crypto y dominancia BTC
    """

    def __init__(self):
        self.last_sentiment = 50  # Neutral por defecto
        self.last_update = None

    def get_fear_greed_index(self) -> float:
        """
        Obtiene el Fear & Greed Index (0-100).
        0 = Extreme Fear, 100 = Extreme Greed.

        En caso de error, retorna el último valor conocido.
        """
        try:
            import urllib.request
            from config import FEAR_GREED_API
            req = urllib.request.Request(FEAR_GREED_API)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                value = int(data["data"][0]["value"])
                self.last_sentiment = value
                self.last_update = data["data"][0].get("timestamp")
                logger.debug(f"Fear & Greed Index: {value}")
                return value
        except Exception as e:
            logger.warning(f"Error obteniendo Fear & Greed: {e}. Usando último: {self.last_sentiment}")
            return self.last_sentiment
