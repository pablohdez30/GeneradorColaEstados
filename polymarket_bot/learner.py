"""
Learning System: Adapts strategy weights based on past performance.
Uses a simple reinforcement learning approach:
- Strategies that perform well get higher weights
- Strategies that underperform get lower weights
- Weights are bounded to prevent any strategy from being completely eliminated
"""

import logging
from typing import Optional

from .config import BotConfig
from .database import Database
from .strategies.base import TradeSignal

logger = logging.getLogger(__name__)


class LearningSystem:
    """
    Adaptive learning system that adjusts strategy weights
    based on historical performance.
    """

    MIN_WEIGHT = 0.05  # Minimum weight (5%) to keep all strategies active
    MAX_WEIGHT = 0.70  # Maximum weight (70%) to prevent over-reliance

    def __init__(self, config: BotConfig, db: Database):
        self.config = config
        self.db = db
        self.weights = dict(config.strategy_weights)
        self.learning_rate = config.learning_rate
        self.performance_window = config.performance_window

    def combine_signals(self, signals: list[TradeSignal]) -> Optional[TradeSignal]:
        """
        Combine multiple strategy signals into a single weighted decision.
        Returns the best signal if the weighted consensus is strong enough.
        """
        if not signals:
            return None

        # Group signals by (outcome, direction)
        buy_yes_signals = [s for s in signals if s.is_buy and s.target_outcome == "yes"]
        buy_no_signals = [s for s in signals if s.is_buy and s.target_outcome == "no"]

        # Calculate weighted scores
        yes_score = self._weighted_score(buy_yes_signals)
        no_score = self._weighted_score(buy_no_signals)

        best_signals = buy_yes_signals if yes_score >= no_score else buy_no_signals
        best_score = max(yes_score, no_score)

        if not best_signals or best_score < self.config.min_confidence:
            return None

        # Pick the highest confidence signal from the winning group
        best = max(best_signals, key=lambda s: s.confidence)

        # Adjust confidence by the consensus score
        combined_confidence = min(0.95, (best.confidence + best_score) / 2)

        return TradeSignal(
            signal=best.signal,
            confidence=combined_confidence,
            target_outcome=best.target_outcome,
            suggested_size=best.suggested_size,
            reason=f"[Combined {len(best_signals)} signals] {best.reason}",
            strategy_name=best.strategy_name,
        )

    def _weighted_score(self, signals: list[TradeSignal]) -> float:
        """Calculate weighted confidence score for a group of signals."""
        if not signals:
            return 0.0

        total = 0.0
        weight_sum = 0.0
        for signal in signals:
            weight = self.weights.get(signal.strategy_name, 0.33)
            total += signal.confidence * weight
            weight_sum += weight

        # Normalize by total weight so score stays in 0-1 range
        return total / weight_sum if weight_sum > 0 else 0.0

    def update_weights(self):
        """
        Update strategy weights based on recent performance.
        Strategies with better win rates and P&L get higher weights.
        """
        scores = {}

        for strategy_name in self.weights:
            stats = self.db.get_strategy_stats(strategy_name)
            total = stats.get("total", 0)

            if total < 3:
                # Not enough data — keep current weight
                scores[strategy_name] = self.weights[strategy_name]
                continue

            wins = stats.get("wins", 0) or 0
            win_rate = wins / total if total > 0 else 0

            avg_pnl = stats.get("avg_pnl_pct", 0) or 0
            total_pnl = stats.get("total_pnl", 0) or 0

            # Composite score: 60% win rate + 40% normalized P&L
            pnl_score = max(-1, min(1, avg_pnl * 10))  # Normalize
            score = 0.6 * win_rate + 0.4 * (pnl_score + 1) / 2

            scores[strategy_name] = score

        if not scores:
            return

        # Normalize scores to weights
        total_score = sum(scores.values())
        if total_score <= 0:
            return

        new_weights = {}
        for name, score in scores.items():
            raw_weight = score / total_score
            # Apply learning rate (gradual adjustment)
            old_weight = self.weights.get(name, 0.33)
            new_weight = old_weight + self.learning_rate * (raw_weight - old_weight)
            # Clamp to bounds
            new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
            new_weights[name] = new_weight

        # Re-normalize to sum to 1.0
        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            new_weights = {k: v / weight_sum for k, v in new_weights.items()}

        # Log changes
        for name in new_weights:
            old = self.weights.get(name, 0)
            new = new_weights[name]
            if abs(old - new) > 0.01:
                logger.info(f"Weight update: {name} {old:.1%} → {new:.1%}")

        self.weights = new_weights
        self.config.strategy_weights = new_weights

        # Persist
        self.db.save_strategy_weights(self.weights)

    def get_performance_report(self) -> dict:
        """Get a report of all strategy performances."""
        report = {}
        for strategy_name in self.weights:
            stats = self.db.get_strategy_stats(strategy_name)
            total = stats.get("total", 0)
            wins = stats.get("wins", 0) or 0
            report[strategy_name] = {
                "weight": self.weights[strategy_name],
                "total_trades": total,
                "winning_trades": wins,
                "win_rate": wins / total if total > 0 else 0,
                "total_pnl": stats.get("total_pnl", 0) or 0,
                "avg_pnl_pct": stats.get("avg_pnl_pct", 0) or 0,
            }
        return report
