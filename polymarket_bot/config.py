"""
Configuration for the Polymarket Trading Bot.
"""

import os
from dataclasses import dataclass, field


@dataclass
class BotConfig:
    """Main configuration for the trading bot."""

    # API endpoints
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"

    # Paper trading settings
    initial_balance: float = 10_000.0  # Starting paper money in USDC
    max_position_size: float = 500.0   # Max per single position
    max_open_positions: int = 20       # Max concurrent positions
    min_trade_size: float = 5.0        # Minimum trade amount

    # Risk management
    stop_loss_pct: float = 0.25        # Close position if down 25%
    take_profit_pct: float = 0.50      # Close position if up 50%
    max_portfolio_risk: float = 0.30   # Max 30% of balance at risk

    # Strategy weights (used by the learning system)
    strategy_weights: dict = field(default_factory=lambda: {
        "momentum": 0.33,
        "value": 0.34,
        "mean_reversion": 0.33,
    })

    # Bot behavior
    scan_interval_seconds: int = 300   # Scan markets every 5 minutes
    min_confidence: float = 0.6        # Minimum confidence to trade (0-1)
    min_liquidity: float = 1000.0      # Minimum market liquidity in USDC
    min_volume: float = 500.0          # Minimum 24h volume

    # Learning system
    learning_rate: float = 0.05        # How fast strategy weights adapt
    performance_window: int = 50       # Number of past trades to evaluate

    # Database
    db_path: str = "polymarket_bot.db"

    # Logging
    log_level: str = "INFO"
