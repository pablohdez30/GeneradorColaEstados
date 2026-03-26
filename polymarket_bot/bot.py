"""
Main Bot Orchestrator: Coordinates all components to run the autonomous trading bot.
"""

import logging
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from .api_client import Market, PolymarketAPIClient
from .config import BotConfig
from .dashboard import display_dashboard, display_startup_banner
from .database import Database
from .learner import LearningSystem
from .paper_trader import PaperTradingEngine
from .strategies import MeanReversionStrategy, MomentumStrategy, ValueStrategy

logger = logging.getLogger(__name__)


class PolymarketBot:
    """
    Autonomous Polymarket trading bot.
    Scans markets, generates signals, executes paper trades,
    and learns from results.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self.running = False
        self._setup_logging()

        # Initialize components
        self.db = Database(self.config.db_path)
        self.api = PolymarketAPIClient(
            self.config.gamma_api_url, self.config.clob_api_url)
        self.engine = PaperTradingEngine(self.config, self.db)
        self.learner = LearningSystem(self.config, self.db)

        # Initialize strategies
        self.strategies = [
            MomentumStrategy(),
            ValueStrategy(),
            MeanReversionStrategy(),
        ]

        # Market price cache (for position checking)
        self._price_cache: dict[str, dict[str, float]] = {}
        self._market_active_cache: dict[str, bool] = {}
        self._last_scan: Optional[str] = None
        self._scan_count = 0
        self._warmup_data: dict[str, int] = {}  # market_id -> data points seen

    def _setup_logging(self):
        """Configure logging."""
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.FileHandler("polymarket_bot.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def run(self):
        """Main bot loop."""
        self.running = True

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        display_startup_banner(self.config)
        logger.info("Bot starting...")

        print("Fetching initial market data...")
        time.sleep(1)

        while self.running:
            try:
                self._run_cycle()
                self._last_scan = datetime.now().strftime("%H:%M:%S")

                # Display dashboard
                display_dashboard(self.engine, self.learner,
                                  last_scan=self._last_scan)

                # Wait for next cycle
                if self.running:
                    logger.info(
                        f"Next scan in {self.config.scan_interval_seconds}s...")
                    self._interruptible_sleep(self.config.scan_interval_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in bot cycle: {e}", exc_info=True)
                self._interruptible_sleep(30)  # Wait before retry

        self._cleanup()

    def _run_cycle(self):
        """Execute one full trading cycle."""
        self._scan_count += 1
        logger.info(f"--- Scan #{self._scan_count} ---")

        # Step 1: Fetch markets
        markets = self.api.get_active_markets(limit=50)
        if not markets:
            logger.warning("No markets fetched, skipping cycle")
            return

        # Filter markets by liquidity and volume
        tradeable = [
            m for m in markets
            if m.liquidity >= self.config.min_liquidity
            and m.volume_24h >= self.config.min_volume
            and m.active
            and 0.05 < m.outcome_yes_price < 0.95
        ]

        logger.info(f"Found {len(tradeable)} tradeable markets out of {len(markets)}")

        # Step 2: Check existing positions (with resolution detection)
        self._update_price_cache(markets)
        self._update_active_cache(markets)
        closed = self.engine.check_positions(
            self._get_cached_price, self._get_market_active)

        if closed:
            logger.info(f"Closed {len(closed)} positions")
            # Update learning after closing positions
            for result in closed:
                strategy_name = result["strategy"]
                for strategy in self.strategies:
                    if strategy.name == strategy_name:
                        strategy.record_result(result["pnl"] > 0)
                        break

        # Step 3: Analyze markets and generate signals
        for market in tradeable:
            if not self.running:
                break

            try:
                self._analyze_and_trade(market)
            except Exception as e:
                logger.debug(f"Error analyzing {market.condition_id}: {e}")
                continue

        # Step 4: Update learning weights periodically
        if self._scan_count % 5 == 0:
            self.learner.update_weights()

        # Step 5: Record market snapshots
        for market in tradeable[:20]:
            self.db.record_market_snapshot(
                market.condition_id, market.question,
                market.outcome_yes_price, market.outcome_no_price,
                market.volume_24h, market.liquidity,
            )

    def _analyze_and_trade(self, market: Market):
        """Analyze a single market and potentially trade."""
        # Get price history
        token_id = market.token_ids[0] if market.token_ids else None
        price_history = []
        order_book = {"bids": [], "asks": []}

        if token_id:
            price_history = self.api.get_price_history(token_id, interval="1w")
            order_book = self.api.get_order_book(token_id)

        # Warm-up check: skip markets without enough historical data
        data_points = len(price_history)
        self._warmup_data[market.condition_id] = data_points
        if data_points < self.config.warmup_min_data_points:
            logger.debug(
                f"Warm-up: {market.condition_id} has {data_points} points "
                f"(need {self.config.warmup_min_data_points}), skipping")
            return

        # Get signals from all strategies
        signals = []
        for strategy in self.strategies:
            try:
                signal = strategy.analyze(market, price_history, order_book)
                if signal and signal.is_actionable:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Strategy {strategy.name} error: {e}")
                continue

        if not signals:
            return

        # Combine signals through learning system
        combined = self.learner.combine_signals(signals)
        if not combined:
            return

        logger.info(
            f"Signal: {combined.signal.value} {combined.target_outcome} "
            f"on '{market.question[:50]}' "
            f"(confidence={combined.confidence:.1%})")

        # Execute trade with order book for slippage simulation
        if combined.is_buy:
            self.engine.execute_buy(combined, market, order_book=order_book)

    def _update_price_cache(self, markets: list[Market]):
        """Update the price cache with current market data."""
        for market in markets:
            self._price_cache[market.condition_id] = {
                "yes": market.outcome_yes_price,
                "no": market.outcome_no_price,
            }

    def _update_active_cache(self, markets: list[Market]):
        """Update market active status cache for resolution detection."""
        fetched_ids = set()
        for market in markets:
            self._market_active_cache[market.condition_id] = market.active
            fetched_ids.add(market.condition_id)

        # Markets we have positions in but weren't in the active fetch
        # are likely resolved/closed — mark them for re-check
        for pos in self.engine.get_open_positions():
            if pos.market_id not in fetched_ids:
                # Fetch individually to check if resolved
                fresh = self.api.get_market_by_id(pos.market_id)
                if fresh:
                    self._market_active_cache[pos.market_id] = fresh.active
                    self._price_cache[pos.market_id] = {
                        "yes": fresh.outcome_yes_price,
                        "no": fresh.outcome_no_price,
                    }
                else:
                    # Can't fetch — assume resolved
                    self._market_active_cache[pos.market_id] = False

    def _get_market_active(self, market_id: str) -> Optional[bool]:
        """Check if a market is still active (not resolved)."""
        return self._market_active_cache.get(market_id)

    def _get_cached_price(self, market_id: str, outcome: str) -> Optional[float]:
        """Get a cached price for position checking."""
        cached = self._price_cache.get(market_id)
        if cached:
            return cached.get(outcome)

        # Try to fetch fresh data
        market = self.api.get_market_by_id(market_id)
        if market:
            self._price_cache[market.condition_id] = {
                "yes": market.outcome_yes_price,
                "no": market.outcome_no_price,
            }
            return market.outcome_yes_price if outcome == "yes" else market.outcome_no_price

        return None

    def _interruptible_sleep(self, seconds: int):
        """Sleep that can be interrupted by shutdown signal."""
        for _ in range(seconds):
            if not self.running:
                break
            time.sleep(1)

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        print("\n")
        logger.info("Shutting down bot...")
        self.running = False

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Saving final state...")
        self.learner.update_weights()
        self.db.close()
        logger.info("Bot stopped.")

    def run_single_scan(self):
        """Run a single scan cycle (useful for testing)."""
        display_startup_banner(self.config)
        self._run_cycle()
        display_dashboard(self.engine, self.learner, last_scan="just now")
        self.db.close()

    def show_status(self):
        """Display current status without trading."""
        display_dashboard(self.engine, self.learner, last_scan="N/A (status only)")
        self.db.close()
