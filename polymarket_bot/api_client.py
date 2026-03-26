"""
Polymarket API client for fetching market data.
Uses the Gamma API (public, no auth required) for market information
and the CLOB API for order book data.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class Market:
    """Represents a Polymarket prediction market."""
    condition_id: str
    question: str
    description: str
    outcome_yes_price: float
    outcome_no_price: float
    volume_24h: float
    liquidity: float
    end_date: str
    category: str
    active: bool
    slug: str
    token_ids: list

    @property
    def implied_probability_yes(self) -> float:
        return self.outcome_yes_price

    @property
    def implied_probability_no(self) -> float:
        return self.outcome_no_price

    @property
    def spread(self) -> float:
        total = self.outcome_yes_price + self.outcome_no_price
        return abs(total - 1.0)


class PolymarketAPIClient:
    """Client for reading public Polymarket data."""

    def __init__(self, gamma_url: str, clob_url: str):
        self.gamma_url = gamma_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketBot/1.0",
        })
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # Rate limit: 2 req/sec

    def _rate_limit(self):
        """Simple rate limiter."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """Make a rate-limited GET request."""
        self._rate_limit()
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            return {}

    def get_active_markets(self, limit: int = 100, offset: int = 0) -> list[Market]:
        """Fetch active markets from Gamma API."""
        params = {
            "limit": limit,
            "offset": offset,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        }
        data = self._get(f"{self.gamma_url}/markets", params)

        if not data:
            return []

        markets = []
        for item in data if isinstance(data, list) else data.get("data", []):
            try:
                market = self._parse_market(item)
                if market:
                    markets.append(market)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipping market: {e}")
                continue

        logger.info(f"Fetched {len(markets)} active markets")
        return markets

    def get_market_by_id(self, condition_id: str) -> Optional[Market]:
        """Fetch a specific market by condition ID."""
        data = self._get(f"{self.gamma_url}/markets/{condition_id}")
        if data:
            return self._parse_market(data)
        return None

    def get_market_by_slug(self, slug: str) -> Optional[Market]:
        """Fetch a specific market by slug."""
        params = {"slug": slug}
        data = self._get(f"{self.gamma_url}/markets", params)
        if data:
            items = data if isinstance(data, list) else data.get("data", [])
            if items:
                return self._parse_market(items[0])
        return None

    def get_price_history(self, token_id: str, interval: str = "1d",
                          fidelity: int = 60) -> list[dict]:
        """
        Fetch price history for a token.
        interval: 1d, 1w, 1m, all
        fidelity: resolution in minutes
        """
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        data = self._get(f"{self.clob_url}/prices-history", params)
        if data and "history" in data:
            return data["history"]
        return data if isinstance(data, list) else []

    def get_order_book(self, token_id: str) -> dict:
        """Fetch order book for a token."""
        params = {"token_id": token_id}
        data = self._get(f"{self.clob_url}/book", params)
        return data if data else {"bids": [], "asks": []}

    def get_events(self, limit: int = 50) -> list[dict]:
        """Fetch events (groups of related markets)."""
        params = {
            "limit": limit,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        }
        data = self._get(f"{self.gamma_url}/events", params)
        if isinstance(data, list):
            return data
        return data.get("data", []) if data else []

    def _parse_market(self, item: dict) -> Optional[Market]:
        """Parse a market item from API response into a Market object."""
        tokens = item.get("tokens", [])
        clobTokenIds = item.get("clobTokenIds", "")

        yes_price = 0.5
        no_price = 0.5
        token_ids = []

        if tokens:
            for token in tokens:
                outcome = token.get("outcome", "").lower()
                price = float(token.get("price", 0.5))
                if outcome == "yes":
                    yes_price = price
                elif outcome == "no":
                    no_price = price
                if token.get("token_id"):
                    token_ids.append(token["token_id"])
        elif "outcomePrices" in item:
            try:
                import json
                prices = json.loads(item["outcomePrices"]) if isinstance(
                    item["outcomePrices"], str) else item["outcomePrices"]
                if len(prices) >= 2:
                    yes_price = float(prices[0])
                    no_price = float(prices[1])
            except (json.JSONDecodeError, IndexError):
                pass

        if clobTokenIds:
            try:
                import json
                parsed = json.loads(clobTokenIds) if isinstance(
                    clobTokenIds, str) else clobTokenIds
                if isinstance(parsed, list):
                    token_ids = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        condition_id = (item.get("conditionId") or item.get("condition_id")
                        or item.get("id", ""))

        volume = float(item.get("volume24hr", 0) or item.get("volume", 0) or 0)
        liquidity = float(item.get("liquidity", 0) or 0)

        return Market(
            condition_id=str(condition_id),
            question=item.get("question", ""),
            description=item.get("description", "")[:500],
            outcome_yes_price=yes_price,
            outcome_no_price=no_price,
            volume_24h=volume,
            liquidity=liquidity,
            end_date=item.get("endDate", item.get("end_date_iso", "")),
            category=item.get("category", item.get("groupItemTitle", "")),
            active=bool(item.get("active", True)),
            slug=item.get("slug", ""),
            token_ids=token_ids,
        )
