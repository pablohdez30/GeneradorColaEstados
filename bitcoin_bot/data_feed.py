"""
data_feed.py — Feed de datos en tiempo real via CCXT + WebSocket
=================================================================
Responsabilidades:
  1. Mantener una conexión WebSocket permanente a Binance (o Testnet)
  2. Acumular velas OHLCV en un buffer circular en memoria
  3. Proveer snapshots del order book (bid/ask, profundidad, imbalance)
  4. Obtener señales externas: Fear & Greed Index, Dominancia BTC
  5. Persistir velas en SQLite periódicamente

Decisión de diseño: CCXT Pro para WebSocket unificado que abstrae
diferencias entre exchanges. Si Pro no está disponible, fallback a
polling REST cada 1 segundo (suficiente para scalping en 1m).
"""

import asyncio
import time
import json
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, List, Any
import threading

import ccxt
import pandas as pd
import numpy as np
import aiohttp

from config import CONFIG
from logger import LOG, EventType
from database import DB

# Intentar importar ccxt.pro (WebSocket nativo)
try:
    import ccxt.pro as ccxtpro
    CCXT_PRO_AVAILABLE = True
except ImportError:
    CCXT_PRO_AVAILABLE = False
    LOG.warning(EventType.SYSTEM,
                "ccxt.pro no disponible — usando polling REST cada 1s")


# ─────────────────────────────────────────────
#  ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────
class Candle:
    """Vela OHLCV inmutable."""
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts: int, o: float, h: float, l: float, c: float, v: float):
        self.ts = ts          # timestamp Unix ms
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v

    def to_dict(self) -> Dict:
        return {
            "ts": datetime.fromtimestamp(self.ts / 1000, tz=timezone.utc).isoformat(),
            "open": self.open, "high": self.high,
            "low": self.low, "close": self.close,
            "volume": self.volume,
        }


class OrderBookSnapshot:
    """Snapshot del libro de órdenes con métricas derivadas."""

    def __init__(self, bids: List, asks: List):
        self.bids = bids[:20]   # top 20 niveles
        self.asks = asks[:20]
        self.timestamp = time.time()

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_pct(self) -> float:
        """Spread bid/ask como porcentaje del mid."""
        if self.mid_price == 0:
            return 0.0
        return (self.best_ask - self.best_bid) / self.mid_price

    @property
    def bid_volume(self) -> float:
        """Volumen total en el lado comprador (top 10 niveles)."""
        return sum(b[1] for b in self.bids[:10])

    @property
    def ask_volume(self) -> float:
        """Volumen total en el lado vendedor (top 10 niveles)."""
        return sum(a[1] for a in self.asks[:10])

    @property
    def imbalance(self) -> float:
        """
        Imbalance del order book: +1 = presión compradora, -1 = presión vendedora.
        Fórmula: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Señal útil: imbalance > 0.2 sugiere subida inminente.
        """
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.0
        return (self.bid_volume - self.ask_volume) / total

    @property
    def large_bid_wall(self) -> Optional[float]:
        """Detecta bid wall: nivel con volumen > 3× la media."""
        if not self.bids:
            return None
        avg_vol = sum(b[1] for b in self.bids) / len(self.bids)
        for price, vol in self.bids:
            if vol > avg_vol * 3:
                return price
        return None

    @property
    def large_ask_wall(self) -> Optional[float]:
        """Detecta ask wall (resistencia de oferta masiva)."""
        if not self.asks:
            return None
        avg_vol = sum(a[1] for a in self.asks) / len(self.asks)
        for price, vol in self.asks:
            if vol > avg_vol * 3:
                return price
        return None


# ─────────────────────────────────────────────
#  CLASE PRINCIPAL DATA FEED
# ─────────────────────────────────────────────
class DataFeed:
    """
    Feed de datos unificado para el bot de scalping.

    Provee una interfaz sincrónica simple al resto del sistema aunque
    internamente use asyncio para las conexiones WebSocket.
    """

    def __init__(self):
        cfg = CONFIG.exchange
        self.symbol = cfg.symbol
        self.timeframe = cfg.timeframe

        # Buffer circular de velas (máx 500 velas en memoria)
        self._candles: deque = deque(maxlen=500)
        self._candles_lock = threading.Lock()

        # Último order book
        self._orderbook: Optional[OrderBookSnapshot] = None
        self._ob_lock = threading.Lock()

        # Señales externas
        self._fear_greed: int = 50          # neutro por defecto
        self._fear_greed_label: str = "Neutral"
        self._btc_dominance: float = 0.0
        self._last_external_update: float = 0.0

        # Callbacks registrados (strategy_engine los usa)
        self._on_candle_close: Optional[Callable] = None

        # Estado de conexión
        self._running = False
        self._last_candle_ts: int = 0

        # Exchange REST (fallback y carga inicial)
        self._exchange = self._create_exchange()

        LOG.log(EventType.SYSTEM, f"DataFeed inicializado — {self.symbol} [{self.timeframe}]")

    def _create_exchange(self) -> ccxt.Exchange:
        """Crea instancia REST de CCXT configurada para Binance."""
        cfg = CONFIG.exchange
        options: Dict[str, Any] = {"defaultType": "spot"}

        if cfg.use_testnet:
            # Binance Testnet (paper trading con API real pero dinero ficticio)
            exchange = ccxt.binance({
                "apiKey": cfg.testnet_api_key or "dummy_key",
                "secret": cfg.testnet_api_secret or "dummy_secret",
                "options": options,
                "urls": {
                    "api": {
                        "public": "https://testnet.binance.vision/api",
                        "private": "https://testnet.binance.vision/api",
                    }
                },
            })
        else:
            exchange = ccxt.binance({
                "apiKey": cfg.api_key,
                "secret": cfg.api_secret,
                "options": options,
            })

        exchange.load_markets()
        return exchange

    # ── CARGA INICIAL ───────────────────────────

    def load_historical_candles(self, limit: int = 200):
        """
        Carga las últimas N velas históricas via REST para tener
        suficientes datos para calcular indicadores desde el inicio.
        """
        LOG.log(EventType.SYSTEM,
                f"Cargando {limit} velas históricas {self.symbol}...")
        try:
            ohlcv = self._exchange.fetch_ohlcv(
                self.symbol, self.timeframe, limit=limit
            )
            with self._candles_lock:
                for bar in ohlcv:
                    ts, o, h, l, c, v = bar
                    self._candles.append(Candle(ts, o, h, l, c, v))
            LOG.log(EventType.SYSTEM,
                    f"{len(self._candles)} velas cargadas. "
                    f"Última: ${self._candles[-1].close:,.2f}")

            # Persistir en SQLite
            DB.save_candles([c.to_dict() for c in self._candles], self.timeframe)
        except Exception as e:
            LOG.error(EventType.SYSTEM, f"Error cargando histórico: {e}")
            raise

    # ── POLLING REST (FALLBACK) ─────────────────

    def _poll_candles_rest(self):
        """
        Loop de polling: cada segundo consulta la vela actual via REST.
        Cuando el timestamp cambia, la vela anterior se considera cerrada
        y se notifica al strategy_engine.
        """
        while self._running:
            try:
                ohlcv = self._exchange.fetch_ohlcv(
                    self.symbol, self.timeframe, limit=2
                )
                if not ohlcv:
                    time.sleep(1)
                    continue

                current_bar = ohlcv[-1]
                ts, o, h, l, c, v = current_bar

                # Nueva vela cerrada detectada
                if self._last_candle_ts and ts != self._last_candle_ts:
                    closed_bar = ohlcv[-2]
                    c_ts, c_o, c_h, c_l, c_c, c_v = closed_bar
                    candle = Candle(c_ts, c_o, c_h, c_l, c_c, c_v)
                    with self._candles_lock:
                        self._candles.append(candle)
                    DB.save_candles([candle.to_dict()], self.timeframe)
                    if self._on_candle_close:
                        self._on_candle_close(candle)

                self._last_candle_ts = ts

                # Obtener order book cada ciclo
                self._fetch_orderbook_rest()

                time.sleep(1)
            except Exception as e:
                LOG.error(EventType.SYSTEM, f"Error en polling REST: {e}")
                time.sleep(5)

    def _fetch_orderbook_rest(self):
        """Obtiene snapshot del order book via REST."""
        try:
            ob = self._exchange.fetch_order_book(
                self.symbol, limit=CONFIG.exchange.orderbook_depth
            )
            with self._ob_lock:
                self._orderbook = OrderBookSnapshot(ob["bids"], ob["asks"])
        except Exception as e:
            LOG.debug(EventType.SYSTEM, f"Error order book: {e}")

    # ── WEBSOCKET (CCXT PRO) ────────────────────

    async def _ws_candles(self, exchange_ws):
        """Recibe velas en tiempo real via WebSocket (CCXT Pro)."""
        while self._running:
            try:
                ohlcv = await exchange_ws.watch_ohlcv(
                    self.symbol, self.timeframe
                )
                if not ohlcv:
                    continue

                for bar in ohlcv:
                    ts, o, h, l, c, v = bar
                    if self._last_candle_ts and ts != self._last_candle_ts:
                        # La vela anterior se cerró
                        candle = Candle(self._last_candle_ts,
                                        *self._last_bar[1:])
                        with self._candles_lock:
                            self._candles.append(candle)
                        DB.save_candles([candle.to_dict()], self.timeframe)
                        if self._on_candle_close:
                            self._on_candle_close(candle)
                    self._last_candle_ts = ts
                    self._last_bar = bar
            except Exception as e:
                LOG.error(EventType.SYSTEM, f"WS candles error: {e}")
                await asyncio.sleep(5)

    async def _ws_orderbook(self, exchange_ws):
        """Recibe order book en tiempo real via WebSocket."""
        while self._running:
            try:
                ob = await exchange_ws.watch_order_book(
                    self.symbol, limit=CONFIG.exchange.orderbook_depth
                )
                with self._ob_lock:
                    self._orderbook = OrderBookSnapshot(
                        ob["bids"], ob["asks"]
                    )
            except Exception as e:
                LOG.debug(EventType.SYSTEM, f"WS orderbook error: {e}")
                await asyncio.sleep(2)

    # ── SEÑALES EXTERNAS ────────────────────────

    async def _update_external_signals(self):
        """
        Obtiene Fear & Greed Index y Dominancia BTC cada hora.
        APIs gratuitas, sin autenticación.
        """
        while self._running:
            now = time.time()
            interval = CONFIG.external.fear_greed_update_interval
            if now - self._last_external_update < interval:
                await asyncio.sleep(60)
                continue

            async with aiohttp.ClientSession() as session:
                # Fear & Greed Index
                try:
                    async with session.get(
                        CONFIG.external.fear_greed_url, timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        data = await resp.json(content_type=None)
                        fg = data["data"][0]
                        self._fear_greed = int(fg["value"])
                        self._fear_greed_label = fg["value_classification"]
                        LOG.external_signal(
                            "Fear&Greed",
                            self._fear_greed,
                            self._fear_greed_label,
                        )
                except Exception as e:
                    LOG.warning(EventType.EXTERNAL_SIGNAL, f"F&G error: {e}")

                # Dominancia BTC (CoinGecko)
                try:
                    async with session.get(
                        CONFIG.external.coingecko_dominance_url,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        data = await resp.json()
                        dom = data["data"]["market_cap_percentage"].get("btc", 0)
                        self._btc_dominance = dom
                        LOG.external_signal("BTC_Dominance", f"{dom:.1f}%",
                                            "alta" if dom > 55 else "media")
                        # Persistir
                        DB.save_external_signal({
                            "fear_greed": self._fear_greed,
                            "fear_greed_label": self._fear_greed_label,
                            "btc_dominance": self._btc_dominance,
                            "market_cap_usdt": data["data"].get(
                                "total_market_cap", {}).get("usd", 0),
                        })
                except Exception as e:
                    LOG.warning(EventType.EXTERNAL_SIGNAL, f"CoinGecko error: {e}")

            self._last_external_update = time.time()
            await asyncio.sleep(60)

    # ── INICIO / PARADA ─────────────────────────

    def start(self, on_candle_close: Optional[Callable] = None):
        """
        Arranca el feed en un hilo/proceso separado.
        on_candle_close: callback(Candle) llamado por cada vela cerrada.
        """
        self._on_candle_close = on_candle_close
        self._running = True

        # Carga histórica sincrónica antes de arrancar
        self.load_historical_candles(limit=CONFIG.indicators.min_candles + 50)

        if CCXT_PRO_AVAILABLE:
            thread = threading.Thread(
                target=self._run_async_loop, daemon=True, name="DataFeed-WS"
            )
        else:
            thread = threading.Thread(
                target=self._poll_candles_rest, daemon=True, name="DataFeed-Poll"
            )
        thread.start()

        # Hilo separado para señales externas
        ext_thread = threading.Thread(
            target=self._run_external_loop, daemon=True, name="ExternalSignals"
        )
        ext_thread.start()

        LOG.log(EventType.SYSTEM, "DataFeed arrancado")

    def _run_async_loop(self):
        """Ejecuta el event loop de asyncio en el hilo del DataFeed."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_main_loop())

    def _run_external_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._update_external_signals())

    async def _ws_main_loop(self):
        """Lanza tareas WS en paralelo (velas + order book)."""
        exchange_ws = getattr(ccxtpro, "binance")({
            "options": {"defaultType": "spot"}
        })
        try:
            await asyncio.gather(
                self._ws_candles(exchange_ws),
                self._ws_orderbook(exchange_ws),
            )
        finally:
            await exchange_ws.close()

    def stop(self):
        self._running = False
        LOG.log(EventType.SYSTEM, "DataFeed detenido")

    # ── ACCESO A DATOS ──────────────────────────

    def get_candles_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Devuelve las velas como DataFrame de pandas.
        Columnas: ts, open, high, low, close, volume
        """
        with self._candles_lock:
            candles = list(self._candles)
        if limit:
            candles = candles[-limit:]
        if not candles:
            return pd.DataFrame()

        data = {
            "ts": [c.ts for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        return df

    @property
    def current_price(self) -> float:
        """Precio actual (close de la última vela)."""
        with self._candles_lock:
            if self._candles:
                return self._candles[-1].close
        return 0.0

    @property
    def orderbook(self) -> Optional[OrderBookSnapshot]:
        with self._ob_lock:
            return self._orderbook

    @property
    def fear_greed_index(self) -> int:
        return self._fear_greed

    @property
    def fear_greed_label(self) -> str:
        return self._fear_greed_label

    @property
    def btc_dominance(self) -> float:
        return self._btc_dominance

    @property
    def is_ready(self) -> bool:
        """True cuando hay suficientes velas para calcular indicadores."""
        with self._candles_lock:
            return len(self._candles) >= CONFIG.indicators.min_candles
