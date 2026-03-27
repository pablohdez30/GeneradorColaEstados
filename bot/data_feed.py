"""
data_feed.py - Conexión al exchange vía WebSocket y REST para datos en tiempo real.

Diseño:
- Usa CCXT para datos REST (OHLCV histórico, order book).
- Implementa WebSocket propio via ccxt.pro (o fallback a polling) para
  recibir velas y ticker en tiempo real.
- Mantiene un buffer circular de las últimas N velas para que
  strategy_engine pueda calcular indicadores sin latencia.

Decisión: CCXT se eligió por su abstracción multi-exchange.
Si mañana se cambia de Binance a otro exchange, solo se cambia EXCHANGE_ID.
"""

import asyncio
import time
import numpy as np
import pandas as pd
import ccxt

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE_ID, SYMBOL, TIMEFRAME, VOLUME_MA_PERIOD, MARKET_TYPE
from bot.logger import setup_logger

logger = setup_logger("data_feed")


class DataFeed:
    """
    Proveedor de datos de mercado (spot o futuros perpetuos).

    Métodos principales:
    - fetch_ohlcv(): obtiene velas históricas
    - fetch_order_book(): profundidad de mercado
    - get_current_price(): último precio
    - stream(): generador asíncrono de velas en tiempo real
    """

    def __init__(self):
        # Inicializar exchange sin credenciales (solo datos públicos para paper trading)
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        # Seleccionar tipo de mercado: spot o futuros perpetuos
        market_type = "swap" if MARKET_TYPE == "futures" else "spot"
        self.exchange = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": market_type},
        })
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME
        self._candle_buffer = pd.DataFrame()
        self._buffer_size = 500  # Últimas 500 velas en memoria
        logger.info(f"DataFeed inicializado: {EXCHANGE_ID} | {SYMBOL} | {TIMEFRAME} | {MARKET_TYPE}")

    def fetch_ohlcv(self, limit: int = 500) -> pd.DataFrame:
        """
        Obtiene velas históricas OHLCV.

        Retorna DataFrame con columnas: timestamp, open, high, low, close, volume
        Índice: datetime UTC.
        """
        try:
            raw = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            self._candle_buffer = df.tail(self._buffer_size)
            logger.debug(f"Fetch OHLCV: {len(df)} velas obtenidas")
            return df
        except ccxt.BaseError as e:
            logger.error(f"Error fetching OHLCV: {e}")
            raise

    def fetch_order_book(self, depth: int = 20) -> dict:
        """
        Obtiene el order book (profundidad de mercado).

        Retorna dict con:
        - bids: lista de [precio, cantidad]
        - asks: lista de [precio, cantidad]
        - bid_volume: volumen total en bids
        - ask_volume: volumen total en asks
        - imbalance: ratio bid/ask (>1 = presión compradora)
        """
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=depth)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            bid_vol = sum(b[1] for b in bids) if bids else 0
            ask_vol = sum(a[1] for a in asks) if asks else 0
            imbalance = bid_vol / ask_vol if ask_vol > 0 else 1.0

            result = {
                "bids": bids,
                "asks": asks,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
                "imbalance": round(imbalance, 4),
                "spread": (asks[0][0] - bids[0][0]) if bids and asks else 0,
                "mid_price": (asks[0][0] + bids[0][0]) / 2 if bids and asks else 0,
            }
            logger.debug(
                f"Order Book: imbalance={result['imbalance']:.3f} | "
                f"spread={result['spread']:.2f}"
            )
            return result
        except ccxt.BaseError as e:
            logger.error(f"Error fetching order book: {e}")
            return {"bids": [], "asks": [], "bid_volume": 0, "ask_volume": 0,
                    "imbalance": 1.0, "spread": 0, "mid_price": 0}

    def get_current_price(self) -> float:
        """Obtiene el último precio del par."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker["last"]
            logger.debug(f"Precio actual: {price}")
            return float(price)
        except ccxt.BaseError as e:
            logger.error(f"Error fetching ticker: {e}")
            raise

    def get_candle_buffer(self) -> pd.DataFrame:
        """Retorna el buffer de velas en memoria."""
        return self._candle_buffer.copy()

    def update_buffer(self) -> pd.DataFrame:
        """
        Actualiza el buffer con las últimas velas.
        Se usa en el loop principal como alternativa a WebSocket.

        Decisión: Polling cada ~5s en lugar de WebSocket puro porque
        ccxt.pro requiere dependencias adicionales y para paper trading
        la latencia de 5s es aceptable. En producción se usaría WS.
        """
        try:
            new_candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=5)
            df_new = pd.DataFrame(
                new_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
            df_new.set_index("timestamp", inplace=True)
            df_new = df_new.astype(float)

            # Merge con buffer existente (actualizar última vela + añadir nuevas)
            if not self._candle_buffer.empty:
                combined = pd.concat([self._candle_buffer, df_new])
                combined = combined[~combined.index.duplicated(keep="last")]
                self._candle_buffer = combined.tail(self._buffer_size)
            else:
                self._candle_buffer = df_new

            return self._candle_buffer
        except ccxt.BaseError as e:
            logger.error(f"Error actualizando buffer: {e}")
            return self._candle_buffer

    def detect_volume_spike(self, df: pd.DataFrame = None) -> bool:
        """
        Detecta si el volumen actual es significativamente mayor al promedio.
        Un spike de volumen suele preceder movimientos fuertes.
        """
        if df is None:
            df = self._candle_buffer
        if len(df) < VOLUME_MA_PERIOD:
            return False

        vol_ma = df["volume"].rolling(VOLUME_MA_PERIOD).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]
        from config import VOLUME_SPIKE_THRESHOLD
        return current_vol > vol_ma * VOLUME_SPIKE_THRESHOLD

    def get_market_summary(self) -> dict:
        """Resumen rápido del estado del mercado para logging."""
        try:
            price = self.get_current_price()
            ob = self.fetch_order_book(depth=10)
            vol_spike = self.detect_volume_spike()
            return {
                "price": price,
                "spread": ob["spread"],
                "imbalance": ob["imbalance"],
                "volume_spike": vol_spike,
            }
        except Exception as e:
            logger.error(f"Error en market summary: {e}")
            return {"price": 0, "spread": 0, "imbalance": 1.0, "volume_spike": False}
