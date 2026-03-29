"""
analyzer.py — Análisis técnico multi-timeframe.

Calcula indicadores en 15m, 1h y 4h simultáneamente.
Diseñado desde cero, sin dependencias del bot anterior.
"""

import numpy as np
import pandas as pd
import ccxt

from sniper_bot.config import (
    EXCHANGE_ID, SYMBOL, TIMEFRAMES,
    EMA_FAST, EMA_SLOW, RSI_PERIOD, ADX_PERIOD, ATR_PERIOD,
)


class MarketAnalyzer:
    """Obtiene datos de mercado y calcula indicadores en múltiples timeframes."""

    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        self.candle_buffers = {}  # {timeframe: DataFrame}

    def fetch_candles(self, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Obtiene velas de Binance para un timeframe."""
        try:
            raw = self.exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            self.candle_buffers[timeframe] = df
            return df
        except Exception as e:
            print(f"Error fetching {timeframe}: {e}")
            return self.candle_buffers.get(timeframe, pd.DataFrame())

    def fetch_all_timeframes(self) -> dict[str, pd.DataFrame]:
        """Obtiene velas para todos los timeframes configurados."""
        result = {}
        for tf in TIMEFRAMES:
            df = self.fetch_candles(tf)
            if not df.empty:
                result[tf] = df
        return result

    def get_current_price(self) -> float:
        """Precio actual."""
        try:
            ticker = self.exchange.fetch_ticker(SYMBOL)
            return ticker["last"]
        except Exception:
            return 0.0

    # ── Indicadores técnicos ──────────────────────────────────

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr_vals = MarketAnalyzer.atr(df, period)
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr_vals.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period).mean() / atr_vals.replace(0, np.nan))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        return dx.ewm(alpha=1 / period).mean()

    @staticmethod
    def volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
        """Volumen actual vs media."""
        vol_ma = df["volume"].rolling(period).mean()
        if vol_ma.iloc[-1] > 0:
            return df["volume"].iloc[-1] / vol_ma.iloc[-1]
        return 1.0

    def analyze(self, df: pd.DataFrame) -> dict:
        """Calcula todos los indicadores para un DataFrame."""
        if df.empty or len(df) < 50:
            return {}

        close = df["close"]
        ema_f = self.ema(close, EMA_FAST)
        ema_s = self.ema(close, EMA_SLOW)
        rsi_val = self.rsi(close)
        adx_val = self.adx(df)
        atr_val = self.atr(df)
        vol_ratio = self.volume_ratio(df)

        # Detectar cruce EMA reciente (últimas 3 velas)
        ema_cross_up = False
        ema_cross_down = False
        for i in range(-3, 0):
            if i - 1 >= -len(ema_f):
                prev_diff = ema_f.iloc[i - 1] - ema_s.iloc[i - 1]
                curr_diff = ema_f.iloc[i] - ema_s.iloc[i]
                if prev_diff <= 0 and curr_diff > 0:
                    ema_cross_up = True
                elif prev_diff >= 0 and curr_diff < 0:
                    ema_cross_down = True

        return {
            "price": round(close.iloc[-1], 2),
            "ema_fast": round(ema_f.iloc[-1], 2),
            "ema_slow": round(ema_s.iloc[-1], 2),
            "ema_above": ema_f.iloc[-1] > ema_s.iloc[-1],
            "ema_cross_up": ema_cross_up,
            "ema_cross_down": ema_cross_down,
            "rsi": round(rsi_val.iloc[-1], 1),
            "adx": round(adx_val.iloc[-1], 1),
            "atr": round(atr_val.iloc[-1], 2),
            "volume_ratio": round(vol_ratio, 2),
        }

    def analyze_all(self) -> dict[str, dict]:
        """Analiza todos los timeframes y retorna indicadores de cada uno."""
        candles = self.fetch_all_timeframes()
        results = {}
        for tf, df in candles.items():
            indicators = self.analyze(df)
            if indicators:
                results[tf] = indicators
        return results
