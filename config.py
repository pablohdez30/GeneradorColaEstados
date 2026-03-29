"""
Configuración central del bot de trend following BTC/USDT.

Estrategia: Seguir tendencias en timeframe 15m.
Solo operar cuando hay tendencia clara. Sin apalancamiento.
"""

# ─── Exchange ────────────────────────────────────────────────────
EXCHANGE_ID = "binance"
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"  # 15 minutos: menos ruido que 1m
MARKET_TYPE = "futures"  # Futuros para poder hacer shorts
LEVERAGE = 1  # SIN apalancamiento hasta que sea rentable

# ─── Paper Trading ───────────────────────────────────────────────
INITIAL_BALANCE = 10_000.0
PAPER_MODE = True
FEE_RATE = 0.0004  # 0.04% comisión futuros Binance

# ─── Gestión de Riesgo ──────────────────────────────────────────
MAX_RISK_PER_TRADE = 0.01  # 1% del capital por operación
MAX_OPEN_POSITIONS = 1  # Solo 1 posición a la vez
MAX_DRAWDOWN = 0.10  # 10% drawdown máximo
TRAILING_STOP_PCT = 0.005  # 0.5% trailing stop
MAX_TRADE_DURATION_MINUTES = 240  # 4 horas máximo por operación

# ─── Indicadores ─────────────────────────────────────────────────
# Solo usamos EMA cross + RSI como filtro + ADX para confirmar tendencia
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # No comprar si RSI > 70, no vender si RSI < 30
RSI_OVERBOUGHT = 70
ADX_PERIOD = 14
ADX_MIN_TREND = 20  # ADX > 20 = hay tendencia, operar. ADX < 20 = no operar.

# ─── ATR para stops ──────────────────────────────────────────────
ATR_PERIOD = 14
STOP_ATR_MULTIPLIER = 1.5  # Stop loss = 1.5x ATR
TP_ATR_MULTIPLIER = 3.0    # Take profit = 3x ATR (ratio 1:2)

# ─── Take Profit (cierre completo, sin escalonado) ───────────────
TAKE_PROFIT_LEVELS = [
    (0.99, 1.0),  # Placeholder: se calcula dinámicamente con ATR
]

# ─── Machine Learning ───────────────────────────────────────────
ML_RETRAIN_INTERVAL = 50
ML_MIN_TRADES_FOR_TRAINING = 30
ML_LEARNING_RATE = 0.001
ML_DISCOUNT_FACTOR = 0.95

# ─── Régimen de Mercado ─────────────────────────────────────────
REGIME_LOOKBACK = 100
VOLATILITY_HIGH_THRESHOLD = 0.02
TREND_ADX_THRESHOLD = 20  # Coincide con ADX_MIN_TREND

# Parámetros legacy (usados por módulos existentes)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_THRESHOLD = 1.5

# ─── Base de Datos ───────────────────────────────────────────────
DB_PATH = "bot/data/trading_bot.db"

# ─── Dashboard ───────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_UPDATE_INTERVAL = 30  # Actualizar cada 30s (no cada 5s)

# ─── Logging ─────────────────────────────────────────────────────
LOG_FILE = "bot/data/bot.log"
LOG_LEVEL = "INFO"

# ─── Señales Externas ───────────────────────────────────────────
FEAR_GREED_API = "https://api.alternative.me/fng/"
ENABLE_SENTIMENT = True
SENTIMENT_WEIGHT = 0.05  # Peso bajo: solo confirma, no decide
