"""
Configuración central del bot de scalping BTC/USDT.

Todas las constantes y parámetros ajustables se centralizan aquí
para facilitar el tuning sin tocar lógica de negocio.
"""

# ─── Exchange ────────────────────────────────────────────────────
EXCHANGE_ID = "binance"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"  # Velas de 1 minuto para scalping
MARKET_TYPE = "futures"  # "spot" o "futures" (perpetual)

# ─── Apalancamiento ─────────────────────────────────────────────
# Apalancamiento dinámico según confianza de la señal
LEVERAGE_LOW = 2    # Confianza 25-40% → x2
LEVERAGE_MID = 3    # Confianza 40-60% → x3
LEVERAGE_HIGH = 4   # Confianza >60%   → x4
LEVERAGE = 3        # Default (usado como fallback)

# Cooldown tras pérdida (evitar overtrading)
COOLDOWN_AFTER_LOSS_SECONDS = 120  # 2 minutos de pausa tras trade perdedor

# ─── Paper Trading ───────────────────────────────────────────────
INITIAL_BALANCE = 10_000.0  # USDT ficticios
PAPER_MODE = True  # NUNCA operar en real sin cambio manual explícito
FEE_RATE = 0.0004  # 0.04% comisión futuros Binance (taker)

# ─── Gestión de Riesgo ──────────────────────────────────────────
# Riesgo dinámico según confianza (confidence-based sizing)
RISK_LOW = 0.005       # Confianza 25-40% → 0.5% del capital
RISK_MID = 0.007       # Confianza 40-60% → 0.7% del capital
RISK_HIGH = 0.010      # Confianza >60%   → 1.0% del capital
MAX_RISK_PER_TRADE = 0.007  # Fallback por defecto
MAX_OPEN_POSITIONS = 3  # Posiciones normales simultáneas
MAX_OPEN_POSITIONS_EXTRA = 4  # Máximo absoluto (incluye slot extra para señales >60%)
HIGH_CONFIDENCE_THRESHOLD = 0.60  # Señales con >60% confianza usan el slot extra
MAX_DRAWDOWN = 0.15  # 15% drawdown máximo antes de pausar
TRAILING_STOP_PCT = 0.003  # 0.3% trailing stop (más ajustado)
MAX_TRADE_DURATION_MINUTES = 60  # 1 hora máximo por operación

# ─── Indicadores Técnicos ───────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD = 2.0

EMA_FAST = 9
EMA_SLOW = 21

VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_THRESHOLD = 1.5  # 1.5x volumen promedio = spike

# ─── Take Profit Escalonado ─────────────────────────────────────
# Cada tupla: (% del target, % de la posición a cerrar)
TAKE_PROFIT_LEVELS = [
    (0.002, 0.33),  # +0.2% → cerrar 33%
    (0.004, 0.33),  # +0.4% → cerrar 33%
    (0.007, 0.34),  # +0.7% → cerrar 34% restante
]

# ─── Machine Learning ───────────────────────────────────────────
ML_RETRAIN_INTERVAL = 50  # Reentrenar cada 50 trades cerrados
ML_MIN_TRADES_FOR_TRAINING = 30  # Mínimo de trades para empezar a entrenar
ML_LEARNING_RATE = 0.001
ML_DISCOUNT_FACTOR = 0.95  # Gamma para Q-Learning

# ─── Régimen de Mercado ─────────────────────────────────────────
REGIME_LOOKBACK = 100  # Velas para detectar régimen
VOLATILITY_HIGH_THRESHOLD = 0.02  # >2% = alta volatilidad
TREND_ADX_THRESHOLD = 25  # ADX > 25 = tendencia

# ─── Base de Datos ───────────────────────────────────────────────
DB_PATH = "bot/data/trading_bot.db"

# ─── Dashboard ───────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_UPDATE_INTERVAL = 5  # Segundos entre actualizaciones

# ─── Logging ─────────────────────────────────────────────────────
LOG_FILE = "bot/data/bot.log"
LOG_LEVEL = "INFO"

# ─── Señales Externas ───────────────────────────────────────────
FEAR_GREED_API = "https://api.alternative.me/fng/"
ENABLE_SENTIMENT = True
SENTIMENT_WEIGHT = 0.1  # Peso del sentimiento en la decisión final
