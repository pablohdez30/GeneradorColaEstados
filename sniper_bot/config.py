"""
Configuración del Sniper Bot — Bot de alertas de alta confianza.

Solo alerta cuando se alinean 3+ condiciones simultáneamente.
NO ejecuta trades, solo envía alertas por Telegram.
"""

# ─── Exchange ────────────────────────────────────────────────────
EXCHANGE_ID = "binance"
SYMBOL = "BTC/USDT"

# Múltiples timeframes para confirmación
TIMEFRAMES = ["15m", "1h", "4h"]

# ─── Indicadores ─────────────────────────────────────────────────
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ADX_PERIOD = 14

# Umbrales para señal de alta confianza
ADX_STRONG_TREND = 30       # ADX > 30 = tendencia fuerte
RSI_OVERSOLD = 30           # No shortear debajo de 30
RSI_OVERBOUGHT = 70         # No comprar encima de 70
VOLUME_SPIKE_THRESHOLD = 2.0  # Volumen 2x por encima de la media

# ─── Scoring ─────────────────────────────────────────────────────
# Cada condición cumplida suma puntos. Mínimo 3 para alertar.
MIN_SCORE_TO_ALERT = 3
SCORES = {
    "adx_strong": 1,          # ADX > 30
    "ema_cross_recent": 2,    # Cruce EMA en últimas 3 velas (vale doble)
    "ema_aligned": 1,         # EMAs alineadas en la dirección
    "rsi_favorable": 1,       # RSI en zona favorable
    "volume_spike": 1,        # Volumen inusual
    "fear_greed_extreme": 1,  # Sentimiento extremo
    "multi_tf_aligned": 2,    # Múltiples timeframes confirman (vale doble)
}

# ─── Apalancamiento sugerido ─────────────────────────────────────
# Basado en score total
LEVERAGE_BY_SCORE = {
    3: 2,   # Score 3 → sugerir x2
    4: 3,   # Score 4 → sugerir x3
    5: 4,   # Score 5 → sugerir x4
    6: 5,   # Score 6+ → sugerir x5
}

# ─── Risk Management sugerido ────────────────────────────────────
RISK_PCT = 0.02              # Sugerir arriesgar 2% del capital
ATR_PERIOD = 14
STOP_ATR_MULTIPLIER = 1.5    # SL = 1.5x ATR
TP_ATR_MULTIPLIER = 3.0      # TP = 3x ATR (ratio 1:2)

# ─── Telegram ────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "8208306630:AAFAPGUB0GIv1Szuj33pY2VZnX_KDPHWRnM"
TELEGRAM_CHAT_ID = "7300811946"

# ─── Cooldown ────────────────────────────────────────────────────
# No enviar más de 1 alerta cada X minutos (evitar spam)
ALERT_COOLDOWN_MINUTES = 240  # Mínimo 4 horas entre alertas

# ─── Sentimiento ─────────────────────────────────────────────────
FEAR_GREED_API = "https://api.alternative.me/fng/"
FEAR_THRESHOLD = 25          # < 25 = miedo extremo (bueno para longs)
GREED_THRESHOLD = 75         # > 75 = avaricia extrema (bueno para shorts)

# ─── Loop ────────────────────────────────────────────────────────
CHECK_INTERVAL_SECONDS = 60  # Analizar cada 60 segundos

# ─── Base de Datos ───────────────────────────────────────────────
DB_PATH = "sniper_bot/data/sniper.db"
LOG_FILE = "sniper_bot/data/sniper.log"
LOG_LEVEL = "INFO"
