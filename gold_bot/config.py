"""
Configuration for the Gold Trading Bot (Golden Cross Strategy).

Parameters optimized in MetaTrader 5 (row 0.68 - best risk/reward).
"""

# ── Symbol & Exchange ───────────────────────────────────────────
SYMBOL = "XAU/USDT"           # Try this first
FALLBACK_SYMBOL = "PAXG/USDT" # If XAU/USDT not available
EXCHANGE = "bybit"            # Primary exchange
FALLBACK_EXCHANGE = "binance" # Fallback

# ── Paper Trading ───────────────────────────────────────────────
INITIAL_BALANCE = 10_000.0
FEE_RATE = 0.0004             # 0.04% commission
RISK_PCT = 0.01               # 1% risk per trade
LEVERAGE = 1                  # No leverage
MAX_OPEN_POSITIONS = 1
MAX_DRAWDOWN = 0.10           # 10% max drawdown

# ── Optimized Parameters from MT5 (row 0.68) ───────────────────
ADX_MIN = 40                  # Only VERY strong trends
SL_ATR_MULT = 3.0             # Wide stops for gold
TP_RATIO = 4.0                # Let profits run (R:R 1:4)
TOUCH_MARGIN_MULT = 0.9       # Generous pullback zone
AWAY_MARGIN_MULT = 1.4        # Price must really move away first
COOLDOWN_SECONDS = 4 * 3600   # 4 hours (16 bars * 15min)
MAX_TRADE_DURATION_MINUTES = 50 * 60  # 200 bars * 15min = 50 hours
CHECK_INTERVAL_SECONDS = 300  # 5 minutes
TRADE_COOLDOWN_SECONDS = 4 * 3600

# ── Session Filter (UTC) ───────────────────────────────────────
SESSION_START_HOUR = 7
SESSION_END_HOUR = 20

# ── Indicators ──────────────────────────────────────────────────
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
EMA_TREND = 200
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14

# ── Database ────────────────────────────────────────────────────
DB_PATH = "gold_bot/data/gold_trading.db"
LOG_FILE = "gold_bot/data/gold_bot.log"
LOG_LEVEL = "INFO"

# ── Dashboard ───────────────────────────────────────────────────
DASHBOARD_PORT = 8052
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_UPDATE_INTERVAL = 30
