"""
config.py — Configuración centralizada del bot de scalping BTC/USDT
=======================================================================
Todos los parámetros del sistema están aquí para facilitar el ajuste
sin tocar la lógica de negocio. Los valores se cargan desde variables
de entorno cuando están disponibles, con defaults seguros para paper trading.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
#  EXCHANGE
# ─────────────────────────────────────────────
@dataclass
class ExchangeConfig:
    # Binance soporta bots via API; Testnet para paper trading
    exchange_id: str = "binance"
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_SECRET", ""))

    # Usar Testnet de Binance en modo paper (sin dinero real)
    use_testnet: bool = True
    testnet_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_TESTNET_KEY", ""))
    testnet_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_TESTNET_SECRET", ""))

    symbol: str = "BTC/USDT"
    timeframe: str = "1m"          # vela de 1 minuto para scalping
    orderbook_depth: int = 20      # niveles del libro de órdenes a analizar


# ─────────────────────────────────────────────
#  PAPER TRADING
# ─────────────────────────────────────────────
@dataclass
class PaperTradingConfig:
    enabled: bool = True           # SIEMPRE True hasta activación manual
    initial_balance_usdt: float = 10_000.0
    maker_fee: float = 0.001       # 0.1% fee Binance Maker
    taker_fee: float = 0.001       # 0.1% fee Binance Taker
    slippage_pct: float = 0.0005   # 0.05% slippage simulado


# ─────────────────────────────────────────────
#  GESTIÓN DE RIESGO
# ─────────────────────────────────────────────
@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.02        # máximo 2% del capital por operación
    max_open_positions: int = 1             # scalping: 1 posición a la vez
    max_daily_drawdown: float = 0.05        # cierre automático si pierde 5% en el día
    max_total_drawdown: float = 0.15        # parada de emergencia al 15% drawdown total

    # Stop-loss dinámico basado en ATR
    atr_multiplier_sl: float = 1.5          # SL = entrada ± 1.5×ATR
    atr_multiplier_tp1: float = 1.0         # TP1 = entrada ± 1×ATR  (50% de la posición)
    atr_multiplier_tp2: float = 2.5         # TP2 = entrada ± 2.5×ATR (50% restante)

    # Trailing stop
    trailing_stop_activation: float = 0.005 # activar trailing cuando gana 0.5%
    trailing_stop_distance: float = 0.003   # mantener 0.3% de distancia al precio


# ─────────────────────────────────────────────
#  INDICADORES TÉCNICOS
# ─────────────────────────────────────────────
@dataclass
class IndicatorConfig:
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 35.0      # más conservador que 30 para BTC
    rsi_overbought: float = 65.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # EMAs para cruces de tendencia
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50             # EMA de tendencia mayor

    # ATR para volatilidad y stops dinámicos
    atr_period: int = 14

    # Volume Moving Average
    volume_ma_period: int = 20
    volume_spike_multiplier: float = 1.5  # volumen > 1.5× media = spike

    # Stochastic RSI (confirmación adicional)
    stoch_rsi_period: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3

    # Lookback mínimo de velas necesarias para calcular todos los indicadores
    min_candles: int = 100


# ─────────────────────────────────────────────
#  MACHINE LEARNING
# ─────────────────────────────────────────────
@dataclass
class MLConfig:
    # Q-Learning para ajuste adaptativo de parámetros
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    epsilon_start: float = 1.0     # exploración inicial
    epsilon_end: float = 0.05      # mínimo de exploración
    epsilon_decay: float = 0.995

    # Reentrenamiento
    retrain_interval_trades: int = 50   # reentrenar cada 50 operaciones cerradas
    min_trades_for_training: int = 20   # mínimo de trades para entrenar

    # Espacio de estados: número de bins para discretizar cada feature
    state_bins: int = 10

    # Features del estado
    state_features: list = field(default_factory=lambda: [
        "rsi", "macd_hist", "bb_position",
        "ema_crossover", "volume_ratio",
        "atr_normalized", "market_regime",
        "fear_greed_index", "btc_dominance"
    ])

    # Acciones disponibles para el agente
    actions: list = field(default_factory=lambda: [
        "hold", "long", "short",
        "close_long", "close_short"
    ])

    # Pesos de recompensa
    reward_pnl_weight: float = 0.6
    reward_sharpe_weight: float = 0.3
    reward_drawdown_penalty: float = 0.1


# ─────────────────────────────────────────────
#  SEÑALES EXTERNAS
# ─────────────────────────────────────────────
@dataclass
class ExternalSignalsConfig:
    # Fear & Greed Index (alternative.me API — gratuita)
    fear_greed_url: str = "https://api.alternative.me/fng/?limit=1"
    fear_greed_update_interval: int = 3600  # actualizar cada hora

    # BTC Dominance via CoinGecko (gratuita sin key)
    coingecko_dominance_url: str = (
        "https://api.coingecko.com/api/v3/global"
    )
    dominance_update_interval: int = 3600

    # Umbrales para ajustar agresividad
    extreme_fear_threshold: int = 25    # F&G < 25 → posible suelo
    extreme_greed_threshold: int = 75   # F&G > 75 → posible techo


# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────
@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8501
    refresh_interval_seconds: int = 5
    equity_chart_candles: int = 500     # puntos en curva de equity


# ─────────────────────────────────────────────
#  BASE DE DATOS
# ─────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    path: str = "bitcoin_bot/data/trading.db"
    trades_table: str = "trades"
    candles_table: str = "candles"
    metrics_table: str = "metrics"
    signals_table: str = "signals"


# ─────────────────────────────────────────────
#  OPERACIONES (timing)
# ─────────────────────────────────────────────
@dataclass
class TradingConfig:
    max_trade_duration_minutes: int = 240   # cerrar si dura más de 4 horas
    cooldown_after_loss_seconds: int = 60   # esperar 1 min tras stop-loss
    signal_confirmation_candles: int = 2    # confirmar señal en N velas


# ─────────────────────────────────────────────
#  CONFIGURACIÓN GLOBAL (singleton)
# ─────────────────────────────────────────────
@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    paper: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    external: ExternalSignalsConfig = field(default_factory=ExternalSignalsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    # Nivel de log: DEBUG | INFO | WARNING | ERROR
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Modo de régimen de mercado detectado automáticamente
    market_regime: str = "unknown"  # "trending_up" | "trending_down" | "ranging" | "volatile"

    def validate(self):
        """Verifica que la configuración es coherente antes de arrancar."""
        assert 0 < self.risk.max_risk_per_trade <= 0.05, \
            "Riesgo por operación debe ser entre 0% y 5%"
        assert self.paper.enabled or (self.exchange.api_key and self.exchange.api_secret), \
            "En modo real se requieren credenciales de API"
        assert self.risk.max_total_drawdown <= 0.20, \
            "Drawdown máximo no debe superar el 20%"
        return True


# Instancia global que se importa desde todos los módulos
CONFIG = BotConfig()
CONFIG.validate()
