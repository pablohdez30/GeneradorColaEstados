# BTC/USDT Scalping Bot - Paper Trading

## Inicio Rápido

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el bot (paper trading)
python main.py

# Dashboard (en otra terminal)
streamlit run bot/dashboard.py
```

## Arquitectura

```
main.py                 → Orquestador principal
config.py               → Configuración centralizada
bot/
  data_feed.py          → Datos de mercado (CCXT + Binance)
  strategy_engine.py    → Señales multi-indicador con confluencia
  risk_manager.py       → Gestión de riesgo y posiciones
  paper_trader.py       → Simulador de órdenes (paper trading)
  ml_module.py          → Q-Learning adaptativo
  dashboard.py          → Panel Streamlit de métricas
  logger.py             → Registro en SQLite + archivo
  data/                 → Base de datos y modelo ML
```

## Estrategia

Confluencia de indicadores (se requieren ≥2 señales coincidentes):
- RSI (14): Sobreventa/sobrecompra
- MACD (12,26,9): Momentum
- Bollinger Bands (20,2): Reversión desde extremos
- EMA Cross (9/21): Dirección de tendencia
- Patrones de velas japonesas
- Análisis de volumen y order book

## Gestión de Riesgo

- Máximo 2% del capital por trade
- Stop-loss dinámico basado en ATR
- Take-profit escalonado (3 niveles)
- Trailing stop (0.5%)
- Max drawdown 15% → pausa automática
- Circuit breaker: 5 pérdidas consecutivas → cooling off

## Machine Learning

Q-Learning tabular que aprende de cada trade:
- Discretiza estado del mercado (648 estados posibles)
- Se reentrena cada 50 trades cerrados
- Incorpora sentimiento (Fear & Greed Index)
- Peso del ML se incrementa conforme mejora su precisión
