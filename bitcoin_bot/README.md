# ₿ BTC/USDT Scalping Bot

Bot de trading algorítmico para Bitcoin en modo **paper trading** (simulación con dinero ficticio). Utiliza análisis técnico multi-indicador y un agente de Q-Learning adaptativo.

---

## Arquitectura

```
bitcoin_bot/
├── config.py          → Parámetros centralizados (riesgo, indicadores, ML)
├── logger.py          → Logging estructurado JSON + consola coloreada
├── database.py        → Persistencia SQLite (trades, velas, métricas)
├── data_feed.py       → WebSocket Binance via CCXT (velas + order book)
├── strategy_engine.py → RSI + MACD + BB + EMA + StochRSI + Volume + OB
├── risk_manager.py    → Position sizing, SL dinámico, trailing stop
├── paper_trader.py    → Simulador de órdenes con balance virtual
├── ml_module.py       → Q-Learning tabular + encoder de estado
├── dashboard.py       → Streamlit (equity curve, métricas, trades)
└── main.py            → Orquestador y punto de entrada
```

---

## Instalación

```bash
# 1. Instalar TA-Lib (librería C nativa)
# Ubuntu/Debian:
sudo apt-get install -y ta-lib
# macOS:
brew install ta-lib

# 2. Instalar dependencias Python
pip install -r bitcoin_bot/requirements.txt

# 3. Configurar credenciales (solo necesario para conectar a Binance Testnet)
cp bitcoin_bot/.env.example bitcoin_bot/.env
# Editar .env con tu API key de Binance Testnet
```

---

## Uso

```bash
# Arrancar bot con dashboard
python bitcoin_bot/main.py

# Con balance personalizado
python bitcoin_bot/main.py --balance 50000 --risk 0.01

# Sin dashboard (solo CLI)
python bitcoin_bot/main.py --no-dashboard

# Dashboard por separado
streamlit run bitcoin_bot/dashboard.py --server.port 8501
```

**Dashboard:** http://localhost:8501

---

## Estrategia de Trading

### Señales (sistema de puntuación ponderada)

| Indicador | Peso | Condición LONG | Condición SHORT |
|-----------|------|----------------|-----------------|
| RSI(14) | 20% | < 35 | > 65 |
| MACD(12,26,9) | 20% | Cruce alcista | Cruce bajista |
| Bollinger Bands(20) | 15% | Precio en BB inferior | Precio en BB superior |
| EMA(9/21) | 15% | EMA9 cruza sobre EMA21 | EMA9 cruza bajo EMA21 |
| Volumen | 10% | Spike > 1.5× media | Spike > 1.5× media |
| Stoch RSI | 8% | K < 20 (sobreventa) | K > 80 (sobrecompra) |
| Order Book | 7% | Imbalance > 0.10 | Imbalance < -0.10 |
| Patrón velas | 5% | Hammer, Engulfing bull | Shooting star, Engulfing bear |

**Umbral de entrada:** confianza ≥ 60% (ajustado por régimen y ML)

### Gestión de Riesgo

- **Riesgo/operación:** 2% del balance (position sizing automático)
- **Stop-loss:** 1.5 × ATR (dinámico según volatilidad)
- **Take-profit 1:** 1.0 × ATR (cierra 50% de la posición)
- **Take-profit 2:** 2.5 × ATR (cierra el 50% restante)
- **Trailing stop:** activado tras TP1, distancia 0.3%
- **Max drawdown diario:** 5% → parar operaciones del día
- **Max drawdown total:** 15% → parada de emergencia
- **Max duración:** 4 horas → cierre por timeout

### Regímenes de Mercado

El bot ajusta automáticamente sus parámetros según el estado del mercado:

- `trending_up` → solo LONG, umbral más bajo
- `trending_down` → solo SHORT, umbral más bajo
- `ranging` → reversiones a extremos de Bollinger, umbral más alto
- `volatile` → umbral muy alto, mayor confirmación requerida

---

## Machine Learning (Q-Learning)

El agente Q-Learning aprende de cada trade cerrado:

- **Estado:** 9 features discretizadas (RSI, MACD hist, BB position, EMA cross, volumen, ATR, régimen, Fear&Greed, dominancia BTC)
- **Acciones:** hold, long, short, close_long, close_short
- **Recompensa:** P&L normalizado (60%) + Sharpe (30%) + penalización drawdown (10%)
- **Reentrenamiento:** cada 50 trades cerrados
- **Q-table:** guardada en `bitcoin_bot/models/q_table.pkl`

---

## Señales Externas

- **Fear & Greed Index:** [alternative.me](https://alternative.me/crypto/fear-and-greed-index/) — actualizado cada hora
- **Dominancia BTC:** CoinGecko API — actualizada cada hora
- Ambas APIs son gratuitas y no requieren autenticación

---

## Métricas Objetivo

| Métrica | Objetivo | Cómo mejorar si no se alcanza |
|---------|----------|-------------------------------|
| Win Rate | > 55% | Aumentar umbral de confianza |
| Sharpe Ratio | > 1.5 | Reducir riesgo por operación |
| Max Drawdown | < 15% | Reducir `max_risk_per_trade` |

---

## ADVERTENCIA DE SEGURIDAD

> Este bot opera **exclusivamente en modo simulación** (paper trading).
> La variable `CONFIG.paper.enabled = True` es el único mecanismo de seguridad
> que previene conexiones a fondos reales. **No modificar a `False`** hasta
> haber validado el rendimiento durante al menos 3 meses en paper trading
> con resultados consistentes.
