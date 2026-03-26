# Polymarket Autonomous Trading Bot

Bot autónomo de trading para Polymarket con paper money (dinero simulado).

## Características

- **Paper Trading**: Opera con dinero ficticio para probar estrategias sin riesgo
- **3 Estrategias de Trading**:
  - **Momentum**: Detecta tendencias y opera en su dirección (SMA crossover + ROC)
  - **Value**: Busca mercados mal valorados (spreads, desequilibrios en order book)
  - **Mean Reversion**: Identifica condiciones de sobrecompra/sobreventa (Bollinger Bands + RSI)
- **Sistema de Aprendizaje**: Ajusta automáticamente los pesos de cada estrategia según su rendimiento
- **Gestión de Riesgo**: Stop loss, take profit, límite de posiciones y control de exposición
- **Dashboard CLI**: Monitoreo en tiempo real del portafolio y operaciones
- **Base de Datos SQLite**: Persistencia de trades, posiciones y métricas

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Ejecutar el bot en modo continuo
python -m polymarket_bot

# Ejecutar un solo escaneo de mercados
python -m polymarket_bot --scan

# Ver estado actual del portafolio
python -m polymarket_bot --status

# Resetear la base de datos
python -m polymarket_bot --reset

# Configuración personalizada
python -m polymarket_bot --balance 5000 --interval 120 --max-positions 10
```

## Opciones

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--balance` | 10000 | Balance inicial en USDC (paper money) |
| `--interval` | 300 | Intervalo entre escaneos (segundos) |
| `--max-positions` | 20 | Máximo de posiciones abiertas |
| `--max-position-size` | 500 | Máximo por posición (USDC) |
| `--min-confidence` | 0.6 | Confianza mínima para operar (0-1) |
| `--db` | polymarket_bot.db | Ruta de la base de datos |
| `--log-level` | INFO | Nivel de logging |

## Arquitectura

```
polymarket_bot/
├── __init__.py          # Package init
├── __main__.py          # CLI entry point
├── bot.py               # Orquestador principal
├── api_client.py        # Cliente API de Polymarket
├── config.py            # Configuración
├── database.py          # Persistencia SQLite
├── paper_trader.py      # Motor de paper trading
├── learner.py           # Sistema de aprendizaje adaptativo
├── dashboard.py         # Dashboard CLI
└── strategies/
    ├── __init__.py
    ├── base.py           # Interfaz base de estrategias
    ├── momentum.py       # Estrategia de momentum
    ├── value.py          # Estrategia de valor
    └── mean_reversion.py # Estrategia de mean reversion
```

## Cómo funciona

1. **Escaneo**: El bot consulta la API pública de Polymarket para obtener mercados activos
2. **Filtrado**: Descarta mercados con poca liquidez o volumen
3. **Análisis**: Cada estrategia analiza los mercados y genera señales (BUY/SELL)
4. **Combinación**: El sistema de aprendizaje combina las señales ponderando por rendimiento histórico
5. **Ejecución**: Si la confianza combinada supera el umbral, ejecuta la operación en paper money
6. **Monitoreo**: Revisa posiciones abiertas para stop loss / take profit
7. **Aprendizaje**: Cada 5 ciclos, ajusta los pesos de las estrategias según resultados

## Nota

Este bot opera exclusivamente con **paper money** (dinero ficticio). No realiza operaciones reales ni requiere wallet o fondos. Los datos de mercado son reales, obtenidos de la API pública de Polymarket.
