# EMA Pullback Trend Rider - BTC/USDT

## Estrategia de Trading Algoritmico

Bot de paper trading que opera BTC/USDT buscando retrocesos (pullbacks) a la media movil EMA 21 en tendencias fuertes, con confirmacion multi-timeframe.

---

## Concepto

En una tendencia solida, el precio se aleja de la media movil, luego retrocede hacia ella (pullback) y rebota para continuar la tendencia. Este bot detecta ese momento exacto de rebote.

---

## Flujo de Decision

```
Precio BTC/USDT cada 5 minutos
         |
    1. Tendencia fuerte?
       ADX > 25
         |  NO --> HOLD (sin tendencia)
         v
    2. EMAs alineadas?
       BULL: EMA9 > EMA21 > EMA50
       BEAR: EMA9 < EMA21 < EMA50
         |  NO --> HOLD (sin direccion)
         v
    3. Timeframes superiores confirman?
       1H Y 4H deben estar alineados
         |  NO --> HOLD (sin confirmacion)
         v
    4. Pullback a EMA21?
       a) Precio se alejo (> 0.5x ATR)
       b) Precio volvio a tocar EMA21 (< 0.3x ATR)
         |  NO --> HOLD (esperando pullback)
         v
    5. RSI entre 35-65?
       (evita sobrecompra/sobreventa)
         |  NO --> HOLD
         v
    SENAL: BUY o SELL
```

---

## Indicadores Utilizados

| Indicador | Configuracion | Funcion |
|-----------|--------------|---------|
| EMA 9 | Rapida | Detecta momentum a corto plazo |
| EMA 21 | Media | Zona de pullback (nivel de entrada) |
| EMA 50 | Lenta | Confirma tendencia de fondo |
| ADX (14) | Umbral: 25 | Mide fuerza de la tendencia |
| RSI (14) | Rango: 35-65 | Filtra extremos de momentum |
| ATR (14) | Multiplicador SL/TP | Mide volatilidad para stops |

---

## Timeframes

| Timeframe | Rol |
|-----------|-----|
| **15 minutos** | Principal - genera las senales |
| **1 hora** | Confirmacion - EMAs deben estar alineadas |
| **4 horas** | Confirmacion - EMAs deben estar alineadas |

Los 3 timeframes deben estar de acuerdo para abrir una operacion.

---

## Gestion de Riesgo

| Parametro | Valor |
|-----------|-------|
| Stop Loss | 1.5x ATR debajo/encima del precio |
| Take Profit | 2x la distancia del SL (R:R 1:2) |
| Trailing Stop | NO (probado, empeora resultados) |
| Modo | Paper Trading (simulacion con dinero ficticio) |
| Balance inicial | $10,000 USDT |

---

## Parametros del Bot

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| ADX minimo | 25 | Solo opera si hay tendencia fuerte |
| Touch margin | 0.3x ATR | Distancia maxima para considerar "toque" a EMA21 |
| Away margin | 0.5x ATR | Distancia minima que el precio debe alejarse antes |
| RSI min/max | 35 / 65 | Rango permitido de RSI para operar |
| EMA ribbon | 9 / 21 / 50 | Periodos de las medias moviles |

---

## Ejemplo de Operacion LONG

```
1. BTC sube con tendencia: EMA9 > EMA21 > EMA50, ADX = 32
2. 1H y 4H tambien alcistas --> confirmado
3. Precio sube a $85,000 (se aleja de EMA21 en $84,500)
4. Precio retrocede a $84,550 (toca EMA21) --> PULLBACK detectado
5. RSI = 52 (dentro de rango) --> OK
6. --> SENAL BUY a $84,550
7. Stop Loss: $84,550 - (1.5 x ATR) = ~$84,100
8. Take Profit: $84,550 + (2 x $450) = ~$85,450
9. Ratio Riesgo:Beneficio = 1:2
```

---

## Resultados del Backtest

| Metrica | Valor |
|---------|-------|
| Periodo | 6 meses de datos historicos BTC/USDT |
| Rendimiento | **+20.8%** |
| Max Drawdown | 32.6% |
| Estrategia ganadora | Backtest v3 (comparada con 5 estrategias) |

---

## Stack Tecnico

- **Lenguaje**: Python 3
- **Exchange**: Binance (futuros perpetuos)
- **Datos**: API ccxt (tiempo real)
- **Dashboard**: Streamlit (visualizacion web)
- **Base de datos**: SQLite (registro de trades)
- **Servidor**: Hetzner Cloud (ejecucion 24/7)

---

## Estrategias Descartadas

| Estrategia | Resultado | Por que se descarto |
|------------|-----------|-------------------|
| BTC Sniper v2 | +5.2% | Bajo rendimiento |
| Bollinger Squeeze | -12.4% | Perdidas |
| RSI Mean Reversion | -8.7% | Perdidas |
| Scalping Momentum | +3.1% | Bajo rendimiento |
| EMA Pullback + Trailing Stop | -49.7% | Trailing stop destruyo resultados |

---

*Bot desarrollado y optimizado mediante backtesting iterativo (v1 a v4). Actualmente en fase de paper trading (simulacion) para validar rendimiento en tiempo real.*
