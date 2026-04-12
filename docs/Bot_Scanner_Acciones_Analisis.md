# Bot de Deteccion de Oportunidades en Acciones (Medio-Largo Plazo)

## Objetivo

Escanear el mercado cada dia, identificar acciones con alta probabilidad de
revalorizacion en semanas/meses, y enviar alertas por Telegram.

---

## Arquitectura General

```
FASE 1: ESCANEO DIARIO (cada noche despues del cierre)
   |
   |-- Universo: S&P 500 + NASDAQ 100 + Russell 1000 (~1,500 acciones)
   |
   |-- FILTRO 1: Fundamental (descartar basura)
   |   - Market Cap > $1B
   |   - Volumen medio 30 dias > 500K
   |   - Precio > $5 (evitar penny stocks)
   |
   |-- FILTRO 2: Tecnico (detectar potencial)
   |   --> Puntuacion 0-100 basada en indicadores
   |
   +-- FILTRO 3: Momentum + Catalizadores
       --> Noticias, earnings, sector rotation

FASE 2: SCORING (puntuar cada accion)
   |
   +-- Ranking de las Top 10-20 acciones

FASE 3: ALERTA
   |
   +-- Telegram/Email con las mejores oportunidades
```

---

## FASE 2: Sistema de Puntuacion

Cada accion recibe puntos de 0 a 100.

### Analisis Tecnico (50 puntos)

| Criterio                  | Puntos | Logica                                              |
|---------------------------|--------|-----------------------------------------------------|
| Tendencia alcista         | 10     | EMA 50 > EMA 200 (Golden Cross)                     |
| Pullback saludable        | 10     | Precio cerca de EMA 50, no lejos                    |
| RSI en zona optima        | 5      | RSI entre 40-60 (no sobrecomprado)                  |
| MACD positivo             | 5      | MACD cruzando al alza                               |
| Volumen creciente         | 10     | Volumen ultimas 5 sesiones > media 20               |
| Rotura de resistencia     | 10     | Precio superando maximos de 52 semanas              |

### Analisis Fundamental (30 puntos)

| Criterio                  | Puntos | Logica                                              |
|---------------------------|--------|-----------------------------------------------------|
| Revenue creciendo         | 10     | Ingresos YoY > 10%                                  |
| EPS positivo y creciendo  | 10     | Beneficio por accion mejorando                       |
| Deuda controlada          | 5      | Debt/Equity < 1.5                                   |
| Margen neto saludable     | 5      | > 10%                                               |

### Momentum y Catalizadores (20 puntos)

| Criterio                           | Puntos | Logica                                     |
|------------------------------------|--------|--------------------------------------------|
| Fuerza relativa vs S&P 500        | 10     | Supera al indice en 1-3 meses              |
| Earnings proximos positivos        | 5      | Estimaciones de analistas al alza          |
| Sector en rotacion alcista         | 5      | El sector esta recibiendo flujo de capital  |

---

## Ejemplo de Alerta

```
SENAL: 3 acciones detectadas (2026-04-12)

1. NVDA - Nvidia (Puntuacion: 87/100)
   - Tecnico: Golden Cross, pullback a EMA50, volumen +40%
   - Fundamental: Revenue +95% YoY, margen 55%
   - Catalizador: Earnings en 2 semanas, sector AI fuerte
   - Zona de entrada: $850-$870
   - Stop Loss sugerido: $800 (-6%)
   - Objetivo: $1,050 (+20%)

2. LLY - Eli Lilly (Puntuacion: 82/100)
   - Tecnico: Rotura de resistencia $780, RSI 55
   - Fundamental: Revenue +35% YoY (Mounjaro/Zepbound)
   - Catalizador: Resultados FDA pendientes
   - Zona de entrada: $780-$800
   - Stop Loss sugerido: $740 (-6%)
   - Objetivo: $920 (+16%)

3. AVGO - Broadcom (Puntuacion: 78/100)
   - Tecnico: EMA ribbon alcista, volumen +25%
   - Fundamental: Revenue +20% YoY, margen 45%
   - Catalizador: Demanda AI/chips creciente
   - Zona de entrada: $170-$175
   - Stop Loss sugerido: $160 (-6%)
   - Objetivo: $210 (+20%)
```

---

## Fuentes de Datos

| Dato                  | Fuente                        | Coste              |
|-----------------------|-------------------------------|---------------------|
| Precios historicos    | Yahoo Finance / yfinance      | Gratis              |
| Fundamentales         | Financial Modeling Prep API   | Gratis (250/dia)    |
| Noticias/Sentimiento  | NewsAPI / Finnhub             | Gratis (limitado)   |
| Datos en tiempo real  | Alpaca API                    | Gratis              |
| Estimaciones analistas| Yahoo Finance                 | Gratis              |

---

## Stack Tecnico

```
Python 3
  |-- yfinance              Datos de precios
  |-- pandas / numpy        Calculos
  |-- ta                    Indicadores tecnicos
  |-- requests              APIs fundamentales
  |-- schedule              Ejecucion diaria automatica
  |-- python-telegram-bot   Alertas por Telegram
  +-- SQLite                Historial de senales
```

---

## Diferencia con el Bot BTC actual

|                   | Bot BTC (actual)                | Bot Scanner Acciones             |
|-------------------|---------------------------------|----------------------------------|
| Objetivo          | Ejecutar trades automaticamente | Detectar oportunidades y avisar  |
| Frecuencia        | Cada 5 minutos                  | 1 vez al dia (cierre de mercado) |
| Accion            | Compra/vende solo               | Tu decides si comprar            |
| Horizonte         | Horas (intradia)                | Semanas/meses                    |
| Complejidad       | Media                           | Alta (mas datos, mas acciones)   |

---

## Gestion de Riesgo

- Riesgo por operacion: 1-2% del capital total
- Maximo 5-10 posiciones abiertas simultaneamente
- Diversificacion: maximo 2 acciones del mismo sector
- Stop Loss obligatorio en cada operacion (-5% a -8%)
- Take Profit parcial: vender 50% en +15%, dejar el resto correr
- Revision semanal del portfolio

---

## Fases de Desarrollo

```
Fase 1: Scanner tecnico basico (1-2 semanas)
  --> Escanear S&P 500 con indicadores tecnicos
  --> Ranking diario Top 10
  --> Alerta por Telegram

Fase 2: Anadir fundamentales (1 semana mas)
  --> Integrar datos de revenue, EPS, deuda
  --> Mejorar puntuacion

Fase 3: Backtesting (1-2 semanas)
  --> Las acciones con score > 80 realmente suben?
  --> Validar con datos historicos 2020-2024

Fase 4: Mejoras (continuo)
  --> Sentimiento de noticias
  --> Deteccion de sectores en rotacion
  --> Machine Learning para mejorar scoring
```

---

## Consideraciones Legales

- Pattern Day Trader (PDT): con menos de $25,000 en EEUU,
  solo 3 day-trades por semana. Este bot es medio-largo plazo,
  por lo que NO aplica esta restriccion.
- Este bot NO ejecuta ordenes. Solo detecta y avisa.
  La decision final de inversion es siempre del usuario.
- Operar en bolsa conlleva riesgo de perdida de capital.

---

Analisis elaborado como parte del proyecto GeneradorColaEstados.
