"""
dashboard.py — Panel de control en tiempo real (Streamlit)
============================================================
Muestra todas las métricas del bot en un dashboard web actualizado
cada N segundos (por defecto 5 segundos).

Secciones:
  1. Header: estado del bot, balance, equity, modo (paper/live)
  2. Métricas clave: Win Rate, Sharpe Ratio, Max Drawdown, P&L total
  3. Curva de equity en tiempo real (gráfico de línea)
  4. Posición abierta actual (si existe)
  5. Historial de trades recientes (tabla)
  6. Indicadores activos del último análisis
  7. Estado del modelo ML (Q-table, epsilon, recompensa media)

Ejecución:
  streamlit run bitcoin_bot/dashboard.py --server.port 8501
"""

import time
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from database import DB
from config import CONFIG


# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Scalping Bot — Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para aspecto profesional
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #313244;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; margin: 5px 0; }
    .metric-label { font-size: 12px; color: #888; text-transform: uppercase; }
    .win  { color: #a6e3a1; }
    .loss { color: #f38ba8; }
    .neutral { color: #cdd6f4; }
    .paper-badge {
        background: #fab387; color: #1e1e2e;
        padding: 3px 12px; border-radius: 20px;
        font-size: 12px; font-weight: bold;
    }
    .live-badge {
        background: #f38ba8; color: #1e1e2e;
        padding: 3px 12px; border-radius: 20px;
        font-size: 12px; font-weight: bold;
    }
    div[data-testid="stMetricValue"] { font-size: 22px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FUNCIONES DE DATOS
# ─────────────────────────────────────────────
@st.cache_data(ttl=5)  # caché de 5 segundos
def load_equity_curve(limit: int = 500) -> pd.DataFrame:
    """Carga la curva de equity desde la BD."""
    data = DB.get_equity_curve(limit=limit)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


@st.cache_data(ttl=5)
def load_recent_trades(limit: int = 50) -> pd.DataFrame:
    """Carga los últimos N trades cerrados."""
    trades = DB.get_closed_trades(limit=limit)
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    if "entry_ts" in df.columns:
        df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    if "exit_ts" in df.columns:
        df["exit_ts"] = pd.to_datetime(df["exit_ts"])
    return df


@st.cache_data(ttl=5)
def load_trade_stats() -> Dict:
    return DB.get_trade_stats()


@st.cache_data(ttl=60)
def load_latest_external() -> Optional[Dict]:
    return DB.get_latest_external_signal()


# ─────────────────────────────────────────────
#  COMPONENTES UI
# ─────────────────────────────────────────────
def render_header():
    """Header con estado del bot y modo de operación."""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        mode_html = (
            '<span class="paper-badge">📝 PAPER TRADING</span>'
            if CONFIG.paper.enabled
            else '<span class="live-badge">🔴 LIVE TRADING</span>'
        )
        st.markdown(
            f"# ₿ BTC/USDT Scalping Bot &nbsp;&nbsp; {mode_html}",
            unsafe_allow_html=True
        )
        st.caption(f"Exchange: Binance | Timeframe: {CONFIG.exchange.timeframe} | "
                   f"Par: {CONFIG.exchange.symbol} | "
                   f"Actualizado: {datetime.now().strftime('%H:%M:%S')}")

    with col2:
        if st.button("🔄 Actualizar"):
            st.cache_data.clear()

    with col3:
        st.metric("Capital Inicial",
                  f"${CONFIG.paper.initial_balance_usdt:,.0f}")


def render_key_metrics(stats: Dict, equity_curve: pd.DataFrame):
    """KPIs principales en 6 columnas."""
    st.markdown("---")
    cols = st.columns(6)

    # Calcular balance actual
    balance = equity_curve["balance_usdt"].iloc[-1] if not equity_curve.empty else CONFIG.paper.initial_balance_usdt
    equity = equity_curve["equity_usdt"].iloc[-1] if not equity_curve.empty else balance
    total_pnl = balance - CONFIG.paper.initial_balance_usdt
    total_return = total_pnl / CONFIG.paper.initial_balance_usdt

    total_trades = stats.get("total", 0) or 0
    wins = stats.get("wins", 0) or 0
    win_rate = wins / total_trades if total_trades > 0 else 0

    # Calcular Sharpe aproximado desde equity curve
    sharpe = 0.0
    if not equity_curve.empty and len(equity_curve) > 2:
        returns = equity_curve["equity_usdt"].pct_change().dropna()
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * (252 * 1440 / max(len(returns), 1)) ** 0.5

    # Calcular Max Drawdown
    max_dd = 0.0
    if not equity_curve.empty:
        eq = equity_curve["equity_usdt"].values
        peak = eq[0]
        for v in eq:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

    pnl_color = "normal" if total_pnl >= 0 else "inverse"
    return_color = "normal" if total_return >= 0 else "inverse"

    with cols[0]:
        st.metric("💰 Balance", f"${balance:,.2f}",
                  delta=f"${total_pnl:+.2f}")

    with cols[1]:
        st.metric("📈 Equity", f"${equity:,.2f}")

    with cols[2]:
        wr_delta = "🎯" if win_rate > 0.55 else "⚠️"
        st.metric(f"🏆 Win Rate {wr_delta}",
                  f"{win_rate:.1%}",
                  delta=f"{total_trades} trades")

    with cols[3]:
        sharpe_delta = "✅" if sharpe > 1.5 else "⚠️"
        st.metric(f"📊 Sharpe {sharpe_delta}", f"{sharpe:.2f}")

    with cols[4]:
        dd_delta = "✅" if max_dd < 0.15 else "🚨"
        st.metric(f"📉 Max Drawdown {dd_delta}",
                  f"{max_dd:.1%}",
                  delta=None)

    with cols[5]:
        st.metric("💹 Retorno Total",
                  f"{total_return:+.2%}",
                  delta=f"${total_pnl:+.2f} USDT")


def render_equity_chart(equity_curve: pd.DataFrame):
    """Curva de equity con área bajo la curva."""
    st.markdown("### 📈 Curva de Equity en Tiempo Real")

    if equity_curve.empty:
        st.info("Sin datos de equity aún. El bot generará datos cuando procese las primeras velas.")
        return

    initial = CONFIG.paper.initial_balance_usdt
    fig = go.Figure()

    # Zona de ganancia/pérdida
    fig.add_hrect(
        y0=initial, y1=equity_curve["equity_usdt"].max() * 1.01,
        fillcolor="rgba(166, 227, 161, 0.05)", line_width=0,
    )
    fig.add_hrect(
        y0=equity_curve["equity_usdt"].min() * 0.99, y1=initial,
        fillcolor="rgba(243, 139, 168, 0.05)", line_width=0,
    )

    # Línea de balance
    fig.add_trace(go.Scatter(
        x=equity_curve["ts"], y=equity_curve["balance_usdt"],
        name="Balance", line=dict(color="#89b4fa", width=1, dash="dot"),
        opacity=0.6,
    ))

    # Línea de equity
    fig.add_trace(go.Scatter(
        x=equity_curve["ts"], y=equity_curve["equity_usdt"],
        name="Equity", line=dict(color="#a6e3a1", width=2),
        fill="tonexty",
        fillcolor="rgba(166, 227, 161, 0.1)",
    ))

    # Línea de capital inicial
    fig.add_hline(
        y=initial, line_dash="dash",
        line_color="#f38ba8", opacity=0.5,
        annotation_text=f"Capital inicial ${initial:,.0f}",
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        legend=dict(
            bgcolor="#313244",
            bordercolor="#45475a",
            borderwidth=1,
        ),
        xaxis=dict(gridcolor="#313244"),
        yaxis=dict(gridcolor="#313244", tickprefix="$", tickformat=",.0f"),
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_open_position(risk_manager_data: Optional[Dict]):
    """Panel de posición abierta."""
    if not risk_manager_data:
        st.info("ℹ️ Sin posición abierta actualmente")
        return

    pos = risk_manager_data
    direction = pos.get("direction", "unknown")
    entry = pos.get("entry_price", 0)
    size = pos.get("size_btc", 0)
    sl = pos.get("stop_loss", 0)
    tp1 = pos.get("take_profit_1", 0)
    tp2 = pos.get("take_profit_2", 0)

    direction_color = "🟢" if direction == "long" else "🔴"
    st.markdown(f"### {direction_color} Posición Abierta — {direction.upper()}")

    cols = st.columns(5)
    with cols[0]:
        st.metric("Entrada", f"${entry:,.2f}")
    with cols[1]:
        st.metric("Tamaño", f"{size:.5f} BTC")
    with cols[2]:
        st.metric("Stop Loss", f"${sl:,.2f}")
    with cols[3]:
        st.metric("TP1", f"${tp1:,.2f}")
    with cols[4]:
        st.metric("TP2", f"${tp2:,.2f}")


def render_recent_trades(trades_df: pd.DataFrame):
    """Tabla de trades recientes con colores."""
    st.markdown("### 📋 Historial de Trades Recientes")

    if trades_df.empty:
        st.info("Sin trades cerrados aún.")
        return

    display_cols = ["entry_ts", "direction", "entry_price", "exit_price",
                    "pnl_usdt", "pnl_pct", "duration_min", "close_reason",
                    "market_regime"]
    available = [c for c in display_cols if c in trades_df.columns]
    df_display = trades_df[available].head(30).copy()

    # Formatear columnas
    if "pnl_usdt" in df_display.columns:
        df_display["pnl_usdt"] = df_display["pnl_usdt"].apply(
            lambda x: f"${x:+.2f}" if pd.notna(x) else "-"
        )
    if "pnl_pct" in df_display.columns:
        df_display["pnl_pct"] = df_display["pnl_pct"].apply(
            lambda x: f"{x:+.2%}" if pd.notna(x) else "-"
        )
    if "entry_price" in df_display.columns:
        df_display["entry_price"] = df_display["entry_price"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "-"
        )
    if "exit_price" in df_display.columns:
        df_display["exit_price"] = df_display["exit_price"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "-"
        )
    if "duration_min" in df_display.columns:
        df_display["duration_min"] = df_display["duration_min"].apply(
            lambda x: f"{x:.0f}m" if pd.notna(x) else "-"
        )

    df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
    st.dataframe(df_display, use_container_width=True, height=300)


def render_pnl_distribution(trades_df: pd.DataFrame):
    """Distribución de P&L como histograma."""
    if trades_df.empty or "pnl_usdt" not in trades_df.columns:
        return

    st.markdown("### 📊 Distribución de P&L")
    pnls = trades_df["pnl_usdt"].dropna()

    fig = go.Figure(go.Histogram(
        x=pnls,
        nbinsx=30,
        marker_color=[
            "#a6e3a1" if v >= 0 else "#f38ba8" for v in pnls
        ],
        opacity=0.8,
    ))
    fig.add_vline(x=0, line_color="#fab387", line_dash="dash")
    fig.update_layout(
        height=250,
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        xaxis=dict(title="P&L (USDT)", gridcolor="#313244"),
        yaxis=dict(title="Frecuencia", gridcolor="#313244"),
        bargap=0.05, margin=dict(l=10, r=10, t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_regime_breakdown(trades_df: pd.DataFrame):
    """Pie chart de regímenes de mercado."""
    if trades_df.empty or "market_regime" not in trades_df.columns:
        return

    regime_counts = trades_df["market_regime"].value_counts()
    fig = px.pie(
        values=regime_counts.values,
        names=regime_counts.index,
        color_discrete_sequence=["#89b4fa", "#a6e3a1", "#f38ba8", "#fab387", "#cba6f7"],
        hole=0.4,
    )
    fig.update_layout(
        height=250,
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="#313244"),
    )
    return fig


def render_ml_status():
    """Panel de estado del módulo ML."""
    st.markdown("### 🧠 Estado del Modelo ML (Q-Learning)")

    import os
    model_exists = os.path.exists("bitcoin_bot/models/q_table.pkl")

    if not model_exists:
        st.warning("⚠️ Modelo ML no entrenado aún. Se entrenará automáticamente "
                   f"cuando haya {CONFIG.ml.min_trades_for_training} trades cerrados.")
        return

    try:
        import pickle
        with open("bitcoin_bot/models/q_table.pkl", "rb") as f:
            model_data = pickle.load(f)

        cols = st.columns(4)
        with cols[0]:
            st.metric("Estados Q-table", f"{len(model_data.get('q_table', {})):,}")
        with cols[1]:
            st.metric("Exploración (ε)",
                      f"{model_data.get('epsilon', 0):.3f}")
        with cols[2]:
            st.metric("Episodios Totales",
                      f"{model_data.get('total_episodes', 0):,}")
        with cols[3]:
            saved_at = model_data.get("saved_at", "N/A")
            if saved_at != "N/A":
                saved_at = saved_at[:19]
            st.metric("Último entreno", saved_at)
    except Exception:
        st.error("Error cargando datos del modelo ML")


def render_external_signals(external: Optional[Dict]):
    """Panel de señales externas."""
    st.markdown("### 🌐 Señales Externas")

    if not external:
        st.info("Sin datos externos. Se actualizan cada hora.")
        return

    cols = st.columns(3)
    fg = external.get("fear_greed", 50)
    fg_label = external.get("fear_greed_label", "Neutral")
    fg_color = (
        "#a6e3a1" if fg < 25 else
        "#f38ba8" if fg > 75 else
        "#fab387"
    )
    dom = external.get("btc_dominance", 0)

    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fear & Greed Index</div>
            <div class="metric-value" style="color:{fg_color}">{fg}</div>
            <div class="metric-label">{fg_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        dom_color = "#89b4fa" if dom > 55 else "#cba6f7"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BTC Dominance</div>
            <div class="metric-value" style="color:{dom_color}">{dom:.1f}%</div>
            <div class="metric-label">{'Alta' if dom > 55 else 'Normal'}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        ts = external.get("ts", "")[:19] if external.get("ts") else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Última actualización</div>
            <div class="metric-value" style="font-size:16px">{ts}</div>
            <div class="metric-label">UTC</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    """Sidebar con configuración y atajos."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")

        st.markdown("**Riesgo por operación**")
        st.info(f"{CONFIG.risk.max_risk_per_trade:.0%}")

        st.markdown("**Max Drawdown**")
        st.info(f"{CONFIG.risk.max_total_drawdown:.0%}")

        st.markdown("**Timeframe**")
        st.info(CONFIG.exchange.timeframe)

        st.markdown("**Capital Virtual**")
        st.info(f"${CONFIG.paper.initial_balance_usdt:,.0f} USDT")

        st.markdown("---")
        st.markdown("**Indicadores activos**")
        for ind in ["RSI(14)", "MACD(12,26,9)", "BB(20,2σ)",
                    "EMA(9,21,50)", "ATR(14)", "StochRSI", "Volume"]:
            st.markdown(f"• {ind}")

        st.markdown("---")
        st.markdown("**ML**")
        st.info(f"Q-Learning\nε-decay: {CONFIG.ml.epsilon_decay}")
        st.info(f"Reentrenar cada {CONFIG.ml.retrain_interval_trades} trades")

        st.markdown("---")
        st.caption("⚠️ **MODO SIMULACIÓN** — Sin dinero real")


# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────
def main():
    """Punto de entrada del dashboard."""
    render_sidebar()
    render_header()

    # Cargar datos
    equity_curve = load_equity_curve()
    trades_df = load_recent_trades()
    stats = load_trade_stats()
    external = load_latest_external()

    # KPIs
    render_key_metrics(stats, equity_curve)

    st.markdown("---")

    # Posición abierta (leer de DB)
    open_trade = DB.get_open_trade()
    render_open_position(open_trade)

    st.markdown("---")

    # Curva de equity
    render_equity_chart(equity_curve)

    # Dos columnas: distribución P&L + régimen
    col1, col2 = st.columns(2)
    with col1:
        render_pnl_distribution(trades_df)
    with col2:
        st.markdown("### 🗂️ Distribución por Régimen")
        fig = render_regime_breakdown(trades_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Historial de trades
    render_recent_trades(trades_df)

    # Señales externas
    render_external_signals(external)

    # Estado ML
    st.markdown("---")
    render_ml_status()

    # Auto-refresh
    refresh_interval = CONFIG.dashboard.refresh_interval_seconds
    time.sleep(refresh_interval)
    st.rerun()


if __name__ == "__main__":
    main()
