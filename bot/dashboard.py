"""
dashboard.py - Panel de métricas en tiempo real con Streamlit.

Diseño:
──────
Se eligió Streamlit sobre Dash por:
1. Menor boilerplate (menos código para mismo resultado)
2. Auto-refresh nativo
3. Más pythónico y fácil de mantener

Métricas mostradas:
- Balance y PnL en tiempo real
- Win rate, Sharpe ratio, max drawdown
- Curva de equity interactiva
- Tabla de trades recientes
- Estado del modelo ML
- Indicadores técnicos actuales
- Estado del risk manager

Ejecutar: streamlit run bot/dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
import json

from config import DB_PATH, INITIAL_BALANCE


def load_trades() -> pd.DataFrame:
    """Carga trades cerrados desde SQLite."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE status='CLOSED' ORDER BY id DESC",
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


def load_equity_curve() -> pd.DataFrame:
    """Carga curva de equity desde SQLite."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM equity_curve ORDER BY id", conn
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


def load_decisions(limit: int = 50) -> pd.DataFrame:
    """Carga últimas decisiones del bot."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM decisions ORDER BY id DESC LIMIT {limit}",
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Calcula métricas de rendimiento."""
    if trades_df.empty:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_pnl": 0, "sharpe": 0, "max_dd": 0,
            "avg_duration": 0, "best_trade": 0, "worst_trade": 0,
        }

    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] <= 0).sum()
    total = len(trades_df)

    pnl_series = trades_df["pnl_pct"].dropna()
    avg_return = pnl_series.mean() if len(pnl_series) > 0 else 0
    std_return = pnl_series.std() if len(pnl_series) > 1 else 1
    sharpe = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0

    # Max drawdown desde la curva de equity
    equity_df = load_equity_curve()
    max_dd = equity_df["drawdown"].max() if not equity_df.empty and "drawdown" in equity_df.columns else 0

    return {
        "total_trades": total,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": wins / total if total > 0 else 0,
        "total_pnl": trades_df["pnl"].sum(),
        "avg_pnl": trades_df["pnl"].mean(),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "best_trade": trades_df["pnl"].max(),
        "worst_trade": trades_df["pnl"].min(),
    }


def run_dashboard():
    """Ejecuta el dashboard de Streamlit."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit no instalado. Ejecuta: pip install streamlit")
        print("Luego: streamlit run bot/dashboard.py")
        return

    st.set_page_config(
        page_title="BTC Trend Following - Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 BTC EMA Pullback Trend Rider (Paper Trading) - Dashboard")
    st.markdown("---")

    # Auto-refresh cada 10 segundos
    from config import DASHBOARD_UPDATE_INTERVAL

    # ── Métricas principales ───────────────────────────────────
    trades_df = load_trades()
    metrics = compute_metrics(trades_df)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        balance = INITIAL_BALANCE + metrics["total_pnl"]
        st.metric("Balance", f"${balance:,.2f}", f"${metrics['total_pnl']:+,.2f}")
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}",
                   f"{metrics['wins']}W / {metrics['losses']}L")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe']:.4f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics['max_dd']:.2%}")
    with col5:
        st.metric("Total Trades", metrics["total_trades"])

    st.markdown("---")

    # ── Curva de Equity ────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Curva de Equity")
        equity_df = load_equity_curve()
        if not equity_df.empty:
            st.line_chart(equity_df.set_index("timestamp")["balance"])
        else:
            st.info("Sin datos de equity aún. El bot debe ejecutar trades primero.")

    with col_right:
        st.subheader("Distribución de PnL")
        if not trades_df.empty and "pnl" in trades_df.columns:
            # trades_df ya viene ORDER BY id DESC, así que .head(20) son los más recientes
            recent_pnl = trades_df["pnl"].head(20).iloc[::-1]  # Invertir para orden cronológico
            recent_pnl.index = range(1, len(recent_pnl) + 1)
            st.bar_chart(recent_pnl)
        else:
            st.info("Sin trades cerrados aún.")

    st.markdown("---")

    # ── Trades Recientes ───────────────────────────────────────
    st.subheader("Últimos Trades")
    if not trades_df.empty:
        display_cols = ["id", "direction", "entry_price", "exit_price",
                        "pnl", "pnl_pct", "market_regime", "status"]
        available_cols = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(trades_df[available_cols].head(20), use_container_width=True)
    else:
        st.info("Sin trades registrados.")

    # ── Decisiones del Bot ─────────────────────────────────────
    st.subheader("Últimas Señales (Sniper Scoring)")
    decisions_df = load_decisions(20)
    if not decisions_df.empty:
        display_cols = []
        if "timestamp" in decisions_df.columns:
            decisions_df["hora"] = pd.to_datetime(decisions_df["timestamp"]).dt.strftime("%H:%M")
            display_cols.append("hora")
        if "action" in decisions_df.columns:
            display_cols.append("action")
        if "confidence" in decisions_df.columns:
            decisions_df["confianza"] = (decisions_df["confidence"] * 100).round(0).astype(str) + "%"
            display_cols.append("confianza")
        if "reason" in decisions_df.columns:
            display_cols.append("reason")
        if display_cols:
            st.dataframe(decisions_df[display_cols], use_container_width=True)
        else:
            st.dataframe(decisions_df, use_container_width=True)
    else:
        st.info("Esperando primera señal...")

    # ── Estado del Modelo ML ───────────────────────────────────
    st.markdown("---")
    st.subheader("Estado del Modelo ML")
    try:
        import pickle
        model_path = Path("bot/data/q_table.pkl")
        if model_path.exists():
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            ml_col1, ml_col2, ml_col3 = st.columns(3)
            with ml_col1:
                st.metric("Estados Aprendidos", len(data.get("q_table", {})))
            with ml_col2:
                st.metric("Epsilon", f"{data.get('epsilon', 1.0):.4f}")
            with ml_col3:
                st.metric("Entrenamientos", data.get("training_count", 0))
        else:
            st.info("Modelo ML no entrenado aún.")
    except Exception as e:
        st.warning(f"Error cargando modelo ML: {e}")

    # ── Métricas detalladas ────────────────────────────────────
    with st.expander("Métricas Detalladas"):
        st.json({
            "best_trade": f"${metrics['best_trade']:.2f}" if metrics['best_trade'] else "N/A",
            "worst_trade": f"${metrics['worst_trade']:.2f}" if metrics['worst_trade'] else "N/A",
            "avg_pnl": f"${metrics['avg_pnl']:.2f}" if metrics['avg_pnl'] else "N/A",
            "initial_balance": f"${INITIAL_BALANCE:,.2f}",
        })


if __name__ == "__main__":
    run_dashboard()
