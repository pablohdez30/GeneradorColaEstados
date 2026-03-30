"""
dashboard_trend.py - Dashboard del Bot Trend Following.

Usa base de datos separada (trading_bot_trend.db).
Ejecutar: streamlit run bot/dashboard_trend.py --server.port 8051
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd

from config import INITIAL_BALANCE

DB_PATH_TREND = "bot/data/trading_bot_trend.db"


def load_trades() -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH_TREND) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE status='CLOSED' ORDER BY id DESC", conn)
        return df
    except Exception:
        return pd.DataFrame()


def load_equity_curve() -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH_TREND) as conn:
            df = pd.read_sql_query("SELECT * FROM equity_curve ORDER BY id", conn)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


def load_decisions(limit: int = 50) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH_TREND) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM decisions ORDER BY id DESC LIMIT {limit}", conn)
        return df
    except Exception:
        return pd.DataFrame()


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "avg_pnl": 0, "sharpe": 0, "max_dd": 0,
                "best_trade": 0, "worst_trade": 0}

    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] <= 0).sum()
    total = len(trades_df)

    pnl_series = trades_df["pnl_pct"].dropna()
    avg_return = pnl_series.mean() if len(pnl_series) > 0 else 0
    std_return = pnl_series.std() if len(pnl_series) > 1 else 1
    sharpe = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0

    equity_df = load_equity_curve()
    max_dd = equity_df["drawdown"].max() if not equity_df.empty and "drawdown" in equity_df.columns else 0

    return {
        "total_trades": total, "wins": int(wins), "losses": int(losses),
        "win_rate": wins / total if total > 0 else 0,
        "total_pnl": trades_df["pnl"].sum(), "avg_pnl": trades_df["pnl"].mean(),
        "sharpe": sharpe, "max_dd": max_dd,
        "best_trade": trades_df["pnl"].max(), "worst_trade": trades_df["pnl"].min(),
    }


def run_dashboard():
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit no instalado.")
        return

    st.set_page_config(
        page_title="BTC Trend Following - Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 BTC Trend Following Bot (Original) - Dashboard")
    st.markdown("*Estrategia: EMA 9/21 Cross + ADX + RSI | Timeframe: 15m*")
    st.markdown("---")

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

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Curva de Equity")
        equity_df = load_equity_curve()
        if not equity_df.empty:
            st.line_chart(equity_df.set_index("timestamp")["balance"])
        else:
            st.info("Sin datos de equity aún.")

    with col_right:
        st.subheader("Distribución de PnL")
        if not trades_df.empty and "pnl" in trades_df.columns:
            recent_pnl = trades_df["pnl"].head(20).iloc[::-1]
            recent_pnl.index = range(1, len(recent_pnl) + 1)
            st.bar_chart(recent_pnl)
        else:
            st.info("Sin trades cerrados aún.")

    st.markdown("---")

    st.subheader("Últimos Trades")
    if not trades_df.empty:
        display_cols = ["id", "direction", "entry_price", "exit_price",
                        "pnl", "pnl_pct", "market_regime", "status"]
        available_cols = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(trades_df[available_cols].head(20), use_container_width=True)
    else:
        st.info("Sin trades registrados.")

    st.subheader("Últimas Señales")
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
        st.info("Esperando primera señal...")


if __name__ == "__main__":
    run_dashboard()
