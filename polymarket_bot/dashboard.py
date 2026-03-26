"""
CLI Dashboard: Displays bot status, portfolio, positions, and performance.
"""

import os
from datetime import datetime


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2%}"


def color(text: str, code: str) -> str:
    """ANSI color wrapper."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return f"{colors.get(code, '')}{text}{colors['reset']}"


def display_dashboard(engine, learner, api_client=None, last_scan=None):
    """Display the main dashboard."""
    clear_screen()
    summary = engine.get_summary()
    positions = engine.get_open_positions()
    perf = learner.get_performance_report()
    recent = engine.db.get_recent_trades(limit=10)

    # Header
    print(color("=" * 70, "cyan"))
    print(color("     POLYMARKET AUTONOMOUS TRADING BOT", "bold"))
    print(color("     Paper Trading Mode", "yellow"))
    print(color("=" * 70, "cyan"))
    print()

    # Portfolio Overview
    total_return = summary["total_return"]
    return_color = "green" if total_return >= 0 else "red"

    print(color("  PORTFOLIO", "bold"))
    print(f"  Balance (Cash):     {color(format_currency(summary['balance']), 'cyan')}")
    print(f"  Portfolio Value:    {color(format_currency(summary['portfolio_value']), 'bold')}")
    print(f"  Peak Value:         {format_currency(summary.get('peak_value', summary['portfolio_value']))}")
    print(f"  Initial Balance:    {format_currency(summary['initial_balance'])}")
    print(f"  Total Return:       {color(format_pct(total_return), return_color)}")
    drawdown = summary.get("drawdown", 0)
    dd_color = "red" if drawdown > 0.1 else "yellow" if drawdown > 0.05 else "green"
    print(f"  Drawdown from Peak: {color(format_pct(-drawdown), dd_color)}")
    print(f"  Open Positions:     {summary['open_positions']}")
    if summary.get("circuit_breaker_active"):
        print(f"  {color('  !! CIRCUIT BREAKER ACTIVE - Trading paused !!', 'red')}")
    print()

    # Trading Stats
    print(color("  TRADING STATS", "bold"))
    print(f"  Total Trades:       {summary['total_trades']}")
    print(f"  Winning:            {color(str(summary['winning_trades']), 'green')}")
    print(f"  Losing:             {color(str(summary['losing_trades']), 'red')}")
    wr = summary['win_rate']
    wr_color = "green" if wr >= 0.5 else "red" if wr > 0 else "dim"
    print(f"  Win Rate:           {color(format_pct(wr), wr_color)}")
    print()

    # Strategy Performance
    print(color("  STRATEGY WEIGHTS & PERFORMANCE", "bold"))
    print(f"  {'Strategy':<20} {'Weight':>8} {'Trades':>8} {'Win Rate':>10} {'P&L':>12}")
    print(f"  {'-'*58}")
    for name, data in perf.items():
        w = format_pct(data["weight"])
        t = str(data["total_trades"])
        wr_val = format_pct(data["win_rate"]) if data["total_trades"] > 0 else "  N/A"
        pnl = format_currency(data["total_pnl"])
        pnl_c = "green" if data["total_pnl"] > 0 else "red" if data["total_pnl"] < 0 else "dim"
        print(f"  {name:<20} {w:>8} {t:>8} {wr_val:>10} {color(pnl, pnl_c):>20}")
    print()

    # Open Positions
    if positions:
        print(color("  OPEN POSITIONS", "bold"))
        print(f"  {'Market':<35} {'Side':>5} {'Entry':>7} {'Current':>9} {'P&L':>10} {'Strategy':>14}")
        print(f"  {'-'*82}")
        for pos in positions:
            question = pos.market_question[:33] if pos.market_question else pos.market_id[:33]
            pnl = pos.unrealized_pnl(pos.entry_price)  # Approximate
            pnl_str = format_currency(pnl) if pnl != 0 else "  --"
            print(f"  {question:<35} {pos.outcome:>5} {pos.entry_price:>7.3f} "
                  f"{'   --':>9} {pnl_str:>10} {pos.strategy:>14}")
        print()

    # Recent Trades
    if recent:
        print(color("  RECENT TRADES", "bold"))
        print(f"  {'Time':<20} {'Market':<25} {'Side':>5} {'Price':>7} {'P&L':>10} {'Status':>8}")
        print(f"  {'-'*77}")
        for trade in recent[:8]:
            ts = trade["timestamp"][:16] if trade["timestamp"] else ""
            q = (trade["market_question"] or "")[:23]
            pnl = trade.get("pnl")
            pnl_str = format_currency(pnl) if pnl is not None else "  --"
            pnl_c = "green" if pnl and pnl > 0 else "red" if pnl and pnl < 0 else "dim"
            status = trade["status"]
            status_c = "green" if status == "closed" else "yellow"
            print(f"  {ts:<20} {q:<25} {trade['outcome']:>5} {trade['price']:>7.3f} "
                  f"{color(pnl_str, pnl_c):>18} {color(status, status_c):>16}")
        print()

    # Footer
    scan_info = f" | Last scan: {last_scan}" if last_scan else ""
    print(color(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{scan_info}", "dim"))
    print(color("  Press Ctrl+C to stop the bot", "dim"))
    print()


def display_startup_banner(config):
    """Show startup configuration."""
    print(color("=" * 70, "cyan"))
    print(color("     POLYMARKET AUTONOMOUS TRADING BOT v1.0", "bold"))
    print(color("=" * 70, "cyan"))
    print()
    print(color("  Configuration:", "bold"))
    print(f"    Initial Balance:    {format_currency(config.initial_balance)}")
    print(f"    Max Position Size:  {format_currency(config.max_position_size)}")
    print(f"    Max Open Positions: {config.max_open_positions}")
    print(f"    Stop Loss:          {config.stop_loss_pct:.0%}")
    print(f"    Take Profit:        {config.take_profit_pct:.0%}")
    print(f"    Min Confidence:     {config.min_confidence:.0%}")
    print(f"    Scan Interval:      {config.scan_interval_seconds}s")
    print(f"    Database:           {config.db_path}")
    print()
    print(color("  Risk Management:", "bold"))
    cb_status = "ON" if config.circuit_breaker_enabled else "OFF"
    print(f"    Circuit Breaker:    {cb_status} (triggers at {config.circuit_breaker_drawdown_pct:.0%} drawdown)")
    print(f"    CB Cooldown:        {config.circuit_breaker_cooldown_seconds}s")
    slip_status = "ON" if config.slippage_enabled else "OFF"
    print(f"    Slippage Sim:       {slip_status} (max {config.max_slippage_bps:.0f}bps)")
    print(f"    Warm-up Period:     {config.warmup_min_data_points} data points")
    print()
    print(color("  Strategies:", "bold"))
    for name, weight in config.strategy_weights.items():
        print(f"    {name:<20} weight: {weight:.0%}")
    print()
    print(color("  Mode: PAPER TRADING (no real money)", "yellow"))
    print()
