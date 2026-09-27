"""
telegram/messages.py — Telegram message formatting.

All messages include the mandatory disclaimer:
⚠️ MANUAL EXECUTION ONLY — BOT DOES NOT PLACE ORDERS.

No capital, position size, or account balance in any message.
"""

from datetime import datetime, timezone
from strategy.intraday import SetupResult
from config.settings import STRATEGY_VERSION
from config.strategy_config import INTRADAY_EXPIRY_MINUTES, SWING_EXPIRY_HOURS


def format_signal_message(setup: SetupResult, signal_id: str) -> str:
    """
    Format a complete Telegram signal message per the spec.
    """
    # Header
    if setup.mode == "INTRADAY":
        mode_label = "INTRADAY | 15M"
    else:
        mode_label = "SWING | 1H"

    direction_emoji = "📈" if setup.direction == "LONG" else "📉"
    direction_label = "LONG" if setup.direction == "LONG" else "SHORT"

    # Coin emoji map
    coin_map = {"BTCUSD": "₿ BTC", "ETHUSD": "Ξ ETH", "SOLUSD": "◎ SOL"}
    coin_label = coin_map.get(setup.symbol, f"🪙 {setup.symbol.replace('USD','')}")

    # Expiry
    if setup.mode == "INTRADAY":
        expiry_label = f"{INTRADAY_EXPIRY_MINUTES} minutes"
    else:
        expiry_label = f"{SWING_EXPIRY_HOURS} hours"

    # 4H bias from description
    h4_bias = _extract_bias(setup.h4_summary)
    h1_bias = _extract_bias(setup.h1_summary)

    # Format entry/sl/tp with appropriate precision
    entry_str = _format_price(setup.symbol, setup.entry)
    sl_str    = _format_price(setup.symbol, setup.sl)
    tp_str    = _format_price(setup.symbol, setup.tp)

    msg = f"""🚨 DELTA SIGNAL

{coin_label}
📊 {mode_label}

{direction_emoji} Direction: {direction_label}

🎯 Entry: {entry_str}
🛑 SL:    {sl_str}
💰 TP:    {tp_str}

📐 RR: 1:{setup.rr}

━━━━━━━━━━━━━━

🧭 4H:
{h4_bias}

📊 1H:
{h1_bias}

⚡ 15M:
{_format_m15_lines(setup.m15_summary)}

💧 Liquidity:
{setup.liquidity_type.replace('_', ' ').title()} @ {_format_price(setup.symbol, setup.liquidity_level)}

🎯 Sweep Score:
{setup.sweep_score}/100

⭐ Signal Quality:
{setup.quality.value}

🔐 Signal ID:
{signal_id}

⏳ Expiry:
{expiry_label}

━━━━━━━━━━━━━━
⚠️ MANUAL EXECUTION ONLY
Bot does NOT place orders."""

    return msg.strip()


def format_outcome_message(signal_id: str, outcome: str, exit_price: float, r_multiple: float) -> str:
    """Format an outcome update notification."""
    if outcome == "TP_HIT":
        emoji = "✅"
        label = "TP HIT"
        r_str = f"+{r_multiple:.2f}R"
    elif outcome == "SL_HIT":
        emoji = "❌"
        label = "STOP LOSS HIT"
        r_str = f"-1.0R"
    else:
        emoji = "⏰"
        label = "EXPIRED"
        r_str = "0R"

    return (
        f"{emoji} SIGNAL UPDATE\n\n"
        f"ID: {signal_id}\n"
        f"Outcome: {label}\n"
        f"Price: {exit_price:.4f}\n"
        f"Result: {r_str}\n\n"
        f"━━━━━━━━━━━━━━\n"
        f"⚠️ MANUAL EXECUTION ONLY\n"
        f"Bot does NOT place orders."
    )


def format_weekly_report(
    week_start: str,
    week_end:   str,
    stats:      dict,
    coin_stats: dict,
    tf_stats:   dict,
    dir_stats:  dict,
    liq_stats:  dict,
    observations: list,
    learning:     list,
    suggestions:  list,
) -> str:
    """Format the weekly performance report."""

    def pct(n, d):
        return f"{n/d*100:.1f}%" if d > 0 else "N/A"

    report = f"""📊 WEEKLY BOT REPORT
Strategy: {STRATEGY_VERSION}

Period:
{week_start} → {week_end}

━━━━━━━━━━━━━━

📌 SIGNALS

Total Signals:      {stats.get('total_signals', 0)}
Valid Signals:      {stats.get('valid_signals', 0)}
Rejected Setups:    {stats.get('rejected', 0)}

📌 TRADE OUTCOMES

Entries Reached:    {stats.get('entries_reached', 0)}
TP Hits:            {stats.get('tp_hits', 0)}
SL Hits:            {stats.get('sl_hits', 0)}
Expired:            {stats.get('expired', 0)}

Win Rate:           {pct(stats.get('tp_hits',0), stats.get('entries_reached',1))}
Loss Rate:          {pct(stats.get('sl_hits',0), stats.get('entries_reached',1))}

Average RR:         1:{stats.get('avg_rr', 0):.2f}
Average R:          {stats.get('avg_r', 0):+.2f}R
Total R:            {stats.get('total_r', 0):+.2f}R

Profit Factor:      {stats.get('profit_factor', 0):.2f}
Max Drawdown:       {stats.get('max_drawdown_r', 0):.2f}R
Max Consec. Losses: {stats.get('max_consec_losses', 0)}
Max Consec. Wins:   {stats.get('max_consec_wins', 0)}

━━━━━━━━━━━━━━

🪙 COIN PERFORMANCE
"""
    for coin, cs in coin_stats.items():
        report += f"\n{coin}: {cs.get('tp',0)}W / {cs.get('sl',0)}L | {cs.get('total_r',0):+.2f}R"

    report += f"""

━━━━━━━━━━━━━━

⏱ TIMEFRAME

INTRADAY | 15M: {tf_stats.get('INTRADAY',{}).get('tp',0)}W / {tf_stats.get('INTRADAY',{}).get('sl',0)}L
SWING | 1H:     {tf_stats.get('SWING',{}).get('tp',0)}W / {tf_stats.get('SWING',{}).get('sl',0)}L

━━━━━━━━━━━━━━

📈 DIRECTION

LONG:  {dir_stats.get('LONG',{}).get('tp',0)}W / {dir_stats.get('LONG',{}).get('sl',0)}L
SHORT: {dir_stats.get('SHORT',{}).get('tp',0)}W / {dir_stats.get('SHORT',{}).get('sl',0)}L

━━━━━━━━━━━━━━

💧 LIQUIDITY

Best liquidity setup:  {liq_stats.get('best', 'N/A')}
Weakest liquidity:     {liq_stats.get('worst', 'N/A')}

━━━━━━━━━━━━━━

🔎 OBSERVATIONS
"""
    for i, obs in enumerate(observations[:5], 1):
        report += f"\n{i}. {obs}"

    report += f"""

━━━━━━━━━━━━━━

🧠 LEARNING
"""
    for i, item in enumerate(learning[:5], 1):
        report += f"\n{i}. {item}"

    report += f"""

━━━━━━━━━━━━━━

🛠 SUGGESTED NEXT VERSION
"""
    if suggestions:
        for i, s in enumerate(suggestions[:3], 1):
            report += f"\n{i}. {s}"
    else:
        report += "\n⚠️ INSUFFICIENT DATA\nContinue paper trading.\nNo strategy modification recommended."

    report += f"""

━━━━━━━━━━━━━━
⚠️ NO STRATEGY CHANGES WERE
APPLIED AUTOMATICALLY."""

    return report


def format_status_message(
    bot_version: str,
    last_scan: str,
    active_count: int,
    monitored_symbols: list,
    mode: str,
) -> str:
    return (
        f"🤖 BOT STATUS\n\n"
        f"Version: {bot_version}\n"
        f"Mode: {mode}\n"
        f"Last Scan: {last_scan}\n"
        f"Active Signals: {active_count}\n"
        f"Monitoring: {', '.join(monitored_symbols)}\n\n"
        f"⚠️ SIGNAL-ONLY BOT\n"
        f"Does NOT place orders."
    )


def format_recommendation(rec: dict) -> str:
    return (
        f"📋 STRATEGY RECOMMENDATION\n\n"
        f"Current Rule:\n{rec.get('current_rule','')}\n\n"
        f"Observation:\n{rec.get('observation','')}\n\n"
        f"Evidence:\n{rec.get('evidence','')}\n\n"
        f"Proposed Change:\n{rec.get('proposed_change','')}\n\n"
        f"Expected Benefit:\n{rec.get('expected_benefit','')}\n\n"
        f"Possible Downside:\n{rec.get('possible_downside','')}\n\n"
        f"Trades Analyzed: {rec.get('trades_analyzed',0)}\n"
        f"Confidence: {rec.get('confidence','')}\n"
        f"Backtest Required: {'Yes' if rec.get('backtest_required') else 'No'}\n\n"
        f"Status: {rec.get('status','PENDING_REVIEW')}\n"
        f"ID: {rec.get('rec_id','')}\n\n"
        f"━━━━━━━━━━━━━━\n"
        f"⚠️ NO CHANGES APPLIED AUTOMATICALLY.\n"
        f"Human approval required."
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_bias(description: str) -> str:
    if "BULLISH" in description.upper():
        return "Bullish"
    elif "BEARISH" in description.upper():
        return "Bearish"
    return "Neutral"


def _format_price(symbol: str, price: float) -> str:
    """Format price with appropriate decimal places."""
    if price is None:
        return "N/A"
    if "BTC" in symbol:
        return f"{price:,.2f}"
    elif "ETH" in symbol:
        return f"{price:,.2f}"
    else:
        return f"{price:,.4f}"


def _format_m15_lines(m15_summary: str) -> str:
    """Convert pipe-separated 15M summary to line-per-item."""
    return "\n".join(part.strip() for part in m15_summary.split("|") if part.strip())
