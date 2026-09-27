"""
telegram/commands.py — Telegram bot command handlers.

Commands:
  /status       — Bot health and active signals count
  /signals      — Recent signals (last 24h)
  /active       — Currently active (ENTRY_REACHED) signals
  /history      — Last N completed signals with outcomes
  /weekly       — Latest weekly report
  /performance  — Summary stats
  /learning     — Detected performance patterns
  /recommendations — Pending strategy improvements
  /version      — Current strategy version

All commands are informational. No order placement possible.
"""

import logging
from datetime import datetime, timezone, timedelta
from config.settings import STRATEGY_VERSION, MONITORED_SYMBOLS
from database.journal import (
    get_active_signals, get_recent_signals, get_completed_signals,
    get_latest_weekly_report, get_recent_observations, get_pending_recommendations
)
from telegram.bot import send_message_sync

logger = logging.getLogger(__name__)

# Store last scan time (updated by main.py)
_last_scan_time: str = "Not scanned yet"


def set_last_scan_time(ts: str):
    global _last_scan_time
    _last_scan_time = ts


def handle_status():
    active = get_active_signals()
    text = (
        f"🤖 BOT STATUS\n\n"
        f"Strategy Version: {STRATEGY_VERSION}\n"
        f"Last Scan: {_last_scan_time}\n"
        f"Active Signals: {len(active)}\n"
        f"Monitoring: {', '.join(MONITORED_SYMBOLS)}\n"
        f"Timeframes: 4H | 1H | 15M\n\n"
        f"⚠️ SIGNAL-ONLY MODE\n"
        f"Does NOT place orders."
    )
    send_message_sync(text)


def handle_signals():
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    signals = get_recent_signals(limit=20)
    recent = [s for s in signals if (s.get("signal_timestamp") or "") >= since]

    if not recent:
        send_message_sync("📭 No signals in the last 24 hours.")
        return

    lines = [f"📊 SIGNALS (Last 24h)\n\n"]
    for s in recent[:10]:
        outcome = s.get("outcome") or s.get("state") or "PENDING"
        lines.append(
            f"• {s['symbol']} {s['direction']} [{s['mode']}] "
            f"Quality: {s['quality']} | {outcome} | RR: 1:{s['rr']}\n"
            f"  ID: {s['signal_id']}"
        )
    send_message_sync("\n".join(lines))


def handle_active():
    active = get_active_signals()
    if not active:
        send_message_sync("✅ No active signals right now.")
        return

    lines = [f"🔴 ACTIVE SIGNALS ({len(active)})\n\n"]
    for s in active:
        lines.append(
            f"• {s['symbol']} {s['direction']} | Entry: {s['entry']:.4f}\n"
            f"  SL: {s['sl']:.4f} | TP: {s['tp']:.4f}\n"
            f"  ID: {s['signal_id']}"
        )
    send_message_sync("\n".join(lines))


def handle_history(n: int = 10):
    completed = get_completed_signals()[-n:]
    if not completed:
        send_message_sync("📭 No completed trades yet.")
        return

    lines = [f"📜 TRADE HISTORY (Last {n})\n\n"]
    for s in completed:
        r = s.get("r_multiple")
        r_str = f"{r:+.2f}R" if r is not None else "?"
        lines.append(
            f"• {s['symbol']} {s['direction']} | {s.get('outcome','?')} | {r_str}\n"
            f"  Quality: {s['quality']} | Score: {s['sweep_score']}"
        )
    send_message_sync("\n".join(lines))


def handle_weekly():
    report = get_latest_weekly_report()
    if not report:
        send_message_sync("📭 No weekly report available yet. Reports are sent every Monday.")
        return
    from telegram.bot import send_weekly_report
    send_weekly_report(report["report_text"])


def handle_performance():
    completed = get_completed_signals()
    if not completed:
        send_message_sync(
            "⚠️ INSUFFICIENT DATA\n\n"
            "No completed trades to analyze.\n"
            "Continue paper trading."
        )
        return

    tp   = [s for s in completed if s.get("outcome") == "TP_HIT"]
    sl   = [s for s in completed if s.get("outcome") == "SL_HIT"]
    exp  = [s for s in completed if s.get("outcome") == "EXPIRED"]
    total = len(completed)
    entries = len([s for s in completed if s.get("outcome") in ("TP_HIT","SL_HIT")])

    r_vals = [s["r_multiple"] for s in completed if s.get("r_multiple") is not None]
    total_r  = sum(r_vals)
    avg_r    = total_r / len(r_vals) if r_vals else 0
    win_rate = len(tp) / entries * 100 if entries > 0 else 0

    pos_r = sum(r for r in r_vals if r > 0)
    neg_r = abs(sum(r for r in r_vals if r < 0))
    pf    = pos_r / neg_r if neg_r > 0 else float("inf")

    send_message_sync(
        f"📈 PERFORMANCE SUMMARY\n\n"
        f"Total Signals: {total}\n"
        f"Entries Reached: {entries}\n"
        f"TP Hits: {len(tp)}\n"
        f"SL Hits: {len(sl)}\n"
        f"Expired: {len(exp)}\n\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Total R: {total_r:+.2f}\n"
        f"Avg R: {avg_r:+.2f}\n"
        f"Profit Factor: {pf:.2f}\n\n"
        f"⚠️ SIGNAL-ONLY BOT\nDoes NOT place orders."
    )


def handle_learning():
    obs = get_recent_observations(limit=10)
    if not obs:
        send_message_sync("🧠 No learning observations yet. Need more completed trade data.")
        return

    lines = ["🧠 LEARNING OBSERVATIONS\n\n"]
    for o in obs[:7]:
        lines.append(f"• {o.get('observation_text','')}")
    send_message_sync("\n".join(lines))


def handle_recommendations():
    recs = get_pending_recommendations()
    if not recs:
        send_message_sync("📋 No pending recommendations. Monitoring for sufficient data …")
        return

    for rec in recs[:3]:
        from telegram.messages import format_recommendation
        send_message_sync(format_recommendation(rec))


def handle_version():
    send_message_sync(
        f"📦 STRATEGY VERSION\n\n"
        f"Current: {STRATEGY_VERSION}\n"
        f"Timeframes: 4H | 1H | 15M\n"
        f"Modes: INTRADAY | SWING\n\n"
        f"⚠️ No automatic version changes.\n"
        f"Human approval required for all updates."
    )


# Command dispatcher
COMMAND_MAP = {
    "/status":          handle_status,
    "/signals":         handle_signals,
    "/active":          handle_active,
    "/history":         handle_history,
    "/weekly":          handle_weekly,
    "/performance":     handle_performance,
    "/learning":        handle_learning,
    "/recommendations": handle_recommendations,
    "/version":         handle_version,
}


def dispatch_command(command: str):
    """Dispatch an incoming Telegram command to its handler."""
    handler = COMMAND_MAP.get(command.split()[0].lower())
    if handler:
        try:
            handler()
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            send_message_sync(f"❌ Error processing {command}: {str(e)[:100]}")
    else:
        send_message_sync(
            f"Unknown command: {command}\n\n"
            f"Available: {', '.join(COMMAND_MAP.keys())}"
        )
