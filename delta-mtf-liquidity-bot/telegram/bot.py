"""
telegram/bot.py — Telegram bot initialization and message sending.

All messages include the disclaimer: ⚠️ MANUAL EXECUTION ONLY.
Bot NEVER places orders.
"""

import logging
import asyncio
from typing import Optional

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

# Lazy import of telegram library
_application = None


def _get_application():
    global _application
    if _application is None:
        try:
            from telegram.ext import Application
            _application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        except ImportError:
            logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            raise
    return _application


def send_message_sync(text: str, chat_id: Optional[str] = None) -> bool:
    """
    Send a Telegram message synchronously.
    Wraps the async send in a new event loop.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set — message not sent")
        logger.info(f"[TELEGRAM MOCK]\n{text}")
        return False

    target = chat_id or TELEGRAM_CHAT_ID

    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": target,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        # Fallback to plain text if HTML fails
        resp = requests.post(url, json=payload, timeout=10)
        if not resp.ok:
            payload["parse_mode"] = "Markdown"
            resp = requests.post(url, json=payload, timeout=10)
        if not resp.ok:
            del payload["parse_mode"]
            resp = requests.post(url, json=payload, timeout=10)

        if resp.ok:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Telegram API error: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram send error: {e}")
        return False


def send_signal(setup, signal_id: str) -> bool:
    """Format and send a signal to Telegram."""
    from telegram.messages import format_signal_message
    text = format_signal_message(setup, signal_id)
    return send_message_sync(text)


def send_outcome_notification(
    signal_id:  str,
    outcome:    str,
    exit_price: float,
    r_multiple: float,
) -> bool:
    """Send a TP/SL/EXPIRED outcome notification."""
    from telegram.messages import format_outcome_message
    text = format_outcome_message(signal_id, outcome, exit_price, r_multiple)
    return send_message_sync(text)


def send_weekly_report(report_text: str) -> bool:
    """Send weekly report to Telegram."""
    # Split into chunks if too long (Telegram limit = 4096 chars)
    MAX_LEN = 4000
    if len(report_text) <= MAX_LEN:
        return send_message_sync(report_text)

    chunks = [report_text[i:i+MAX_LEN] for i in range(0, len(report_text), MAX_LEN)]
    success = True
    for i, chunk in enumerate(chunks):
        prefix = f"[{i+1}/{len(chunks)}]\n" if len(chunks) > 1 else ""
        if not send_message_sync(prefix + chunk):
            success = False
    return success
