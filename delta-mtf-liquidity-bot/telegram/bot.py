"""
telegram/bot.py — Telegram transport layer.

Responsibilities:
- Send validated trading signals
- Send signal lifecycle outcome notifications
- Send weekly reports
- Send generic synchronous messages

SIGNAL ONLY.
No order placement is performed here.
"""

import logging
from typing import Optional

import requests

from config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)

from telegram.messages import (
    format_signal_message,
    format_outcome_message,
)


logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot"


def _telegram_configured() -> bool:
    """Check whether Telegram credentials are configured."""
    return bool(
        TELEGRAM_BOT_TOKEN
        and TELEGRAM_CHAT_ID
    )


def is_telegram_configured() -> bool:
    """Public Telegram configuration check."""
    return _telegram_configured()


def _send_text(
    text: str,
    chat_id: Optional[str] = None,
) -> bool:
    """Send a plain text Telegram message."""

    if not text:
        logger.warning(
            "Telegram message is empty — not sending."
        )
        return False

    if not _telegram_configured():
        logger.warning(
            "Telegram credentials missing — message not sent."
        )
        return False

    target_chat_id = chat_id or TELEGRAM_CHAT_ID

    if not target_chat_id:
        logger.warning(
            "Telegram chat ID missing — message not sent."
        )
        return False

    url = (
        f"{TELEGRAM_API_BASE}"
        f"{TELEGRAM_BOT_TOKEN}"
        f"/sendMessage"
    )

    payload = {
        "chat_id": target_chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=15,
        )

        if not response.ok:
            logger.error(
                "Telegram API HTTP error: status=%s response=%s",
                response.status_code,
                response.text[:500],
            )
            return False

        try:
            data = response.json()
        except ValueError:
            logger.error(
                "Telegram API returned invalid JSON: %s",
                response.text[:500],
            )
            return False

        if not data.get("ok"):
            logger.error(
                "Telegram API rejected message: %s",
                data,
            )
            return False

        logger.debug(
            "Telegram message sent successfully."
        )

        return True

    except requests.RequestException as e:
        logger.error(
            "Telegram network error: %s",
            e,
        )
        return False

    except Exception as e:
        logger.error(
            "Unexpected Telegram error: %s",
            e,
            exc_info=True,
        )
        return False


def send_message_sync(text: str) -> bool:
    """
    Send a generic Telegram message synchronously.
    """

    return _send_text(text)


def send_signal(
    setup,
    signal_id: str,
) -> bool:
    """
    Format and send a validated trading signal.
    """

    try:
        message = format_signal_message(
            setup,
            signal_id,
        )

        return _send_text(message)

    except Exception as e:
        logger.error(
            "Failed to format/send signal %s: %s",
            signal_id,
            e,
            exc_info=True,
        )
        return False


def send_outcome_notification(
    signal_id: str,
    outcome: str,
    exit_price: float,
    r_multiple: float,
) -> bool:
    """
    Send signal lifecycle outcome notification.

    Supported outcomes:
        TP_HIT
        SL_HIT
        EXPIRED
    """

    try:
        message = format_outcome_message(
            signal_id,
            outcome,
            exit_price,
            r_multiple,
        )

        return _send_text(message)

    except Exception as e:
        logger.error(
            "Failed to send outcome notification "
            "for signal %s: %s",
            signal_id,
            e,
            exc_info=True,
        )
        return False


def send_weekly_report(
    report_text: str,
) -> bool:
    """
    Send generated weekly performance report.
    """

    if not report_text:
        logger.warning(
            "Weekly report is empty — not sending."
        )
        return False

    return _send_text(report_text)


if __name__ == "__main__":
    print("=" * 72)
    print("TELEGRAM BOT MODULE SELF-TEST")
    print("=" * 72)

    print(
        "Credentials configured :",
        _telegram_configured(),
    )

    print(
        "send_message_sync      :",
        callable(send_message_sync),
    )

    print(
        "send_signal            :",
        callable(send_signal),
    )

    print(
        "send_outcome_notification :",
        callable(send_outcome_notification),
    )

    print(
        "send_weekly_report     :",
        callable(send_weekly_report),
    )

    print(
        "is_telegram_configured :",
        callable(is_telegram_configured),
    )

    print("=" * 72)
    print(
        "Telegram transport module loaded successfully."
    )
    print("=" * 72)