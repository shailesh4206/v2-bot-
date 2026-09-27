"""
main.py — Bot orchestrator and main entry point.

Runs:
1. Signal scan loop (every 15M candle close)
2. Signal lifecycle monitor (outcome tracking)
3. Weekly report scheduler (every Monday 09:00 IST)
4. Telegram command polling

Signal-only bot. Does NOT place orders.
Uses ONLY 4H + 1H + 15M. NO 5M anywhere.
"""

import sys
import time
import signal
import logging
import threading
from datetime import datetime, timezone

from database.models import initialize_database
from signals.generator import generate_signals
from signals.validator import validate_signal, approve_and_register
from signals.lifecycle import SignalMonitor
from database.journal import save_signal
from telegram.bot import send_signal, send_message_sync
from telegram.commands import set_last_scan_time, dispatch_command
from weekly.report import generate_weekly_report
from config.settings import (
    SCAN_INTERVAL_SECONDS, PAPER_MODE, LIVE_SIGNAL_MODE,
    STRATEGY_VERSION, MONITORED_SYMBOLS, LOG_FILE, LOG_LEVEL,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
import os
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class DeltaMTFBot:
    """
    Main bot orchestrator.

    SIGNAL ONLY. NO ORDER PLACEMENT.
    """

    def __init__(self):
        self._running      = False
        self._monitor      = SignalMonitor()
        self._scan_count   = 0
        self._last_weekly  = None

    def start(self):
        """Start all bot components."""
        logger.info("=" * 60)
        logger.info(f"🚀 DELTA MTF LIQUIDITY BOT STARTING")
        logger.info(f"   Strategy:  {STRATEGY_VERSION}")
        logger.info(f"   Mode:      {'PAPER' if PAPER_MODE else 'LIVE SIGNAL'}")
        logger.info(f"   Symbols:   {MONITORED_SYMBOLS}")
        logger.info(f"   Timeframes: 4H | 1H | 15M (NO 5M)")
        logger.info(f"   ⚠️  SIGNAL-ONLY — BOT DOES NOT PLACE ORDERS")
        logger.info("=" * 60)

        # Init DB
        initialize_database()

        if PAPER_MODE:
            from paper.engine import PaperTradingEngine
            PaperTradingEngine().start()
            return

        if LIVE_SIGNAL_MODE:
            self._running = True
            self._start_telegram_polling()
            self._run_main_loop()

    def _run_main_loop(self):
        """Main scan + lifecycle loop."""
        send_message_sync(
            f"🟢 BOT ONLINE\n\n"
            f"Strategy: {STRATEGY_VERSION}\n"
            f"Monitoring: {', '.join(MONITORED_SYMBOLS)}\n"
            f"Timeframes: 4H | 1H | 15M\n\n"
            f"⚠️ SIGNAL-ONLY MODE\n"
            f"Bot does NOT place orders."
        )

        while self._running:
            try:
                self._do_scan()
                self._monitor.check_all_active()
                self._check_weekly_report()
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)

            logger.info(f"Scan #{self._scan_count} complete. Sleeping {SCAN_INTERVAL_SECONDS}s …")
            time.sleep(SCAN_INTERVAL_SECONDS)

    def _do_scan(self):
        """Run one full signal scan cycle."""
        self._scan_count += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        set_last_scan_time(f"Scan #{self._scan_count} @ {now_iso[:19]}Z")
        logger.info(f"━━━ Scan #{self._scan_count} @ {now_iso[:19]}Z ━━━")

        setups = generate_signals()
        logger.info(f"Found {len(setups)} valid setup(s)")

        for setup in setups:
            try:
                approved, reason, _ = validate_signal(setup)

                if not approved:
                    logger.info(f"  ⛔ {setup.symbol} {setup.direction} rejected: {reason}")
                    continue

                import uuid
                signal_id = str(uuid.uuid4())[:8].upper()

                # Save to journal
                save_signal(setup, signal_id=signal_id)

                # Register in duplicate guard
                approve_and_register(setup, signal_id)

                # Send to Telegram
                ok = send_signal(setup, signal_id)
                if ok:
                    logger.info(f"  ✅ Signal sent: {signal_id} {setup.symbol} {setup.direction} {setup.mode} Q:{setup.quality.value}")
                else:
                    logger.warning(f"  ⚠️  Telegram send failed for {signal_id}")

            except Exception as e:
                logger.error(f"Error processing setup for {setup.symbol}: {e}", exc_info=True)

    def _check_weekly_report(self):
        """Check if it's time to send the weekly report (Monday 09:00 IST)."""
        from config.settings import WEEKLY_REPORT_DAY, WEEKLY_REPORT_HOUR, WEEKLY_REPORT_MINUTE
        now = datetime.now(timezone.utc)
        # IST = UTC+5:30
        ist_hour   = (now.hour + 5) % 24
        ist_minute = (now.minute + 30) % 60

        is_monday = now.weekday() == WEEKLY_REPORT_DAY
        is_report_time = (
            ist_hour == WEEKLY_REPORT_HOUR and
            WEEKLY_REPORT_MINUTE <= ist_minute < WEEKLY_REPORT_MINUTE + 20
        )
        today_str = now.strftime("%Y-%m-%d")

        if is_monday and is_report_time and self._last_weekly != today_str:
            logger.info("📊 Generating weekly report …")
            try:
                generate_weekly_report(send_to_telegram=True)
                self._last_weekly = today_str
            except Exception as e:
                logger.error(f"Weekly report error: {e}")

    def _start_telegram_polling(self):
        """Start Telegram update polling in a background thread."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials missing — command polling disabled")
            return

        thread = threading.Thread(target=self._poll_telegram, daemon=True, name="TelegramPoller")
        thread.start()

    def _poll_telegram(self):
        """Poll Telegram for incoming commands."""
        import requests

        offset = 0
        logger.info("Telegram command polling started")

        while self._running:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
                resp = requests.get(url, params={"offset": offset, "timeout": 10}, timeout=15)

                if not resp.ok:
                    time.sleep(5)
                    continue

                updates = resp.json().get("result", [])
                for update in updates:
                    offset = update["update_id"] + 1
                    msg = update.get("message") or update.get("channel_post")
                    if msg:
                        text = msg.get("text", "")
                        if text.startswith("/"):
                            logger.info(f"Command received: {text}")
                            try:
                                dispatch_command(text.strip())
                            except Exception as e:
                                logger.error(f"Command error: {e}")

            except Exception as e:
                logger.debug(f"Telegram poll error: {e}")
                time.sleep(5)


def handle_shutdown(signum, frame):
    logger.info("Shutdown signal received")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    bot = DeltaMTFBot()
    bot.start()
