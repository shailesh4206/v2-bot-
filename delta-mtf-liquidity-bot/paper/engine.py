"""
paper/engine.py — Paper Trading Engine.

Uses REAL live market data but NEVER places orders.
Tracks signals, entries, TP, SL, expiry in the same journal as live mode
but with is_paper=1 flag.

Run with: python paper.py
"""

import logging
import time
from datetime import datetime, timezone

from signals.generator import generate_signals
from signals.validator import validate_signal, approve_and_register
from signals.lifecycle import SignalMonitor
from database.journal import save_signal
from database.models import initialize_database
from telegram.bot import send_signal
from telegram.commands import set_last_scan_time
from config.settings import SCAN_INTERVAL_SECONDS, STRATEGY_VERSION
from config.strategy_config import INTRADAY_EXPIRY_MINUTES

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Paper trading engine — mirrors live mode exactly except:
    - Signals marked as is_paper=True in DB
    - Weekly stats tracked separately
    - No impact on live signal dedup guard (optional — configurable)
    """

    def __init__(self):
        self._running = False
        self._monitor = SignalMonitor()
        self._scan_count = 0

    def start(self):
        """Start the paper trading loop."""
        logger.info(f"📄 Paper Trading Engine started | Strategy {STRATEGY_VERSION}")
        initialize_database()
        self._running = True

        while self._running:
            try:
                self._scan()
                self._monitor.check_all_active()
            except Exception as e:
                logger.error(f"Paper scan error: {e}", exc_info=True)

            logger.info(f"Paper scan #{self._scan_count} complete. Next in {SCAN_INTERVAL_SECONDS}s …")
            time.sleep(SCAN_INTERVAL_SECONDS)

    def stop(self):
        self._running = False

    def _scan(self):
        self._scan_count += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        set_last_scan_time(f"PAPER #{self._scan_count} @ {now_iso}")

        setups = generate_signals()
        logger.info(f"Paper scan: {len(setups)} setup(s) found")

        for setup in setups:
            approved, reason, _ = validate_signal(setup)
            if not approved:
                logger.debug(f"[PAPER] {setup.symbol} {setup.direction} rejected: {reason}")
                continue

            import uuid
            signal_id = "P" + str(uuid.uuid4())[:7].upper()

            # Save as paper signal
            save_signal(setup, signal_id=signal_id)

            # Override is_paper flag directly
            from database.models import get_connection
            conn = get_connection()
            conn.execute("UPDATE signals SET is_paper = 1 WHERE signal_id = ?", (signal_id,))
            conn.commit()
            conn.close()

            # Register in duplicate guard
            approve_and_register(setup, signal_id)

            # Send to Telegram with [PAPER] prefix
            from telegram.bot import send_message_sync
            from telegram.messages import format_signal_message
            msg = "[📄 PAPER TRADE]\n\n" + format_signal_message(setup, signal_id)
            send_message_sync(msg)
            logger.info(f"[PAPER] Signal sent: {signal_id} {setup.symbol} {setup.direction}")
