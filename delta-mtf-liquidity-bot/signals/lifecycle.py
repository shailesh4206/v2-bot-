"""
signals/lifecycle.py — Signal lifecycle monitoring.

After a signal is sent, monitors:
- Has entry price been reached?
- Has TP been hit?
- Has SL been hit?
- Has the signal expired?

Uses REST polling + optional WebSocket pricing.
Only confirmed market price data marks outcomes — never guessed.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List

from database.journal import (
    get_active_signals, record_outcome, update_signal_state, get_signal
)
from market_data.delta_client import DeltaClient
from telegram.bot import send_outcome_notification

logger = logging.getLogger(__name__)


class SignalMonitor:
    """
    Monitors all active signals and updates their lifecycle state.

    Can be called periodically (polling mode) or
    driven by WebSocket price callbacks.
    """

    def __init__(self, price_monitor=None):
        """
        Args:
            price_monitor: Optional PriceMonitor (websocket) instance.
                           If None, uses REST polling for prices.
        """
        self._price_monitor = price_monitor
        self._client = DeltaClient()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current market price for a symbol."""
        # Try WebSocket first (faster)
        if self._price_monitor:
            price = self._price_monitor.get_price(symbol)
            if price:
                return price

        # Fallback to REST
        try:
            ticker = self._client.get_ticker(symbol)
            if ticker:
                price = (
                    ticker.get("close") or
                    ticker.get("last_price") or
                    ticker.get("mark_price")
                )
                if price:
                    return float(price)
        except Exception as e:
            logger.warning(f"Could not get price for {symbol}: {e}")

        return None

    def check_all_active(self):
        """Check all active signals and update their states."""
        active = get_active_signals()

        if not active:
            return

        logger.info(f"Checking {len(active)} active signals …")

        for signal in active:
            try:
                self._check_signal(signal)
            except Exception as e:
                logger.error(f"Error checking signal {signal.get('signal_id')}: {e}")

    def _check_signal(self, signal: Dict):
        """Check a single signal against current market price."""
        signal_id  = signal["signal_id"]
        symbol     = signal["symbol"]
        direction  = signal["direction"]
        state      = signal["state"]
        entry      = signal["entry"]
        sl         = signal["sl"]
        tp         = signal["tp"]
        expiry_ts  = signal["expiry_timestamp"]

        # ── Expiry check ──────────────────────────────────────────────────
        if expiry_ts:
            now = datetime.now(timezone.utc)
            exp = datetime.fromisoformat(expiry_ts)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)

            if now > exp and state in ("SIGNAL_SENT", "DETECTED"):
                logger.info(f"Signal {signal_id} EXPIRED")
                record_outcome(
                    signal_id=signal_id,
                    outcome="EXPIRED",
                    exit_price=0.0,
                    exit_timestamp=now.isoformat(),
                    r_multiple=0.0,
                    notes="Signal expired — entry not reached in time",
                )
                try:
                    send_outcome_notification(signal_id, "EXPIRED", 0.0, 0.0)
                except Exception:
                    pass
                return

        # ── Get current price ─────────────────────────────────────────────
        current_price = self.get_current_price(symbol)
        if current_price is None:
            logger.debug(f"No price for {symbol} — skipping")
            return

        # ── Entry check ───────────────────────────────────────────────────
        if state == "SIGNAL_SENT":
            entry_hit = False
            if direction == "LONG" and current_price <= entry:
                entry_hit = True
            elif direction == "SHORT" and current_price >= entry:
                entry_hit = True

            if entry_hit:
                logger.info(f"Signal {signal_id}: ENTRY REACHED @ {current_price}")
                update_signal_state(signal_id, "ACTIVE")
                record_outcome(
                    signal_id=signal_id,
                    outcome="ACTIVE",
                    exit_price=current_price,
                    entry_timestamp=datetime.now(timezone.utc).isoformat(),
                    r_multiple=None,
                    notes=f"Entry reached @ {current_price}",
                )
            return  # Don't check TP/SL until entry is reached

        # ── TP / SL check ─────────────────────────────────────────────────
        if state == "ACTIVE":
            now_iso = datetime.now(timezone.utc).isoformat()
            entry_ts = signal.get("entry_timestamp", now_iso)

            # Calculate R-multiple
            risk = abs(entry - sl) if entry and sl else 1.0

            if direction == "LONG":
                tp_hit = current_price >= tp
                sl_hit = current_price <= sl
                r_at_current = (current_price - entry) / risk if risk > 0 else 0
            else:
                tp_hit = current_price <= tp
                sl_hit = current_price >= sl
                r_at_current = (entry - current_price) / risk if risk > 0 else 0

            if tp_hit:
                r_multiple = abs(tp - entry) / risk if risk > 0 else 0
                logger.info(f"Signal {signal_id}: TP HIT @ {current_price} (+{r_multiple:.2f}R)")
                record_outcome(
                    signal_id=signal_id,
                    outcome="TP_HIT",
                    exit_price=current_price,
                    entry_timestamp=entry_ts,
                    exit_timestamp=now_iso,
                    r_multiple=round(r_multiple, 2),
                    notes=f"TP hit @ {current_price}",
                )
                try:
                    send_outcome_notification(signal_id, "TP_HIT", current_price, r_multiple)
                except Exception:
                    pass

            elif sl_hit:
                r_multiple = -1.0  # Full stop = -1R by convention
                logger.info(f"Signal {signal_id}: SL HIT @ {current_price} (-1R)")
                record_outcome(
                    signal_id=signal_id,
                    outcome="SL_HIT",
                    exit_price=current_price,
                    entry_timestamp=entry_ts,
                    exit_timestamp=now_iso,
                    r_multiple=-1.0,
                    notes=f"SL hit @ {current_price}",
                )
                try:
                    send_outcome_notification(signal_id, "SL_HIT", current_price, -1.0)
                except Exception:
                    pass
