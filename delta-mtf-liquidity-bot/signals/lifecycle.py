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
from datetime import datetime, timezone
from typing import Dict, Optional, Any

from database.journal import (
    get_active_signals,
    record_outcome,
    update_signal_state,
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

    # ------------------------------------------------------------------
    # PRICE NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_price_from_response(response: Any) -> Optional[float]:
        """
        Safely extract a market price from Delta ticker response.

        Delta responses may arrive as:
        - dict
        - list containing a dict
        - nested dict/list structures

        Never assumes `.get()` exists on the root object.
        """

        if response is None:
            return None

        candidates = []

        if isinstance(response, dict):
            candidates.append(response)

            # Common nested API wrappers
            for key in ("result", "data", "ticker"):
                value = response.get(key)

                if isinstance(value, dict):
                    candidates.append(value)

                elif isinstance(value, list):
                    candidates.extend(
                        item for item in value
                        if isinstance(item, dict)
                    )

        elif isinstance(response, list):
            candidates.extend(
                item for item in response
                if isinstance(item, dict)
            )

        for item in candidates:
            for key in (
                "close",
                "last_price",
                "mark_price",
                "price",
                "last",
            ):
                value = item.get(key)

                if value is None:
                    continue

                try:
                    price = float(value)

                    if price > 0:
                        return price

                except (TypeError, ValueError):
                    continue

        return None

    # ------------------------------------------------------------------
    # CURRENT PRICE
    # ------------------------------------------------------------------

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current market price for a symbol."""

        # --------------------------------------------------------------
        # Try WebSocket first
        # --------------------------------------------------------------
        if self._price_monitor:
            try:
                price = self._price_monitor.get_price(symbol)

                if price is not None:
                    price = float(price)

                    if price > 0:
                        return price

            except Exception as e:
                logger.debug(
                    f"WebSocket price unavailable for {symbol}: {e}"
                )

        # --------------------------------------------------------------
        # REST fallback
        # --------------------------------------------------------------
        try:
            ticker = self._client.get_ticker(symbol)

            price = self._extract_price_from_response(ticker)

            if price is not None:
                return price

            logger.warning(
                f"Could not extract valid price for {symbol} "
                f"from ticker response type={type(ticker).__name__}"
            )

        except Exception as e:
            logger.warning(
                f"Could not get price for {symbol}: {e}"
            )

        return None

    # ------------------------------------------------------------------
    # ACTIVE SIGNALS
    # ------------------------------------------------------------------

    def check_all_active(self):
        """Check all active signals and update their states."""

        active = get_active_signals()

        if not active:
            return

        logger.info(
            f"Checking {len(active)} active signals …"
        )

        for signal in active:
            try:
                self._check_signal(signal)

            except Exception as e:
                signal_id = (
                    signal.get("signal_id")
                    if isinstance(signal, dict)
                    else "UNKNOWN"
                )

                logger.error(
                    f"Error checking signal {signal_id}: {e}"
                )

    # ------------------------------------------------------------------
    # SINGLE SIGNAL
    # ------------------------------------------------------------------

    def _check_signal(self, signal: Dict):
        """Check a single signal against current market price."""

        signal_id = signal["signal_id"]
        symbol = signal["symbol"]
        direction = signal["direction"]
        state = signal["state"]

        entry = float(signal["entry"])
        sl = float(signal["sl"])
        tp = float(signal["tp"])

        expiry_ts = signal.get("expiry_timestamp")

        # --------------------------------------------------------------
        # EXPIRY CHECK
        # --------------------------------------------------------------

        if expiry_ts:
            now = datetime.now(timezone.utc)

            exp = datetime.fromisoformat(expiry_ts)

            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)

            if now > exp and state in ("SIGNAL_SENT", "DETECTED"):

                logger.info(
                    f"Signal {signal_id} EXPIRED"
                )

                record_outcome(
                    signal_id=signal_id,
                    outcome="EXPIRED",
                    exit_price=0.0,
                    exit_timestamp=now.isoformat(),
                    r_multiple=0.0,
                    notes="Signal expired — entry not reached in time",
                )

                try:
                    send_outcome_notification(
                        signal_id,
                        "EXPIRED",
                        0.0,
                        0.0,
                    )
                except Exception:
                    pass

                return

        # --------------------------------------------------------------
        # GET CURRENT PRICE
        # --------------------------------------------------------------

        current_price = self.get_current_price(symbol)

        if current_price is None:
            logger.debug(
                f"No valid price for {symbol} — skipping"
            )
            return

        # --------------------------------------------------------------
        # ENTRY CHECK
        # --------------------------------------------------------------

        if state == "SIGNAL_SENT":

            entry_hit = False

            if direction == "LONG":
                if current_price <= entry:
                    entry_hit = True

            elif direction == "SHORT":
                if current_price >= entry:
                    entry_hit = True

            if entry_hit:

                logger.info(
                    f"Signal {signal_id}: "
                    f"ENTRY REACHED @ {current_price}"
                )

                update_signal_state(
                    signal_id,
                    "ACTIVE",
                )

                record_outcome(
                    signal_id=signal_id,
                    outcome="ACTIVE",
                    exit_price=current_price,
                    entry_timestamp=datetime.now(
                        timezone.utc
                    ).isoformat(),
                    r_multiple=None,
                    notes=f"Entry reached @ {current_price}",
                )

            # Do not check TP/SL until entry is reached
            return

        # --------------------------------------------------------------
        # TP / SL CHECK
        # --------------------------------------------------------------

        if state == "ACTIVE":

            now_iso = datetime.now(
                timezone.utc
            ).isoformat()

            entry_ts = signal.get(
                "entry_timestamp",
                now_iso,
            )

            risk = (
                abs(entry - sl)
                if entry is not None and sl is not None
                else 1.0
            )

            if risk <= 0:
                logger.warning(
                    f"Signal {signal_id}: invalid risk distance"
                )
                return

            # ----------------------------------------------------------
            # LONG
            # ----------------------------------------------------------

            if direction == "LONG":

                tp_hit = current_price >= tp
                sl_hit = current_price <= sl

            # ----------------------------------------------------------
            # SHORT
            # ----------------------------------------------------------

            else:

                tp_hit = current_price <= tp
                sl_hit = current_price >= sl

            # ----------------------------------------------------------
            # TP HIT
            # ----------------------------------------------------------

            if tp_hit:

                r_multiple = (
                    abs(tp - entry) / risk
                    if risk > 0
                    else 0
                )

                logger.info(
                    f"Signal {signal_id}: "
                    f"TP HIT @ {current_price} "
                    f"(+{r_multiple:.2f}R)"
                )

                record_outcome(
                    signal_id=signal_id,
                    outcome="TP_HIT",
                    exit_price=current_price,
                    entry_timestamp=entry_ts,
                    exit_timestamp=now_iso,
                    r_multiple=round(
                        r_multiple,
                        2,
                    ),
                    notes=f"TP hit @ {current_price}",
                )

                try:
                    send_outcome_notification(
                        signal_id,
                        "TP_HIT",
                        current_price,
                        r_multiple,
                    )
                except Exception:
                    pass

            # ----------------------------------------------------------
            # SL HIT
            # ----------------------------------------------------------

            elif sl_hit:

                r_multiple = -1.0

                logger.info(
                    f"Signal {signal_id}: "
                    f"SL HIT @ {current_price} "
                    f"(-1R)"
                )

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
                    send_outcome_notification(
                        signal_id,
                        "SL_HIT",
                        current_price,
                        -1.0,
                    )
                except Exception:
                    pass