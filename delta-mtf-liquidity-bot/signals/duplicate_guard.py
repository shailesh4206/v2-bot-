"""
signals/duplicate_guard.py — Prevent sending duplicate signals.

Fingerprint = symbol + timeframe + direction + setup_candle_ts + liquidity_level + strategy_version

A new signal is allowed when:
- A genuinely new setup has formed (different candle, different level)
- The previous signal for the same setup has been resolved or expired

Fingerprints are stored in SQLite with an expiry timestamp.
"""

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from database.models import get_connection
from config.settings import STRATEGY_VERSION
from config.strategy_config import INTRADAY_EXPIRY_MINUTES, SWING_EXPIRY_HOURS

logger = logging.getLogger(__name__)


def _make_fingerprint(
    symbol:         str,
    timeframe:      str,
    direction:      str,
    liquidity_level: float,
    mode:           str,
    strategy_version: str = STRATEGY_VERSION,
) -> str:
    """Create a deterministic fingerprint for a setup."""
    # Round level to 4 significant figures to handle floating-point noise
    level_rounded = round(liquidity_level, 4)
    raw = f"{symbol}|{timeframe}|{direction}|{level_rounded}|{mode}|{strategy_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def is_duplicate(
    symbol:          str,
    timeframe:       str,
    direction:       str,
    liquidity_level: float,
    mode:            str,
) -> bool:
    """
    Return True if an active (non-expired) signal with the same fingerprint exists.
    """
    fp = _make_fingerprint(symbol, timeframe, direction, liquidity_level, mode)
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT id FROM duplicate_guard
            WHERE fingerprint = ? AND expires_at > ?
        """, (fp, now)).fetchone()
        return row is not None
    finally:
        conn.close()


def register_signal(
    signal_id:       str,
    symbol:          str,
    timeframe:       str,
    direction:       str,
    liquidity_level: float,
    mode:            str,
):
    """Register a new signal fingerprint to prevent future duplicates."""
    fp = _make_fingerprint(symbol, timeframe, direction, liquidity_level, mode)
    now = datetime.now(timezone.utc)

    if mode == "INTRADAY":
        expiry = (now + timedelta(minutes=INTRADAY_EXPIRY_MINUTES)).isoformat()
    else:
        expiry = (now + timedelta(hours=SWING_EXPIRY_HOURS)).isoformat()

    conn = get_connection()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO duplicate_guard
            (fingerprint, signal_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (fp, signal_id, now.isoformat(), expiry))
        conn.commit()
        logger.debug(f"Registered fingerprint {fp[:8]}… for {signal_id}")
    finally:
        conn.close()


def cleanup_expired():
    """Remove expired fingerprints from the guard table."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        result = conn.execute(
            "DELETE FROM duplicate_guard WHERE expires_at <= ?", (now,)
        )
        conn.commit()
        if result.rowcount:
            logger.debug(f"Cleaned {result.rowcount} expired fingerprints")
    finally:
        conn.close()
