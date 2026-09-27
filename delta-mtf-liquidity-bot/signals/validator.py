"""
signals/validator.py — Pre-send signal validation gate.

Validates a SetupResult before it is sent to Telegram:
1. Not a duplicate
2. Quality gate (A+ or A only by default)
3. Entry/SL/TP sanity checks
4. RR minimum check
5. Registers in duplicate guard if approved
"""

import logging
from typing import Optional, Tuple
from strategy.intraday import SetupResult
from strategy.filters import SignalQuality
from signals.duplicate_guard import is_duplicate, register_signal, cleanup_expired
from config.strategy_config import MIN_RR_INTRADAY, MIN_RR_SWING

logger = logging.getLogger(__name__)


def validate_signal(setup: SetupResult) -> Tuple[bool, str, Optional[str]]:
    """
    Validate a setup before sending to Telegram.

    Returns:
        (approved: bool, reason: str, signal_id: str | None)
    """
    # ── Cleanup expired duplicates first ───────────────────────────────────
    cleanup_expired()

    # ── Quality gate ───────────────────────────────────────────────────────
    if not setup.should_send:
        return False, f"Quality gate: {setup.rejection_reason}", None

    if setup.quality == SignalQuality.REJECTED:
        return False, "Signal quality is REJECTED", None

    if setup.quality == SignalQuality.B:
        return False, "B-quality signals are not sent to Telegram (configurable)", None

    # ── Basic sanity checks ────────────────────────────────────────────────
    if setup.entry is None or setup.sl is None or setup.tp is None:
        return False, "Invalid Entry/SL/TP (None values)", None

    if setup.entry <= 0 or setup.sl <= 0 or setup.tp <= 0:
        return False, "Entry/SL/TP must be positive prices", None

    if setup.direction == "LONG":
        if setup.sl >= setup.entry:
            return False, f"LONG SL ({setup.sl}) must be below entry ({setup.entry})", None
        if setup.tp <= setup.entry:
            return False, f"LONG TP ({setup.tp}) must be above entry ({setup.entry})", None
    else:
        if setup.sl <= setup.entry:
            return False, f"SHORT SL ({setup.sl}) must be above entry ({setup.entry})", None
        if setup.tp >= setup.entry:
            return False, f"SHORT TP ({setup.tp}) must be below entry ({setup.entry})", None

    # ── RR check ───────────────────────────────────────────────────────────
    min_rr = MIN_RR_INTRADAY if setup.mode == "INTRADAY" else MIN_RR_SWING
    if setup.rr is None or setup.rr < min_rr:
        return False, f"RR {setup.rr} < minimum {min_rr}", None

    # ── Duplicate check ────────────────────────────────────────────────────
    if setup.liquidity_level and is_duplicate(
        symbol=setup.symbol,
        timeframe=setup.timeframe,
        direction=setup.direction,
        liquidity_level=setup.liquidity_level,
        mode=setup.mode,
    ):
        return False, "Duplicate signal for same setup (still active)", None

    return True, "Signal approved", None


def approve_and_register(setup: SetupResult, signal_id: str) -> bool:
    """Register a validated signal in the duplicate guard."""
    try:
        if setup.liquidity_level:
            register_signal(
                signal_id=signal_id,
                symbol=setup.symbol,
                timeframe=setup.timeframe,
                direction=setup.direction,
                liquidity_level=setup.liquidity_level,
                mode=setup.mode,
            )
        return True
    except Exception as e:
        logger.error(f"Failed to register signal {signal_id}: {e}")
        return False
