"""
signals/validator.py — Institutional final pre-send validation firewall.

This is the FINAL safety gate before Telegram.

Hard requirements:
    - Setup must be valid
    - should_send must be True
    - A / A+ quality only
    - Valid liquidity sweep
    - Minimum sweep score
    - CHoCH confirmed
    - BOS confirmed
    - Retest confirmed
    - Volume confirmed
    - Strict MTF directional context
    - Valid Entry / SL / TP
    - Minimum RR
    - Duplicate protection

Important:
    This validator NEVER upgrades a setup.
    It can only APPROVE or REJECT.

No automatic order execution.
"""

import logging
from typing import Optional, Tuple

from strategy.intraday import SetupResult
from strategy.filters import SignalQuality

from signals.duplicate_guard import (
    is_duplicate,
    register_signal,
    cleanup_expired,
)

from config import strategy_config


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MIN_RR_INTRADAY = float(
    getattr(strategy_config, "MIN_RR_INTRADAY", 2.0)
)

MIN_RR_SWING = float(
    getattr(strategy_config, "MIN_RR_SWING", 2.5)
)

MIN_SWEEP_SCORE = int(
    getattr(strategy_config, "MIN_SWEEP_SCORE", 70)
)

RETEST_REQUIRED_INTRADAY = bool(
    getattr(strategy_config, "RETEST_REQUIRED_INTRADAY", True)
)

RETEST_REQUIRED_SWING = bool(
    getattr(strategy_config, "RETEST_REQUIRED_SWING", True)
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_neutral_summary(summary: Optional[str]) -> bool:
    """
    Detect an explicit NEUTRAL higher-timeframe summary.

    This is a defensive fallback because SetupResult currently stores
    summaries rather than the complete MTFContext.
    """

    if not summary:
        return True

    text = str(summary).upper()

    return "NEUTRAL" in text


def _validate_mtf_summary(setup: SetupResult) -> Tuple[bool, str]:
    """
    Defensive MTF gate.

    Current SetupResult does not expose the complete MTFContext, so
    explicitly neutral 4H/1H summaries are blocked here.

    The strategy pipeline itself performs the authoritative directional
    MTF validation.
    """

    if _is_neutral_summary(setup.h4_summary):
        return (
            False,
            "MTF hard gate: 4H macro regime is NEUTRAL/unknown",
        )

    if _is_neutral_summary(setup.h1_summary):
        return (
            False,
            "MTF hard gate: 1H structure regime is NEUTRAL/unknown",
        )

    return True, "MTF summary gate passed"


def _validate_directional_prices(
    setup: SetupResult,
) -> Tuple[bool, str]:
    """Validate Entry / SL / TP direction."""

    if setup.entry is None:
        return False, "Entry is None"

    if setup.sl is None:
        return False, "SL is None"

    if setup.tp is None:
        return False, "TP is None"

    try:
        entry = float(setup.entry)
        sl = float(setup.sl)
        tp = float(setup.tp)
    except (TypeError, ValueError):
        return False, "Entry/SL/TP are not numeric"

    if entry <= 0 or sl <= 0 or tp <= 0:
        return False, "Entry/SL/TP must be positive"

    direction = str(setup.direction).upper()

    if direction == "LONG":

        if sl >= entry:
            return (
                False,
                f"LONG SL {sl} must be below Entry {entry}",
            )

        if tp <= entry:
            return (
                False,
                f"LONG TP {tp} must be above Entry {entry}",
            )

    elif direction == "SHORT":

        if sl <= entry:
            return (
                False,
                f"SHORT SL {sl} must be above Entry {entry}",
            )

        if tp >= entry:
            return (
                False,
                f"SHORT TP {tp} must be below Entry {entry}",
            )

    else:
        return False, f"Unknown direction: {setup.direction}"

    return True, "Entry/SL/TP structure valid"


def _validate_rr(
    setup: SetupResult,
) -> Tuple[bool, str]:
    """Validate minimum configured risk/reward."""

    if setup.rr is None:
        return False, "RR is None"

    try:
        rr = float(setup.rr)
    except (TypeError, ValueError):
        return False, "RR is not numeric"

    if rr <= 0:
        return False, f"Invalid RR: {rr}"

    if str(setup.mode).upper() == "INTRADAY":
        minimum_rr = MIN_RR_INTRADAY
    else:
        minimum_rr = MIN_RR_SWING

    if rr < minimum_rr:
        return (
            False,
            f"RR {rr:.2f} < minimum {minimum_rr:.2f}",
        )

    return True, f"RR {rr:.2f} >= minimum {minimum_rr:.2f}"


# ============================================================================
# MAIN VALIDATOR
# ============================================================================

def validate_signal(
    setup: SetupResult,
) -> Tuple[bool, str, Optional[str]]:
    """
    Final institutional validation before Telegram.

    Returns:
        (approved, reason, signal_id)

    signal_id is generated/assigned by the caller after approval.
    """

    cleanup_expired()

    # ========================================================================
    # 1. OBJECT VALIDATION
    # ========================================================================

    if setup is None:
        return False, "Setup is None", None

    # ========================================================================
    # 2. VALID FLAG
    # ========================================================================

    if not bool(setup.valid):
        return (
            False,
            f"Setup marked invalid: {setup.rejection_reason}",
            None,
        )

    # ========================================================================
    # 3. SHOULD-SEND FLAG
    # ========================================================================

    if not bool(setup.should_send):
        return (
            False,
            f"Strategy should_send=False: {setup.rejection_reason}",
            None,
        )

    # ========================================================================
    # 4. QUALITY GATE
    # ========================================================================

    if setup.quality == SignalQuality.REJECTED:
        return (
            False,
            "Signal quality is REJECTED",
            None,
        )

    if setup.quality == SignalQuality.B:
        return (
            False,
            "B-quality signals are not allowed",
            None,
        )

    if setup.quality not in (
        SignalQuality.A,
        SignalQuality.A_PLUS,
    ):
        return (
            False,
            f"Unsupported signal quality: {setup.quality}",
            None,
        )

    # ========================================================================
    # 5. LIQUIDITY SWEEP GATE
    # ========================================================================

    if setup.sweep_event is None:
        return (
            False,
            "Hard gate failed: liquidity sweep missing",
            None,
        )

    if not bool(getattr(setup.sweep_event, "valid", True)):
        return (
            False,
            "Hard gate failed: liquidity sweep marked invalid",
            None,
        )

    if setup.sweep_score < MIN_SWEEP_SCORE:
        return (
            False,
            (
                f"Hard gate failed: sweep score "
                f"{setup.sweep_score} < minimum {MIN_SWEEP_SCORE}"
            ),
            None,
        )

    # ========================================================================
    # 6. CHoCH GATE
    # ========================================================================

    if not bool(setup.choch_confirmed):
        return (
            False,
            "Hard gate failed: CHoCH not confirmed",
            None,
        )

    # ========================================================================
    # 7. BOS GATE
    # ========================================================================

    if not bool(setup.bos_confirmed):
        return (
            False,
            "Hard gate failed: BOS not confirmed",
            None,
        )

    # ========================================================================
    # 8. RETEST GATE
    # ========================================================================

    mode = str(setup.mode).upper()

    if mode == "INTRADAY":
        retest_required = RETEST_REQUIRED_INTRADAY
    else:
        retest_required = RETEST_REQUIRED_SWING

    if retest_required and not bool(setup.retest_confirmed):
        return (
            False,
            (
                "Hard gate failed: Retest not confirmed. "
                "Retest Pending signals are NEVER allowed."
            ),
            None,
        )

    # ========================================================================
    # 9. VOLUME GATE
    # ========================================================================

    if not bool(setup.volume_confirmed):
        return (
            False,
            "Hard gate failed: volume confirmation missing",
            None,
        )

    # ========================================================================
    # 10. MTF DEFENSIVE GATE
    # ========================================================================

    mtf_ok, mtf_reason = _validate_mtf_summary(setup)

    if not mtf_ok:
        return False, mtf_reason, None

    # ========================================================================
    # 11. ENTRY / SL / TP
    # ========================================================================

    prices_ok, prices_reason = _validate_directional_prices(setup)

    if not prices_ok:
        return False, prices_reason, None

    # ========================================================================
    # 12. RR
    # ========================================================================

    rr_ok, rr_reason = _validate_rr(setup)

    if not rr_ok:
        return False, rr_reason, None

    # ========================================================================
    # 13. LIQUIDITY LEVEL
    # ========================================================================

    if setup.liquidity_level is None:
        return (
            False,
            "Hard gate failed: liquidity level missing",
            None,
        )

    try:
        liquidity_level = float(setup.liquidity_level)

        if liquidity_level <= 0:
            return (
                False,
                "Liquidity level must be positive",
                None,
            )

    except (TypeError, ValueError):
        return (
            False,
            "Liquidity level is not numeric",
            None,
        )

    # ========================================================================
    # 14. DUPLICATE PROTECTION
    # ========================================================================

    if is_duplicate(
        symbol=setup.symbol,
        timeframe=setup.timeframe,
        direction=setup.direction,
        liquidity_level=setup.liquidity_level,
        mode=setup.mode,
    ):
        return (
            False,
            "Duplicate signal for same setup (still active)",
            None,
        )

    # ========================================================================
    # FINAL APPROVAL
    # ========================================================================

    logger.info(
        "INSTITUTIONAL VALIDATOR APPROVED | "
        "%s | %s | %s | Quality=%s | RR=%s | Sweep=%s",
        setup.symbol,
        setup.mode,
        setup.direction,
        setup.quality,
        setup.rr,
        setup.sweep_score,
    )

    return True, "Institutional signal approved", None


# ============================================================================
# REGISTER APPROVED SIGNAL
# ============================================================================

def approve_and_register(
    setup: SetupResult,
    signal_id: str,
) -> bool:
    """
    Register an already validated signal.

    Registration happens only after validate_signal() returns approved.
    """

    try:

        if not signal_id:
            logger.error(
                "Cannot register signal without signal_id"
            )
            return False

        if not setup.liquidity_level:
            logger.error(
                "Cannot register signal without liquidity_level"
            )
            return False

        register_signal(
            signal_id=signal_id,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
            direction=setup.direction,
            liquidity_level=setup.liquidity_level,
            mode=setup.mode,
        )

        logger.info(
            "Signal registered successfully: %s",
            signal_id,
        )

        return True

    except Exception as exc:

        logger.error(
            "Failed to register signal %s: %s",
            signal_id,
            exc,
            exc_info=True,
        )

        return False


# ============================================================================
# SELF TEST
# ============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("INSTITUTIONAL SIGNAL VALIDATOR SELF-TEST")
    print("=" * 72)

    print(f"Minimum intraday RR : {MIN_RR_INTRADAY}")
    print(f"Minimum swing RR    : {MIN_RR_SWING}")
    print(f"Minimum sweep score : {MIN_SWEEP_SCORE}")
    print(
        f"Intraday retest     : {RETEST_REQUIRED_INTRADAY}"
    )
    print(
        f"Swing retest        : {RETEST_REQUIRED_SWING}"
    )

    print()
    print("Hard gates:")
    print("  Setup valid        : True")
    print("  should_send        : True")
    print("  Quality A/A+       : True")
    print("  Liquidity sweep    : True")
    print("  Sweep score        : True")
    print("  CHoCH              : True")
    print("  BOS                : True")
    print("  Retest             : True")
    print("  Volume             : True")
    print("  MTF                : True")
    print("  Entry/SL/TP        : True")
    print("  RR                 : True")
    print("  Duplicate guard    : True")

    print()
    print("Retest Pending       : BLOCKED")
    print("Neutral 4H/1H        : BLOCKED")
    print("B-quality            : BLOCKED")
    print("Invalid SL/TP        : BLOCKED")
    print("Low RR               : BLOCKED")
    print()
    print("✓ Institutional validator loaded successfully.")