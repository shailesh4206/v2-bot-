"""
filters.py — Signal filters and quality classification.

Handles:
- BTC correlation filter (for ETH, SOL, and other altcoins)
- Chase protection (ATR-based entry distance check)
- Signal quality classification: A+ / A / B / REJECTED
- Countertrend protection
"""

import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from indicators.trend import TrendAnalysis, TrendBias, TrendStrength
from strategy.mtf import MTFContext, MTFStatus
from config.strategy_config import (
    BTC_CORRELATION_FILTER_ENABLED,
    BTC_BLOCK_ALT_LONG_ON_STRONG_BEAR,
    BTC_BLOCK_ALT_SHORT_ON_STRONG_BULL,
    BTC_DOWNGRADE_ALT_LONG_ON_BEAR,
    BTC_DOWNGRADE_ALT_SHORT_ON_BULL,
    CHASE_PROTECTION_ATR_MULTIPLIER,
    SEND_QUALITY_A_PLUS,
    SEND_QUALITY_A,
    SEND_QUALITY_B,
    A_PLUS_REQUIRE_MTF_ALIGNED,
    A_PLUS_REQUIRE_RETEST,
    A_PLUS_REQUIRE_VOLUME_CONFIRM,
    A_PLUS_MIN_SWEEP_SCORE,
    A_PLUS_MIN_RR_INTRADAY,
    A_PLUS_MIN_RR_SWING,
    A_PLUS_NO_MTF_CONFLICT,
    A_REQUIRE_RETEST,
    A_REQUIRE_VOLUME_CONFIRM,
    A_MIN_SWEEP_SCORE,
    A_MIN_RR_INTRADAY,
    A_MIN_RR_SWING,
    MIN_RR_INTRADAY,
    MIN_RR_SWING,
)
from config.settings import BTC_SYMBOL

logger = logging.getLogger(__name__)


class SignalQuality(str, Enum):
    A_PLUS   = "A+"
    A        = "A"
    B        = "B"
    REJECTED = "REJECTED"


@dataclass
class QualityResult:
    quality:      SignalQuality
    should_send:  bool
    reason:       str


@dataclass
class BTCFilterResult:
    blocked:    bool
    downgraded: bool
    reason:     str


def apply_btc_correlation_filter(
    symbol:     str,
    direction:  str,
    btc_h4:     Optional[TrendAnalysis],
    btc_h1:     Optional[TrendAnalysis],
) -> BTCFilterResult:
    """
    Apply BTC correlation filter for non-BTC altcoins.

    Args:
        symbol:    Trading symbol being analyzed
        direction: "LONG" or "SHORT"
        btc_h4:    BTC 4H trend analysis
        btc_h1:    BTC 1H trend analysis

    Returns:
        BTCFilterResult with blocked/downgraded flags and reason
    """
    # BTC itself is exempt
    if symbol == BTC_SYMBOL or not BTC_CORRELATION_FILTER_ENABLED:
        return BTCFilterResult(blocked=False, downgraded=False, reason="BTC filter N/A")

    if btc_h4 is None or btc_h1 is None:
        return BTCFilterResult(blocked=False, downgraded=False, reason="BTC data unavailable")

    btc_strongly_bearish = (
        btc_h4.bias == TrendBias.BEARISH and btc_h4.strength == TrendStrength.STRONG
        or btc_h1.bias == TrendBias.BEARISH and btc_h1.strength == TrendStrength.STRONG
    )

    btc_strongly_bullish = (
        btc_h4.bias == TrendBias.BULLISH and btc_h4.strength == TrendStrength.STRONG
        or btc_h1.bias == TrendBias.BULLISH and btc_h1.strength == TrendStrength.STRONG
    )

    # Block ALT LONG when BTC is strongly bearish
    if direction == "LONG" and btc_strongly_bearish:
        if BTC_BLOCK_ALT_LONG_ON_STRONG_BEAR:
            return BTCFilterResult(
                blocked=True,
                downgraded=False,
                reason="BTC STRONGLY BEARISH — ALT LONG blocked by correlation filter",
            )
        elif BTC_DOWNGRADE_ALT_LONG_ON_BEAR:
            return BTCFilterResult(
                blocked=False,
                downgraded=True,
                reason="BTC BEARISH — ALT LONG downgraded by correlation filter",
            )

    # Block ALT SHORT when BTC is strongly bullish
    if direction == "SHORT" and btc_strongly_bullish:
        if BTC_BLOCK_ALT_SHORT_ON_STRONG_BULL:
            return BTCFilterResult(
                blocked=True,
                downgraded=False,
                reason="BTC STRONGLY BULLISH — ALT SHORT blocked by correlation filter",
            )
        elif BTC_DOWNGRADE_ALT_SHORT_ON_BULL:
            return BTCFilterResult(
                blocked=False,
                downgraded=True,
                reason="BTC BULLISH — ALT SHORT downgraded by correlation filter",
            )

    return BTCFilterResult(blocked=False, downgraded=False, reason="BTC correlation: neutral")


def check_chase_protection(
    entry_price:   float,
    structure_ref: float,
    current_atr:   float,
    direction:     str,
) -> tuple:
    """
    Reject signals if entry is too far from the structure reference.

    Args:
        entry_price:   Proposed entry price
        structure_ref: BOS level or swing point used as reference
        current_atr:   Current ATR value
        direction:     "LONG" or "SHORT"

    Returns:
        (chase_ok: bool, reason: str)
    """
    if current_atr <= 0:
        return True, "ATR not available — chase check skipped"

    max_distance = CHASE_PROTECTION_ATR_MULTIPLIER * current_atr
    distance     = abs(entry_price - structure_ref)

    if distance > max_distance:
        return False, (
            f"CHASE PROTECTION: Entry {entry_price:.4f} is "
            f"{distance:.4f} from structure {structure_ref:.4f} "
            f"(max {max_distance:.4f} = {CHASE_PROTECTION_ATR_MULTIPLIER}×ATR)"
        )

    return True, "Chase check passed"


def classify_signal_quality(
    sweep_score:     int,
    rr:              float,
    mode:            str,           # "INTRADAY" or "SWING"
    mtf_status:      MTFStatus,
    retest_present:  bool,
    volume_confirmed: bool,
    btc_downgraded:  bool = False,
) -> QualityResult:
    """
    Classify signal as A+, A, B, or REJECTED.

    Args:
        sweep_score:      Liquidity sweep score (0–100)
        rr:               Risk:Reward ratio
        mode:             Strategy mode
        mtf_status:       Multi-timeframe alignment status
        retest_present:   Whether retest was detected
        volume_confirmed: Whether volume confirmation passed
        btc_downgraded:   Whether BTC correlation filter downgraded the signal

    Returns:
        QualityResult
    """
    min_rr     = MIN_RR_INTRADAY if mode == "INTRADAY" else MIN_RR_SWING
    a_plus_rr  = A_PLUS_MIN_RR_INTRADAY if mode == "INTRADAY" else A_PLUS_MIN_RR_SWING
    a_rr       = A_MIN_RR_INTRADAY if mode == "INTRADAY" else A_MIN_RR_SWING

    # ── REJECTED ─────────────────────────────────
    if rr < min_rr:
        return QualityResult(
            quality=SignalQuality.REJECTED,
            should_send=False,
            reason=f"RR {rr:.2f} < minimum {min_rr}",
        )

    if sweep_score < A_MIN_SWEEP_SCORE:
        return QualityResult(
            quality=SignalQuality.REJECTED,
            should_send=False,
            reason=f"Sweep score {sweep_score} < minimum {A_MIN_SWEEP_SCORE}",
        )

    if mtf_status == MTFStatus.MAJOR_CONFLICT:
        return QualityResult(
            quality=SignalQuality.REJECTED,
            should_send=False,
            reason="MTF MAJOR CONFLICT",
        )

    # ── A+ ────────────────────────────────────────
    if (
        sweep_score >= A_PLUS_MIN_SWEEP_SCORE
        and rr >= a_plus_rr
        and (not A_PLUS_NO_MTF_CONFLICT or mtf_status == MTFStatus.ALIGNED)
        and (not A_PLUS_REQUIRE_RETEST or retest_present)
        and (not A_PLUS_REQUIRE_VOLUME_CONFIRM or volume_confirmed)
        and not btc_downgraded
    ):
        return QualityResult(
            quality=SignalQuality.A_PLUS,
            should_send=SEND_QUALITY_A_PLUS,
            reason="A+ quality: all conditions met",
        )

    # ── A ─────────────────────────────────────────
    if (
        sweep_score >= A_MIN_SWEEP_SCORE
        and rr >= a_rr
        and (not A_REQUIRE_VOLUME_CONFIRM or volume_confirmed)
    ):
        return QualityResult(
            quality=SignalQuality.A,
            should_send=SEND_QUALITY_A,
            reason="A quality: most conditions met",
        )

    # ── B ─────────────────────────────────────────
    return QualityResult(
        quality=SignalQuality.B,
        should_send=SEND_QUALITY_B,
        reason="B quality: setup valid but below A threshold",
    )
