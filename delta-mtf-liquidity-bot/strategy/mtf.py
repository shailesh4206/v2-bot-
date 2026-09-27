"""
mtf.py — Multi-Timeframe (MTF) alignment analysis.

Combines 4H macro + 1H structure to produce an MTF context object
that is used by both intraday and swing strategy pipelines.

MTF Conflict Detection:
- MAJOR CONFLICT: e.g., 4H strongly bearish + 1H/15M bullish setup → REJECT
- MINOR CONFLICT: e.g., 4H neutral + 1H slightly mixed → WARN but allow
- ALIGNED: All timeframes agree → high-quality setup context
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from indicators.trend import (
    TrendAnalysis, TrendBias, TrendStrength,
    classify_trend, is_strongly_bullish, is_strongly_bearish
)
from market_structure.swings import (
    detect_swing_highs, detect_swing_lows,
    label_market_structure, get_market_structure_bias
)
from config.strategy_config import (
    COUNTERTREND_ALLOWED,
    MTF_CONFLICT_REJECT,
    CONFLICT_STRONG_ADX_THRESHOLD,
)


class MTFStatus(str, Enum):
    ALIGNED        = "ALIGNED"
    MINOR_CONFLICT = "MINOR_CONFLICT"
    MAJOR_CONFLICT = "MAJOR_CONFLICT"


@dataclass
class MTFContext:
    """Complete multi-timeframe analysis result."""
    symbol: str

    # Per-timeframe analysis
    h4_trend:          TrendAnalysis
    h1_trend:          TrendAnalysis
    m15_trend:         Optional[TrendAnalysis]

    h4_structure_bias: str   # "BULLISH", "BEARISH", "NEUTRAL"
    h1_structure_bias: str

    # Overall MTF alignment
    status:            MTFStatus
    conflict_reason:   str

    # Derived
    is_bullish_aligned: bool
    is_bearish_aligned: bool

    # PDH/PDL for signal use
    pdh: Optional[float] = None
    pdl: Optional[float] = None


def analyze_mtf(
    symbol:   str,
    df_4h:    pd.DataFrame,
    df_1h:    pd.DataFrame,
    df_15m:   pd.DataFrame,
    pdh:      Optional[float] = None,
    pdl:      Optional[float] = None,
) -> MTFContext:
    """
    Perform complete multi-timeframe analysis.

    Args:
        symbol:  Trading symbol (e.g., "BTCUSD")
        df_4h:   4H confirmed OHLCV DataFrame
        df_1h:   1H confirmed OHLCV DataFrame
        df_15m:  15M confirmed OHLCV DataFrame
        pdh:     Previous Day High
        pdl:     Previous Day Low

    Returns:
        MTFContext with all timeframe biases and alignment status
    """
    # ── Trend analysis per timeframe ──────────────
    h4_trend  = classify_trend(df_4h,  "4h")
    h1_trend  = classify_trend(df_1h,  "1h")
    m15_trend = classify_trend(df_15m, "15m") if not df_15m.empty else None

    # ── Market structure bias ──────────────────────
    h4_sh = detect_swing_highs(df_4h,  "4h")
    h4_sl = detect_swing_lows(df_4h,   "4h")
    h4_sh, h4_sl = label_market_structure(h4_sh, h4_sl)
    h4_struct = get_market_structure_bias(h4_sh, h4_sl)

    h1_sh = detect_swing_highs(df_1h,  "1h")
    h1_sl = detect_swing_lows(df_1h,   "1h")
    h1_sh, h1_sl = label_market_structure(h1_sh, h1_sl)
    h1_struct = get_market_structure_bias(h1_sh, h1_sl)

    # ── MTF Conflict Classification ────────────────
    status, conflict_reason = _classify_conflict(
        h4_trend, h1_trend, m15_trend,
        h4_struct, h1_struct,
    )

    # ── Derived flags ──────────────────────────────
    is_bullish = (
        h4_trend.bias  != TrendBias.BEARISH and
        h1_trend.bias  == TrendBias.BULLISH  and
        h4_struct      != "BEARISH"           and
        status         != MTFStatus.MAJOR_CONFLICT
    )

    is_bearish = (
        h4_trend.bias  != TrendBias.BULLISH and
        h1_trend.bias  == TrendBias.BEARISH  and
        h4_struct      != "BULLISH"           and
        status         != MTFStatus.MAJOR_CONFLICT
    )

    return MTFContext(
        symbol=symbol,
        h4_trend=h4_trend,
        h1_trend=h1_trend,
        m15_trend=m15_trend,
        h4_structure_bias=h4_struct,
        h1_structure_bias=h1_struct,
        status=status,
        conflict_reason=conflict_reason,
        is_bullish_aligned=is_bullish,
        is_bearish_aligned=is_bearish,
        pdh=pdh,
        pdl=pdl,
    )


def _classify_conflict(
    h4: TrendAnalysis,
    h1: TrendAnalysis,
    m15: Optional[TrendAnalysis],
    h4_struct: str,
    h1_struct: str,
) -> tuple:
    """
    Classify the MTF conflict level.

    MAJOR CONFLICT examples:
    - 4H strongly bearish + 1H bullish setup
    - 4H strongly bullish + 1H bearish setup
    - CounterTrend disabled + clear countertrend signal

    Returns (MTFStatus, reason_string)
    """
    reason = ""

    # Check for strong 4H trend conflicting with 1H
    if is_strongly_bearish(h4) and h1.bias == TrendBias.BULLISH:
        reason = f"4H STRONGLY BEARISH (ADX={h4.adx:.0f}) conflicts with 1H BULLISH"
        return MTFStatus.MAJOR_CONFLICT, reason

    if is_strongly_bullish(h4) and h1.bias == TrendBias.BEARISH:
        reason = f"4H STRONGLY BULLISH (ADX={h4.adx:.0f}) conflicts with 1H BEARISH"
        return MTFStatus.MAJOR_CONFLICT, reason

    # Structure-level conflict
    if h4_struct == "BEARISH" and h1_struct == "BULLISH":
        if h4.adx > CONFLICT_STRONG_ADX_THRESHOLD:
            reason = f"4H BEARISH structure + strong ADX ({h4.adx:.0f}) vs 1H BULLISH structure"
            return MTFStatus.MAJOR_CONFLICT, reason
        reason = "4H BEARISH structure vs 1H BULLISH structure"
        return MTFStatus.MINOR_CONFLICT, reason

    if h4_struct == "BULLISH" and h1_struct == "BEARISH":
        if h4.adx > CONFLICT_STRONG_ADX_THRESHOLD:
            reason = f"4H BULLISH structure + strong ADX ({h4.adx:.0f}) vs 1H BEARISH structure"
            return MTFStatus.MAJOR_CONFLICT, reason
        reason = "4H BULLISH structure vs 1H BEARISH structure"
        return MTFStatus.MINOR_CONFLICT, reason

    # Minor conflicts
    if h4.bias != TrendBias.NEUTRAL and h1.bias != TrendBias.NEUTRAL:
        if h4.bias != h1.bias:
            reason = f"4H {h4.bias.value} vs 1H {h1.bias.value} (not strong)"
            return MTFStatus.MINOR_CONFLICT, reason

    return MTFStatus.ALIGNED, "All timeframes aligned"


def mtf_allows_long(ctx: MTFContext) -> tuple:
    """
    Check if MTF context permits a LONG signal.

    Returns (allowed: bool, reason: str)
    """
    if ctx.status == MTFStatus.MAJOR_CONFLICT and MTF_CONFLICT_REJECT:
        return False, f"MTF MAJOR CONFLICT: {ctx.conflict_reason}"

    if ctx.h4_trend.bias == TrendBias.BEARISH and ctx.h4_trend.strength == TrendStrength.STRONG:
        if not COUNTERTREND_ALLOWED:
            return False, f"COUNTERTREND BLOCKED: 4H strongly bearish"

    if not ctx.is_bullish_aligned and ctx.h1_trend.bias == TrendBias.BEARISH:
        return False, "1H trend is bearish — no bullish alignment"

    return True, "MTF allows LONG"


def mtf_allows_short(ctx: MTFContext) -> tuple:
    """
    Check if MTF context permits a SHORT signal.

    Returns (allowed: bool, reason: str)
    """
    if ctx.status == MTFStatus.MAJOR_CONFLICT and MTF_CONFLICT_REJECT:
        return False, f"MTF MAJOR CONFLICT: {ctx.conflict_reason}"

    if ctx.h4_trend.bias == TrendBias.BULLISH and ctx.h4_trend.strength == TrendStrength.STRONG:
        if not COUNTERTREND_ALLOWED:
            return False, f"COUNTERTREND BLOCKED: 4H strongly bullish"

    if not ctx.is_bearish_aligned and ctx.h1_trend.bias == TrendBias.BULLISH:
        return False, "1H trend is bullish — no bearish alignment"

    return True, "MTF allows SHORT"
