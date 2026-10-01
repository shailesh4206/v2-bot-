"""
strategy/mtf.py — Institutional Multi-Timeframe alignment engine.

Hierarchy:
    4H = Macro regime
    1H = Structure regime
    15M = Execution context

Hard rules:
    - Neutral is NOT aligned.
    - 4H and 1H must have directional agreement.
    - Countertrend is blocked when disabled.
    - Major conflicts are rejected.
    - 15M is execution context, not a replacement for 4H/1H.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from indicators.trend import (
    TrendAnalysis,
    TrendBias,
    TrendStrength,
    classify_trend,
    is_strongly_bullish,
    is_strongly_bearish,
)

from market_structure.swings import (
    detect_swing_highs,
    detect_swing_lows,
    label_market_structure,
    get_market_structure_bias,
)

from config.strategy_config import (
    COUNTERTREND_ALLOWED,
    MTF_CONFLICT_REJECT,
    CONFLICT_STRONG_ADX_THRESHOLD,
)


class MTFStatus(str, Enum):
    ALIGNED = "ALIGNED"
    MINOR_CONFLICT = "MINOR_CONFLICT"
    MAJOR_CONFLICT = "MAJOR_CONFLICT"


@dataclass
class MTFContext:
    """Complete multi-timeframe analysis result."""

    symbol: str

    h4_trend: TrendAnalysis
    h1_trend: TrendAnalysis
    m15_trend: Optional[TrendAnalysis]

    h4_structure_bias: str
    h1_structure_bias: str

    status: MTFStatus
    conflict_reason: str

    is_bullish_aligned: bool
    is_bearish_aligned: bool

    pdh: Optional[float] = None
    pdl: Optional[float] = None


def _directional_alignment(
    h4_trend: TrendAnalysis,
    h1_trend: TrendAnalysis,
    h4_struct: str,
    h1_struct: str,
):
    """
    Strict institutional directional alignment.

    LONG requires:
        4H BULLISH
        1H BULLISH
        4H structure not BEARISH
        1H structure not BEARISH

    SHORT requires:
        4H BEARISH
        1H BEARISH
        4H structure not BULLISH
        1H structure not BULLISH

    NEUTRAL is NEVER considered aligned.
    """

    bullish = (
        h4_trend.bias == TrendBias.BULLISH
        and h1_trend.bias == TrendBias.BULLISH
        and h4_struct != "BEARISH"
        and h1_struct != "BEARISH"
    )

    bearish = (
        h4_trend.bias == TrendBias.BEARISH
        and h1_trend.bias == TrendBias.BEARISH
        and h4_struct != "BULLISH"
        and h1_struct != "BULLISH"
    )

    return bullish, bearish


def analyze_mtf(
    symbol: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    pdh: Optional[float] = None,
    pdl: Optional[float] = None,
) -> MTFContext:
    """
    Perform strict 4H -> 1H -> 15M MTF analysis.

    Caller must provide confirmed/closed candle data.
    """

    if df_4h is None or df_4h.empty:
        raise ValueError(f"{symbol}: 4H dataframe is empty")

    if df_1h is None or df_1h.empty:
        raise ValueError(f"{symbol}: 1H dataframe is empty")

    if df_15m is None or df_15m.empty:
        raise ValueError(f"{symbol}: 15M dataframe is empty")

    # ==========================================================
    # TREND ANALYSIS
    # ==========================================================

    h4_trend = classify_trend(df_4h, "4h")
    h1_trend = classify_trend(df_1h, "1h")
    m15_trend = classify_trend(df_15m, "15m")

    # ==========================================================
    # MARKET STRUCTURE
    # ==========================================================

    h4_sh = detect_swing_highs(df_4h, "4h")
    h4_sl = detect_swing_lows(df_4h, "4h")
    h4_sh, h4_sl = label_market_structure(h4_sh, h4_sl)
    h4_struct = get_market_structure_bias(h4_sh, h4_sl)

    h1_sh = detect_swing_highs(df_1h, "1h")
    h1_sl = detect_swing_lows(df_1h, "1h")
    h1_sh, h1_sl = label_market_structure(h1_sh, h1_sl)
    h1_struct = get_market_structure_bias(h1_sh, h1_sl)

    # ==========================================================
    # CONFLICT CLASSIFICATION
    # ==========================================================

    status, conflict_reason = _classify_conflict(
        h4_trend=h4_trend,
        h1_trend=h1_trend,
        m15_trend=m15_trend,
        h4_struct=h4_struct,
        h1_struct=h1_struct,
    )

    # ==========================================================
    # STRICT ALIGNMENT
    # ==========================================================

    is_bullish, is_bearish = _directional_alignment(
        h4_trend=h4_trend,
        h1_trend=h1_trend,
        h4_struct=h4_struct,
        h1_struct=h1_struct,
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
    h4_trend: TrendAnalysis,
    h1_trend: TrendAnalysis,
    m15_trend: Optional[TrendAnalysis],
    h4_struct: str,
    h1_struct: str,
) -> tuple:
    """
    Classify MTF relationship.

    Neutral is NOT considered aligned.
    """

    # ==========================================================
    # STRONG 4H VS OPPOSITE 1H
    # ==========================================================

    if (
        is_strongly_bearish(h4_trend)
        and h1_trend.bias == TrendBias.BULLISH
    ):
        return (
            MTFStatus.MAJOR_CONFLICT,
            (
                f"4H STRONGLY BEARISH "
                f"(ADX={h4_trend.adx:.1f}) "
                f"conflicts with 1H BULLISH"
            ),
        )

    if (
        is_strongly_bullish(h4_trend)
        and h1_trend.bias == TrendBias.BEARISH
    ):
        return (
            MTFStatus.MAJOR_CONFLICT,
            (
                f"4H STRONGLY BULLISH "
                f"(ADX={h4_trend.adx:.1f}) "
                f"conflicts with 1H BEARISH"
            ),
        )

    # ==========================================================
    # STRUCTURE CONFLICT
    # ==========================================================

    if h4_struct == "BEARISH" and h1_struct == "BULLISH":
        if h4_trend.adx > CONFLICT_STRONG_ADX_THRESHOLD:
            return (
                MTFStatus.MAJOR_CONFLICT,
                (
                    f"4H BEARISH structure + strong ADX "
                    f"({h4_trend.adx:.1f}) "
                    f"vs 1H BULLISH structure"
                ),
            )

        return (
            MTFStatus.MINOR_CONFLICT,
            "4H BEARISH structure vs 1H BULLISH structure",
        )

    if h4_struct == "BULLISH" and h1_struct == "BEARISH":
        if h4_trend.adx > CONFLICT_STRONG_ADX_THRESHOLD:
            return (
                MTFStatus.MAJOR_CONFLICT,
                (
                    f"4H BULLISH structure + strong ADX "
                    f"({h4_trend.adx:.1f}) "
                    f"vs 1H BEARISH structure"
                ),
            )

        return (
            MTFStatus.MINOR_CONFLICT,
            "4H BULLISH structure vs 1H BEARISH structure",
        )

    # ==========================================================
    # OPPOSITE NON-NEUTRAL TRENDS
    # ==========================================================

    if (
        h4_trend.bias != TrendBias.NEUTRAL
        and h1_trend.bias != TrendBias.NEUTRAL
        and h4_trend.bias != h1_trend.bias
    ):
        return (
            MTFStatus.MINOR_CONFLICT,
            (
                f"4H {h4_trend.bias.value} "
                f"vs 1H {h1_trend.bias.value}"
            ),
        )

    # ==========================================================
    # NEUTRAL = NOT ALIGNED
    # ==========================================================

    if h4_trend.bias == TrendBias.NEUTRAL:
        return (
            MTFStatus.MINOR_CONFLICT,
            "4H macro regime is NEUTRAL - directional setup blocked",
        )

    if h1_trend.bias == TrendBias.NEUTRAL:
        return (
            MTFStatus.MINOR_CONFLICT,
            "1H structure regime is NEUTRAL - directional setup blocked",
        )

    if h4_struct == "NEUTRAL":
        return (
            MTFStatus.MINOR_CONFLICT,
            "4H market structure is NEUTRAL - directional setup blocked",
        )

    if h1_struct == "NEUTRAL":
        return (
            MTFStatus.MINOR_CONFLICT,
            "1H market structure is NEUTRAL - directional setup blocked",
        )

    # ==========================================================
    # TRUE ALIGNMENT
    # ==========================================================

    if (
        h4_trend.bias == h1_trend.bias
        and h4_trend.bias != TrendBias.NEUTRAL
    ):
        return (
            MTFStatus.ALIGNED,
            "4H and 1H directional regimes aligned",
        )

    return (
        MTFStatus.MINOR_CONFLICT,
        "MTF directional alignment not confirmed",
    )


def mtf_allows_long(ctx: MTFContext) -> tuple:
    """
    Final hard gate for LONG.
    """

    if (
        ctx.status == MTFStatus.MAJOR_CONFLICT
        and MTF_CONFLICT_REJECT
    ):
        return (
            False,
            f"MTF MAJOR CONFLICT: {ctx.conflict_reason}",
        )

    if not ctx.is_bullish_aligned:
        return (
            False,
            "LONG blocked: strict 4H + 1H bullish alignment not confirmed",
        )

    if (
        ctx.h4_trend.bias == TrendBias.BEARISH
        and ctx.h4_trend.strength == TrendStrength.STRONG
        and not COUNTERTREND_ALLOWED
    ):
        return (
            False,
            "COUNTERTREND BLOCKED: 4H strongly bearish",
        )

    return True, "MTF allows LONG"


def mtf_allows_short(ctx: MTFContext) -> tuple:
    """
    Final hard gate for SHORT.
    """

    if (
        ctx.status == MTFStatus.MAJOR_CONFLICT
        and MTF_CONFLICT_REJECT
    ):
        return (
            False,
            f"MTF MAJOR CONFLICT: {ctx.conflict_reason}",
        )

    if not ctx.is_bearish_aligned:
        return (
            False,
            "SHORT blocked: strict 4H + 1H bearish alignment not confirmed",
        )

    if (
        ctx.h4_trend.bias == TrendBias.BULLISH
        and ctx.h4_trend.strength == TrendStrength.STRONG
        and not COUNTERTREND_ALLOWED
    ):
        return (
            False,
            "COUNTERTREND BLOCKED: 4H strongly bullish",
        )

    return True, "MTF allows SHORT"


if __name__ == "__main__":
    print("=" * 72)
    print("INSTITUTIONAL MTF ENGINE SELF-TEST")
    print("=" * 72)
    print("Hierarchy              : 4H -> 1H -> 15M")
    print("4H role                : Macro regime")
    print("1H role                : Structure regime")
    print("15M role               : Execution context")
    print("Neutral = aligned      : False")
    print("4H + 1H required       : True")
    print("Major conflict reject  :", MTF_CONFLICT_REJECT)
    print("Countertrend allowed   :", COUNTERTREND_ALLOWED)
    print("Directional hard gate  : True")
    print("Closed candle model    : True")
    print()
    print("✓ Institutional MTF engine loaded successfully.")