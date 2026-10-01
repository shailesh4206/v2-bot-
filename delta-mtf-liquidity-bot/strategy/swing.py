"""
strategy/swing.py — Institutional Swing Strategy Pipeline.

Hierarchy:
    4H Macro Regime
        ↓
    1H Structure
        ↓
    1H Liquidity Pool
        ↓
    1H Liquidity Sweep
        ↓
    1H CHoCH
        ↓
    1H BOS
        ↓
    1H Retest
        ↓
    Volume / BTC / Volatility / Chase Filters
        ↓
    SL / TP / RR
        ↓
    Quality Engine
        ↓
    Final Signal

Signal:
    SWING | 1H

Rules:
- Closed candles only.
- No future leakage.
- Strict 4H → 1H → 15M hierarchy.
- Neutral MTF never qualifies.
- CHoCH must occur after liquidity sweep.
- BOS must occur after CHoCH.
- CHoCH and BOS cannot be the same candle.
- Retest required.
- Volume confirmation required.
- Minimum RR enforced.
- Signal-only. No order execution.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import pandas as pd

from strategy.mtf import (
    MTFContext,
    mtf_allows_long,
    mtf_allows_short,
)

from strategy.filters import (
    apply_btc_correlation_filter,
    check_chase_protection,
    classify_signal_quality,
    SignalQuality,
)

from strategy.intraday import (
    _check_retest,
    _calculate_rr,
    SetupResult,
)

from market_structure.swings import (
    detect_swing_highs,
    detect_swing_lows,
    label_market_structure,
)

from market_structure.choch import (
    CHoCHDirection,
    get_latest_choch,
)

from market_structure.bos import (
    BOSDirection,
    get_latest_bos,
)

from liquidity.pools import (
    detect_liquidity_pools,
)

from liquidity.sweep import (
    detect_sweeps,
    SweepDirection,
)

from liquidity.scoring import (
    score_sweep,
)

from indicators.volatility import (
    get_current_atr,
)

from indicators.volume import (
    analyze_volume,
)

from config.strategy_config import (
    ATR_BUFFER_MULTIPLIER,
    MIN_RR_SWING,
    RETEST_REQUIRED_SWING,
    RETEST_MAX_CANDLES_AFTER_BOS_SWING,
    BOS_MAX_CANDLES_AFTER_CHOCH,
    CHOCH_BOS_SAME_BREAK_FORBIDDEN,
)

logger = logging.getLogger(__name__)


# ============================================================================
# PUBLIC PIPELINE
# ============================================================================

def run_swing_pipeline(
    symbol: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    mtf_ctx: MTFContext,
    btc_h4_trend=None,
    btc_h1_trend=None,
    pdh: Optional[float] = None,
    pdl: Optional[float] = None,
) -> List[SetupResult]:
    """
    Run institutional swing pipeline.

    Evaluates both LONG and SHORT.

    Only:
        valid=True
        should_send=True

    setups are returned.
    """

    results: List[SetupResult] = []

    for direction in ("LONG", "SHORT"):

        try:
            result = _evaluate_swing(
                symbol=symbol,
                direction=direction,
                df_4h=df_4h,
                df_1h=df_1h,
                df_15m=df_15m,
                mtf_ctx=mtf_ctx,
                btc_h4_trend=btc_h4_trend,
                btc_h1_trend=btc_h1_trend,
                pdh=pdh,
                pdl=pdl,
            )

            results.append(result)

        except Exception as exc:

            logger.error(
                "%s %s swing evaluation failed: %s",
                symbol,
                direction,
                exc,
                exc_info=True,
            )

    return [
        result
        for result in results
        if result.valid and result.should_send
    ]


# ============================================================================
# EVALUATE ONE DIRECTION
# ============================================================================

def _evaluate_swing(
    symbol: str,
    direction: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    mtf_ctx: MTFContext,
    btc_h4_trend=None,
    btc_h1_trend=None,
    pdh: Optional[float] = None,
    pdl: Optional[float] = None,
) -> SetupResult:

    def reject(reason: str) -> SetupResult:

        return SetupResult(
            valid=False,
            direction=direction,
            mode="SWING",
            symbol=symbol,
            timeframe="1h",
            entry=None,
            sl=None,
            tp=None,
            rr=None,

            h4_summary=(
                mtf_ctx.h4_trend.description
                if getattr(mtf_ctx, "h4_trend", None)
                else ""
            ),

            h1_summary=(
                mtf_ctx.h1_trend.description
                if getattr(mtf_ctx, "h1_trend", None)
                else ""
            ),

            m15_summary=(
                mtf_ctx.m15_trend.description
                if getattr(mtf_ctx, "m15_trend", None)
                else ""
            ),

            sweep_event=None,
            sweep_score=0,
            liquidity_type="",
            liquidity_level=None,

            choch_confirmed=False,
            bos_confirmed=False,
            retest_confirmed=False,
            volume_confirmed=False,

            quality=SignalQuality.REJECTED,
            should_send=False,
            rejection_reason=reason,
        )

    # ========================================================================
    # DATA VALIDATION
    # ========================================================================

    if df_4h is None or df_4h.empty:
        return reject("4H data unavailable")

    if df_1h is None or df_1h.empty:
        return reject("1H data unavailable")

    if df_15m is None or df_15m.empty:
        return reject("15M data unavailable")

    if len(df_4h) < 50:
        return reject("Insufficient 4H candles")

    if len(df_1h) < 50:
        return reject("Insufficient 1H candles")

    if len(df_15m) < 50:
        return reject("Insufficient 15M candles")

    # ========================================================================
    # STRICT MTF
    # ========================================================================

    if direction == "LONG":

        mtf_ok, mtf_reason = mtf_allows_long(
            mtf_ctx
        )

    else:

        mtf_ok, mtf_reason = mtf_allows_short(
            mtf_ctx
        )

    if not mtf_ok:
        return reject(
            f"MTF rejected: {mtf_reason}"
        )

    # ------------------------------------------------------------------------
    # Explicit 4H / 1H directional gate
    # ------------------------------------------------------------------------

    h4_bias = (
        mtf_ctx.h4_trend.bias.value
        if getattr(mtf_ctx, "h4_trend", None)
        else "UNKNOWN"
    )

    h1_bias = (
        mtf_ctx.h1_trend.bias.value
        if getattr(mtf_ctx, "h1_trend", None)
        else "UNKNOWN"
    )

    if direction == "LONG":

        if h4_bias != "BULLISH":
            return reject(
                f"LONG requires 4H BULLISH, got {h4_bias}"
            )

        if h1_bias != "BULLISH":
            return reject(
                f"LONG requires 1H BULLISH, got {h1_bias}"
            )

    else:

        if h4_bias != "BEARISH":
            return reject(
                f"SHORT requires 4H BEARISH, got {h4_bias}"
            )

        if h1_bias != "BEARISH":
            return reject(
                f"SHORT requires 1H BEARISH, got {h1_bias}"
            )

    # ========================================================================
    # BTC CORRELATION
    # ========================================================================

    try:

        btc_filter = apply_btc_correlation_filter(
            symbol,
            direction,
            btc_h4_trend,
            btc_h1_trend,
        )

        if btc_filter.blocked:

            return reject(
                f"BTC correlation blocked: "
                f"{btc_filter.reason}"
            )

    except Exception as exc:

        logger.warning(
            "%s BTC correlation filter failed: %s",
            symbol,
            exc,
        )

        return reject(
            "BTC correlation validation failed"
        )

    # ========================================================================
    # 1H SWING STRUCTURE
    # ========================================================================

    sh_1h = detect_swing_highs(
        df_1h,
        "1h",
    )

    sl_1h = detect_swing_lows(
        df_1h,
        "1h",
    )

    if not sh_1h or not sl_1h:

        return reject(
            "Insufficient confirmed 1H swing structure"
        )

    sh_1h, sl_1h = label_market_structure(
        sh_1h,
        sl_1h,
    )

    if not sh_1h or not sl_1h:

        return reject(
            "Unable to classify 1H market structure"
        )

    # ========================================================================
    # 4H STRUCTURE
    # ========================================================================

    sh_4h = detect_swing_highs(
        df_4h,
        "4h",
    )

    sl_4h = detect_swing_lows(
        df_4h,
        "4h",
    )

    if not sh_4h or not sl_4h:

        return reject(
            "Insufficient confirmed 4H structure"
        )

    sh_4h, sl_4h = label_market_structure(
        sh_4h,
        sl_4h,
    )

    # ========================================================================
    # LIQUIDITY POOLS
    # ========================================================================

    try:

        pools_1h = detect_liquidity_pools(
            df_1h,
            sh_1h,
            sl_1h,
            "1h",
            pdh,
            pdl,
        )

    except TypeError:

        # Compatibility fallback for older pool API.
        pools_1h = detect_liquidity_pools(
            df_1h,
            sh_1h,
            sl_1h,
            "1h",
        )

    try:

        pools_4h = detect_liquidity_pools(
            df_4h,
            sh_4h,
            sl_4h,
            "4h",
            pdh,
            pdl,
        )

    except TypeError:

        pools_4h = detect_liquidity_pools(
            df_4h,
            sh_4h,
            sl_4h,
            "4h",
        )

    if not pools_1h:

        return reject(
            "No 1H liquidity pools detected"
        )

    # ========================================================================
    # LIQUIDITY SWEEP
    # ========================================================================

    sweep_direction = (
        SweepDirection.BULLISH
        if direction == "LONG"
        else SweepDirection.BEARISH
    )

    try:

        sweeps = detect_sweeps(
            df_1h,
            pools_1h,
            lookback=12,
        )

    except TypeError:

        sweeps = detect_sweeps(
            df_1h,
            pools_1h,
        )

    if not sweeps:

        return reject(
            "No 1H liquidity sweep detected"
        )

    directional_sweeps = [
        s
        for s in sweeps
        if getattr(s, "direction", None)
        == sweep_direction
    ]

    if not directional_sweeps:

        return reject(
            f"No {direction} liquidity sweep detected"
        )

    # Latest directional sweep.
    sweep = directional_sweeps[-1]

    # ========================================================================
    # SWEEP INDEX
    # ========================================================================

    sweep_idx = getattr(
        sweep,
        "break_candle_idx",
        None,
    )

    if sweep_idx is None:

        sweep_idx = getattr(
            sweep,
            "candle_idx",
            None,
        )

    if sweep_idx is None:

        sweep_idx = getattr(
            sweep,
            "sweep_candle_idx",
            None,
        )

    if sweep_idx is None:

        return reject(
            "Sweep event has no candle index"
        )

    try:

        sweep_idx = int(sweep_idx)

    except Exception:

        return reject(
            "Invalid sweep candle index"
        )

    if sweep_idx < 0:

        return reject(
            "Negative sweep candle index"
        )

    if sweep_idx >= len(df_1h):

        return reject(
            "Sweep candle index outside 1H data"
        )

    # Closed candle requirement.
    if sweep_idx >= len(df_1h) - 1:

        return reject(
            "Sweep is on unconfirmed/current candle"
        )

    # ========================================================================
    # ATR
    # ========================================================================

    try:

        atr_val = get_current_atr(
            df_1h
        )

    except Exception as exc:

        logger.error(
            "%s ATR calculation failed: %s",
            symbol,
            exc,
        )

        return reject(
            "ATR calculation failed"
        )

    if atr_val is None or atr_val <= 0:

        return reject(
            "Invalid 1H ATR"
        )

    atr_val = float(atr_val)

    current_price = float(
        df_1h["close"].iloc[-1]
    )

    # ========================================================================
    # OVEREXTENSION
    # ========================================================================

    sweep_level = getattr(
        getattr(sweep, "pool", None),
        "level",
        None,
    )

    if sweep_level is None:

        sweep_level = getattr(
            sweep,
            "liquidity_level",
            None,
        )

    if sweep_level is None:

        return reject(
            "Sweep liquidity level unavailable"
        )

    sweep_level = float(
        sweep_level
    )

    distance_from_sweep = abs(
        current_price - sweep_level
    )

    atr_overextended = (
        distance_from_sweep
        > 3.0 * atr_val
    )

    # ========================================================================
    # CHoCH — STRICTLY AFTER SWEEP
    # ========================================================================

    if direction == "LONG":

        choch_dir = (
            CHoCHDirection.BULLISH
        )

        bos_dir = (
            BOSDirection.BULLISH
        )

    else:

        choch_dir = (
            CHoCHDirection.BEARISH
        )

        bos_dir = (
            BOSDirection.BEARISH
        )

    try:

        choch_event = get_latest_choch(
            df=df_1h,
            swing_highs=sh_1h,
            swing_lows=sl_1h,
            direction=choch_dir,
            lookback=12,
            after_candle_idx=sweep_idx,
        )

    except TypeError:

        # If implementation uses positional names.
        choch_event = get_latest_choch(
            df_1h,
            sh_1h,
            sl_1h,
            choch_dir,
            lookback=12,
            after_candle_idx=sweep_idx,
        )

    if choch_event is None:

        return reject(
            f"No valid {direction} CHoCH after sweep"
        )

    choch_idx = getattr(
        choch_event,
        "break_candle_idx",
        None,
    )

    if choch_idx is None:

        return reject(
            "CHoCH event has no candle index"
        )

    choch_idx = int(
        choch_idx
    )

    # ------------------------------------------------------------------------
    # Strict chronology
    # ------------------------------------------------------------------------

    if choch_idx <= sweep_idx:

        return reject(
            "Invalid chronology: CHoCH must occur after sweep"
        )

    # ------------------------------------------------------------------------
    # CHoCH timing
    # ------------------------------------------------------------------------

    max_choch_distance = 8

    if (
        choch_idx - sweep_idx
        > max_choch_distance
    ):

        return reject(
            "CHoCH occurred too far after sweep"
        )

    # ========================================================================
    # BOS — STRICTLY AFTER CHoCH
    # ========================================================================

    try:

        bos_event = get_latest_bos(
            df=df_1h,
            swing_highs=sh_1h,
            swing_lows=sl_1h,
            lookback=12,
            direction=bos_dir,
            after_candle_idx=choch_idx,
        )

    except TypeError:

        bos_event = get_latest_bos(
            df_1h,
            sh_1h,
            sl_1h,
            lookback=12,
            direction=bos_dir,
            after_candle_idx=choch_idx,
        )

    if bos_event is None:

        return reject(
            f"No valid {direction} BOS after CHoCH"
        )

    bos_idx = getattr(
        bos_event,
        "break_candle_idx",
        None,
    )

    if bos_idx is None:

        return reject(
            "BOS event has no candle index"
        )

    bos_idx = int(
        bos_idx
    )

    # ------------------------------------------------------------------------
    # BOS chronology
    # ------------------------------------------------------------------------

    if bos_idx <= choch_idx:

        return reject(
            "Invalid chronology: BOS must occur after CHoCH"
        )

    if bos_idx <= sweep_idx:

        return reject(
            "Invalid chronology: BOS must occur after sweep"
        )

    # ------------------------------------------------------------------------
    # Same-break protection
    # ------------------------------------------------------------------------

    if (
        CHOCH_BOS_SAME_BREAK_FORBIDDEN
        and bos_idx == choch_idx
    ):

        return reject(
            "CHoCH and BOS cannot use same candle"
        )

    # ------------------------------------------------------------------------
    # BOS timing
    # ------------------------------------------------------------------------

    if (
        BOS_MAX_CANDLES_AFTER_CHOCH > 0
        and
        bos_idx - choch_idx
        > BOS_MAX_CANDLES_AFTER_CHOCH
    ):

        return reject(
            "BOS occurred too far after CHoCH"
        )

    # ========================================================================
    # SWEEP SCORE
    # ========================================================================

    try:

        score_result = score_sweep(
            sweep=sweep,
            h4_pools=pools_4h,
            h1_pools=pools_1h,
            choch_confirmed=True,
            bos_confirmed=True,
            atr_overextended=atr_overextended,
        )

    except TypeError:

        score_result = score_sweep(
            sweep,
            pools_4h,
            pools_1h,
            True,
            True,
            atr_overextended,
        )

    score_total = int(
        getattr(
            score_result,
            "total",
            0,
        )
    )

    score_valid = bool(
        getattr(
            score_result,
            "is_valid",
            score_total >= 70,
        )
    )

    if not score_valid:

        return reject(
            f"Liquidity sweep score rejected: "
            f"{score_total}/100"
        )

    # ========================================================================
    # RETEST
    # ========================================================================

    try:

        retest_ok = _check_retest(
            df_1h,
            sh_1h,
            sl_1h,
            bos_dir,
            atr_val,
        )

    except Exception as exc:

        logger.warning(
            "%s retest calculation failed: %s",
            symbol,
            exc,
        )

        retest_ok = False

    # ------------------------------------------------------------------------
    # Retest must happen after BOS.
    # ------------------------------------------------------------------------

    candles_after_bos = (
        len(df_1h)
        - 1
        - bos_idx
    )

    if candles_after_bos < 0:

        return reject(
            "Invalid BOS chronology relative to latest candle"
        )

    if (
        RETEST_MAX_CANDLES_AFTER_BOS_SWING > 0
        and
        candles_after_bos
        > RETEST_MAX_CANDLES_AFTER_BOS_SWING
    ):

        retest_ok = False

    if (
        RETEST_REQUIRED_SWING
        and not retest_ok
    ):

        return reject(
            "1H retest required but not confirmed"
        )

    # ========================================================================
    # VOLUME
    # ========================================================================

    try:

        volume_result = analyze_volume(
            df_1h
        )

        volume_ok = bool(
            getattr(
                volume_result,
                "is_confirmed",
                False,
            )
        )

    except Exception as exc:

        logger.warning(
            "%s volume analysis failed: %s",
            symbol,
            exc,
        )

        volume_ok = False

    # Strict institutional gate.
    if not volume_ok:

        return reject(
            "Volume confirmation failed"
        )

    # ========================================================================
    # ENTRY
    # ========================================================================

    entry = current_price

    refined_entry = _try_15m_refinement(
        df_15m=df_15m,
        direction=direction,
        current_price=current_price,
        atr_val=atr_val,
    )

    if refined_entry is not None:

        entry = float(
            refined_entry
        )

    # ========================================================================
    # SL / TP
    # ========================================================================

    sl, tp = _swing_sl_tp(
        direction=direction,
        entry=entry,
        sweep=sweep,
        sh_1h=sh_1h,
        sl_1h=sl_1h,
        atr_val=atr_val,
        pools_1h=pools_1h,
        df_1h=df_1h,
    )

    if sl is None or tp is None:

        return reject(
            "Unable to calculate valid SL/TP"
        )

    # ========================================================================
    # PRICE GEOMETRY
    # ========================================================================

    if direction == "LONG":

        if not (
            sl < entry < tp
        ):

            return reject(
                "Invalid LONG price geometry"
            )

    else:

        if not (
            tp < entry < sl
        ):

            return reject(
                "Invalid SHORT price geometry"
            )

    # ========================================================================
    # RR
    # ========================================================================

    rr = _calculate_rr(
        direction,
        entry,
        sl,
        tp,
    )

    if rr is None:

        return reject(
            "RR calculation failed"
        )

    rr = float(rr)

    if rr <= 0:

        return reject(
            "Invalid RR"
        )

    if rr < MIN_RR_SWING:

        return reject(
            f"RR {rr:.2f} below swing minimum "
            f"{MIN_RR_SWING:.2f}"
        )

    # ========================================================================
    # CHASE PROTECTION
    # ========================================================================

    try:

        chase_ok, chase_reason = (
            check_chase_protection(
                entry,
                sweep_level,
                atr_val,
                direction,
            )
        )

    except Exception as exc:

        logger.warning(
            "%s chase protection failed: %s",
            symbol,
            exc,
        )

        return reject(
            "Chase protection calculation failed"
        )

    if not chase_ok:

        return reject(
            chase_reason
        )

    # ========================================================================
    # QUALITY ENGINE
    # ========================================================================

    try:

        quality_result = classify_signal_quality(
            sweep_score=score_total,
            rr=rr,
            mode="SWING",
            mtf_status=mtf_ctx.status,
            retest_present=retest_ok,
            volume_confirmed=volume_ok,
            btc_downgraded=btc_filter.downgraded,
        )

    except TypeError:

        quality_result = classify_signal_quality(
            score_total,
            rr,
            "SWING",
            mtf_ctx.status,
            retest_ok,
            volume_ok,
            btc_filter.downgraded,
        )

    quality = getattr(
        quality_result,
        "quality",
        SignalQuality.REJECTED,
    )

    should_send = bool(
        getattr(
            quality_result,
            "should_send",
            False,
        )
    )

    quality_reason = getattr(
        quality_result,
        "reason",
        "",
    )

    # Only A / A+.
    if quality not in (
        SignalQuality.A,
        SignalQuality.A_PLUS,
    ):

        return reject(
            f"Quality rejected: {quality.value}"
        )

    if not should_send:

        return reject(
            quality_reason
            or "Quality engine rejected signal"
        )

    # ========================================================================
    # FINAL HARD GATES
    # ========================================================================

    if choch_event is None:

        return reject(
            "Final gate failed: CHoCH"
        )

    if bos_event is None:

        return reject(
            "Final gate failed: BOS"
        )

    if not retest_ok:

        return reject(
            "Final gate failed: Retest"
        )

    if not volume_ok:

        return reject(
            "Final gate failed: Volume"
        )

    if choch_idx <= sweep_idx:

        return reject(
            "Final gate failed: CHoCH chronology"
        )

    if bos_idx <= choch_idx:

        return reject(
            "Final gate failed: BOS chronology"
        )

    if (
        CHOCH_BOS_SAME_BREAK_FORBIDDEN
        and bos_idx == choch_idx
    ):

        return reject(
            "Final gate failed: CHoCH/BOS same candle"
        )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    sweep_text = (
        "Buy-side Liquidity Sweep"
        if direction == "LONG"
        else "Sell-side Liquidity Sweep"
    )

    choch_text = (
        "Bullish CHoCH"
        if direction == "LONG"
        else "Bearish CHoCH"
    )

    bos_text = (
        "Bullish BOS"
        if direction == "LONG"
        else "Bearish BOS"
    )

    m15_text = (
        "15M Entry Refined"
        if refined_entry is not None
        else "15M Context Confirmed"
    )

    m15_summary = (
        f"{sweep_text} | "
        f"{choch_text} | "
        f"{bos_text} | "
        f"Retest Confirmed | "
        f"Volume Confirmed | "
        f"{m15_text}"
    )

    # ========================================================================
    # FINAL VALID SETUP
    # ========================================================================

    return SetupResult(
        valid=True,
        direction=direction,
        mode="SWING",
        symbol=symbol,
        timeframe="1h",

        entry=round(
            float(entry),
            6,
        ),

        sl=round(
            float(sl),
            6,
        ),

        tp=round(
            float(tp),
            6,
        ),

        rr=round(
            float(rr),
            2,
        ),

        h4_summary=(
            mtf_ctx.h4_trend.description
            if getattr(mtf_ctx, "h4_trend", None)
            else ""
        ),

        h1_summary=(
            mtf_ctx.h1_trend.description
            if getattr(mtf_ctx, "h1_trend", None)
            else ""
        ),

        m15_summary=m15_summary,

        sweep_event=sweep,
        sweep_score=score_total,

        liquidity_type=(
            getattr(
                getattr(sweep, "pool", None),
                "pool_type",
                ""
            ).value
            if getattr(
                getattr(sweep, "pool", None),
                "pool_type",
                None
            ) is not None
            else ""
        ),

        liquidity_level=sweep_level,

        choch_confirmed=True,
        bos_confirmed=True,
        retest_confirmed=True,
        volume_confirmed=True,

        quality=quality,
        should_send=True,
        rejection_reason="",
    )


# ============================================================================
# 15M REFINEMENT
# ============================================================================

def _try_15m_refinement(
    df_15m: pd.DataFrame,
    direction: str,
    current_price: float,
    atr_val: float,
) -> Optional[float]:
    """
    15M is refinement only.

    It cannot override 4H/1H structure.
    """

    if df_15m is None or df_15m.empty:
        return None

    if len(df_15m) < 10:
        return None

    try:

        sh_15m = detect_swing_highs(
            df_15m,
            "15m",
        )

        sl_15m = detect_swing_lows(
            df_15m,
            "15m",
        )

        if direction == "LONG":

            if not sl_15m:
                return None

            recent_low = float(
                sl_15m[-1].price
            )

            distance = abs(
                current_price - recent_low
            )

            if distance <= 1.25 * atr_val:

                return float(
                    current_price
                )

        else:

            if not sh_15m:
                return None

            recent_high = float(
                sh_15m[-1].price
            )

            distance = abs(
                current_price - recent_high
            )

            if distance <= 1.25 * atr_val:

                return float(
                    current_price
                )

    except Exception as exc:

        logger.debug(
            "15M refinement unavailable: %s",
            exc,
        )

    return None


# ============================================================================
# SWING SL / TP
# ============================================================================

def _swing_sl_tp(
    direction: str,
    entry: float,
    sweep,
    sh_1h,
    sl_1h,
    atr_val: float,
    pools_1h,
    df_1h: pd.DataFrame,
):
    """
    Institutional swing SL/TP.

    LONG:
        SL below sweep low / structural low.

    SHORT:
        SL above sweep HIGH / structural high.

    The bearish case intentionally uses sweep_high.
    """

    try:

        if entry is None or entry <= 0:
            return None, None

        if atr_val is None or atr_val <= 0:
            return None, None

        buffer = (
            ATR_BUFFER_MULTIPLIER
            * atr_val
        )

        # ====================================================================
        # LONG
        # ====================================================================

        if direction == "LONG":

            sweep_low = float(
                sweep.sweep_low
            )

            structural_low = (
                float(sl_1h[-1].price)
                if sl_1h
                else sweep_low
            )

            structural_low = min(
                sweep_low,
                structural_low,
            )

            sl = (
                structural_low
                - buffer
            )

            # Find nearest meaningful buy-side liquidity
            # above entry.
            targets = []

            for pool in pools_1h:

                try:

                    pool_level = float(
                        pool.level
                    )

                    pool_side = (
                        pool.side.value
                        if hasattr(
                            pool.side,
                            "value",
                        )
                        else str(
                            pool.side
                        )
                    )

                    if (
                        pool_level > entry
                        and
                        "BUY" in pool_side.upper()
                    ):

                        targets.append(
                            pool_level
                        )

                except Exception:
                    continue

            if targets:

                tp = min(
                    targets
                )

            else:

                higher_highs = [
                    float(s.price)
                    for s in sh_1h
                    if float(s.price) > entry
                ]

                if higher_highs:

                    tp = min(
                        higher_highs
                    )

                else:

                    risk = (
                        entry - sl
                    )

                    tp = (
                        entry
                        + 4.0 * risk
                    )

        # ====================================================================
        # SHORT
        # ====================================================================

        else:

            # IMPORTANT:
            # Bearish setup invalidates above sweep HIGH.
            sweep_high = float(
                sweep.sweep_high
            )

            structural_high = (
                float(sh_1h[-1].price)
                if sh_1h
                else sweep_high
            )

            structural_high = max(
                sweep_high,
                structural_high,
            )

            sl = (
                structural_high
                + buffer
            )

            # Find nearest meaningful sell-side liquidity
            # below entry.
            targets = []

            for pool in pools_1h:

                try:

                    pool_level = float(
                        pool.level
                    )

                    pool_side = (
                        pool.side.value
                        if hasattr(
                            pool.side,
                            "value",
                        )
                        else str(
                            pool.side
                        )
                    )

                    if (
                        pool_level < entry
                        and
                        "SELL" in pool_side.upper()
                    ):

                        targets.append(
                            pool_level
                        )

                except Exception:
                    continue

            if targets:

                tp = max(
                    targets
                )

            else:

                lower_lows = [
                    float(s.price)
                    for s in sl_1h
                    if float(s.price) < entry
                ]

                if lower_lows:

                    tp = max(
                        lower_lows
                    )

                else:

                    risk = (
                        sl - entry
                    )

                    tp = (
                        entry
                        - 4.0 * risk
                    )

        # ====================================================================
        # FINAL GEOMETRY
        # ====================================================================

        if direction == "LONG":

            if not (
                sl < entry < tp
            ):

                return None, None

        else:

            if not (
                tp < entry < sl
            ):

                return None, None

        return (
            float(sl),
            float(tp),
        )

    except Exception as exc:

        logger.error(
            "Swing SL/TP calculation failed: %s",
            exc,
            exc_info=True,
        )

        return None, None


# ============================================================================
# MODULE SELF TEST
# ============================================================================

if __name__ == "__main__":

    print(
        "INSTITUTIONAL SWING ENGINE"
    )

    print(
        "Hierarchy       : 4H → 1H → 15M"
    )

    print(
        "Sweep → CHoCH   : REQUIRED"
    )

    print(
        "CHoCH → BOS     : REQUIRED"
    )

    print(
        "Same candle     : FORBIDDEN"
    )

    print(
        "Retest           : REQUIRED"
    )

    print(
        "Volume           : REQUIRED"
    )

    print(
        f"Minimum RR       : {MIN_RR_SWING}"
    )

    print(
        f"BOS max candles  : "
        f"{BOS_MAX_CANDLES_AFTER_CHOCH}"
    )

    print(
        "Signal execution : MANUAL ONLY"
    )

    print(
        "✓ Institutional swing engine loaded."
    )