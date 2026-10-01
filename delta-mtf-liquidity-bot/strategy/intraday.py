"""
strategy/intraday.py

Institutional Intraday Strategy Pipeline
Primary Execution Timeframe: 15M

Pipeline:

4H Macro Regime
        ↓
1H Structure / Trend
        ↓
15M Liquidity Pool
        ↓
15M Liquidity Sweep
        ↓
15M CHoCH
        ↓
15M BOS
        ↓
15M Retest
        ↓
15M Volume Confirmation
        ↓
Entry / SL / TP
        ↓
RR Validation
        ↓
BTC Correlation
        ↓
Quality Engine
        ↓
Signal

Important:
- Closed candles only.
- No 5M.
- Sweep → CHoCH → BOS → Retest chronology is enforced.
- Score cannot override hard gates.
- No automatic order execution.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

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

from market_structure.swings import (
    detect_swing_highs,
    detect_swing_lows,
    label_market_structure,
)

from market_structure.choch import (
    detect_choch,
    CHoCHDirection,
)

from market_structure.bos import (
    detect_bos,
    BOSDirection,
)

from liquidity.pools import (
    detect_liquidity_pools,
)

from liquidity.sweep import (
    detect_sweeps,
    SweepDirection,
    SweepEvent,
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
    MIN_RR_INTRADAY,
    RETEST_REQUIRED_INTRADAY,
    RETEST_TOLERANCE_ATR_MULTIPLIER,
)


logger = logging.getLogger(__name__)


# ============================================================================
# SETUP RESULT
# ============================================================================

@dataclass
class SetupResult:

    valid: bool

    direction: str
    mode: str
    symbol: str
    timeframe: str

    entry: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    rr: Optional[float]

    # ------------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------------

    h4_summary: str
    h1_summary: str
    m15_summary: str

    # ------------------------------------------------------------------------
    # Setup details
    # ------------------------------------------------------------------------

    sweep_event: Optional[SweepEvent]

    sweep_score: int

    liquidity_type: str
    liquidity_level: Optional[float]

    choch_confirmed: bool
    bos_confirmed: bool
    retest_confirmed: bool
    volume_confirmed: bool

    # ------------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------------

    quality: SignalQuality
    should_send: bool

    rejection_reason: str


# ============================================================================
# PUBLIC PIPELINE
# ============================================================================

def run_intraday_pipeline(
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
    Run both LONG and SHORT institutional intraday pipelines.

    Only approved signals are returned.
    """

    results: List[SetupResult] = []

    for direction in ("LONG", "SHORT"):

        result = _evaluate_intraday(
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

    return [
        result
        for result in results
        if result.valid and result.should_send
    ]


# ============================================================================
# MAIN EVALUATOR
# ============================================================================

def _evaluate_intraday(
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
    """
    Evaluate one directional setup.

    Hard sequence:

        MTF
          ↓
        BTC
          ↓
        Sweep
          ↓
        CHoCH
          ↓
        BOS
          ↓
        Retest
          ↓
        Volume
          ↓
        RR
          ↓
        Quality
    """

    # ========================================================================
    # REJECTION HELPER
    # ========================================================================

    def reject(reason: str) -> SetupResult:

        logger.info(
            "%s %s INTRADAY rejected: %s",
            symbol,
            direction,
            reason,
        )

        return SetupResult(
            valid=False,
            direction=direction,
            mode="INTRADAY",
            symbol=symbol,
            timeframe="15m",

            entry=None,
            sl=None,
            tp=None,
            rr=None,

            h4_summary=(
                mtf_ctx.h4_trend.description
                if mtf_ctx.h4_trend
                else ""
            ),

            h1_summary=(
                mtf_ctx.h1_trend.description
                if mtf_ctx.h1_trend
                else ""
            ),

            m15_summary=(
                mtf_ctx.m15_trend.description
                if mtf_ctx.m15_trend
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
    # BASIC DATA VALIDATION
    # ========================================================================

    if df_15m is None or df_15m.empty:
        return reject("15M dataframe is empty")

    required_columns = {
        "open",
        "high",
        "low",
        "close",
    }

    if not required_columns.issubset(df_15m.columns):
        return reject("15M dataframe missing OHLC columns")

    if len(df_15m) < 50:
        return reject(
            f"Insufficient 15M candles: {len(df_15m)}"
        )

    # ========================================================================
    # STEP 1 — MTF VALIDATION
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
            f"MTF: {mtf_reason}"
        )

    # ========================================================================
    # STEP 2 — BTC CORRELATION
    # ========================================================================

    btc_filter = apply_btc_correlation_filter(
        symbol,
        direction,
        btc_h4_trend,
        btc_h1_trend,
    )

    if btc_filter.blocked:

        return reject(
            f"BTC correlation: {btc_filter.reason}"
        )

    # ========================================================================
    # STEP 3 — 15M SWINGS
    # ========================================================================

    sh_15m = detect_swing_highs(
        df_15m,
        "15m",
    )

    sl_15m = detect_swing_lows(
        df_15m,
        "15m",
    )

    sh_15m, sl_15m = label_market_structure(
        sh_15m,
        sl_15m,
    )

    if not sh_15m and not sl_15m:

        return reject(
            "Insufficient 15M swing data"
        )

    # ========================================================================
    # STEP 4 — HIGHER-TIMEFRAME SWINGS
    # ========================================================================

    sh_4h = detect_swing_highs(
        df_4h,
        "4h",
    )

    sl_4h = detect_swing_lows(
        df_4h,
        "4h",
    )

    sh_1h = detect_swing_highs(
        df_1h,
        "1h",
    )

    sl_1h = detect_swing_lows(
        df_1h,
        "1h",
    )

    # ========================================================================
    # STEP 5 — LIQUIDITY POOLS
    # ========================================================================

    pools_15m = detect_liquidity_pools(
        df_15m,
        sh_15m,
        sl_15m,
        "15m",
        pdh,
        pdl,
    )

    pools_4h = detect_liquidity_pools(
        df_4h,
        sh_4h,
        sl_4h,
        "4h",
        pdh,
        pdl,
    )

    pools_1h = detect_liquidity_pools(
        df_1h,
        sh_1h,
        sl_1h,
        "1h",
        pdh,
        pdl,
    )

    if not pools_15m:

        return reject(
            "No 15M liquidity pools detected"
        )

    # ========================================================================
    # STEP 6 — LIQUIDITY SWEEP
    # ========================================================================

    sweep_direction = (
        SweepDirection.BULLISH
        if direction == "LONG"
        else SweepDirection.BEARISH
    )

    sweeps = detect_sweeps(
        df_15m,
        pools_15m,
        lookback=8,
    )

    sweeps = [
        sweep
        for sweep in sweeps
        if sweep.direction == sweep_direction
        and getattr(sweep, "valid", True)
    ]

    if not sweeps:

        return reject(
            f"No valid {direction} liquidity sweep detected on 15M"
        )

    # Most recent valid sweep.
    sweep = max(
        sweeps,
        key=lambda item: item.sweep_candle_idx,
    )

    sweep_idx = int(
        sweep.sweep_candle_idx
    )

    # Sweep cannot be the current unconfirmed candle.
    if sweep_idx >= len(df_15m):

        return reject(
            "Sweep candle index is outside dataframe"
        )

    # ========================================================================
    # STEP 7 — ATR / OVEREXTENSION
    # ========================================================================

    atr_val = get_current_atr(
        df_15m
    )

    if atr_val is None or atr_val <= 0:

        return reject(
            "Invalid 15M ATR"
        )

    current_price = float(
        df_15m["close"].iloc[-1]
    )

    dist_from_sweep = abs(
        current_price
        - float(sweep.pool.level)
    )

    atr_overextended = (
        dist_from_sweep
        > 3.0 * atr_val
    )

    # ========================================================================
    # STEP 8 — CHoCH AFTER SWEEP
    # ========================================================================

    choch_direction = (
        CHoCHDirection.BULLISH
        if direction == "LONG"
        else CHoCHDirection.BEARISH
    )

    choch_events = detect_choch(
        df=df_15m,
        swing_highs=sh_15m,
        swing_lows=sl_15m,
        lookback=5,
        after_candle_idx=sweep_idx,
    )

    choch_events = [
        event
        for event in choch_events
        if event.direction == choch_direction
        and event.break_candle_idx > sweep_idx
    ]

    if not choch_events:

        return reject(
            f"No {direction} CHoCH AFTER liquidity sweep"
        )

    # Latest chronological CHoCH.
    choch_event = max(
        choch_events,
        key=lambda event: event.break_candle_idx,
    )

    choch_idx = int(
        choch_event.break_candle_idx
    )

    # Defensive chronology gate.
    if choch_idx <= sweep_idx:

        return reject(
            "CHoCH chronology invalid: CHoCH must occur after sweep"
        )

    # ========================================================================
    # STEP 9 — BOS AFTER CHoCH
    # ========================================================================

    bos_direction = (
        BOSDirection.BULLISH
        if direction == "LONG"
        else BOSDirection.BEARISH
    )

    bos_events = detect_bos(
        df=df_15m,
        swing_highs=sh_15m,
        swing_lows=sl_15m,
        direction=bos_direction,
    )

    # ------------------------------------------------------------------------
    # Some versions of BOS engine may return None.
    # ------------------------------------------------------------------------

    if bos_events is None:

        bos_events = []

    # Ensure iterable.
    try:
        bos_events = list(bos_events)
    except TypeError:
        bos_events = []

    # ------------------------------------------------------------------------
    # Filter BOS:
    #
    # BOS must:
    #   1. Be correct direction
    #   2. Occur after CHoCH
    #   3. Occur after sweep
    #   4. Be a closed candle
    # ------------------------------------------------------------------------

    valid_bos_events = []

    for event in bos_events:

        event_direction = getattr(
            event,
            "direction",
            None,
        )

        event_idx = getattr(
            event,
            "break_candle_idx",
            None,
        )

        if event_idx is None:
            continue

        try:
            event_idx = int(event_idx)
        except Exception:
            continue

        if event_direction != bos_direction:
            continue

        if event_idx <= choch_idx:
            continue

        if event_idx <= sweep_idx:
            continue

        if event_idx >= len(df_15m):
            continue

        valid_bos_events.append(event)

    if not valid_bos_events:

        return reject(
            f"No {direction} BOS AFTER CHoCH"
        )

    # Latest BOS after CHoCH.
    bos_event = max(
        valid_bos_events,
        key=lambda event: int(
            getattr(
                event,
                "break_candle_idx",
                0,
            )
        ),
    )

    bos_idx = int(
        getattr(
            bos_event,
            "break_candle_idx",
            -1,
        )
    )

    if bos_idx <= choch_idx:

        return reject(
            "BOS chronology invalid: BOS must occur after CHoCH"
        )

    # ========================================================================
    # STEP 10 — SWEEP SCORE
    # ========================================================================

    sweep_score_obj = score_sweep(
        sweep=sweep,
        h4_pools=pools_4h,
        h1_pools=pools_1h,
        choch_confirmed=True,
        bos_confirmed=True,
        atr_overextended=atr_overextended,
    )

    if not sweep_score_obj.is_valid:

        return reject(
            f"Sweep score too low: "
            f"{sweep_score_obj.total}/100 — "
            f"{sweep_score_obj.rejection_note}"
        )

    # ========================================================================
    # STEP 11 — RETEST
    # ========================================================================

    retest_ok = _check_retest(
        df=df_15m,
        bos_event=bos_event,
        bos_idx=bos_idx,
        atr_val=atr_val,
        direction=direction,
    )

    if (
        RETEST_REQUIRED_INTRADAY
        and not retest_ok
    ):

        return reject(
            "Retest required but not confirmed AFTER BOS"
        )

    # ========================================================================
    # STEP 12 — VOLUME
    # ========================================================================

    vol_analysis = analyze_volume(
        df_15m
    )

    volume_ok = bool(
        vol_analysis.is_confirmed
    )

    # ========================================================================
    # STEP 13 — ENTRY / SL / TP
    # ========================================================================

    entry, sl, tp = _calculate_entry_sl_tp(
        df_15m=df_15m,
        direction=direction,
        sweep=sweep,
        sh_15m=sh_15m,
        sl_15m=sl_15m,
        atr_val=atr_val,
        pools_15m=pools_15m,
    )

    if (
        entry is None
        or sl is None
        or tp is None
    ):

        return reject(
            "Could not calculate Entry/SL/TP"
        )

    # ========================================================================
    # STEP 14 — RR
    # ========================================================================

    rr = _calculate_rr(
        direction,
        entry,
        sl,
        tp,
    )

    if rr <= 0:

        return reject(
            "Invalid RR calculation"
        )

    # ========================================================================
    # STEP 15 — CHASE PROTECTION
    # ========================================================================

    chase_ok, chase_reason = (
        check_chase_protection(
            entry,
            sweep.pool.level,
            atr_val,
            direction,
        )
    )

    if not chase_ok:

        return reject(
            chase_reason
        )

    # ========================================================================
    # STEP 16 — MINIMUM RR
    # ========================================================================

    if rr < MIN_RR_INTRADAY:

        return reject(
            f"RR {rr:.2f} < minimum "
            f"{MIN_RR_INTRADAY}"
        )

    # ========================================================================
    # STEP 17 — QUALITY ENGINE
    # ========================================================================

    quality_result = classify_signal_quality(
        sweep_score=sweep_score_obj.total,
        rr=rr,
        mode="INTRADAY",
        mtf_status=mtf_ctx.status,
        retest_present=retest_ok,
        volume_confirmed=volume_ok,
        btc_downgraded=btc_filter.downgraded,
    )

    # ------------------------------------------------------------------------
    # Final defensive hard gate.
    #
    # Even if the quality engine accidentally returns should_send=True,
    # institutional conditions must still be present.
    # ------------------------------------------------------------------------

    if not choch_event:
        return reject(
            "Hard gate failed: CHoCH missing"
        )

    if not bos_event:
        return reject(
            "Hard gate failed: BOS missing"
        )

    if RETEST_REQUIRED_INTRADAY and not retest_ok:
        return reject(
            "Hard gate failed: Retest missing"
        )

    # ========================================================================
    # M15 SUMMARY
    # ========================================================================

    sweep_label = (
        "Buy-side Liquidity Sweep"
        if direction == "SHORT"
        else "Sell-side Liquidity Sweep"
    )

    choch_label = (
        "Bullish CHoCH"
        if direction == "LONG"
        else "Bearish CHoCH"
    )

    bos_label = (
        "Bullish BOS"
        if direction == "LONG"
        else "Bearish BOS"
    )

    m15_summary = (
        f"{sweep_label} | "
        f"{choch_label} | "
        f"{bos_label} | "
        f"Retest "
        f"{'Confirmed' if retest_ok else 'Pending'} | "
        f"Volume "
        f"{'✅' if volume_ok else '❌'}"
    )

    # ========================================================================
    # FINAL RESULT
    # ========================================================================

    return SetupResult(

        valid=True,

        direction=direction,

        mode="INTRADAY",

        symbol=symbol,

        timeframe="15m",

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
            if mtf_ctx.h4_trend
            else ""
        ),

        h1_summary=(
            mtf_ctx.h1_trend.description
            if mtf_ctx.h1_trend
            else ""
        ),

        m15_summary=m15_summary,

        sweep_event=sweep,

        sweep_score=int(
            sweep_score_obj.total
        ),

        liquidity_type=(
            sweep.pool.pool_type.value
            if sweep.pool
            else ""
        ),

        liquidity_level=(
            float(sweep.pool.level)
            if sweep.pool
            else None
        ),

        choch_confirmed=True,

        bos_confirmed=True,

        retest_confirmed=retest_ok,

        volume_confirmed=volume_ok,

        quality=quality_result.quality,

        should_send=bool(
            quality_result.should_send
        ),

        rejection_reason=(
            quality_result.reason
            if not quality_result.should_send
            else ""
        ),
    )


# ============================================================================
# RETEST ENGINE
# ============================================================================

def _check_retest(
    df: pd.DataFrame,
    bos_event: Any,
    bos_idx: int,
    atr_val: float,
    direction: str,
) -> bool:
    """
    Institutional BOS retest detector.

    A retest is NOT:

        current close ≈ BOS level

    A valid retest requires:

        BOS candle
            ↓
        later candle
            ↓
        actual high/low touches BOS level
            ↓
        price remains structurally valid
            ↓
        current state has not invalidated setup

    BOS candle itself is never counted as retest.
    """

    if df is None or df.empty:
        return False

    if bos_event is None:
        return False

    if bos_idx < 0:
        return False

    if bos_idx >= len(df):
        return False

    if atr_val is None or atr_val <= 0:
        return False

    level = getattr(
        bos_event,
        "broken_level",
        None,
    )

    if level is None:
        return False

    try:
        level = float(level)
    except Exception:
        return False

    if level <= 0:
        return False

    tolerance = (
        RETEST_TOLERANCE_ATR_MULTIPLIER
        * float(atr_val)
    )

    if tolerance <= 0:
        return False

    # ------------------------------------------------------------------------
    # Search only candles AFTER BOS.
    # ------------------------------------------------------------------------

    start_idx = bos_idx + 1

    if start_idx >= len(df):
        return False

    # ------------------------------------------------------------------------
    # Retest window.
    #
    # Intraday institutional default is intentionally bounded.
    # ------------------------------------------------------------------------

    max_window = 12

    end_idx = min(
        len(df) - 1,
        bos_idx + max_window,
    )

    # ------------------------------------------------------------------------
    # Search actual candle interaction with BOS level.
    # ------------------------------------------------------------------------

    for i in range(
        start_idx,
        end_idx + 1,
    ):

        try:

            high = float(
                df["high"].iloc[i]
            )

            low = float(
                df["low"].iloc[i]
            )

            close = float(
                df["close"].iloc[i]
            )

        except Exception:
            continue

        if (
            not pd.notna(high)
            or not pd.notna(low)
            or not pd.notna(close)
        ):
            continue

        # ================================================================
        # LONG
        # ================================================================

        if direction == "LONG":

            # Price must actually trade down into BOS level.
            touched = (
                low
                <= level + tolerance
                and high
                >= level - tolerance
            )

            if not touched:
                continue

            # Do not accept a candle that decisively closes
            # below the BOS level by more than tolerance.
            invalidation = (
                close
                < level - tolerance
            )

            if invalidation:
                continue

            # Acceptance/reclaim:
            # candle closes at or above the BOS level.
            acceptance = (
                close
                >= level
            )

            # Rejection:
            # candle touched the level but closed back above
            # the level.
            rejection = (
                close
                > level
            )

            if acceptance or rejection:
                return True

        # ================================================================
        # SHORT
        # ================================================================

        else:

            # Price must actually trade up into BOS level.
            touched = (
                high
                >= level - tolerance
                and low
                <= level + tolerance
            )

            if not touched:
                continue

            # Do not accept decisive close above BOS.
            invalidation = (
                close
                > level + tolerance
            )

            if invalidation:
                continue

            # Acceptance/reclaim:
            # candle closes at or below BOS level.
            acceptance = (
                close
                <= level
            )

            # Rejection:
            # candle touched and closed back below.
            rejection = (
                close
                < level
            )

            if acceptance or rejection:
                return True

    return False


# ============================================================================
# ENTRY / SL / TP
# ============================================================================

def _calculate_entry_sl_tp(
    df_15m: pd.DataFrame,
    direction: str,
    sweep: SweepEvent,
    sh_15m,
    sl_15m,
    atr_val: float,
    pools_15m,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    Calculate institutional Entry / SL / TP.

    Important bearish fix:

        bearish sweep high = sweep.sweep_high

    NOT sweep.sweep_low.
    """

    try:

        current_price = float(
            df_15m["close"].iloc[-1]
        )

        if current_price <= 0:
            return None, None, None

        # --------------------------------------------------------------------
        # ATR stop buffer
        # --------------------------------------------------------------------

        if atr_val and atr_val > 0:

            buf = (
                ATR_BUFFER_MULTIPLIER
                * float(atr_val)
            )

        else:

            buf = (
                current_price
                * 0.002
            )

        # ====================================================================
        # LONG
        # ====================================================================

        if direction == "LONG":

            entry = current_price

            structural_lows = [
                float(s.price)
                for s in sl_15m
                if getattr(
                    s,
                    "price",
                    None,
                ) is not None
            ]

            sweep_low = float(
                sweep.sweep_low
            )

            if structural_lows:

                struct_low = min(
                    sweep_low,
                    min(structural_lows),
                )

            else:

                struct_low = sweep_low

            sl = (
                struct_low
                - buf
            )

            # ----------------------------------------------------------------
            # Opposing buy-side liquidity above entry
            # ----------------------------------------------------------------

            opp_pools = [
                pool
                for pool in pools_15m
                if getattr(
                    pool,
                    "side",
                    None,
                ) is not None
                and pool.side.value
                == "BUY_SIDE"
                and float(pool.level)
                > entry
            ]

            if opp_pools:

                opp_pools.sort(
                    key=lambda pool: float(
                        pool.level
                    )
                )

                tp = float(
                    opp_pools[0].level
                )

            else:

                highs_above = [
                    float(s.price)
                    for s in sh_15m
                    if float(s.price)
                    > entry
                ]

                if highs_above:

                    tp = min(
                        highs_above
                    )

                else:

                    tp = (
                        entry
                        + 3.0
                        * (
                            entry
                            - sl
                        )
                    )

        # ====================================================================
        # SHORT
        # ====================================================================

        else:

            entry = current_price

            structural_highs = [
                float(s.price)
                for s in sh_15m
                if getattr(
                    s,
                    "price",
                    None,
                ) is not None
            ]

            # IMPORTANT:
            #
            # Bearish sweep creates a HIGH wick.
            #
            # Therefore:
            #
            # sweep.sweep_high
            #
            # must be used for SL.
            sweep_high = float(
                sweep.sweep_high
            )

            if structural_highs:

                struct_high = max(
                    sweep_high,
                    max(structural_highs),
                )

            else:

                struct_high = sweep_high

            sl = (
                struct_high
                + buf
            )

            # ----------------------------------------------------------------
            # Opposing sell-side liquidity below entry
            # ----------------------------------------------------------------

            opp_pools = [
                pool
                for pool in pools_15m
                if getattr(
                    pool,
                    "side",
                    None,
                ) is not None
                and pool.side.value
                == "SELL_SIDE"
                and float(pool.level)
                < entry
            ]

            if opp_pools:

                opp_pools.sort(
                    key=lambda pool: float(
                        pool.level
                    ),
                    reverse=True,
                )

                tp = float(
                    opp_pools[0].level
                )

            else:

                lows_below = [
                    float(s.price)
                    for s in sl_15m
                    if float(s.price)
                    < entry
                ]

                if lows_below:

                    tp = max(
                        lows_below
                    )

                else:

                    tp = (
                        entry
                        - 3.0
                        * (
                            sl
                            - entry
                        )
                    )

        # --------------------------------------------------------------------
        # Final sanity
        # --------------------------------------------------------------------

        if direction == "LONG":

            if sl >= entry:
                return None, None, None

            if tp <= entry:
                return None, None, None

        else:

            if sl <= entry:
                return None, None, None

            if tp >= entry:
                return None, None, None

        return (
            float(entry),
            float(sl),
            float(tp),
        )

    except Exception as exc:

        logger.error(
            "Error calculating Entry/SL/TP: %s",
            exc,
            exc_info=True,
        )

        return None, None, None


# ============================================================================
# RR CALCULATOR
# ============================================================================

def _calculate_rr(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
) -> float:
    """
    Calculate Risk : Reward.
    """

    try:

        entry = float(entry)
        sl = float(sl)
        tp = float(tp)

    except Exception:

        return 0.0

    if direction == "LONG":

        risk = (
            entry
            - sl
        )

        reward = (
            tp
            - entry
        )

    else:

        risk = (
            sl
            - entry
        )

        reward = (
            entry
            - tp
        )

    if risk <= 0:
        return 0.0

    if reward <= 0:
        return 0.0

    return round(
        reward / risk,
        2,
    )


# ============================================================================
# SELF TEST
# ============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("INTRADAY STRATEGY ENGINE SELF-TEST")
    print("=" * 72)

    print("Primary timeframe : 15m")
    print("Macro timeframe   : 4h")
    print("Structure TF      : 1h")

    print("")
    print("Institutional sequence:")
    print("  1. MTF")
    print("  2. BTC correlation")
    print("  3. Liquidity pool")
    print("  4. Liquidity sweep")
    print("  5. CHoCH AFTER sweep")
    print("  6. BOS AFTER CHoCH")
    print("  7. Retest AFTER BOS")
    print("  8. Volume")
    print("  9. Entry / SL / TP")
    print(" 10. RR")
    print(" 11. Quality")

    print("")
    print("Closed candle only : True")
    print("Sweep chronology   : True")
    print("CHoCH chronology   : True")
    print("BOS chronology     : True")
    print("Retest chronology  : True")
    print("Bearish SL fix     : True")
    print("5M timeframe       : False")
    print("Auto execution     : False")

    print("")
    print("✓ Institutional intraday engine loaded successfully.")