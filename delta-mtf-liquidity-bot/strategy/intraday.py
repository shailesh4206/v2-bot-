"""
intraday.py — Intraday Strategy Pipeline (15M Primary Timeframe).

Full pipeline:
  4H Macro → 1H Trend/Structure → 15M Liquidity Pool → 15M Liquidity Sweep
  → 15M CHoCH → 15M BOS → 15M Retest → 15M Volume Confirmation
  → RR Check → MTF Validation → Signal

Signal labels: INTRADAY | 15M
Timeframe: 15M for entry, 4H+1H for context.
NO 5M anywhere.
"""

import logging
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

from strategy.mtf import MTFContext, MTFStatus, mtf_allows_long, mtf_allows_short
from strategy.filters import (
    apply_btc_correlation_filter, check_chase_protection,
    classify_signal_quality, SignalQuality
)
from market_structure.swings import detect_swing_highs, detect_swing_lows, label_market_structure
from market_structure.choch import detect_choch, CHoCHDirection, has_recent_choch
from market_structure.bos import detect_bos, BOSDirection, has_recent_bos
from liquidity.pools import detect_liquidity_pools, get_nearest_pool, LiquiditySide
from liquidity.sweep import detect_sweeps, SweepDirection, SweepEvent
from liquidity.scoring import score_sweep
from indicators.volatility import get_current_atr
from indicators.volume import analyze_volume
from config.strategy_config import (
    ATR_BUFFER_MULTIPLIER,
    MIN_RR_INTRADAY,
    RETEST_REQUIRED_INTRADAY,
    RETEST_TOLERANCE_ATR_MULTIPLIER,
)

logger = logging.getLogger(__name__)


@dataclass
class SetupResult:
    valid:             bool
    direction:         str          # "LONG" or "SHORT"
    mode:              str          # "INTRADAY"
    symbol:            str
    timeframe:         str          # "15m"
    entry:             Optional[float]
    sl:                Optional[float]
    tp:                Optional[float]
    rr:                Optional[float]

    # Context
    h4_summary:        str
    h1_summary:        str
    m15_summary:       str

    # Setup details
    sweep_event:       Optional[SweepEvent]
    sweep_score:       int
    liquidity_type:    str
    liquidity_level:   Optional[float]
    choch_confirmed:   bool
    bos_confirmed:     bool
    retest_confirmed:  bool
    volume_confirmed:  bool

    # Quality
    quality:           SignalQuality
    should_send:       bool
    rejection_reason:  str


def run_intraday_pipeline(
    symbol:     str,
    df_4h:      pd.DataFrame,
    df_1h:      pd.DataFrame,
    df_15m:     pd.DataFrame,
    mtf_ctx:    MTFContext,
    btc_h4_trend=None,
    btc_h1_trend=None,
    pdh:        Optional[float] = None,
    pdl:        Optional[float] = None,
) -> List[SetupResult]:
    """
    Run the full intraday strategy pipeline and return valid setups.

    Checks both LONG and SHORT setups.
    Returns a list because multiple independent setups can form simultaneously
    (though typically only 0 or 1 will be valid at any given time).
    """
    results = []

    for direction in ["LONG", "SHORT"]:
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

    return [r for r in results if r.valid and r.should_send]


def _evaluate_intraday(
    symbol:     str,
    direction:  str,
    df_4h:      pd.DataFrame,
    df_1h:      pd.DataFrame,
    df_15m:     pd.DataFrame,
    mtf_ctx:    MTFContext,
    btc_h4_trend=None,
    btc_h1_trend=None,
    pdh:        Optional[float] = None,
    pdl:        Optional[float] = None,
) -> SetupResult:
    """Evaluate a single directional intraday setup."""

    def reject(reason):
        return SetupResult(
            valid=False, direction=direction, mode="INTRADAY", symbol=symbol,
            timeframe="15m", entry=None, sl=None, tp=None, rr=None,
            h4_summary=mtf_ctx.h4_trend.description,
            h1_summary=mtf_ctx.h1_trend.description,
            m15_summary=mtf_ctx.m15_trend.description if mtf_ctx.m15_trend else "",
            sweep_event=None, sweep_score=0, liquidity_type="", liquidity_level=None,
            choch_confirmed=False, bos_confirmed=False, retest_confirmed=False,
            volume_confirmed=False, quality=SignalQuality.REJECTED,
            should_send=False, rejection_reason=reason,
        )

    # ── Step 1: MTF Validation ───────────────────────────
    if direction == "LONG":
        mtf_ok, mtf_reason = mtf_allows_long(mtf_ctx)
    else:
        mtf_ok, mtf_reason = mtf_allows_short(mtf_ctx)

    if not mtf_ok:
        return reject(f"MTF: {mtf_reason}")

    # ── Step 2: BTC Correlation Filter ───────────────────
    btc_filter = apply_btc_correlation_filter(symbol, direction, btc_h4_trend, btc_h1_trend)
    if btc_filter.blocked:
        return reject(btc_filter.reason)

    # ── Step 3: Detect Swings on 15M ─────────────────────
    sh_15m = detect_swing_highs(df_15m, "15m")
    sl_15m = detect_swing_lows(df_15m,  "15m")
    sh_15m, sl_15m = label_market_structure(sh_15m, sl_15m)

    if not sh_15m and not sl_15m:
        return reject("Insufficient 15M swing data")

    # ── Step 4: Detect Liquidity Pools ───────────────────
    sh_4h  = detect_swing_highs(df_4h, "4h")
    sl_4h  = detect_swing_lows(df_4h,  "4h")
    sh_1h  = detect_swing_highs(df_1h, "1h")
    sl_1h  = detect_swing_lows(df_1h,  "1h")

    pools_15m = detect_liquidity_pools(df_15m, sh_15m, sl_15m, "15m", pdh, pdl)
    pools_4h  = detect_liquidity_pools(df_4h,  sh_4h,  sl_4h,  "4h",  pdh, pdl)
    pools_1h  = detect_liquidity_pools(df_1h,  sh_1h,  sl_1h,  "1h",  pdh, pdl)

    if not pools_15m:
        return reject("No 15M liquidity pools detected")

    # ── Step 5: Detect Liquidity Sweep ───────────────────
    sweep_dir = SweepDirection.BULLISH if direction == "LONG" else SweepDirection.BEARISH
    sweeps    = detect_sweeps(df_15m, pools_15m, lookback=8)
    sweeps    = [s for s in sweeps if s.direction == sweep_dir]

    if not sweeps:
        return reject(f"No {direction} liquidity sweep detected on 15M")

    sweep = sweeps[-1]  # Most recent sweep

    # ── Step 6: Score the Sweep ──────────────────────────
    atr_val = get_current_atr(df_15m)

    # Check if overextended
    current_price  = df_15m["close"].iloc[-1]
    dist_from_sweep = abs(current_price - sweep.pool.level)
    atr_overextended = dist_from_sweep > 3.0 * atr_val if atr_val > 0 else False

    # CHoCH and BOS status at time of scoring
    choch_dir = CHoCHDirection.BULLISH if direction == "LONG" else CHoCHDirection.BEARISH
    bos_dir   = BOSDirection.BULLISH   if direction == "LONG" else BOSDirection.BEARISH

    choch_ok = has_recent_choch(df_15m, sh_15m, sl_15m, choch_dir, lookback_candles=15)
    bos_ok   = has_recent_bos(df_15m,   sh_15m, sl_15m, bos_dir,   lookback_candles=15)

    sweep_score_obj = score_sweep(
        sweep=sweep,
        h4_pools=pools_4h,
        h1_pools=pools_1h,
        choch_confirmed=choch_ok,
        bos_confirmed=bos_ok,
        atr_overextended=atr_overextended,
    )

    if not sweep_score_obj.is_valid:
        return reject(f"Sweep score too low: {sweep_score_obj.total}/100 — {sweep_score_obj.rejection_note}")

    # ── Step 7: Require CHoCH ────────────────────────────
    if not choch_ok:
        return reject(f"No {direction} CHoCH detected on 15M within 15 candles")

    # ── Step 8: Require BOS ──────────────────────────────
    if not bos_ok:
        return reject(f"No {direction} BOS confirmed on 15M within 15 candles")

    # ── Step 9: Retest Check ─────────────────────────────
    retest_ok = _check_retest(df_15m, sh_15m, sl_15m, bos_dir, atr_val)

    if RETEST_REQUIRED_INTRADAY and not retest_ok:
        return reject("Retest required but not confirmed")

    # ── Step 10: Volume Confirmation ─────────────────────
    vol_analysis = analyze_volume(df_15m)
    volume_ok    = vol_analysis.is_confirmed

    # ── Step 11: Entry / SL / TP ─────────────────────────
    entry, sl, tp = _calculate_entry_sl_tp(
        df_15m=df_15m,
        direction=direction,
        sweep=sweep,
        sh_15m=sh_15m,
        sl_15m=sl_15m,
        atr_val=atr_val,
        pools_15m=pools_15m,
    )

    if entry is None:
        return reject("Could not calculate Entry/SL/TP")

    rr = _calculate_rr(direction, entry, sl, tp)

    # ── Step 12: Chase Protection ────────────────────────
    chase_ok, chase_reason = check_chase_protection(entry, sweep.pool.level, atr_val, direction)
    if not chase_ok:
        return reject(chase_reason)

    # ── Step 13: RR Validation ───────────────────────────
    if rr < MIN_RR_INTRADAY:
        return reject(f"RR {rr:.2f} < minimum {MIN_RR_INTRADAY}")

    # ── Step 14: Quality Classification ─────────────────
    quality_result = classify_signal_quality(
        sweep_score=sweep_score_obj.total,
        rr=rr,
        mode="INTRADAY",
        mtf_status=mtf_ctx.status,
        retest_present=retest_ok,
        volume_confirmed=volume_ok,
        btc_downgraded=btc_filter.downgraded,
    )

    m15_summary = (
        f"{'Buy' if direction == 'LONG' else 'Sell'}-side Liquidity Sweep | "
        f"{'Bullish' if direction == 'LONG' else 'Bearish'} CHoCH | "
        f"{'Bullish' if direction == 'LONG' else 'Bearish'} BOS | "
        f"{'Retest Confirmed' if retest_ok else 'Retest Pending'} | "
        f"Volume {'✅' if volume_ok else '❌'}"
    )

    return SetupResult(
        valid=True,
        direction=direction,
        mode="INTRADAY",
        symbol=symbol,
        timeframe="15m",
        entry=round(entry, 6),
        sl=round(sl, 6),
        tp=round(tp, 6),
        rr=round(rr, 2),
        h4_summary=mtf_ctx.h4_trend.description,
        h1_summary=mtf_ctx.h1_trend.description,
        m15_summary=m15_summary,
        sweep_event=sweep,
        sweep_score=sweep_score_obj.total,
        liquidity_type=sweep.pool.pool_type.value,
        liquidity_level=sweep.pool.level,
        choch_confirmed=choch_ok,
        bos_confirmed=bos_ok,
        retest_confirmed=retest_ok,
        volume_confirmed=volume_ok,
        quality=quality_result.quality,
        should_send=quality_result.should_send,
        rejection_reason=quality_result.reason if not quality_result.should_send else "",
    )


def _check_retest(df, sh, sl, bos_dir, atr_val) -> bool:
    """Check if price has retested the broken BOS level."""
    from market_structure.bos import get_latest_bos
    bos = get_latest_bos(df, sh, sl, direction=bos_dir)
    if not bos:
        return False

    current_price = df["close"].iloc[-1]
    level = bos.broken_level
    tolerance = RETEST_TOLERANCE_ATR_MULTIPLIER * atr_val if atr_val > 0 else level * 0.002
    return abs(current_price - level) <= tolerance


def _calculate_entry_sl_tp(df_15m, direction, sweep, sh_15m, sl_15m, atr_val, pools_15m):
    """Calculate Entry, Stop Loss, and Take Profit levels."""
    try:
        current_price = df_15m["close"].iloc[-1]
        buf = ATR_BUFFER_MULTIPLIER * atr_val if atr_val > 0 else current_price * 0.002

        if direction == "LONG":
            entry = current_price
            # SL below the swept liquidity low or structural swing low
            struct_low = min(
                sweep.sweep_low,
                sl_15m[-1].price if sl_15m else sweep.sweep_low
            )
            sl = struct_low - buf

            # TP: nearest opposing buy-side liquidity above
            opp_pools = [p for p in pools_15m if p.side.value == "BUY_SIDE" and p.level > entry]
            if opp_pools:
                opp_pools.sort(key=lambda p: p.level)
                tp = opp_pools[0].level
            else:
                # Fallback: SH above
                highs_above = [s.price for s in sh_15m if s.price > entry]
                tp = min(highs_above) if highs_above else entry + 3 * (entry - sl)

        else:  # SHORT
            entry = current_price
            # SL above the swept liquidity high or structural swing high
            struct_high = max(
                sweep.sweep_low,  # For bearish, sweep_low holds the high wick
                sh_15m[-1].price if sh_15m else sweep.sweep_low
            )
            sl = struct_high + buf

            # TP: nearest opposing sell-side liquidity below
            opp_pools = [p for p in pools_15m if p.side.value == "SELL_SIDE" and p.level < entry]
            if opp_pools:
                opp_pools.sort(key=lambda p: p.level, reverse=True)
                tp = opp_pools[0].level
            else:
                lows_below = [s.price for s in sl_15m if s.price < entry]
                tp = max(lows_below) if lows_below else entry - 3 * (sl - entry)

        return entry, sl, tp

    except Exception as e:
        logger.error(f"Error calculating Entry/SL/TP: {e}")
        return None, None, None


def _calculate_rr(direction, entry, sl, tp) -> float:
    """Calculate Risk:Reward ratio."""
    if direction == "LONG":
        risk   = entry - sl
        reward = tp - entry
    else:
        risk   = sl - entry
        reward = entry - tp

    if risk <= 0:
        return 0.0
    return round(reward / risk, 2)
