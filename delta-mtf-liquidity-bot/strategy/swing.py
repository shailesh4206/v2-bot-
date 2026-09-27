"""
swing.py — Swing Strategy Pipeline (1H Primary Timeframe).

Full pipeline:
  4H Macro → 1H Trend/Structure → 1H Liquidity Pool → 1H Liquidity Sweep
  → 1H CHoCH → 1H BOS → Optional 15M Entry Refinement
  → RR Check → MTF Validation → Signal

Signal labels: SWING | 1H
The actual swing structure is based on 1H. 15M is optional refinement only.
NO 5M anywhere.
"""

import logging
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

from strategy.mtf import MTFContext, mtf_allows_long, mtf_allows_short
from strategy.filters import (
    apply_btc_correlation_filter, check_chase_protection,
    classify_signal_quality, SignalQuality
)
from strategy.intraday import _check_retest, _calculate_rr, SetupResult
from market_structure.swings import detect_swing_highs, detect_swing_lows, label_market_structure
from market_structure.choch import CHoCHDirection, has_recent_choch
from market_structure.bos import BOSDirection, has_recent_bos, get_latest_bos
from liquidity.pools import detect_liquidity_pools, LiquiditySide
from liquidity.sweep import detect_sweeps, SweepDirection
from liquidity.scoring import score_sweep
from indicators.volatility import get_current_atr
from indicators.volume import analyze_volume
from config.strategy_config import (
    ATR_BUFFER_MULTIPLIER,
    MIN_RR_SWING,
    RETEST_REQUIRED_SWING,
    RETEST_TOLERANCE_ATR_MULTIPLIER,
)

logger = logging.getLogger(__name__)


def run_swing_pipeline(
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
    """Run the full swing strategy pipeline for both directions."""
    results = []

    for direction in ["LONG", "SHORT"]:
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

    return [r for r in results if r.valid and r.should_send]


def _evaluate_swing(
    symbol, direction, df_4h, df_1h, df_15m, mtf_ctx,
    btc_h4_trend, btc_h1_trend, pdh, pdl,
) -> SetupResult:
    """Evaluate a single directional swing setup on 1H."""

    def reject(reason):
        return SetupResult(
            valid=False, direction=direction, mode="SWING", symbol=symbol,
            timeframe="1h", entry=None, sl=None, tp=None, rr=None,
            h4_summary=mtf_ctx.h4_trend.description,
            h1_summary=mtf_ctx.h1_trend.description,
            m15_summary=mtf_ctx.m15_trend.description if mtf_ctx.m15_trend else "",
            sweep_event=None, sweep_score=0, liquidity_type="", liquidity_level=None,
            choch_confirmed=False, bos_confirmed=False, retest_confirmed=False,
            volume_confirmed=False, quality=SignalQuality.REJECTED,
            should_send=False, rejection_reason=reason,
        )

    # ── Step 1: MTF — require 4H bullish for swing long ───────
    if direction == "LONG":
        if mtf_ctx.h4_trend.bias.value == "BEARISH":
            return reject("Swing LONG requires 4H bullish — 4H is BEARISH")
        mtf_ok, mtf_reason = mtf_allows_long(mtf_ctx)
    else:
        if mtf_ctx.h4_trend.bias.value == "BULLISH":
            return reject("Swing SHORT requires 4H bearish — 4H is BULLISH")
        mtf_ok, mtf_reason = mtf_allows_short(mtf_ctx)

    if not mtf_ok:
        return reject(f"MTF: {mtf_reason}")

    # ── Step 2: BTC Correlation ────────────────────────────────
    btc_filter = apply_btc_correlation_filter(symbol, direction, btc_h4_trend, btc_h1_trend)
    if btc_filter.blocked:
        return reject(btc_filter.reason)

    # ── Step 3: 1H Swing Detection ────────────────────────────
    sh_1h = detect_swing_highs(df_1h, "1h")
    sl_1h = detect_swing_lows(df_1h,  "1h")
    sh_1h, sl_1h = label_market_structure(sh_1h, sl_1h)

    if not sh_1h and not sl_1h:
        return reject("Insufficient 1H swing data")

    # ── Step 4: 1H Liquidity Pools ────────────────────────────
    sh_4h = detect_swing_highs(df_4h, "4h")
    sl_4h = detect_swing_lows(df_4h,  "4h")
    pools_1h = detect_liquidity_pools(df_1h, sh_1h, sl_1h, "1h", pdh, pdl)
    pools_4h = detect_liquidity_pools(df_4h, sh_4h, sl_4h, "4h", pdh, pdl)

    if not pools_1h:
        return reject("No 1H liquidity pools detected")

    # ── Step 5: 1H Liquidity Sweep ────────────────────────────
    sweep_dir = SweepDirection.BULLISH if direction == "LONG" else SweepDirection.BEARISH
    sweeps = detect_sweeps(df_1h, pools_1h, lookback=8)
    sweeps = [s for s in sweeps if s.direction == sweep_dir]

    if not sweeps:
        return reject(f"No {direction} liquidity sweep on 1H")

    sweep = sweeps[-1]
    atr_val = get_current_atr(df_1h)

    # ── Step 6: Score Sweep ───────────────────────────────────
    current_price = df_1h["close"].iloc[-1]
    atr_overext   = abs(current_price - sweep.pool.level) > 3 * atr_val if atr_val > 0 else False

    choch_dir = CHoCHDirection.BULLISH if direction == "LONG" else CHoCHDirection.BEARISH
    bos_dir   = BOSDirection.BULLISH   if direction == "LONG" else BOSDirection.BEARISH

    choch_ok = has_recent_choch(df_1h, sh_1h, sl_1h, choch_dir, lookback_candles=12)
    bos_ok   = has_recent_bos(df_1h,   sh_1h, sl_1h, bos_dir,   lookback_candles=12)

    score = score_sweep(
        sweep=sweep,
        h4_pools=pools_4h,
        h1_pools=pools_1h,
        choch_confirmed=choch_ok,
        bos_confirmed=bos_ok,
        atr_overextended=atr_overext,
    )

    if not score.is_valid:
        return reject(f"1H sweep score too low: {score.total}/100")

    # ── Step 7: CHoCH and BOS ────────────────────────────────
    if not choch_ok:
        return reject(f"No 1H {direction} CHoCH detected")

    if not bos_ok:
        return reject(f"No 1H {direction} BOS confirmed")

    # ── Step 8: Optional 15M Entry Refinement ────────────────
    refined_entry = _try_15m_refinement(df_15m, direction, atr_val, current_price)
    entry = refined_entry if refined_entry else current_price

    # ── Step 9: Retest Check ─────────────────────────────────
    retest_ok = _check_retest(df_1h, sh_1h, sl_1h, bos_dir, atr_val)
    if RETEST_REQUIRED_SWING and not retest_ok:
        return reject("1H retest required but not confirmed")

    # ── Step 10: Volume ───────────────────────────────────────
    vol_analysis = analyze_volume(df_1h)
    volume_ok    = vol_analysis.is_confirmed

    # ── Step 11: Entry / SL / TP ─────────────────────────────
    sl, tp = _swing_sl_tp(direction, entry, sweep, sh_1h, sl_1h, atr_val, pools_1h, df_1h)
    if sl is None:
        return reject("Could not calculate swing SL/TP")

    rr = _calculate_rr(direction, entry, sl, tp)

    # ── Step 12: Chase Protection ────────────────────────────
    chase_ok, chase_reason = check_chase_protection(entry, sweep.pool.level, atr_val, direction)
    if not chase_ok:
        return reject(chase_reason)

    if rr < MIN_RR_SWING:
        return reject(f"Swing RR {rr:.2f} < minimum {MIN_RR_SWING}")

    # ── Step 13: Quality ──────────────────────────────────────
    quality_result = classify_signal_quality(
        sweep_score=score.total,
        rr=rr,
        mode="SWING",
        mtf_status=mtf_ctx.status,
        retest_present=retest_ok,
        volume_confirmed=volume_ok,
        btc_downgraded=btc_filter.downgraded,
    )

    m15_note = "15M entry refined" if refined_entry else "Entry at current 1H close"
    m15_summary = (
        f"{'Buy' if direction == 'LONG' else 'Sell'}-side 1H Liquidity Sweep | "
        f"1H {'Bullish' if direction == 'LONG' else 'Bearish'} CHoCH | "
        f"1H {'Bullish' if direction == 'LONG' else 'Bearish'} BOS | "
        f"{'Retest ✅' if retest_ok else 'Retest Pending'} | "
        f"{m15_note}"
    )

    return SetupResult(
        valid=True, direction=direction, mode="SWING", symbol=symbol,
        timeframe="1h", entry=round(entry, 6), sl=round(sl, 6), tp=round(tp, 6),
        rr=round(rr, 2),
        h4_summary=mtf_ctx.h4_trend.description,
        h1_summary=mtf_ctx.h1_trend.description,
        m15_summary=m15_summary,
        sweep_event=sweep, sweep_score=score.total,
        liquidity_type=sweep.pool.pool_type.value, liquidity_level=sweep.pool.level,
        choch_confirmed=choch_ok, bos_confirmed=bos_ok, retest_confirmed=retest_ok,
        volume_confirmed=volume_ok,
        quality=quality_result.quality, should_send=quality_result.should_send,
        rejection_reason="" if quality_result.should_send else quality_result.reason,
    )


def _try_15m_refinement(df_15m, direction, atr_val, current_price) -> Optional[float]:
    """Optional: refine entry using 15M structure after 1H setup is confirmed."""
    if df_15m.empty or len(df_15m) < 10:
        return None
    try:
        sh = detect_swing_highs(df_15m, "15m")
        sl = detect_swing_lows(df_15m, "15m")
        if not sh and not sl:
            return None

        if direction == "LONG" and sl:
            # Entry near the last 15M swing low (tighter entry)
            recent_low = sl[-1].price
            if abs(current_price - recent_low) / current_price < 0.005:
                return current_price
        elif direction == "SHORT" and sh:
            recent_high = sh[-1].price
            if abs(current_price - recent_high) / current_price < 0.005:
                return current_price
    except Exception:
        pass
    return None


def _swing_sl_tp(direction, entry, sweep, sh_1h, sl_1h, atr_val, pools_1h, df_1h):
    """Calculate SL and TP for swing trades."""
    try:
        buf = ATR_BUFFER_MULTIPLIER * atr_val if atr_val > 0 else entry * 0.003

        if direction == "LONG":
            struct_low = min(sweep.sweep_low, sl_1h[-1].price if sl_1h else sweep.sweep_low)
            sl = struct_low - buf
            opp_pools = [p for p in pools_1h if p.side.value == "BUY_SIDE" and p.level > entry]
            if opp_pools:
                opp_pools.sort(key=lambda p: p.level)
                tp = opp_pools[0].level
            else:
                highs = [s.price for s in sh_1h if s.price > entry]
                tp = min(highs) if highs else entry + 4 * (entry - sl)
        else:
            struct_high = max(sweep.sweep_low, sh_1h[-1].price if sh_1h else sweep.sweep_low)
            sl = struct_high + buf
            opp_pools = [p for p in pools_1h if p.side.value == "SELL_SIDE" and p.level < entry]
            if opp_pools:
                opp_pools.sort(key=lambda p: p.level, reverse=True)
                tp = opp_pools[0].level
            else:
                lows = [s.price for s in sl_1h if s.price < entry]
                tp = max(lows) if lows else entry - 4 * (sl - entry)

        return sl, tp
    except Exception as e:
        logger.error(f"Error in swing SL/TP: {e}")
        return None, None
