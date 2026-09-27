"""
pools.py — Liquidity Pool Detection.

Identifies structurally meaningful liquidity pools around:
- Previous Day High (PDH) / Previous Day Low (PDL)
- Equal Highs / Equal Lows (within tolerance)
- Major swing highs/lows
- Minor swing highs/lows
- Session highs/lows
- Key support/resistance zones

Does NOT classify every random candle high/low as liquidity.
Liquidity must be structurally meaningful.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime, timezone, timedelta

from market_structure.swings import SwingPoint, SwingType
from indicators.volatility import atr as calc_atr
from config.strategy_config import EQUAL_LEVEL_TOLERANCE_PERCENT


class LiquidityType(str, Enum):
    PDH          = "PDH"           # Previous Day High
    PDL          = "PDL"           # Previous Day Low
    EQUAL_HIGHS  = "EQUAL_HIGHS"
    EQUAL_LOWS   = "EQUAL_LOWS"
    SWING_MAJOR  = "SWING_MAJOR"
    SWING_MINOR  = "SWING_MINOR"
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW  = "SESSION_LOW"
    RESISTANCE   = "RESISTANCE"
    SUPPORT      = "SUPPORT"


class LiquiditySide(str, Enum):
    BUY_SIDE  = "BUY_SIDE"    # Stops above a high (short stops)
    SELL_SIDE = "SELL_SIDE"   # Stops below a low  (long stops)


@dataclass
class LiquidityPool:
    pool_type:     LiquidityType
    side:          LiquiditySide
    level:         float              # Price level of the liquidity
    zone_high:     float              # Zone upper boundary
    zone_low:      float              # Zone lower boundary
    touches:       int = 1            # How many times price tested this level
    timeframe:     str = "1h"
    timestamp:     Optional[pd.Timestamp] = None
    strength_note: str = ""


def detect_liquidity_pools(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
    timeframe:   str = "1h",
    pdh:         Optional[float] = None,
    pdl:         Optional[float] = None,
) -> List[LiquidityPool]:
    """
    Detect all meaningful liquidity pools in the DataFrame.

    Args:
        df:          OHLCV DataFrame
        swing_highs: Detected swing highs
        swing_lows:  Detected swing lows
        timeframe:   "4h", "1h", "15m"
        pdh:         Previous Day High (pre-calculated)
        pdl:         Previous Day Low  (pre-calculated)

    Returns:
        List of LiquidityPool objects
    """
    pools: List[LiquidityPool] = []
    if df.empty:
        return pools

    atr_series = calc_atr(df)
    current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    tol = EQUAL_LEVEL_TOLERANCE_PERCENT / 100.0

    # ── 1. Previous Day High / Low ──────────────────────
    if pdh is not None:
        buf = pdh * tol
        pools.append(LiquidityPool(
            pool_type=LiquidityType.PDH,
            side=LiquiditySide.BUY_SIDE,
            level=pdh,
            zone_high=pdh + buf,
            zone_low=pdh - buf,
            touches=_count_touches(df, pdh, buf),
            timeframe=timeframe,
            strength_note="Previous Day High — strong buy-side liquidity",
        ))

    if pdl is not None:
        buf = pdl * tol
        pools.append(LiquidityPool(
            pool_type=LiquidityType.PDL,
            side=LiquiditySide.SELL_SIDE,
            level=pdl,
            zone_high=pdl + buf,
            zone_low=pdl - buf,
            touches=_count_touches(df, pdl, buf),
            timeframe=timeframe,
            strength_note="Previous Day Low — strong sell-side liquidity",
        ))

    # ── 2. Equal Highs / Equal Lows ─────────────────────
    eq_highs = _find_equal_levels(swing_highs, tol)
    for group in eq_highs:
        level = float(np.mean([s.price for s in group]))
        buf   = level * tol
        pools.append(LiquidityPool(
            pool_type=LiquidityType.EQUAL_HIGHS,
            side=LiquiditySide.BUY_SIDE,
            level=level,
            zone_high=level + buf,
            zone_low=level - buf,
            touches=len(group),
            timeframe=timeframe,
            timestamp=group[-1].timestamp,
            strength_note=f"Equal Highs × {len(group)} — clustered buy-side stops",
        ))

    eq_lows = _find_equal_levels(swing_lows, tol)
    for group in eq_lows:
        level = float(np.mean([s.price for s in group]))
        buf   = level * tol
        pools.append(LiquidityPool(
            pool_type=LiquidityType.EQUAL_LOWS,
            side=LiquiditySide.SELL_SIDE,
            level=level,
            zone_high=level + buf,
            zone_low=level - buf,
            touches=len(group),
            timeframe=timeframe,
            timestamp=group[-1].timestamp,
            strength_note=f"Equal Lows × {len(group)} — clustered sell-side stops",
        ))

    # ── 3. Major/Minor Swing Highs/Lows ─────────────────
    # Classify as MAJOR if in top 33% by relative size
    sh_sorted = sorted(swing_highs, key=lambda x: x.price, reverse=True)
    sl_sorted = sorted(swing_lows,  key=lambda x: x.price)
    major_n   = max(1, len(sh_sorted) // 3)

    for i, sh in enumerate(swing_highs):
        if sh in sh_sorted[:major_n]:
            ptype = LiquidityType.SWING_MAJOR
            note  = "Major Swing High — significant buy-side liquidity"
        else:
            ptype = LiquidityType.SWING_MINOR
            note  = "Minor Swing High"
        buf = sh.price * tol
        pools.append(LiquidityPool(
            pool_type=ptype,
            side=LiquiditySide.BUY_SIDE,
            level=sh.price,
            zone_high=sh.price + buf,
            zone_low=sh.price - buf,
            touches=1,
            timeframe=timeframe,
            timestamp=sh.timestamp,
            strength_note=note,
        ))

    for sl in swing_lows:
        if sl in sl_sorted[:major_n]:
            ptype = LiquidityType.SWING_MAJOR
            note  = "Major Swing Low — significant sell-side liquidity"
        else:
            ptype = LiquidityType.SWING_MINOR
            note  = "Minor Swing Low"
        buf = sl.price * tol
        pools.append(LiquidityPool(
            pool_type=ptype,
            side=LiquiditySide.SELL_SIDE,
            level=sl.price,
            zone_high=sl.price + buf,
            zone_low=sl.price - buf,
            touches=1,
            timeframe=timeframe,
            timestamp=sl.timestamp,
            strength_note=note,
        ))

    # Deduplicate pools that are very close to each other
    pools = _deduplicate_pools(pools, tol)

    return pools


def _count_touches(df: pd.DataFrame, level: float, tolerance: float) -> int:
    """Count how many candles touched the level within tolerance."""
    mask = (
        ((df["high"] >= level - tolerance) & (df["high"] <= level + tolerance)) |
        ((df["low"]  >= level - tolerance) & (df["low"]  <= level + tolerance))
    )
    return int(mask.sum())


def _find_equal_levels(
    swings: List[SwingPoint],
    tol: float,
    min_group: int = 2,
) -> List[List[SwingPoint]]:
    """Group swings that are within tolerance% of each other."""
    if len(swings) < min_group:
        return []

    groups: List[List[SwingPoint]] = []
    visited = set()

    for i, a in enumerate(swings):
        if i in visited:
            continue
        group = [a]
        for j, b in enumerate(swings[i + 1:], start=i + 1):
            if j in visited:
                continue
            if abs(a.price - b.price) / max(a.price, 1e-9) <= tol:
                group.append(b)
                visited.add(j)
        if len(group) >= min_group:
            visited.add(i)
            groups.append(group)

    return groups


def _deduplicate_pools(
    pools: List[LiquidityPool],
    tol: float,
) -> List[LiquidityPool]:
    """Remove pools that are within tol% of each other, keeping the strongest."""
    if not pools:
        return []

    # Priority order for keeping
    priority = {
        LiquidityType.PDH:         10,
        LiquidityType.PDL:         10,
        LiquidityType.EQUAL_HIGHS:  9,
        LiquidityType.EQUAL_LOWS:   9,
        LiquidityType.SWING_MAJOR:  7,
        LiquidityType.SESSION_HIGH: 6,
        LiquidityType.SESSION_LOW:  6,
        LiquidityType.SWING_MINOR:  4,
        LiquidityType.RESISTANCE:   5,
        LiquidityType.SUPPORT:      5,
    }

    pools_sorted = sorted(pools, key=lambda p: priority.get(p.pool_type, 0), reverse=True)
    kept: List[LiquidityPool] = []

    for pool in pools_sorted:
        too_close = any(
            abs(pool.level - k.level) / max(k.level, 1e-9) <= tol
            for k in kept
        )
        if not too_close:
            kept.append(pool)

    return kept


def get_nearest_pool(
    pools: List[LiquidityPool],
    current_price: float,
    direction: str,
    max_distance_pct: float = 2.0,
) -> Optional[LiquidityPool]:
    """
    Get the nearest meaningful liquidity pool in the direction of the expected sweep.

    Args:
        direction: "BULLISH" → look below (sell-side), "BEARISH" → look above (buy-side)
        max_distance_pct: Maximum distance from price as a percentage
    """
    candidates = []
    for pool in pools:
        dist_pct = abs(pool.level - current_price) / current_price * 100
        if dist_pct > max_distance_pct:
            continue

        if direction == "BULLISH" and pool.side == LiquiditySide.SELL_SIDE and pool.level < current_price:
            candidates.append((dist_pct, pool))
        elif direction == "BEARISH" and pool.side == LiquiditySide.BUY_SIDE and pool.level > current_price:
            candidates.append((dist_pct, pool))

    if not candidates:
        return None

    # Return the closest pool
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
