"""
pools.py — Institutional Liquidity Pool Detection.

Detects structurally meaningful liquidity around:

- Previous Day High / Low
- Equal Highs / Equal Lows
- Major / Minor confirmed swings
- Session-type structural levels when supplied
- Support / Resistance style structural levels

Design principles:
- Liquidity is structural, not every candle high/low.
- Price clustering is ATR-normalized.
- Equal levels require multiple meaningful touches.
- Major/minor swing classification is based on structural significance,
  not absolute price.
- Freshness and age are tracked.
- Liquidity zones are dynamic.
- Duplicate nearby pools are clustered instead of blindly discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_structure.swings import (
    SwingPoint,
    SwingType,
    are_equal_levels,
)

from indicators.volatility import atr as calc_atr

from config.strategy_config import (
    EQUAL_LEVEL_TOLERANCE_PCT,
    EQUAL_LEVEL_TOLERANCE_ATR,
    MIN_EQUAL_LEVEL_TOUCHES,
    LIQUIDITY_FRESHNESS_BARS,
    LIQUIDITY_MAX_AGE_BARS,
    LIQUIDITY_SATURATION_TOUCHES,
)


# ============================================================================
# ENUMS
# ============================================================================

class LiquidityType(str, Enum):
    PDH = "PDH"
    PDL = "PDL"

    EQUAL_HIGHS = "EQUAL_HIGHS"
    EQUAL_LOWS = "EQUAL_LOWS"

    SWING_MAJOR = "SWING_MAJOR"
    SWING_MINOR = "SWING_MINOR"

    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"

    RESISTANCE = "RESISTANCE"
    SUPPORT = "SUPPORT"


class LiquiditySide(str, Enum):
    BUY_SIDE = "BUY_SIDE"
    SELL_SIDE = "SELL_SIDE"


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class LiquidityPool:
    pool_type: LiquidityType
    side: LiquiditySide

    level: float
    zone_high: float
    zone_low: float

    touches: int = 1

    timeframe: str = "1h"

    timestamp: Optional[pd.Timestamp] = None

    strength_note: str = ""

    # Institutional metadata
    atr_value: float = 0.0
    significance_score: float = 0.0

    age_bars: int = 0
    is_fresh: bool = True
    is_stale: bool = False

    is_saturated: bool = False

    source_count: int = 1

    last_touch_idx: int = -1

    cluster_id: str = ""

    target_valid: bool = True


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def _config_value(
    name: str,
    default,
):
    """
    Safely retrieve configuration.

    Keeps this module compatible if a future config version changes
    a non-critical parameter name.
    """

    try:
        from config import strategy_config

        return getattr(
            strategy_config,
            name,
            default,
        )

    except Exception:
        return default


def _equal_pct() -> float:
    return float(
        _config_value(
            "EQUAL_LEVEL_TOLERANCE_PCT",
            EQUAL_LEVEL_TOLERANCE_PCT,
        )
    )


def _equal_atr() -> float:
    return float(
        _config_value(
            "EQUAL_LEVEL_TOLERANCE_ATR",
            EQUAL_LEVEL_TOLERANCE_ATR,
        )
    )


def _min_equal_touches() -> int:
    return max(
        int(
            _config_value(
                "MIN_EQUAL_LEVEL_TOUCHES",
                MIN_EQUAL_LEVEL_TOUCHES,
            )
        ),
        2,
    )


def _freshness_bars() -> int:
    return max(
        int(
            _config_value(
                "LIQUIDITY_FRESHNESS_BARS",
                LIQUIDITY_FRESHNESS_BARS,
            )
        ),
        1,
    )


def _max_age_bars() -> int:
    return max(
        int(
            _config_value(
                "LIQUIDITY_MAX_AGE_BARS",
                LIQUIDITY_MAX_AGE_BARS,
            )
        ),
        1,
    )


def _saturation_touches() -> int:
    return max(
        int(
            _config_value(
                "LIQUIDITY_SATURATION_TOUCHES",
                LIQUIDITY_SATURATION_TOUCHES,
            )
        ),
        2,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def _clean_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a clean numeric OHLC dataframe."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=[
            "high",
            "low",
            "close",
        ]
    )

    result = result[
        (result["high"] >= result["low"])
        & (result["high"] > 0)
        & (result["low"] > 0)
        & (result["close"] > 0)
    ]

    return result


# ============================================================================
# DYNAMIC TOLERANCE
# ============================================================================

def _dynamic_tolerance(
    level: float,
    atr_value: float,
) -> float:
    """
    Dynamic liquidity-zone tolerance.

    Uses the larger of:
    - percentage tolerance
    - ATR tolerance
    """

    if level <= 0:
        return 0.0

    percentage_tolerance = (
        level
        * _equal_pct()
        / 100.0
    )

    atr_tolerance = (
        max(atr_value, 0.0)
        * _equal_atr()
    )

    return max(
        percentage_tolerance,
        atr_tolerance,
    )


def _zone_for_level(
    level: float,
    atr_value: float,
) -> Tuple[float, float]:
    """Return dynamic upper/lower liquidity zone."""

    tolerance = _dynamic_tolerance(
        level,
        atr_value,
    )

    return (
        level + tolerance,
        level - tolerance,
    )


# ============================================================================
# TOUCH DETECTION
# ============================================================================

def _candle_touches_level(
    high: float,
    low: float,
    level: float,
    tolerance: float,
) -> bool:
    """Return True when candle range intersects liquidity zone."""

    return (
        high >= level - tolerance
        and low <= level + tolerance
    )


def _touch_indices(
    df: pd.DataFrame,
    level: float,
    tolerance: float,
) -> List[int]:
    """Return candle indices that touched a level."""

    if df.empty:
        return []

    result: List[int] = []

    highs = df["high"].to_numpy(
        dtype=float
    )

    lows = df["low"].to_numpy(
        dtype=float
    )

    for i in range(len(df)):
        if _candle_touches_level(
            highs[i],
            lows[i],
            level,
            tolerance,
        ):
            result.append(i)

    return result


def _count_touches(
    df: pd.DataFrame,
    level: float,
    tolerance: float,
) -> int:
    """Count structural candle touches."""

    return len(
        _touch_indices(
            df,
            level,
            tolerance,
        )
    )


# ============================================================================
# EQUAL LEVEL CLUSTERING
# ============================================================================

def _cluster_swings(
    swings: Sequence[SwingPoint],
    atr_value: float,
    min_group: int = 2,
) -> List[List[SwingPoint]]:
    """
    Cluster same-type swings into equal liquidity groups.

    Unlike the old implementation, the comparison is against the
    evolving cluster center rather than only the first swing.
    """

    if len(swings) < min_group:
        return []

    ordered = sorted(
        swings,
        key=lambda swing: swing.price,
    )

    clusters: List[List[SwingPoint]] = []

    for swing in ordered:
        placed = False

        for cluster in clusters:
            center = float(
                np.mean(
                    [
                        item.price
                        for item in cluster
                    ]
                )
            )

            if are_equal_levels(
                center,
                swing.price,
                atr_value,
                _equal_pct(),
                _equal_atr(),
            ):
                cluster.append(swing)
                placed = True
                break

        if not placed:
            clusters.append([swing])

    return [
        cluster
        for cluster in clusters
        if len(cluster) >= min_group
    ]


def _find_equal_levels(
    swings: List[SwingPoint],
    tol: float,
    min_group: int = 2,
) -> List[List[SwingPoint]]:
    """
    Backward-compatible equal-level helper.

    `tol` is interpreted as decimal percentage tolerance.
    ATR-aware clustering is preferred internally.
    """

    if len(swings) < min_group:
        return []

    groups: List[List[SwingPoint]] = []

    ordered = sorted(
        swings,
        key=lambda swing: swing.price,
    )

    current_group: List[SwingPoint] = []

    for swing in ordered:
        if not current_group:
            current_group = [swing]
            continue

        center = float(
            np.mean(
                [
                    item.price
                    for item in current_group
                ]
            )
        )

        if (
            abs(
                swing.price - center
            )
            / max(abs(center), 1e-9)
            <= tol
        ):
            current_group.append(swing)

        else:
            if len(current_group) >= min_group:
                groups.append(
                    current_group
                )

            current_group = [swing]

    if len(current_group) >= min_group:
        groups.append(
            current_group
        )

    return groups


# ============================================================================
# SWING SIGNIFICANCE
# ============================================================================

def _swing_significance(
    swing: SwingPoint,
) -> float:
    """
    Structural significance score for a swing.

    Preference:
    - Larger ATR distance = stronger
    - Existing swing significance is preserved
    """

    value = float(
        getattr(
            swing,
            "significance_atr",
            0.0,
        )
    )

    if value > 0:
        return value

    return 0.0


def _classify_swing_strength(
    swing: SwingPoint,
    all_swings: Sequence[SwingPoint],
) -> Tuple[LiquidityType, str, float]:
    """
    Classify swing as major/minor based on structural significance.

    This deliberately avoids the old "top 33% of absolute price" logic.
    """

    significance = _swing_significance(
        swing
    )

    values = [
        _swing_significance(item)
        for item in all_swings
        if _swing_significance(item) > 0
    ]

    if not values:
        # Confirmed structural swings still count as minor.
        return (
            LiquidityType.SWING_MINOR,
            "Confirmed Swing — structural liquidity",
            35.0,
        )

    percentile = (
        sum(
            value <= significance
            for value in values
        )
        / len(values)
    )

    if (
        significance >= 1.0
        or percentile >= 0.67
    ):
        return (
            LiquidityType.SWING_MAJOR,
            "Major Swing — significant structural liquidity",
            min(
                100.0,
                55.0
                + significance * 10.0,
            ),
        )

    return (
        LiquidityType.SWING_MINOR,
        "Minor Swing — structural liquidity",
        min(
            60.0,
            30.0
            + significance * 10.0,
        ),
    )


# ============================================================================
# FRESHNESS / AGE
# ============================================================================

def _pool_age(
    pool_timestamp: Optional[pd.Timestamp],
    df: pd.DataFrame,
) -> int:
    """Estimate pool age in dataframe bars."""

    if (
        pool_timestamp is None
        or df.empty
    ):
        return 0

    try:
        positions = np.where(
            df.index >= pool_timestamp
        )[0]

        if len(positions) == 0:
            return len(df)

        last_position = int(
            positions[-1]
        )

        return max(
            len(df) - 1 - last_position,
            0,
        )

    except Exception:
        return 0


def _apply_freshness(
    pool: LiquidityPool,
    df: pd.DataFrame,
) -> LiquidityPool:
    """Apply freshness and staleness metadata."""

    age = _pool_age(
        pool.timestamp,
        df,
    )

    pool.age_bars = age

    pool.is_fresh = (
        age <= _freshness_bars()
    )

    pool.is_stale = (
        age > _max_age_bars()
    )

    if pool.is_stale:
        pool.target_valid = False

    return pool


# ============================================================================
# SIGNIFICANCE SCORE
# ============================================================================

def _pool_significance_score(
    pool_type: LiquidityType,
    touches: int,
    age_bars: int,
    swing_score: float = 0.0,
) -> float:
    """
    Compute supporting liquidity significance score.

    This score is NOT a hard gate and must not override
    sweep / structure validation.
    """

    base = {
        LiquidityType.PDH: 80.0,
        LiquidityType.PDL: 80.0,
        LiquidityType.EQUAL_HIGHS: 75.0,
        LiquidityType.EQUAL_LOWS: 75.0,
        LiquidityType.SWING_MAJOR: 65.0,
        LiquidityType.SWING_MINOR: 40.0,
        LiquidityType.SESSION_HIGH: 60.0,
        LiquidityType.SESSION_LOW: 60.0,
        LiquidityType.RESISTANCE: 50.0,
        LiquidityType.SUPPORT: 50.0,
    }.get(
        pool_type,
        25.0,
    )

    touch_bonus = min(
        max(touches - 1, 0) * 5.0,
        20.0,
    )

    age_penalty = min(
        max(age_bars, 0) * 1.0,
        20.0,
    )

    score = (
        base
        + touch_bonus
        + min(
            max(swing_score, 0.0)
            * 0.15,
            15.0,
        )
        - age_penalty
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


# ============================================================================
# CLUSTER / DEDUPLICATION
# ============================================================================

def _pool_priority(
    pool: LiquidityPool,
) -> float:
    """Priority used when merging nearby pools."""

    type_priority = {
        LiquidityType.PDH: 100,
        LiquidityType.PDL: 100,

        LiquidityType.EQUAL_HIGHS: 95,
        LiquidityType.EQUAL_LOWS: 95,

        LiquidityType.SWING_MAJOR: 85,

        LiquidityType.SESSION_HIGH: 75,
        LiquidityType.SESSION_LOW: 75,

        LiquidityType.RESISTANCE: 70,
        LiquidityType.SUPPORT: 70,

        LiquidityType.SWING_MINOR: 50,
    }

    return (
        type_priority.get(
            pool.pool_type,
            20,
        )
        + pool.significance_score * 0.5
        + min(pool.touches, 10) * 2
    )


def _same_side(
    a: LiquidityPool,
    b: LiquidityPool,
) -> bool:
    return a.side == b.side


def _deduplicate_pools(
    pools: List[LiquidityPool],
    tol: float,
) -> List[LiquidityPool]:
    """
    Backward-compatible pool deduplication.

    Pools are only merged when they:
    - have the same liquidity side
    - are within percentage tolerance

    Stronger pools survive.
    """

    if not pools:
        return []

    ordered = sorted(
        pools,
        key=_pool_priority,
        reverse=True,
    )

    kept: List[LiquidityPool] = []

    for pool in ordered:
        duplicate = False

        for existing in kept:
            if not _same_side(
                pool,
                existing,
            ):
                continue

            distance_pct = (
                abs(
                    pool.level
                    - existing.level
                )
                / max(
                    abs(existing.level),
                    1e-9,
                )
                * 100.0
            )

            if distance_pct <= (
                float(tol) * 100.0
            ):
                duplicate = True

                # Merge useful metadata.
                existing.touches = max(
                    existing.touches,
                    pool.touches,
                )

                existing.source_count += (
                    pool.source_count
                )

                if pool.significance_score > existing.significance_score:
                    existing.significance_score = (
                        pool.significance_score
                    )

                break

        if not duplicate:
            kept.append(pool)

    return kept


def _deduplicate_pools_atr(
    pools: List[LiquidityPool],
) -> List[LiquidityPool]:
    """ATR-aware deduplication used by the institutional detector."""

    if not pools:
        return []

    ordered = sorted(
        pools,
        key=_pool_priority,
        reverse=True,
    )

    kept: List[LiquidityPool] = []

    for pool in ordered:
        merged = False

        for existing in kept:
            if pool.side != existing.side:
                continue

            reference_atr = max(
                pool.atr_value,
                existing.atr_value,
            )

            tolerance = _dynamic_tolerance(
                (
                    pool.level
                    + existing.level
                ) / 2.0,
                reference_atr,
            )

            if (
                abs(
                    pool.level
                    - existing.level
                )
                <= tolerance
            ):
                # Keep stronger structural pool.
                existing.touches = max(
                    existing.touches,
                    pool.touches,
                )

                existing.source_count += (
                    pool.source_count
                )

                existing.level = float(
                    (
                        existing.level
                        + pool.level
                    ) / 2.0
                )

                existing.zone_high = max(
                    existing.zone_high,
                    pool.zone_high,
                )

                existing.zone_low = min(
                    existing.zone_low,
                    pool.zone_low,
                )

                existing.is_fresh = (
                    existing.is_fresh
                    or pool.is_fresh
                )

                existing.is_stale = (
                    existing.is_stale
                    and pool.is_stale
                )

                existing.is_saturated = (
                    existing.is_saturated
                    or pool.is_saturated
                )

                existing.significance_score = max(
                    existing.significance_score,
                    pool.significance_score,
                )

                merged = True
                break

        if not merged:
            kept.append(pool)

    return kept


# ============================================================================
# MAIN DETECTOR
# ============================================================================

def detect_liquidity_pools(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    timeframe: str = "1h",
    pdh: Optional[float] = None,
    pdl: Optional[float] = None,
) -> List[LiquidityPool]:
    """
    Detect structurally meaningful liquidity pools.

    Supported:
        4h
        1h
        15m

    Liquidity is generated from:
        PDH / PDL
        Equal highs / lows
        Confirmed major/minor swings
    """

    cleaned = _clean_dataframe(df)

    if cleaned.empty:
        return []

    atr_series = calc_atr(cleaned)

    current_atr = 0.0

    if not atr_series.empty:
        value = atr_series.iloc[-1]

        if (
            pd.notna(value)
            and np.isfinite(value)
        ):
            current_atr = float(value)

    pools: List[LiquidityPool] = []

    # ------------------------------------------------------------------------
    # 1. PREVIOUS DAY HIGH
    # ------------------------------------------------------------------------

    if (
        pdh is not None
        and np.isfinite(pdh)
        and pdh > 0
    ):
        level = float(pdh)

        zone_high, zone_low = _zone_for_level(
            level,
            current_atr,
        )

        touches = _count_touches(
            cleaned,
            level,
            max(
                zone_high - level,
                level - zone_low,
            ),
        )

        pool = LiquidityPool(
            pool_type=LiquidityType.PDH,
            side=LiquiditySide.BUY_SIDE,
            level=level,
            zone_high=zone_high,
            zone_low=zone_low,
            touches=max(touches, 1),
            timeframe=timeframe,
            timestamp=None,
            strength_note=(
                "Previous Day High — "
                "strong buy-side liquidity"
            ),
            atr_value=current_atr,
            source_count=1,
            target_valid=True,
        )

        pool.significance_score = (
            _pool_significance_score(
                pool.pool_type,
                pool.touches,
                0,
            )
        )

        pools.append(
            _apply_freshness(
                pool,
                cleaned,
            )
        )

    # ------------------------------------------------------------------------
    # 2. PREVIOUS DAY LOW
    # ------------------------------------------------------------------------

    if (
        pdl is not None
        and np.isfinite(pdl)
        and pdl > 0
    ):
        level = float(pdl)

        zone_high, zone_low = _zone_for_level(
            level,
            current_atr,
        )

        touches = _count_touches(
            cleaned,
            level,
            max(
                zone_high - level,
                level - zone_low,
            ),
        )

        pool = LiquidityPool(
            pool_type=LiquidityType.PDL,
            side=LiquiditySide.SELL_SIDE,
            level=level,
            zone_high=zone_high,
            zone_low=zone_low,
            touches=max(touches, 1),
            timeframe=timeframe,
            timestamp=None,
            strength_note=(
                "Previous Day Low — "
                "strong sell-side liquidity"
            ),
            atr_value=current_atr,
            source_count=1,
            target_valid=True,
        )

        pool.significance_score = (
            _pool_significance_score(
                pool.pool_type,
                pool.touches,
                0,
            )
        )

        pools.append(
            _apply_freshness(
                pool,
                cleaned,
            )
        )

    # ------------------------------------------------------------------------
    # 3. EQUAL HIGHS
    # ------------------------------------------------------------------------

    equal_high_groups = _cluster_swings(
        swing_highs,
        current_atr,
        min_group=_min_equal_touches(),
    )

    for group in equal_high_groups:
        level = float(
            np.mean(
                [
                    swing.price
                    for swing in group
                ]
            )
        )

        zone_high, zone_low = _zone_for_level(
            level,
            current_atr,
        )

        latest = max(
            group,
            key=lambda swing: swing.candle_idx,
        )

        touches = max(
            len(group),
            _count_touches(
                cleaned,
                level,
                max(
                    zone_high - level,
                    level - zone_low,
                ),
            ),
        )

        pool = LiquidityPool(
            pool_type=LiquidityType.EQUAL_HIGHS,
            side=LiquiditySide.BUY_SIDE,
            level=level,
            zone_high=zone_high,
            zone_low=zone_low,
            touches=touches,
            timeframe=timeframe,
            timestamp=latest.timestamp,
            strength_note=(
                f"Equal Highs × {touches} — "
                "clustered buy-side liquidity"
            ),
            atr_value=current_atr,
            source_count=len(group),
            last_touch_idx=latest.candle_idx,
        )

        pool.significance_score = (
            _pool_significance_score(
                pool.pool_type,
                touches,
                0,
            )
        )

        pool.is_saturated = (
            touches >= _saturation_touches()
        )

        pools.append(
            _apply_freshness(
                pool,
                cleaned,
            )
        )

    # ------------------------------------------------------------------------
    # 4. EQUAL LOWS
    # ------------------------------------------------------------------------

    equal_low_groups = _cluster_swings(
        swing_lows,
        current_atr,
        min_group=_min_equal_touches(),
    )

    for group in equal_low_groups:
        level = float(
            np.mean(
                [
                    swing.price
                    for swing in group
                ]
            )
        )

        zone_high, zone_low = _zone_for_level(
            level,
            current_atr,
        )

        latest = max(
            group,
            key=lambda swing: swing.candle_idx,
        )

        touches = max(
            len(group),
            _count_touches(
                cleaned,
                level,
                max(
                    zone_high - level,
                    level - zone_low,
                ),
            ),
        )

        pool = LiquidityPool(
            pool_type=LiquidityType.EQUAL_LOWS,
            side=LiquiditySide.SELL_SIDE,
            level=level,
            zone_high=zone_high,
            zone_low=zone_low,
            touches=touches,
            timeframe=timeframe,
            timestamp=latest.timestamp,
            strength_note=(
                f"Equal Lows × {touches} — "
                "clustered sell-side liquidity"
            ),
            atr_value=current_atr,
            source_count=len(group),
            last_touch_idx=latest.candle_idx,
        )

        pool.significance_score = (
            _pool_significance_score(
                pool.pool_type,
                touches,
                0,
            )
        )

        pool.is_saturated = (
            touches >= _saturation_touches()
        )

        pools.append(
            _apply_freshness(
                pool,
                cleaned,
            )
        )

    # ------------------------------------------------------------------------
    # 5. CONFIRMED SWING HIGHS
    # ------------------------------------------------------------------------

    for swing in swing_highs:
        pool_type, note, swing_score = (
            _classify_swing_strength(
                swing,
                swing_highs,
            )
        )

        level = float(
            swing.price
        )

        zone_high, zone_low = _zone_for_level(
            level,
            max(
                current_atr,
                swing.atr_value,
            ),
        )

        touches = max(
            1,
            _count_touches(
                cleaned,
                level,
                max(
                    zone_high - level,
                    level - zone_low,
                ),
            ),
        )

        pool = LiquidityPool(
            pool_type=pool_type,
            side=LiquiditySide.BUY_SIDE,
            level=level,
            zone_high=zone_high,
            zone_low=zone_low,
            touches=touches,
            timeframe=timeframe,
            timestamp=swing.timestamp,
            strength_note=note,
            atr_value=max(
                current_atr,
                swing.atr_value,
            ),
            significance_score=(
                _pool_significance_score(
                    pool_type,
                    touches,
                    0,
                    swing_score,
                )
            ),
            source_count=1,
            last_touch_idx=swing.candle_idx,
        )

        pool.is_saturated = (
            touches >= _saturation_touches()
        )

        pools.append(
            _apply_freshness(
                pool,
                cleaned,
            )
        )

    # ------------------------------------------------------------------------
    # 6. CONFIRMED SWING LOWS
    # ------------------------------------------------------------------------

    for swing in swing_lows:
        pool_type, note, swing_score = (
            _classify_swing_strength(
                swing,
                swing_lows,
            )
        )

        level = float(
            swing.price
        )

        zone_high, zone_low = _zone_for_level(
            level,
            max(
                current_atr,
                swing.atr_value,
            ),
        )

        touches = max(
            1,
            _count_touches(
                cleaned,
                level,
                max(
                    zone_high - level,
                    level - zone_low,
                ),
            ),
        )

        pool = LiquidityPool(
            pool_type=pool_type,
            side=LiquiditySide.SELL_SIDE,
            level=level,
            zone_high=zone_high,
            zone_low=zone_low,
            touches=touches,
            timeframe=timeframe,
            timestamp=swing.timestamp,
            strength_note=note,
            atr_value=max(
                current_atr,
                swing.atr_value,
            ),
            significance_score=(
                _pool_significance_score(
                    pool_type,
                    touches,
                    0,
                    swing_score,
                )
            ),
            source_count=1,
            last_touch_idx=swing.candle_idx,
        )

        pool.is_saturated = (
            touches >= _saturation_touches()
        )

        pools.append(
            _apply_freshness(
                pool,
                cleaned,
            )
        )

    # ------------------------------------------------------------------------
    # 7. ATR-AWARE DEDUPLICATION
    # ------------------------------------------------------------------------

    pools = _deduplicate_pools_atr(
        pools
    )

    # ------------------------------------------------------------------------
    # 8. FINAL METADATA REFRESH
    # ------------------------------------------------------------------------

    for index, pool in enumerate(pools):
        if not pool.cluster_id:
            pool.cluster_id = (
                f"{timeframe}-"
                f"{pool.side.value}-"
                f"{round(pool.level, 8)}-"
                f"{index}"
            )

        pool.is_saturated = (
            pool.touches
            >= _saturation_touches()
        )

        # Stale liquidity should not be used as a primary target.
        if pool.is_stale:
            pool.target_valid = False

    return pools


# ============================================================================
# NEAREST LIQUIDITY
# ============================================================================

def get_nearest_pool(
    pools: List[LiquidityPool],
    current_price: float,
    direction: str,
    max_distance_pct: float = 2.0,
) -> Optional[LiquidityPool]:
    """
    Get nearest meaningful liquidity in expected sweep direction.

    BULLISH:
        search below price for SELL_SIDE liquidity.

    BEARISH:
        search above price for BUY_SIDE liquidity.

    Stale pools are excluded.
    """

    if (
        not pools
        or current_price <= 0
    ):
        return None

    direction = str(
        direction
    ).upper()

    candidates = []

    for pool in pools:
        if pool.is_stale:
            continue

        if not pool.target_valid:
            continue

        distance_pct = (
            abs(
                pool.level
                - current_price
            )
            / current_price
            * 100.0
        )

        if distance_pct > max_distance_pct:
            continue

        if (
            direction == "BULLISH"
            and pool.side
            == LiquiditySide.SELL_SIDE
            and pool.level < current_price
        ):
            candidates.append(
                (
                    distance_pct,
                    -pool.significance_score,
                    pool,
                )
            )

        elif (
            direction == "BEARISH"
            and pool.side
            == LiquiditySide.BUY_SIDE
            and pool.level > current_price
        ):
            candidates.append(
                (
                    distance_pct,
                    -pool.significance_score,
                    pool,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return candidates[0][2]


# ============================================================================
# LIQUIDITY TARGET HELPERS
# ============================================================================

def get_pools_by_side(
    pools: List[LiquidityPool],
    side: LiquiditySide,
    include_stale: bool = False,
) -> List[LiquidityPool]:
    """Return liquidity pools for one side."""

    return [
        pool
        for pool in pools
        if pool.side == side
        and (
            include_stale
            or not pool.is_stale
        )
    ]


def get_strongest_pool(
    pools: List[LiquidityPool],
    side: Optional[LiquiditySide] = None,
) -> Optional[LiquidityPool]:
    """Return strongest available liquidity pool."""

    candidates = [
        pool
        for pool in pools
        if not pool.is_stale
        and (
            side is None
            or pool.side == side
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda pool: (
            pool.significance_score,
            pool.touches,
        ),
    )


def liquidity_summary(
    pools: List[LiquidityPool],
) -> str:
    """Compact liquidity summary for logs/debugging."""

    if not pools:
        return "Liquidity: NONE"

    buy_side = sum(
        pool.side == LiquiditySide.BUY_SIDE
        for pool in pools
        if not pool.is_stale
    )

    sell_side = sum(
        pool.side == LiquiditySide.SELL_SIDE
        for pool in pools
        if not pool.is_stale
    )

    fresh = sum(
        pool.is_fresh
        for pool in pools
        if not pool.is_stale
    )

    saturated = sum(
        pool.is_saturated
        for pool in pools
        if not pool.is_stale
    )

    return (
        f"Pools={len(pools)} | "
        f"BuySide={buy_side} | "
        f"SellSide={sell_side} | "
        f"Fresh={fresh} | "
        f"Saturated={saturated}"
    )


# ============================================================================
# MODULE SELF TEST
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    size = 300

    close = (
        100
        + np.cumsum(
            np.random.normal(
                0,
                0.5,
                size,
            )
        )
    )

    high = (
        close
        + np.random.uniform(
            0.1,
            0.8,
            size,
        )
    )

    low = (
        close
        - np.random.uniform(
            0.1,
            0.8,
            size,
        )
    )

    open_price = (
        close
        + np.random.normal(
            0,
            0.2,
            size,
        )
    )

    volume = np.random.uniform(
        100,
        1000,
        size,
    )

    index = pd.date_range(
        end=pd.Timestamp.utcnow(),
        periods=size,
        freq="15min",
    )

    test_df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )

    from market_structure.swings import (
        detect_swing_highs,
        detect_swing_lows,
        label_market_structure,
    )

    highs = detect_swing_highs(
        test_df,
        timeframe="15m",
    )

    lows = detect_swing_lows(
        test_df,
        timeframe="15m",
    )

    highs, lows = label_market_structure(
        highs,
        lows,
    )

    pools = detect_liquidity_pools(
        test_df,
        highs,
        lows,
        timeframe="15m",
    )

    print("=== LIQUIDITY ENGINE TEST ===")
    print(
        liquidity_summary(pools)
    )

    for pool in pools[:10]:
        print(
            pool.pool_type.value,
            pool.side.value,
            pool.level,
            pool.touches,
            pool.significance_score,
            pool.is_fresh,
            pool.is_saturated,
        )