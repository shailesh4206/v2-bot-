"""
liquidity/sweep.py
Institutional Liquidity Sweep Detection Engine.

Pipeline:

Liquidity Pool
    ↓
Sweep / Wick Penetration
    ↓
Reclaim / Rejection
    ↓
ATR Depth Validation
    ↓
Wick/Body Validation
    ↓
Displacement
    ↓
Volume / RVOL
    ↓
Volatility Safety
    ↓
Fresh Liquidity
    ↓
Validated Sweep Event

IMPORTANT:
- Sweep alone NEVER creates a trading signal.
- Downstream CHoCH → BOS → Retest must be independently confirmed.
- Only closed candles should be supplied.
- No future candle may be used to validate a historical sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from liquidity.pools import LiquidityPool, LiquiditySide
from indicators.volatility import atr as calc_atr, classify_volatility
from indicators.volume import get_rvol_at_index

from config.strategy_config import (
    SWEEP_MIN_DEPTH_ATR,
    SWEEP_MAX_DEPTH_ATR,
    SWEEP_MIN_WICK_BODY_RATIO,
    SWEEP_MIN_BODY_ATR,

    SWEEP_RECLAIM_REQUIRED,
    SWEEP_REJECTION_REQUIRED,
    SWEEP_MIN_DEPTH_REQUIRED,

    DISPLACEMENT_REQUIRED,
    DISPLACEMENT_ATR_MULTIPLIER,
    DISPLACEMENT_BODY_PERCENT,
    DISPLACEMENT_MAX_CANDLES_AFTER_SWEEP,
    DISPLACEMENT_VOLUME_CONFIRM,
    DISPLACEMENT_MIN_RVOL,

    ABNORMAL_VOLATILITY_ENABLED,
    ABNORMAL_VOLATILITY_BLOCK_EXTREME,
)


# ============================================================================
# ENUMS
# ============================================================================

class SweepDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class SweepQuality(str, Enum):
    INVALID = "INVALID"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    INSTITUTIONAL = "INSTITUTIONAL"


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class SweepEvent:
    direction: SweepDirection
    pool: LiquidityPool

    sweep_candle_idx: int
    sweep_timestamp: pd.Timestamp

    # Geometry
    sweep_low: float
    sweep_high: float
    close_price: float
    liquidity_level: float

    wick_size: float
    body_size: float
    candle_range: float
    wick_to_body_ratio: float

    # ATR
    atr_value: float
    sweep_depth: float
    sweep_depth_atr: float
    body_atr: float
    range_atr: float

    # Volume
    sweep_volume: float
    avg_volume: float
    rvol: float

    # Reclaim / rejection
    reclaim_distance: float
    reclaim_distance_atr: float
    rejection_strength: str

    # Displacement
    displacement_confirmed: bool
    displacement_candle_idx: Optional[int]
    displacement_body_atr: float
    displacement_body_percent: float

    # Volatility
    volatility_regime: str
    volatility_allowed: bool
    abnormal_volatility: bool
    volatility_warning: Optional[str]

    # Pool quality
    pool_is_fresh: bool
    pool_is_stale: bool
    pool_is_saturated: bool
    pool_significance: float

    # Final quality
    quality: SweepQuality
    valid: bool

    # Hard-gate diagnostics
    rejection_confirmed: bool
    reclaim_confirmed: bool
    depth_valid: bool
    volume_confirmed: bool

    reasons: tuple[str, ...] = ()


# ============================================================================
# DATA VALIDATION
# ============================================================================

_REQUIRED_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize OHLCV data."""

    if df is None or df.empty:
        return pd.DataFrame()

    missing = [
        column
        for column in _REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        return pd.DataFrame()

    out = df.copy()

    for column in _REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    out = out.dropna(
        subset=list(_REQUIRED_COLUMNS)
    )

    if out.empty:
        return pd.DataFrame()

    valid_ohlc = (
        (out["high"] >= out["low"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
        & (out["volume"] >= 0)
    )

    out = out.loc[valid_ohlc].copy()

    return out


# ============================================================================
# CANDLE HELPERS
# ============================================================================

def _candle_metrics(row: pd.Series) -> dict:
    """Return normalized candle geometry."""

    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])

    candle_range = max(h - l, 0.0)
    body_size = abs(c - o)

    upper_wick = max(
        h - max(o, c),
        0.0,
    )

    lower_wick = max(
        min(o, c) - l,
        0.0,
    )

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "range": candle_range,
        "body": body_size,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
    }


def _safe_ratio(
    numerator: float,
    denominator: float,
    fallback: float = 0.0,
) -> float:

    if denominator <= 0:
        return fallback

    return float(numerator / denominator)


# ============================================================================
# REJECTION
# ============================================================================

def _rejection_strength(
    wick_to_body_ratio: float,
) -> str:

    if wick_to_body_ratio >= 2.0:
        return "STRONG"

    if wick_to_body_ratio >= 1.0:
        return "MODERATE"

    return "WEAK"


# ============================================================================
# DISPLACEMENT
# ============================================================================

def _is_displacement_candle(
    df: pd.DataFrame,
    idx: int,
    direction: SweepDirection,
    atr_series: pd.Series,
) -> tuple[bool, float, float]:

    if idx < 0 or idx >= len(df):
        return False, 0.0, 0.0

    row = df.iloc[idx]

    metrics = _candle_metrics(row)

    atr_value = float(
        atr_series.iloc[idx]
    )

    if not np.isfinite(atr_value) or atr_value <= 0:
        return False, 0.0, 0.0

    body_atr = (
        metrics["body"] / atr_value
    )

    body_percent = (
        metrics["body"] / metrics["range"] * 100.0
        if metrics["range"] > 0
        else 0.0
    )

    if direction == SweepDirection.BULLISH:
        directional = (
            metrics["close"] > metrics["open"]
        )
    else:
        directional = (
            metrics["close"] < metrics["open"]
        )

    confirmed = (
        directional
        and body_atr >= DISPLACEMENT_ATR_MULTIPLIER
        and body_percent >= DISPLACEMENT_BODY_PERCENT
    )

    return (
        bool(confirmed),
        float(body_atr),
        float(body_percent),
    )


def _find_displacement_after_sweep(
    df: pd.DataFrame,
    sweep_idx: int,
    direction: SweepDirection,
    atr_series: pd.Series,
) -> tuple[bool, Optional[int], float, float]:

    max_bars = max(
        0,
        int(DISPLACEMENT_MAX_CANDLES_AFTER_SWEEP),
    )

    if max_bars <= 0:
        return False, None, 0.0, 0.0

    end_idx = min(
        len(df) - 1,
        sweep_idx + max_bars,
    )

    # IMPORTANT:
    # Start AFTER sweep candle.
    for idx in range(
        sweep_idx + 1,
        end_idx + 1,
    ):

        confirmed, body_atr, body_percent = (
            _is_displacement_candle(
                df,
                idx,
                direction,
                atr_series,
            )
        )

        if confirmed:
            return (
                True,
                idx,
                body_atr,
                body_percent,
            )

    return False, None, 0.0, 0.0


# ============================================================================
# VOLUME
# ============================================================================

def _get_volume_confirmation(
    df: pd.DataFrame,
    idx: int,
) -> tuple[float, bool]:

    try:
        rvol = float(
            get_rvol_at_index(
                df,
                idx,
            )
        )
    except Exception:
        rvol = 0.0

    if not np.isfinite(rvol):
        rvol = 0.0

    confirmed = (
        rvol >= DISPLACEMENT_MIN_RVOL
    )

    return (
        rvol,
        bool(confirmed),
    )


# ============================================================================
# VOLATILITY
# ============================================================================

def _volatility_status(
    df: pd.DataFrame,
    idx: int,
) -> tuple[str, bool, bool, Optional[str]]:

    try:

        context = df.iloc[
            : idx + 1
        ].copy()

        if len(context) < 30:
            return (
                "UNKNOWN",
                False,
                True,
                "Insufficient volatility history",
            )

        analysis = classify_volatility(
            context
        )

        regime = getattr(
            analysis,
            "regime",
            "UNKNOWN",
        )

        if hasattr(regime, "value"):
            regime_value = regime.value
        else:
            regime_value = str(regime)

        abnormal = bool(
            getattr(
                analysis,
                "is_abnormal",
                False,
            )
        )

        block = bool(
            getattr(
                analysis,
                "block_new_signal",
                False,
            )
        )

        risk_allowed = bool(
            getattr(
                analysis,
                "risk_allowed",
                not block,
            )
        )

        warning = getattr(
            analysis,
            "warning",
            None,
        )

        allowed = (
            risk_allowed
            and not block
        )

        if (
            ABNORMAL_VOLATILITY_ENABLED
            and ABNORMAL_VOLATILITY_BLOCK_EXTREME
            and abnormal
        ):
            allowed = False

        return (
            regime_value,
            bool(allowed),
            abnormal,
            warning,
        )

    except Exception:
        # Fail closed.
        return (
            "UNKNOWN",
            False,
            True,
            "Volatility validation failed",
        )


# ============================================================================
# QUALITY
# ============================================================================

def _quality_from_metrics(
    *,
    wick_ratio: float,
    depth_atr: float,
    rvol: float,
    displacement: bool,
    volatility_allowed: bool,
    pool_fresh: bool,
    reclaim_atr: float,
) -> SweepQuality:

    if not volatility_allowed:
        return SweepQuality.INVALID

    if not pool_fresh:
        return SweepQuality.INVALID

    if not displacement:
        return SweepQuality.WEAK

    if (
        wick_ratio >= 2.0
        and depth_atr >= SWEEP_MIN_DEPTH_ATR
        and rvol >= max(
            1.0,
            DISPLACEMENT_MIN_RVOL,
        )
        and reclaim_atr >= 0.10
    ):
        return SweepQuality.INSTITUTIONAL

    if (
        wick_ratio >= 1.5
        and depth_atr >= SWEEP_MIN_DEPTH_ATR
    ):
        return SweepQuality.STRONG

    return SweepQuality.MODERATE


# ============================================================================
# SWEEP DETECTION
# ============================================================================

def detect_sweeps(
    df: pd.DataFrame,
    pools: List[LiquidityPool],
    lookback: int = 10,
) -> List[SweepEvent]:
    """
    Detect validated liquidity sweeps.

    IMPORTANT:
    This function assumes the caller supplies CLOSED candles only.

    A sweep requires:

    1. Fresh liquidity
    2. Wick penetration
    3. Reclaim/rejection
    4. Valid ATR depth
    5. Valid wick/body ratio
    6. Valid candle body
    7. Displacement after sweep
    8. Volume confirmation when configured
    9. Volatility safety
    """

    sweeps: List[SweepEvent] = []

    data = _prepare_dataframe(df)

    if data.empty or not pools:
        return sweeps

    if len(data) < 30:
        return sweeps

    atr_series = pd.to_numeric(
        calc_atr(data),
        errors="coerce",
    )

    lookback = max(
        1,
        int(lookback),
    )

    start_idx = max(
        0,
        len(data) - lookback,
    )

    for i in range(
        start_idx,
        len(data),
    ):

        candle = data.iloc[i]

        metrics = _candle_metrics(
            candle
        )

        o = metrics["open"]
        h = metrics["high"]
        l = metrics["low"]
        c = metrics["close"]

        volume = float(
            candle["volume"]
        )

        atr_value = (
            float(atr_series.iloc[i])
            if pd.notna(atr_series.iloc[i])
            else 0.0
        )

        if (
            not np.isfinite(atr_value)
            or atr_value <= 0
        ):
            continue

        if metrics["range"] <= 0:
            continue

        # ------------------------------------------------------------
        # Volatility safety
        # ------------------------------------------------------------

        (
            volatility_regime,
            volatility_allowed,
            abnormal_volatility,
            volatility_warning,
        ) = _volatility_status(
            data,
            i,
        )

        if (
            ABNORMAL_VOLATILITY_ENABLED
            and ABNORMAL_VOLATILITY_BLOCK_EXTREME
            and not volatility_allowed
        ):
            continue

        # ------------------------------------------------------------
        # Volume
        # ------------------------------------------------------------

        rvol, volume_confirmed = (
            _get_volume_confirmation(
                data,
                i,
            )
        )

        volume_window_start = max(
            0,
            i - 20,
        )

        previous_volumes = data.iloc[
            volume_window_start:i
        ]["volume"]

        avg_volume = (
            float(previous_volumes.mean())
            if len(previous_volumes) > 0
            else 0.0
        )

        # ------------------------------------------------------------
        # Pool loop
        # ------------------------------------------------------------

        for pool in pools:

            pool_stale = bool(
                getattr(
                    pool,
                    "is_stale",
                    False,
                )
            )

            pool_fresh = bool(
                getattr(
                    pool,
                    "is_fresh",
                    not pool_stale,
                )
            )

            pool_saturated = bool(
                getattr(
                    pool,
                    "is_saturated",
                    False,
                )
            )

            if (
                pool_stale
                or not pool_fresh
                or pool_saturated
            ):
                continue

            level = float(
                pool.level
            )

            if (
                not np.isfinite(level)
                or level <= 0
            ):
                continue

            # ========================================================
            # BULLISH SWEEP
            # Sell-side liquidity swept
            # ========================================================

            if pool.side == LiquiditySide.SELL_SIDE:

                swept = (
                    l < level
                )

                reclaimed = (
                    c > level
                )

                if SWEEP_RECLAIM_REQUIRED:
                    if not reclaimed:
                        continue

                if not swept:
                    continue

                sweep_depth = (
                    level - l
                )

                depth_atr = (
                    sweep_depth / atr_value
                )

                if SWEEP_MIN_DEPTH_REQUIRED:
                    if (
                        depth_atr
                        < SWEEP_MIN_DEPTH_ATR
                    ):
                        continue

                if (
                    depth_atr
                    > SWEEP_MAX_DEPTH_ATR
                ):
                    continue

                wick_size = (
                    level - l
                )

                body_size = (
                    metrics["body"]
                )

                wick_ratio = _safe_ratio(
                    wick_size,
                    body_size,
                    fallback=99.0,
                )

                if (
                    wick_ratio
                    < SWEEP_MIN_WICK_BODY_RATIO
                ):
                    continue

                body_atr = (
                    body_size
                    / atr_value
                )

                if (
                    body_atr
                    < SWEEP_MIN_BODY_ATR
                ):
                    continue

                # Explicit rejection.
                rejection_confirmed = (
                    c > level
                    and wick_size > 0
                )

                if (
                    SWEEP_REJECTION_REQUIRED
                    and not rejection_confirmed
                ):
                    continue

                reclaim_distance = (
                    c - level
                )

                reclaim_distance_atr = (
                    reclaim_distance
                    / atr_value
                )

                rejection_strength = (
                    _rejection_strength(
                        wick_ratio
                    )
                )

                # Displacement must happen AFTER sweep.
                (
                    displacement_confirmed,
                    displacement_idx,
                    displacement_body_atr,
                    displacement_body_percent,
                ) = _find_displacement_after_sweep(
                    data,
                    i,
                    SweepDirection.BULLISH,
                    atr_series,
                )

                if (
                    DISPLACEMENT_REQUIRED
                    and not displacement_confirmed
                ):
                    continue

                if (
                    DISPLACEMENT_VOLUME_CONFIRM
                    and not volume_confirmed
                ):
                    continue

                quality = _quality_from_metrics(
                    wick_ratio=wick_ratio,
                    depth_atr=depth_atr,
                    rvol=rvol,
                    displacement=displacement_confirmed,
                    volatility_allowed=volatility_allowed,
                    pool_fresh=pool_fresh,
                    reclaim_atr=reclaim_distance_atr,
                )

                if quality == SweepQuality.INVALID:
                    continue

                reasons = (
                    "SELL_SIDE_LIQUIDITY_SWEPT",
                    "CLOSE_RECLAIMED_LEVEL",
                    "ATR_DEPTH_VALID",
                    "REJECTION_VALID",
                    "DISPLACEMENT_CONFIRMED",
                )

                sweeps.append(
                    SweepEvent(
                        direction=SweepDirection.BULLISH,
                        pool=pool,
                        sweep_candle_idx=i,
                        sweep_timestamp=data.index[i],
                        sweep_low=l,
                        sweep_high=h,
                        close_price=c,
                        liquidity_level=level,
                        wick_size=wick_size,
                        body_size=body_size,
                        candle_range=metrics["range"],
                        wick_to_body_ratio=wick_ratio,
                        atr_value=atr_value,
                        sweep_depth=sweep_depth,
                        sweep_depth_atr=depth_atr,
                        body_atr=body_atr,
                        range_atr=(
                            metrics["range"]
                            / atr_value
                        ),
                        sweep_volume=volume,
                        avg_volume=avg_volume,
                        rvol=rvol,
                        reclaim_distance=reclaim_distance,
                        reclaim_distance_atr=reclaim_distance_atr,
                        rejection_strength=rejection_strength,
                        displacement_confirmed=displacement_confirmed,
                        displacement_candle_idx=displacement_idx,
                        displacement_body_atr=displacement_body_atr,
                        displacement_body_percent=displacement_body_percent,
                        volatility_regime=volatility_regime,
                        volatility_allowed=volatility_allowed,
                        abnormal_volatility=abnormal_volatility,
                        volatility_warning=volatility_warning,
                        pool_is_fresh=pool_fresh,
                        pool_is_stale=pool_stale,
                        pool_is_saturated=pool_saturated,
                        pool_significance=float(
                            getattr(
                                pool,
                                "significance_score",
                                0.0,
                            )
                        ),
                        quality=quality,
                        valid=True,
                        rejection_confirmed=True,
                        reclaim_confirmed=reclaimed,
                        depth_valid=True,
                        volume_confirmed=volume_confirmed,
                        reasons=reasons,
                    )
                )

            # ========================================================
            # BEARISH SWEEP
            # Buy-side liquidity swept
            # ========================================================

            elif pool.side == LiquiditySide.BUY_SIDE:

                swept = (
                    h > level
                )

                reclaimed = (
                    c < level
                )

                if SWEEP_RECLAIM_REQUIRED:
                    if not reclaimed:
                        continue

                if not swept:
                    continue

                sweep_depth = (
                    h - level
                )

                depth_atr = (
                    sweep_depth / atr_value
                )

                if SWEEP_MIN_DEPTH_REQUIRED:
                    if (
                        depth_atr
                        < SWEEP_MIN_DEPTH_ATR
                    ):
                        continue

                if (
                    depth_atr
                    > SWEEP_MAX_DEPTH_ATR
                ):
                    continue

                wick_size = (
                    h - level
                )

                body_size = (
                    metrics["body"]
                )

                wick_ratio = _safe_ratio(
                    wick_size,
                    body_size,
                    fallback=99.0,
                )

                if (
                    wick_ratio
                    < SWEEP_MIN_WICK_BODY_RATIO
                ):
                    continue

                body_atr = (
                    body_size
                    / atr_value
                )

                if (
                    body_atr
                    < SWEEP_MIN_BODY_ATR
                ):
                    continue

                rejection_confirmed = (
                    c < level
                    and wick_size > 0
                )

                if (
                    SWEEP_REJECTION_REQUIRED
                    and not rejection_confirmed
                ):
                    continue

                reclaim_distance = (
                    level - c
                )

                reclaim_distance_atr = (
                    reclaim_distance
                    / atr_value
                )

                rejection_strength = (
                    _rejection_strength(
                        wick_ratio
                    )
                )

                (
                    displacement_confirmed,
                    displacement_idx,
                    displacement_body_atr,
                    displacement_body_percent,
                ) = _find_displacement_after_sweep(
                    data,
                    i,
                    SweepDirection.BEARISH,
                    atr_series,
                )

                if (
                    DISPLACEMENT_REQUIRED
                    and not displacement_confirmed
                ):
                    continue

                if (
                    DISPLACEMENT_VOLUME_CONFIRM
                    and not volume_confirmed
                ):
                    continue

                quality = _quality_from_metrics(
                    wick_ratio=wick_ratio,
                    depth_atr=depth_atr,
                    rvol=rvol,
                    displacement=displacement_confirmed,
                    volatility_allowed=volatility_allowed,
                    pool_fresh=pool_fresh,
                    reclaim_atr=reclaim_distance_atr,
                )

                if quality == SweepQuality.INVALID:
                    continue

                reasons = (
                    "BUY_SIDE_LIQUIDITY_SWEPT",
                    "CLOSE_REJECTED_LEVEL",
                    "ATR_DEPTH_VALID",
                    "REJECTION_VALID",
                    "DISPLACEMENT_CONFIRMED",
                )

                sweeps.append(
                    SweepEvent(
                        direction=SweepDirection.BEARISH,
                        pool=pool,
                        sweep_candle_idx=i,
                        sweep_timestamp=data.index[i],
                        sweep_low=l,
                        sweep_high=h,
                        close_price=c,
                        liquidity_level=level,
                        wick_size=wick_size,
                        body_size=body_size,
                        candle_range=metrics["range"],
                        wick_to_body_ratio=wick_ratio,
                        atr_value=atr_value,
                        sweep_depth=sweep_depth,
                        sweep_depth_atr=depth_atr,
                        body_atr=body_atr,
                        range_atr=(
                            metrics["range"]
                            / atr_value
                        ),
                        sweep_volume=volume,
                        avg_volume=avg_volume,
                        rvol=rvol,
                        reclaim_distance=reclaim_distance,
                        reclaim_distance_atr=reclaim_distance_atr,
                        rejection_strength=rejection_strength,
                        displacement_confirmed=displacement_confirmed,
                        displacement_candle_idx=displacement_idx,
                        displacement_body_atr=displacement_body_atr,
                        displacement_body_percent=displacement_body_percent,
                        volatility_regime=volatility_regime,
                        volatility_allowed=volatility_allowed,
                        abnormal_volatility=abnormal_volatility,
                        volatility_warning=volatility_warning,
                        pool_is_fresh=pool_fresh,
                        pool_is_stale=pool_stale,
                        pool_is_saturated=pool_saturated,
                        pool_significance=float(
                            getattr(
                                pool,
                                "significance_score",
                                0.0,
                            )
                        ),
                        quality=quality,
                        valid=True,
                        rejection_confirmed=True,
                        reclaim_confirmed=reclaimed,
                        depth_valid=True,
                        volume_confirmed=volume_confirmed,
                        reasons=reasons,
                    )
                )

    # =========================================================================
    # SORT
    # =========================================================================

    sweeps.sort(
        key=lambda s: (
            s.sweep_candle_idx,
            s.sweep_timestamp,
        )
    )

    # =========================================================================
    # DEDUPLICATE
    # =========================================================================

    unique: dict[tuple, SweepEvent] = {}

    quality_rank = {
        SweepQuality.INVALID: 0,
        SweepQuality.WEAK: 1,
        SweepQuality.MODERATE: 2,
        SweepQuality.STRONG: 3,
        SweepQuality.INSTITUTIONAL: 4,
    }

    for sweep in sweeps:

        cluster_id = getattr(
            sweep.pool,
            "cluster_id",
            None,
        )

        key = (
            sweep.sweep_candle_idx,
            sweep.direction,
            (
                cluster_id
                if cluster_id is not None
                else round(
                    sweep.liquidity_level,
                    8,
                )
            ),
        )

        existing = unique.get(key)

        if existing is None:
            unique[key] = sweep
            continue

        if (
            quality_rank[sweep.quality]
            > quality_rank[existing.quality]
        ):
            unique[key] = sweep

    result = list(
        unique.values()
    )

    result.sort(
        key=lambda s: (
            s.sweep_candle_idx,
            s.sweep_timestamp,
        )
    )

    return result


# ============================================================================
# LATEST SWEEP
# ============================================================================

def get_latest_sweep(
    df: pd.DataFrame,
    pools: List[LiquidityPool],
    direction: Optional[SweepDirection] = None,
    lookback: int = 10,
) -> Optional[SweepEvent]:

    sweeps = detect_sweeps(
        df=df,
        pools=pools,
        lookback=lookback,
    )

    if direction is not None:
        sweeps = [
            sweep
            for sweep in sweeps
            if sweep.direction == direction
        ]

    if not sweeps:
        return None

    return sweeps[-1]


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def is_institutional_sweep(
    sweep: Optional[SweepEvent],
) -> bool:

    if sweep is None:
        return False

    return bool(
        sweep.valid
        and sweep.quality
        == SweepQuality.INSTITUTIONAL
    )


def sweep_has_displacement(
    sweep: Optional[SweepEvent],
) -> bool:

    if sweep is None:
        return False

    return bool(
        sweep.valid
        and sweep.displacement_confirmed
    )


def sweep_is_valid(
    sweep: Optional[SweepEvent],
) -> bool:

    if sweep is None:
        return False

    if not sweep.valid:
        return False

    if not sweep.reclaim_confirmed:
        return False

    if not sweep.rejection_confirmed:
        return False

    if not sweep.depth_valid:
        return False

    if not sweep.pool_is_fresh:
        return False

    if sweep.pool_is_stale:
        return False

    if sweep.pool_is_saturated:
        return False

    if not sweep.volatility_allowed:
        return False

    if (
        DISPLACEMENT_REQUIRED
        and not sweep.displacement_confirmed
    ):
        return False

    if (
        DISPLACEMENT_VOLUME_CONFIRM
        and not sweep.volume_confirmed
    ):
        return False

    return True


# ============================================================================
# SUMMARY
# ============================================================================

def sweep_summary(
    sweep: Optional[SweepEvent],
) -> dict:

    if sweep is None:
        return {
            "valid": False,
            "direction": None,
            "quality": SweepQuality.INVALID.value,
        }

    return {
        "valid": sweep.valid,
        "direction": sweep.direction.value,
        "quality": sweep.quality.value,
        "timestamp": str(
            sweep.sweep_timestamp
        ),
        "liquidity_level": sweep.liquidity_level,
        "sweep_depth_atr": round(
            sweep.sweep_depth_atr,
            3,
        ),
        "wick_body_ratio": round(
            sweep.wick_to_body_ratio,
            3,
        ),
        "rvol": round(
            sweep.rvol,
            3,
        ),
        "displacement": (
            sweep.displacement_confirmed
        ),
        "displacement_candle": (
            sweep.displacement_candle_idx
        ),
        "volume_confirmed": (
            sweep.volume_confirmed
        ),
        "volatility_regime": (
            sweep.volatility_regime
        ),
        "pool_fresh": (
            sweep.pool_is_fresh
        ),
        "pool_saturated": (
            sweep.pool_is_saturated
        ),
        "reclaim_atr": round(
            sweep.reclaim_distance_atr,
            3,
        ),
    }


# ============================================================================
# SELF TEST
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LIQUIDITY SWEEP ENGINE SELF-TEST")
    print("=" * 70)

    print(
        "SweepDirection:",
        [x.value for x in SweepDirection],
    )

    print(
        "SweepQuality:",
        [x.value for x in SweepQuality],
    )

    print(
        "Reclaim required:",
        SWEEP_RECLAIM_REQUIRED,
    )

    print(
        "Rejection required:",
        SWEEP_REJECTION_REQUIRED,
    )

    print(
        "Min depth required:",
        SWEEP_MIN_DEPTH_REQUIRED,
    )

    print(
        "Displacement required:",
        DISPLACEMENT_REQUIRED,
    )

    print(
        "Min depth ATR:",
        SWEEP_MIN_DEPTH_ATR,
    )

    print(
        "Max depth ATR:",
        SWEEP_MAX_DEPTH_ATR,
    )

    print(
        "Min wick/body:",
        SWEEP_MIN_WICK_BODY_RATIO,
    )

    print(
        "Min displacement ATR:",
        DISPLACEMENT_ATR_MULTIPLIER,
    )

    print(
        "Min displacement body %:",
        DISPLACEMENT_BODY_PERCENT,
    )

    print(
        "Min displacement RVOL:",
        DISPLACEMENT_MIN_RVOL,
    )

    print("\n✓ Institutional sweep engine loaded successfully.")