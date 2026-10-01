"""
volume.py — Institution-Grade Volume / RVOL Engine.

Used for:
- Liquidity sweep confirmation
- Displacement confirmation
- Entry confirmation
- RVOL analysis
- Abnormal volume spike detection
- Volume regime classification

Design principles:
- Closed-candle data only.
- Volume is confirmation, NOT a standalone trigger.
- Extreme volume does not automatically mean bullish/bearish.
- Median + SMA baselines are used to reduce sensitivity to one
  abnormal candle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config.strategy_config import (
    VOLUME_AVG_PERIOD,
    VOLUME_MEDIAN_PERIOD,
    VOLUME_CONFIRM_MULTIPLIER,
    RVOL_CONFIRM_MIN,
    RVOL_STRONG_MIN,
    RVOL_EXTREME_MIN,
    EXTREME_VOLUME_IS_NOT_A_TRIGGER,
    VOLUME_DIRECTION_CONFIRMATION,
)


# ============================================================
# ENUMS
# ============================================================

class VolumeRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class VolumeDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


# ============================================================
# RESULT DATACLASS
# ============================================================

@dataclass
class VolumeAnalysis:
    # --------------------------------------------------------
    # Backward-compatible fields
    # --------------------------------------------------------

    current_volume: float
    avg_volume: float
    volume_ratio: float
    is_confirmed: bool
    description: str

    # --------------------------------------------------------
    # Institution-grade fields
    # --------------------------------------------------------

    median_volume: float = 0.0

    rvol_sma: float = 0.0
    rvol_median: float = 0.0

    regime: VolumeRegime = VolumeRegime.UNKNOWN

    is_elevated: bool = False
    is_high: bool = False
    is_extreme: bool = False
    is_spike: bool = False

    direction: VolumeDirection = VolumeDirection.NEUTRAL

    directional_confirmation: bool = False

    data_sufficient: bool = False


# ============================================================
# BASIC VALIDATION
# ============================================================

def _clean_volume_series(
    df: pd.DataFrame,
) -> pd.Series:
    """Return clean numeric volume series."""

    if df.empty or "volume" not in df.columns:
        return pd.Series(
            index=df.index,
            dtype=float,
        )

    volume = pd.to_numeric(
        df["volume"],
        errors="coerce",
    )

    volume = volume.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Negative volume is invalid.
    volume = volume.where(volume >= 0)

    return volume


# ============================================================
# VOLUME SMA
# ============================================================

def volume_sma(
    df: pd.DataFrame,
    period: int = VOLUME_AVG_PERIOD,
) -> pd.Series:
    """
    Rolling simple moving average of volume.
    """

    if period <= 0:
        raise ValueError(
            "Volume SMA period must be positive"
        )

    volume = _clean_volume_series(df)

    return volume.rolling(
        window=period,
        min_periods=period,
    ).mean()


# ============================================================
# VOLUME MEDIAN
# ============================================================

def volume_median(
    df: pd.DataFrame,
    period: int = VOLUME_MEDIAN_PERIOD,
) -> pd.Series:
    """
    Rolling median volume.

    Median is more robust than SMA when the dataset contains
    occasional extreme volume spikes.
    """

    if period <= 0:
        raise ValueError(
            "Volume median period must be positive"
        )

    volume = _clean_volume_series(df)

    return volume.rolling(
        window=period,
        min_periods=period,
    ).median()


# ============================================================
# HIGH VOLUME
# ============================================================

def is_high_volume(
    candle_volume: float,
    avg_volume: float,
    multiplier: float = VOLUME_CONFIRM_MULTIPLIER,
) -> bool:
    """
    Backward-compatible high-volume check.

    True when:
        candle volume >= multiplier × baseline volume
    """

    if (
        candle_volume <= 0
        or avg_volume <= 0
        or multiplier <= 0
    ):
        return False

    return candle_volume >= (
        multiplier * avg_volume
    )


# ============================================================
# RVOL
# ============================================================

def calculate_rvol(
    current_volume: float,
    baseline_volume: float,
) -> float:
    """
    Relative Volume:

        RVOL = Current Volume / Baseline Volume
    """

    if (
        current_volume <= 0
        or baseline_volume <= 0
    ):
        return 0.0

    return float(
        current_volume / baseline_volume
    )


# ============================================================
# VOLUME REGIME
# ============================================================

def classify_volume_regime(
    rvol: float,
) -> VolumeRegime:
    """
    Classify volume using RVOL.

    Approximation:

        < 0.70     LOW
        0.70–1.00  NORMAL
        1.00–1.50  ELEVATED
        1.50–2.50  HIGH
        >= 2.50    EXTREME
    """

    if rvol <= 0:
        return VolumeRegime.UNKNOWN

    if rvol < 0.70:
        return VolumeRegime.LOW

    if rvol < RVOL_CONFIRM_MIN:
        return VolumeRegime.NORMAL

    if rvol < RVOL_STRONG_MIN:
        return VolumeRegime.ELEVATED

    if rvol < RVOL_EXTREME_MIN:
        return VolumeRegime.HIGH

    return VolumeRegime.EXTREME


# ============================================================
# CANDLE DIRECTION
# ============================================================

def candle_volume_direction(
    df: pd.DataFrame,
) -> VolumeDirection:
    """
    Determine the direction of the latest candle.

    Volume itself has no direction.
    Direction comes from candle price movement.
    """

    if df.empty:
        return VolumeDirection.NEUTRAL

    required = {"open", "close"}

    if not required.issubset(df.columns):
        return VolumeDirection.NEUTRAL

    try:
        open_price = float(df["open"].iloc[-1])
        close_price = float(df["close"].iloc[-1])
    except (TypeError, ValueError, IndexError):
        return VolumeDirection.NEUTRAL

    if close_price > open_price:
        return VolumeDirection.BULLISH

    if close_price < open_price:
        return VolumeDirection.BEARISH

    return VolumeDirection.NEUTRAL


# ============================================================
# DIRECTIONAL CONFIRMATION
# ============================================================

def confirms_direction(
    analysis: VolumeAnalysis,
    direction: Optional[str],
) -> bool:
    """
    Check whether volume-backed candle direction agrees with
    requested trade direction.

    direction examples:
        LONG
        SHORT
        BULLISH
        BEARISH
    """

    if not direction:
        return False

    direction = str(direction).upper()

    if direction in {
        "LONG",
        "BUY",
        "BULLISH",
    }:
        return (
            analysis.direction
            == VolumeDirection.BULLISH
        )

    if direction in {
        "SHORT",
        "SELL",
        "BEARISH",
    }:
        return (
            analysis.direction
            == VolumeDirection.BEARISH
        )

    return False


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_volume(
    df: pd.DataFrame,
    period: int = VOLUME_AVG_PERIOD,
    multiplier: float = VOLUME_CONFIRM_MULTIPLIER,
) -> VolumeAnalysis:
    """
    Analyze the most recent closed candle.

    Baselines:
        - SMA volume
        - Median volume

    RVOL:
        Current / SMA
        Current / Median

    The SMA RVOL is used for compatibility and primary
    confirmation.

    The median RVOL provides robustness during abnormal spikes.
    """

    if (
        df.empty
        or "volume" not in df.columns
        or len(df) < max(period, VOLUME_MEDIAN_PERIOD)
    ):
        return VolumeAnalysis(
            current_volume=0.0,
            avg_volume=0.0,
            volume_ratio=0.0,
            is_confirmed=False,
            description="Insufficient volume data",
            data_sufficient=False,
        )

    volume = _clean_volume_series(df)

    if volume.isna().iloc[-1]:
        return VolumeAnalysis(
            current_volume=0.0,
            avg_volume=0.0,
            volume_ratio=0.0,
            is_confirmed=False,
            description="Invalid latest volume",
            data_sufficient=False,
        )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    sma_series = volume_sma(
        df,
        period=period,
    )

    median_series = volume_median(
        df,
        period=VOLUME_MEDIAN_PERIOD,
    )

    current_volume = float(
        volume.iloc[-1]
    )

    avg_volume = (
        float(sma_series.iloc[-1])
        if pd.notna(sma_series.iloc[-1])
        else 0.0
    )

    median_volume = (
        float(median_series.iloc[-1])
        if pd.notna(median_series.iloc[-1])
        else 0.0
    )

    # --------------------------------------------------------
    # RVOL
    # --------------------------------------------------------

    rvol_sma = calculate_rvol(
        current_volume,
        avg_volume,
    )

    rvol_median = calculate_rvol(
        current_volume,
        median_volume,
    )

    # Backward-compatible ratio.
    volume_ratio = rvol_sma

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    confirmed = is_high_volume(
        current_volume,
        avg_volume,
        multiplier,
    )

    # Also require the configured minimum RVOL.
    confirmed = (
        confirmed
        and rvol_sma >= RVOL_CONFIRM_MIN
    )

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    regime = classify_volume_regime(
        rvol_sma
    )

    is_elevated = (
        regime
        in {
            VolumeRegime.ELEVATED,
            VolumeRegime.HIGH,
            VolumeRegime.EXTREME,
        }
    )

    is_high = (
        regime
        in {
            VolumeRegime.HIGH,
            VolumeRegime.EXTREME,
        }
    )

    is_extreme = (
        regime == VolumeRegime.EXTREME
    )

    # --------------------------------------------------------
    # Spike
    # --------------------------------------------------------

    # A very large deviation from the baseline is treated as
    # a spike. It is information, not an automatic entry.
    is_spike = (
        rvol_sma >= RVOL_EXTREME_MIN
        or rvol_median >= RVOL_EXTREME_MIN
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = candle_volume_direction(
        df
    )

    directional_confirmation = (
        confirmed
        and direction
        != VolumeDirection.NEUTRAL
    )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    status = (
        "CONFIRMED"
        if confirmed
        else "LOW"
    )

    spike_status = (
        " | SPIKE"
        if is_spike
        else ""
    )

    desc = (
        f"Volume={current_volume:,.0f} "
        f"| SMA{period}={avg_volume:,.0f} "
        f"| Median{VOLUME_MEDIAN_PERIOD}="
        f"{median_volume:,.0f} "
        f"| RVOL={rvol_sma:.2f}x "
        f"| MedianRVOL={rvol_median:.2f}x "
        f"| Regime={regime.value} "
        f"| Direction={direction.value} "
        f"| {status}"
        f"{spike_status}"
    )

    return VolumeAnalysis(
        current_volume=current_volume,
        avg_volume=avg_volume,
        volume_ratio=volume_ratio,
        is_confirmed=confirmed,
        description=desc,
        median_volume=median_volume,
        rvol_sma=rvol_sma,
        rvol_median=rvol_median,
        regime=regime,
        is_elevated=is_elevated,
        is_high=is_high,
        is_extreme=is_extreme,
        is_spike=is_spike,
        direction=direction,
        directional_confirmation=(
            directional_confirmation
        ),
        data_sufficient=True,
    )


# ============================================================
# INDEX-BASED VOLUME CHECK
# ============================================================

def get_volume_at_index(
    df: pd.DataFrame,
    idx: int,
    avg_volume: float,
    multiplier: float = VOLUME_CONFIRM_MULTIPLIER,
) -> bool:
    """
    Backward-compatible check for historical candle volume.
    """

    if (
        idx < 0
        or idx >= len(df)
        or "volume" not in df.columns
    ):
        return False

    try:
        volume = float(
            df["volume"].iloc[idx]
        )
    except (TypeError, ValueError):
        return False

    return is_high_volume(
        volume,
        avg_volume,
        multiplier,
    )


# ============================================================
# HISTORICAL RVOL
# ============================================================

def get_rvol_at_index(
    df: pd.DataFrame,
    idx: int,
    period: int = VOLUME_AVG_PERIOD,
) -> float:
    """
    Calculate RVOL for any historical candle.

    Uses only candles available up to that point.
    """

    if (
        idx < 0
        or idx >= len(df)
        or "volume" not in df.columns
    ):
        return 0.0

    if idx < period:
        return 0.0

    volume = _clean_volume_series(df)

    current = volume.iloc[idx]

    baseline = (
        volume.iloc[
            idx - period:idx
        ].mean()
    )

    if (
        pd.isna(current)
        or pd.isna(baseline)
        or baseline <= 0
    ):
        return 0.0

    return float(
        current / baseline
    )


# ============================================================
# SPIKE DETECTION
# ============================================================

def detect_volume_spike(
    df: pd.DataFrame,
    period: int = VOLUME_AVG_PERIOD,
    extreme_rvol: float = RVOL_EXTREME_MIN,
) -> bool:
    """
    Detect abnormal volume spike on the latest candle.

    This is intentionally separate from confirmation.

    A spike can represent:
        - liquidation
        - news
        - breakout
        - stop hunt
        - institutional activity

    Therefore spike != directional signal.
    """

    analysis = analyze_volume(
        df,
        period=period,
    )

    if not analysis.data_sufficient:
        return False

    return (
        analysis.rvol_sma >= extreme_rvol
        or analysis.rvol_median >= extreme_rvol
    )


# ============================================================
# SAFE SUMMARY
# ============================================================

def volume_summary(
    analysis: VolumeAnalysis,
) -> dict:
    """
    Serializable representation for:
        Telegram
        database
        logging
        debugging
    """

    return {
        "current_volume": round(
            analysis.current_volume,
            4,
        ),
        "avg_volume": round(
            analysis.avg_volume,
            4,
        ),
        "median_volume": round(
            analysis.median_volume,
            4,
        ),
        "volume_ratio": round(
            analysis.volume_ratio,
            4,
        ),
        "rvol_sma": round(
            analysis.rvol_sma,
            4,
        ),
        "rvol_median": round(
            analysis.rvol_median,
            4,
        ),
        "is_confirmed": analysis.is_confirmed,
        "is_elevated": analysis.is_elevated,
        "is_high": analysis.is_high,
        "is_extreme": analysis.is_extreme,
        "is_spike": analysis.is_spike,
        "regime": analysis.regime.value,
        "direction": analysis.direction.value,
        "directional_confirmation": (
            analysis.directional_confirmation
        ),
        "data_sufficient": analysis.data_sufficient,
        "description": analysis.description,
    }