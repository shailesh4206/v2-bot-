"""
volume.py — Volume analysis and confirmation.

Used for:
- Confirming entry signals (volume > N-period average)
- Scoring liquidity sweeps (high volume at sweep = stronger signal)
- Detecting anomalous volume spikes
"""

import pandas as pd
from dataclasses import dataclass
from config.strategy_config import VOLUME_AVG_PERIOD, VOLUME_CONFIRM_MULTIPLIER


@dataclass
class VolumeAnalysis:
    current_volume:  float
    avg_volume:      float
    volume_ratio:    float   # current / avg
    is_confirmed:    bool    # True if current > multiplier × avg
    description:     str


def volume_sma(df: pd.DataFrame, period: int = VOLUME_AVG_PERIOD) -> pd.Series:
    """Rolling simple moving average of volume."""
    return df["volume"].rolling(window=period).mean()


def is_high_volume(
    candle_volume: float,
    avg_volume: float,
    multiplier: float = VOLUME_CONFIRM_MULTIPLIER,
) -> bool:
    """True if candle volume exceeds the threshold."""
    if avg_volume <= 0:
        return False
    return candle_volume >= (multiplier * avg_volume)


def analyze_volume(
    df: pd.DataFrame,
    period: int = VOLUME_AVG_PERIOD,
    multiplier: float = VOLUME_CONFIRM_MULTIPLIER,
) -> VolumeAnalysis:
    """
    Analyze volume on the most recent closed candle.

    Args:
        df:         OHLCV DataFrame
        period:     Rolling average period
        multiplier: Confirmation threshold multiplier

    Returns:
        VolumeAnalysis dataclass
    """
    if df.empty or len(df) < 2:
        return VolumeAnalysis(
            current_volume=0.0,
            avg_volume=0.0,
            volume_ratio=0.0,
            is_confirmed=False,
            description="Insufficient data",
        )

    vol_avg = volume_sma(df, period)

    # Use second-to-last row if last row is the open (unconfirmed) candle
    # In live mode we only pass confirmed candles, so iloc[-1] is always safe.
    current_vol = float(df["volume"].iloc[-1])
    avg_vol     = float(vol_avg.iloc[-1]) if not pd.isna(vol_avg.iloc[-1]) else 0.0
    ratio       = current_vol / avg_vol if avg_vol > 0 else 0.0
    confirmed   = is_high_volume(current_vol, avg_vol, multiplier)

    desc = (
        f"Volume={current_vol:,.0f} | Avg{period}={avg_vol:,.0f} | "
        f"Ratio={ratio:.2f}x | {'✅ CONFIRMED' if confirmed else '❌ LOW'}"
    )

    return VolumeAnalysis(
        current_volume=current_vol,
        avg_volume=avg_vol,
        volume_ratio=ratio,
        is_confirmed=confirmed,
        description=desc,
    )


def get_volume_at_index(
    df: pd.DataFrame,
    idx: int,
    avg_volume: float,
    multiplier: float = VOLUME_CONFIRM_MULTIPLIER,
) -> bool:
    """Check if the candle at position idx had high volume vs. average."""
    if idx < 0 or idx >= len(df):
        return False
    vol = float(df["volume"].iloc[idx])
    return is_high_volume(vol, avg_volume, multiplier)
