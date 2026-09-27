"""
volatility.py — ATR (Average True Range) and volatility classification.

ATR is used for:
- SL buffer calculation
- Chase protection
- Swing size validation
- Liquidity level significance
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from config.strategy_config import ADX_PERIOD


class VolatilityRegime(str, Enum):
    LOW    = "LOW"
    NORMAL = "NORMAL"
    HIGH   = "HIGH"


@dataclass
class VolatilityAnalysis:
    atr:    float
    atr_pct: float          # ATR as % of close price
    regime: VolatilityRegime
    description: str


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range calculation."""
    high  = df["high"]
    low   = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder smoothing.

    Returns:
        pd.Series of ATR values (same length as df)
    """
    if df.empty or len(df) < period + 1:
        return pd.Series([float("nan")] * len(df), index=df.index)
    tr = true_range(df)
    atr_series = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr_series


def get_current_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ATR value."""
    atr_series = atr(df, period)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return 0.0
    return float(atr_series.iloc[-1])


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as a percentage of the current close price."""
    atr_val = get_current_atr(df, period)
    last_close = df["close"].iloc[-1] if not df.empty else 1.0
    if last_close == 0:
        return 0.0
    return (atr_val / last_close) * 100.0


def classify_volatility(df: pd.DataFrame, period: int = 14) -> VolatilityAnalysis:
    """
    Classify current volatility regime.

    Uses ATR percentile over recent history:
    - LOW:    current ATR < 33rd percentile of last 100 ATR values
    - NORMAL: 33rd–67th percentile
    - HIGH:   > 67th percentile
    """
    if df.empty or len(df) < period + 10:
        return VolatilityAnalysis(
            atr=0.0,
            atr_pct=0.0,
            regime=VolatilityRegime.NORMAL,
            description="Insufficient data for volatility classification",
        )

    atr_series = atr(df, period).dropna()
    recent_atr = atr_series.iloc[-100:] if len(atr_series) >= 100 else atr_series
    current_atr = float(atr_series.iloc[-1])
    current_atr_pct = (current_atr / df["close"].iloc[-1]) * 100 if df["close"].iloc[-1] else 0.0

    p33 = float(recent_atr.quantile(0.33))
    p67 = float(recent_atr.quantile(0.67))

    if current_atr < p33:
        regime = VolatilityRegime.LOW
    elif current_atr <= p67:
        regime = VolatilityRegime.NORMAL
    else:
        regime = VolatilityRegime.HIGH

    desc = (
        f"ATR={current_atr:.4f} ({current_atr_pct:.2f}%) | "
        f"Regime={regime.value} | p33={p33:.4f} p67={p67:.4f}"
    )

    return VolatilityAnalysis(
        atr=current_atr,
        atr_pct=current_atr_pct,
        regime=regime,
        description=desc,
    )
