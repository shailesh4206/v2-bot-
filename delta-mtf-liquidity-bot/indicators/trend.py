"""
trend.py — Trend indicators: EMA, ADX, trend classification.

Used across all timeframes (4H, 1H, 15M).
NO 5M references.
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from config.strategy_config import (
    ADX_PERIOD,
    ADX_STRONG_TREND_THRESHOLD,
    ADX_WEAK_TREND_THRESHOLD,
    EMA_FAST_4H, EMA_SLOW_4H,
    EMA_FAST_1H, EMA_SLOW_1H,
    EMA_FAST_15M, EMA_SLOW_15M,
)


class TrendBias(str, Enum):
    BULLISH      = "BULLISH"
    BEARISH      = "BEARISH"
    NEUTRAL      = "NEUTRAL"


class TrendStrength(str, Enum):
    STRONG  = "STRONG"
    NORMAL  = "NORMAL"
    WEAK    = "WEAK"
    RANGING = "RANGING"


@dataclass
class TrendAnalysis:
    bias:         TrendBias
    strength:     TrendStrength
    adx:          float
    ema_fast:     float
    ema_slow:     float
    price_vs_ema: str        # "ABOVE_FAST", "BELOW_FAST", "ABOVE_SLOW", "BELOW_SLOW"
    description:  str


# ─────────────────────────────────────────────
# Core indicator functions
# ─────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(H-L, |H-prevC|, |L-prevC|)."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs(),
    ], axis=1).max(axis=1)
    return tr


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """
    Average Directional Index (Wilder smoothing).
    Returns the ADX series.
    """
    high  = df["high"]
    low   = df["low"]

    plus_dm  = high.diff()
    minus_dm = -low.diff()

    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = true_range(df)

    # Wilder smoothing
    atr_s    = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val


def classify_trend(
    df: pd.DataFrame,
    timeframe: str,
) -> TrendAnalysis:
    """
    Classify the trend for a given timeframe using EMA + ADX.

    Args:
        df:        OHLCV DataFrame (index = datetime)
        timeframe: "4h", "1h", "15m"

    Returns:
        TrendAnalysis dataclass
    """
    if df.empty or len(df) < 210:
        return TrendAnalysis(
            bias=TrendBias.NEUTRAL,
            strength=TrendStrength.RANGING,
            adx=0.0,
            ema_fast=0.0,
            ema_slow=0.0,
            price_vs_ema="UNKNOWN",
            description="Insufficient data",
        )

    # Choose EMA periods based on timeframe
    tf_map = {
        "4h":  (EMA_FAST_4H,  EMA_SLOW_4H),
        "1h":  (EMA_FAST_1H,  EMA_SLOW_1H),
        "15m": (EMA_FAST_15M, EMA_SLOW_15M),
    }
    fast_period, slow_period = tf_map.get(timeframe, (50, 200))

    ema_fast_series = ema(df["close"], fast_period)
    ema_slow_series = ema(df["close"], slow_period)
    adx_series      = adx(df)

    last_close    = df["close"].iloc[-1]
    last_ema_fast = ema_fast_series.iloc[-1]
    last_ema_slow = ema_slow_series.iloc[-1]
    last_adx      = adx_series.iloc[-1]

    # Price vs EMA
    if last_close > last_ema_fast > last_ema_slow:
        price_vs_ema = "ABOVE_BOTH"
        raw_bias = TrendBias.BULLISH
    elif last_close < last_ema_fast < last_ema_slow:
        price_vs_ema = "BELOW_BOTH"
        raw_bias = TrendBias.BEARISH
    elif last_close > last_ema_slow:
        price_vs_ema = "ABOVE_SLOW"
        raw_bias = TrendBias.BULLISH
    elif last_close < last_ema_slow:
        price_vs_ema = "BELOW_SLOW"
        raw_bias = TrendBias.BEARISH
    else:
        price_vs_ema = "MIXED"
        raw_bias = TrendBias.NEUTRAL

    # EMA alignment check
    if last_ema_fast > last_ema_slow and raw_bias == TrendBias.BULLISH:
        bias = TrendBias.BULLISH
    elif last_ema_fast < last_ema_slow and raw_bias == TrendBias.BEARISH:
        bias = TrendBias.BEARISH
    else:
        bias = TrendBias.NEUTRAL

    # ADX → Strength
    if last_adx > ADX_STRONG_TREND_THRESHOLD:
        strength = TrendStrength.STRONG
    elif last_adx > ADX_WEAK_TREND_THRESHOLD:
        strength = TrendStrength.NORMAL
    elif last_adx > 10:
        strength = TrendStrength.WEAK
    else:
        strength = TrendStrength.RANGING

    description = (
        f"{timeframe.upper()} | {bias.value} {strength.value} "
        f"| EMA{fast_period}={last_ema_fast:.2f} EMA{slow_period}={last_ema_slow:.2f} "
        f"| ADX={last_adx:.1f}"
    )

    return TrendAnalysis(
        bias=bias,
        strength=strength,
        adx=last_adx,
        ema_fast=last_ema_fast,
        ema_slow=last_ema_slow,
        price_vs_ema=price_vs_ema,
        description=description,
    )


def is_strongly_bullish(analysis: TrendAnalysis) -> bool:
    return (
        analysis.bias == TrendBias.BULLISH
        and analysis.strength == TrendStrength.STRONG
    )


def is_strongly_bearish(analysis: TrendAnalysis) -> bool:
    return (
        analysis.bias == TrendBias.BEARISH
        and analysis.strength == TrendStrength.STRONG
    )
