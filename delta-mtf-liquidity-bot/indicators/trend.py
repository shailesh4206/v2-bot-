"""
trend.py — Institution-Grade Trend / Regime Engine.

Used across:
    4H = Macro Regime
    1H = Structure Regime
    15M = Execution Regime

NO 5M references.

Design principles:
- Closed-candle analysis only.
- No future leakage.
- EMA alignment + price location + EMA slope.
- ADX measures trend strength, not direction.
- +DI / -DI provides directional confirmation.
- Neutral/conflicting conditions remain neutral.
- Insufficient/invalid data never produces a bullish/bearish signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from enum import Enum
from dataclasses import dataclass
from typing import Optional

from config.strategy_config import (
    ADX_PERIOD,
    ADX_STRONG_TREND_THRESHOLD,
    ADX_WEAK_TREND_THRESHOLD,
    ADX_EXTREME_TREND_THRESHOLD,
    DI_CONFIRMATION_ENABLED,
    DI_MIN_SEPARATION,
    EMA_FAST_4H,
    EMA_SLOW_4H,
    EMA_FAST_1H,
    EMA_SLOW_1H,
    EMA_FAST_15M,
    EMA_SLOW_15M,
    EMA_SLOPE_LOOKBACK,
    EMA_SLOPE_MIN_PERCENT,
)


# ============================================================
# ENUMS
# ============================================================

class TrendBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TrendStrength(str, Enum):
    STRONG = "STRONG"
    NORMAL = "NORMAL"
    WEAK = "WEAK"
    RANGING = "RANGING"


class TrendRegime(str, Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    WEAK_BULLISH = "WEAK_BULLISH"
    WEAK_BEARISH = "WEAK_BEARISH"
    RANGING = "RANGING"
    CONFLICTED = "CONFLICTED"


# ============================================================
# RESULT DATACLASS
# ============================================================

@dataclass
class TrendAnalysis:
    bias: TrendBias
    strength: TrendStrength

    adx: float

    ema_fast: float
    ema_slow: float

    price_vs_ema: str

    description: str

    # --------------------------------------------------------
    # Directional movement
    # --------------------------------------------------------

    plus_di: float = 0.0
    minus_di: float = 0.0
    di_spread: float = 0.0

    # --------------------------------------------------------
    # EMA structure
    # --------------------------------------------------------

    ema_fast_slope_percent: float = 0.0
    ema_slow_slope_percent: float = 0.0

    ema_aligned: bool = False
    slope_confirmed: bool = False

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    regime: TrendRegime = TrendRegime.RANGING

    # --------------------------------------------------------
    # Data state
    # --------------------------------------------------------

    data_sufficient: bool = False
    closed_candle_confirmed: bool = True

    # --------------------------------------------------------
    # Raw price information
    # --------------------------------------------------------

    last_close: float = 0.0


# ============================================================
# CORE INDICATORS
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    adjust=False is used so the indicator behaves consistently
    with live sequential candle processing.
    """

    if period <= 0:
        raise ValueError("EMA period must be positive")

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""

    if period <= 0:
        raise ValueError("SMA period must be positive")

    return series.rolling(
        window=period,
        min_periods=period,
    ).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range:

        max(
            High - Low,
            abs(High - Previous Close),
            abs(Low - Previous Close)
        )
    """

    required = {"high", "low", "close"}

    if not required.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    previous_close = pd.to_numeric(
        df["close"],
        errors="coerce",
    ).shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.replace([np.inf, -np.inf], np.nan)


# ============================================================
# DIRECTIONAL MOVEMENT
# ============================================================

def directional_movement(
    df: pd.DataFrame,
    period: int = ADX_PERIOD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns:

        ADX
        +DI
        -DI

    Uses Wilder-style exponential smoothing.
    """

    if period <= 0:
        raise ValueError("ADX period must be positive")

    if df.empty:
        empty = pd.Series(index=df.index, dtype=float)
        return empty, empty, empty

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
        dtype=float,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
        dtype=float,
    )

    tr = true_range(df)

    # Wilder smoothing.
    atr_smoothed = tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_smoothed = plus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    safe_atr = atr_smoothed.replace(0.0, np.nan)

    plus_di = (
        100.0 * plus_smoothed / safe_atr
    )

    minus_di = (
        100.0 * minus_smoothed / safe_atr
    )

    di_sum = (
        plus_di + minus_di
    ).replace(0.0, np.nan)

    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / di_sum
    )

    adx_value = dx.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return (
        adx_value.replace([np.inf, -np.inf], np.nan),
        plus_di.replace([np.inf, -np.inf], np.nan),
        minus_di.replace([np.inf, -np.inf], np.nan),
    )


def adx(
    df: pd.DataFrame,
    period: int = ADX_PERIOD,
) -> pd.Series:
    """
    Backward-compatible ADX-only function.
    """

    adx_series, _, _ = directional_movement(
        df,
        period=period,
    )

    return adx_series


def plus_di(
    df: pd.DataFrame,
    period: int = ADX_PERIOD,
) -> pd.Series:
    """Return +DI series."""

    _, plus_series, _ = directional_movement(
        df,
        period=period,
    )

    return plus_series


def minus_di(
    df: pd.DataFrame,
    period: int = ADX_PERIOD,
) -> pd.Series:
    """Return -DI series."""

    _, _, minus_series = directional_movement(
        df,
        period=period,
    )

    return minus_series


# ============================================================
# DATA VALIDATION
# ============================================================

def _validate_ohlcv(df: pd.DataFrame) -> bool:
    """Basic OHLC validation."""

    required = {"open", "high", "low", "close"}

    if df.empty or not required.issubset(df.columns):
        return False

    try:
        numeric = df[
            ["open", "high", "low", "close"]
        ].apply(pd.to_numeric, errors="coerce")

        if numeric.isna().any().any():
            return False

        if not np.isfinite(numeric.to_numpy()).all():
            return False

        if (numeric["high"] < numeric["low"]).any():
            return False

        if (numeric["high"] < numeric["open"]).any():
            return False

        if (numeric["high"] < numeric["close"]).any():
            return False

        if (numeric["low"] > numeric["open"]).any():
            return False

        if (numeric["low"] > numeric["close"]).any():
            return False

        if (numeric["close"] <= 0).any():
            return False

    except Exception:
        return False

    return True


# ============================================================
# TIMEFRAME SETTINGS
# ============================================================

def _get_ema_periods(
    timeframe: str,
) -> Optional[tuple[int, int]]:
    """
    Return configured EMA periods.

    Only 4H, 1H and 15M are supported.
    """

    tf_map = {
        "4h": (
            EMA_FAST_4H,
            EMA_SLOW_4H,
        ),
        "1h": (
            EMA_FAST_1H,
            EMA_SLOW_1H,
        ),
        "15m": (
            EMA_FAST_15M,
            EMA_SLOW_15M,
        ),
    }

    return tf_map.get(
        str(timeframe).lower()
    )


def _minimum_required_bars(
    fast_period: int,
    slow_period: int,
    adx_period: int,
) -> int:
    """
    Conservative data requirement.

    Slow EMA + ADX need enough warm-up candles.
    """

    return max(
        slow_period + 20,
        adx_period * 3 + 10,
        EMA_SLOPE_LOOKBACK + slow_period,
    )


# ============================================================
# EMA SLOPE
# ============================================================

def _slope_percent(
    series: pd.Series,
    lookback: int,
) -> float:
    """
    Percentage slope between current EMA and EMA N candles ago.
    """

    if len(series) <= lookback:
        return 0.0

    current = series.iloc[-1]
    previous = series.iloc[-1 - lookback]

    if pd.isna(current) or pd.isna(previous):
        return 0.0

    if previous == 0:
        return 0.0

    return float(
        ((current - previous) / abs(previous)) * 100.0
    )


# ============================================================
# TREND CLASSIFICATION
# ============================================================

def classify_trend(
    df: pd.DataFrame,
    timeframe: str,
) -> TrendAnalysis:
    """
    Classify trend/regime for 4H, 1H or 15M.

    Important:
        This function expects the dataframe to represent
        CONFIRMED/CLOSED candles.

    It does not use future candles and does not create
    look-ahead bias.

    Direction is determined using:

        1. Price location
        2. EMA alignment
        3. +DI / -DI
        4. EMA slope
        5. ADX strength

    ADX itself does NOT determine direction.
    """

    timeframe = str(timeframe).lower()

    periods = _get_ema_periods(timeframe)

    if periods is None:
        return TrendAnalysis(
            bias=TrendBias.NEUTRAL,
            strength=TrendStrength.RANGING,
            adx=0.0,
            ema_fast=0.0,
            ema_slow=0.0,
            price_vs_ema="UNKNOWN",
            description=f"Unsupported timeframe: {timeframe}",
            regime=TrendRegime.RANGING,
            data_sufficient=False,
        )

    fast_period, slow_period = periods

    required_bars = _minimum_required_bars(
        fast_period,
        slow_period,
        ADX_PERIOD,
    )

    # --------------------------------------------------------
    # Basic data checks
    # --------------------------------------------------------

    if not _validate_ohlcv(df):
        return TrendAnalysis(
            bias=TrendBias.NEUTRAL,
            strength=TrendStrength.RANGING,
            adx=0.0,
            ema_fast=0.0,
            ema_slow=0.0,
            price_vs_ema="UNKNOWN",
            description=(
                f"{timeframe.upper()} | Invalid OHLC data"
            ),
            regime=TrendRegime.RANGING,
            data_sufficient=False,
        )

    if len(df) < required_bars:
        return TrendAnalysis(
            bias=TrendBias.NEUTRAL,
            strength=TrendStrength.RANGING,
            adx=0.0,
            ema_fast=0.0,
            ema_slow=0.0,
            price_vs_ema="UNKNOWN",
            description=(
                f"{timeframe.upper()} | "
                f"Insufficient data "
                f"({len(df)}/{required_bars})"
            ),
            regime=TrendRegime.RANGING,
            data_sufficient=False,
        )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    close = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    ema_fast_series = ema(
        close,
        fast_period,
    )

    ema_slow_series = ema(
        close,
        slow_period,
    )

    adx_series, plus_di_series, minus_di_series = (
        directional_movement(
            df,
            period=ADX_PERIOD,
        )
    )

    last_close = float(close.iloc[-1])

    last_ema_fast = float(
        ema_fast_series.iloc[-1]
    )

    last_ema_slow = float(
        ema_slow_series.iloc[-1]
    )

    last_adx = float(
        adx_series.iloc[-1]
    ) if pd.notna(adx_series.iloc[-1]) else 0.0

    last_plus_di = float(
        plus_di_series.iloc[-1]
    ) if pd.notna(plus_di_series.iloc[-1]) else 0.0

    last_minus_di = float(
        minus_di_series.iloc[-1]
    ) if pd.notna(minus_di_series.iloc[-1]) else 0.0

    di_spread = abs(
        last_plus_di - last_minus_di
    )

    # --------------------------------------------------------
    # EMA slope
    # --------------------------------------------------------

    fast_slope = _slope_percent(
        ema_fast_series,
        EMA_SLOPE_LOOKBACK,
    )

    slow_slope = _slope_percent(
        ema_slow_series,
        EMA_SLOPE_LOOKBACK,
    )

    slope_confirmed = (
        abs(fast_slope) >= EMA_SLOPE_MIN_PERCENT
        or abs(slow_slope) >= EMA_SLOPE_MIN_PERCENT
    )

    # --------------------------------------------------------
    # Price location
    # --------------------------------------------------------

    if (
        last_close > last_ema_fast
        and last_close > last_ema_slow
    ):
        price_vs_ema = "ABOVE_BOTH"

    elif (
        last_close < last_ema_fast
        and last_close < last_ema_slow
    ):
        price_vs_ema = "BELOW_BOTH"

    elif last_close > last_ema_fast:
        price_vs_ema = "ABOVE_FAST"

    elif last_close < last_ema_fast:
        price_vs_ema = "BELOW_FAST"

    elif last_close > last_ema_slow:
        price_vs_ema = "ABOVE_SLOW"

    elif last_close < last_ema_slow:
        price_vs_ema = "BELOW_SLOW"

    else:
        price_vs_ema = "MIXED"

    # --------------------------------------------------------
    # EMA alignment
    # --------------------------------------------------------

    bullish_ema_alignment = (
        last_ema_fast > last_ema_slow
    )

    bearish_ema_alignment = (
        last_ema_fast < last_ema_slow
    )

    ema_aligned = (
        bullish_ema_alignment
        or bearish_ema_alignment
    )

    # --------------------------------------------------------
    # Directional confirmation
    # --------------------------------------------------------

    bullish_di = (
        last_plus_di > last_minus_di
        and di_spread >= DI_MIN_SEPARATION
    )

    bearish_di = (
        last_minus_di > last_plus_di
        and di_spread >= DI_MIN_SEPARATION
    )

    if not DI_CONFIRMATION_ENABLED:
        bullish_di = True
        bearish_di = True

    # --------------------------------------------------------
    # Raw directional state
    # --------------------------------------------------------

    bullish_price = (
        last_close > last_ema_fast
        and last_close > last_ema_slow
    )

    bearish_price = (
        last_close < last_ema_fast
        and last_close < last_ema_slow
    )

    bullish_slope = (
        fast_slope > 0
        and slow_slope >= 0
    )

    bearish_slope = (
        fast_slope < 0
        and slow_slope <= 0
    )

    # --------------------------------------------------------
    # Direction decision
    # --------------------------------------------------------

    bullish_score = 0
    bearish_score = 0

    if bullish_price:
        bullish_score += 1

    if bearish_price:
        bearish_score += 1

    if bullish_ema_alignment:
        bullish_score += 1

    if bearish_ema_alignment:
        bearish_score += 1

    if bullish_di:
        bullish_score += 1

    if bearish_di:
        bearish_score += 1

    if bullish_slope:
        bullish_score += 1

    if bearish_slope:
        bearish_score += 1

    # Require strong agreement.
    if bullish_score >= 3 and bullish_score > bearish_score:
        bias = TrendBias.BULLISH

    elif bearish_score >= 3 and bearish_score > bullish_score:
        bias = TrendBias.BEARISH

    else:
        bias = TrendBias.NEUTRAL

    # --------------------------------------------------------
    # ADX strength
    # --------------------------------------------------------

    if last_adx >= ADX_EXTREME_TREND_THRESHOLD:
        strength = TrendStrength.STRONG

    elif last_adx >= ADX_STRONG_TREND_THRESHOLD:
        strength = TrendStrength.STRONG

    elif last_adx >= ADX_WEAK_TREND_THRESHOLD:
        strength = TrendStrength.NORMAL

    elif last_adx >= 10.0:
        strength = TrendStrength.WEAK

    else:
        strength = TrendStrength.RANGING

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    if (
        bias == TrendBias.BULLISH
        and strength == TrendStrength.STRONG
    ):
        regime = TrendRegime.TRENDING_BULLISH

    elif (
        bias == TrendBias.BEARISH
        and strength == TrendStrength.STRONG
    ):
        regime = TrendRegime.TRENDING_BEARISH

    elif bias == TrendBias.BULLISH:
        regime = TrendRegime.WEAK_BULLISH

    elif bias == TrendBias.BEARISH:
        regime = TrendRegime.WEAK_BEARISH

    else:
        # Strong EMA conflict or lack of directional agreement.
        if (
            bullish_score > 0
            and bearish_score > 0
            and abs(bullish_score - bearish_score) <= 1
        ):
            regime = TrendRegime.CONFLICTED
        else:
            regime = TrendRegime.RANGING

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    description = (
        f"{timeframe.upper()} | "
        f"{bias.value} {strength.value} "
        f"| Regime={regime.value} "
        f"| EMA{fast_period}={last_ema_fast:.2f} "
        f"EMA{slow_period}={last_ema_slow:.2f} "
        f"| ADX={last_adx:.1f} "
        f"| +DI={last_plus_di:.1f} "
        f"-DI={last_minus_di:.1f} "
        f"| DIΔ={di_spread:.1f} "
        f"| Slope={fast_slope:.3f}%"
    )

    return TrendAnalysis(
        bias=bias,
        strength=strength,
        adx=last_adx,
        ema_fast=last_ema_fast,
        ema_slow=last_ema_slow,
        price_vs_ema=price_vs_ema,
        description=description,
        plus_di=last_plus_di,
        minus_di=last_minus_di,
        di_spread=di_spread,
        ema_fast_slope_percent=fast_slope,
        ema_slow_slope_percent=slow_slope,
        ema_aligned=ema_aligned,
        slope_confirmed=slope_confirmed,
        regime=regime,
        data_sufficient=True,
        closed_candle_confirmed=True,
        last_close=last_close,
    )


# ============================================================
# DIRECTION HELPERS
# ============================================================

def is_strongly_bullish(
    analysis: TrendAnalysis,
) -> bool:
    """
    Strong bullish trend.

    Requires:
    - bullish bias
    - strong ADX
    - positive directional movement
    """

    return (
        analysis.bias == TrendBias.BULLISH
        and analysis.strength == TrendStrength.STRONG
        and analysis.plus_di > analysis.minus_di
        and analysis.data_sufficient
    )


def is_strongly_bearish(
    analysis: TrendAnalysis,
) -> bool:
    """
    Strong bearish trend.

    Requires:
    - bearish bias
    - strong ADX
    - negative directional movement
    """

    return (
        analysis.bias == TrendBias.BEARISH
        and analysis.strength == TrendStrength.STRONG
        and analysis.minus_di > analysis.plus_di
        and analysis.data_sufficient
    )


def is_bullish(
    analysis: TrendAnalysis,
) -> bool:
    """Basic bullish classification."""

    return (
        analysis.bias == TrendBias.BULLISH
        and analysis.data_sufficient
    )


def is_bearish(
    analysis: TrendAnalysis,
) -> bool:
    """Basic bearish classification."""

    return (
        analysis.bias == TrendBias.BEARISH
        and analysis.data_sufficient
    )


def is_ranging(
    analysis: TrendAnalysis,
) -> bool:
    """Return True when market lacks usable directional trend."""

    return (
        analysis.regime
        in {
            TrendRegime.RANGING,
            TrendRegime.CONFLICTED,
        }
    )


# ============================================================
# REGIME HELPERS
# ============================================================

def trend_allows_long(
    analysis: TrendAnalysis,
    strong_only: bool = False,
) -> bool:
    """
    Check whether the trend context permits LONG setups.

    This is a context filter, NOT an entry trigger.
    """

    if not analysis.data_sufficient:
        return False

    if analysis.regime in {
        TrendRegime.TRENDING_BULLISH,
        TrendRegime.WEAK_BULLISH,
    }:
        if strong_only:
            return analysis.regime == TrendRegime.TRENDING_BULLISH

        return True

    return False


def trend_allows_short(
    analysis: TrendAnalysis,
    strong_only: bool = False,
) -> bool:
    """
    Check whether the trend context permits SHORT setups.

    This is a context filter, NOT an entry trigger.
    """

    if not analysis.data_sufficient:
        return False

    if analysis.regime in {
        TrendRegime.TRENDING_BEARISH,
        TrendRegime.WEAK_BEARISH,
    }:
        if strong_only:
            return analysis.regime == TrendRegime.TRENDING_BEARISH

        return True

    return False


# ============================================================
# CONFLICT HELPERS
# ============================================================

def has_directional_conflict(
    analysis: TrendAnalysis,
) -> bool:
    """
    Detect conflicting directional evidence.

    Example:
        EMA bullish
        but DI bearish
        and slope bearish

    Such a state should not be treated as a clean trend.
    """

    if not analysis.data_sufficient:
        return True

    bullish_evidence = 0
    bearish_evidence = 0

    if analysis.ema_fast > analysis.ema_slow:
        bullish_evidence += 1
    elif analysis.ema_fast < analysis.ema_slow:
        bearish_evidence += 1

    if analysis.plus_di > analysis.minus_di:
        bullish_evidence += 1
    elif analysis.minus_di > analysis.plus_di:
        bearish_evidence += 1

    if analysis.ema_fast_slope_percent > 0:
        bullish_evidence += 1
    elif analysis.ema_fast_slope_percent < 0:
        bearish_evidence += 1

    return (
        bullish_evidence > 0
        and bearish_evidence > 0
        and abs(
            bullish_evidence - bearish_evidence
        ) <= 1
    )


# ============================================================
# SAFE SUMMARY
# ============================================================

def trend_summary(
    analysis: TrendAnalysis,
) -> dict:
    """
    Convert TrendAnalysis into a serializable dictionary.

    Useful for:
        Telegram
        database snapshots
        logging
        debugging
    """

    return {
        "bias": analysis.bias.value,
        "strength": analysis.strength.value,
        "regime": analysis.regime.value,
        "adx": round(analysis.adx, 2),
        "plus_di": round(analysis.plus_di, 2),
        "minus_di": round(analysis.minus_di, 2),
        "di_spread": round(analysis.di_spread, 2),
        "ema_fast": round(analysis.ema_fast, 8),
        "ema_slow": round(analysis.ema_slow, 8),
        "ema_fast_slope_percent": round(
            analysis.ema_fast_slope_percent,
            4,
        ),
        "ema_slow_slope_percent": round(
            analysis.ema_slow_slope_percent,
            4,
        ),
        "ema_aligned": analysis.ema_aligned,
        "slope_confirmed": analysis.slope_confirmed,
        "price_vs_ema": analysis.price_vs_ema,
        "last_close": round(
            analysis.last_close,
            8,
        ),
        "data_sufficient": analysis.data_sufficient,
        "closed_candle_confirmed": (
            analysis.closed_candle_confirmed
        ),
        "description": analysis.description,
    }