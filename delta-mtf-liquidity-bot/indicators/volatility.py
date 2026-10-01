"""
volatility.py — Institutional volatility engine.

Responsibilities:
- ATR calculation using Wilder smoothing
- ATR percentage and volatility regime
- ATR expansion / contraction detection
- Abnormal volatility and spike detection
- Candle range/body abnormality detection
- Volatility risk gate support
- SL buffer / chase protection / swing validation support

Design principles:
- Closed-candle aware
- No future leakage
- Dynamic ATR-based thresholds
- Extreme volatility can block new signals
- Score must never override hard volatility gates
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config.strategy_config import (
    ATR_PERIOD,
    VOLATILITY_BASELINE_PERIOD,
    VOLATILITY_EXPANSION_THRESHOLD,
    VOLATILITY_WARNING_MULTIPLIER,
    VOLATILITY_EXTREME_MULTIPLIER,
    VOLATILITY_RANGE_ATR_HIGH,
    VOLATILITY_RANGE_ATR_EXTREME,
    VOLATILITY_BODY_ATR_EXTREME,
    ABNORMAL_VOLATILITY_ENABLED,
    ABNORMAL_VOLATILITY_BLOCK_EXTREME,
    SPIKE_RANGE_ATR,
    SPIKE_BODY_ATR,
)


# ---------------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------------

class VolatilityRegime(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


# ---------------------------------------------------------------------------
# DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class VolatilityAnalysis:
    """
    Complete volatility snapshot.

    Original fields are preserved for backward compatibility:
        atr
        atr_pct
        regime
        description
    """

    atr: float
    atr_pct: float
    regime: VolatilityRegime
    description: str

    # Institutional fields
    atr_baseline: float = 0.0
    atr_expansion_ratio: float = 1.0

    range_atr_ratio: float = 0.0
    body_atr_ratio: float = 0.0

    is_expanding: bool = False
    is_abnormal: bool = False
    is_spike: bool = False
    is_extreme: bool = False

    data_sufficient: bool = False
    closed_candle_confirmed: bool = True

    risk_allowed: bool = True
    block_new_signal: bool = False

    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {"high", "low", "close"}


def _validate_ohlc(df: pd.DataFrame) -> bool:
    """Validate minimum OHLC structure and finite numeric values."""

    if df is None or df.empty:
        return False

    if not _REQUIRED_COLUMNS.issubset(df.columns):
        return False

    try:
        values = df[["high", "low", "close"]].tail(1).astype(float)

        if not np.isfinite(values.to_numpy()).all():
            return False

        high = float(values["high"].iloc[-1])
        low = float(values["low"].iloc[-1])
        close = float(values["close"].iloc[-1])

        if high <= 0 or low <= 0 or close <= 0:
            return False

        if high < low:
            return False

    except (TypeError, ValueError, IndexError):
        return False

    return True


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric OHLC dataframe with invalid rows removed."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    for column in ("open", "high", "low", "close", "volume"):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result = result.replace([np.inf, -np.inf], np.nan)

    result = result.dropna(
        subset=["high", "low", "close"]
    )

    result = result[
        (result["high"] >= result["low"])
        & (result["high"] > 0)
        & (result["low"] > 0)
        & (result["close"] > 0)
    ]

    return result


# ---------------------------------------------------------------------------
# TRUE RANGE / ATR
# ---------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range.

    TR = max(
        high-low,
        abs(high-prev_close),
        abs(low-prev_close)
    )
    """

    if df is None or df.empty:
        return pd.Series(dtype=float)

    required = {"high", "low", "close"}

    if not required.issubset(df.columns):
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr


def atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """
    Calculate Wilder-style ATR.

    The returned series has the same index/length as the input.
    """

    if df is None or df.empty:
        return pd.Series(dtype=float)

    period = max(int(period), 1)

    if len(df) < period:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    tr = true_range(df)

    if tr.empty:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    # Wilder smoothing.
    atr_series = tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr_series


def get_current_atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> float:
    """Return the latest ATR value."""

    if df is None or df.empty:
        return 0.0

    atr_series = atr(df, period)

    if atr_series.empty:
        return 0.0

    value = atr_series.iloc[-1]

    if pd.isna(value) or not np.isfinite(value):
        return 0.0

    return float(value)


# ---------------------------------------------------------------------------
# ATR PERCENTAGE
# ---------------------------------------------------------------------------

def atr_percent(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> float:
    """Return ATR as percentage of latest close."""

    if df is None or df.empty:
        return 0.0

    current_atr = get_current_atr(df, period)

    try:
        close = float(df["close"].iloc[-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0

    if close <= 0 or not np.isfinite(close):
        return 0.0

    return float((current_atr / close) * 100.0)


# ---------------------------------------------------------------------------
# ATR BASELINE
# ---------------------------------------------------------------------------

def atr_baseline(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    baseline_period: int = VOLATILITY_BASELINE_PERIOD,
) -> float:
    """
    Calculate historical ATR baseline.

    Uses only completed ATR observations available in the dataframe.
    """

    if df is None or df.empty:
        return 0.0

    baseline_period = max(int(baseline_period), 5)

    atr_series = atr(df, period).dropna()

    if atr_series.empty:
        return 0.0

    recent = atr_series.tail(baseline_period)

    if recent.empty:
        return 0.0

    value = float(recent.median())

    if not np.isfinite(value) or value <= 0:
        return 0.0

    return value


def atr_expansion_ratio(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    baseline_period: int = VOLATILITY_BASELINE_PERIOD,
) -> float:
    """
    Current ATR / historical ATR baseline.

    > 1.0  = volatility expansion
    < 1.0  = volatility contraction
    """

    current = get_current_atr(df, period)
    baseline = atr_baseline(
        df,
        period,
        baseline_period,
    )

    if current <= 0 or baseline <= 0:
        return 1.0

    return float(current / baseline)


# ---------------------------------------------------------------------------
# CANDLE GEOMETRY
# ---------------------------------------------------------------------------

def candle_range(df: pd.DataFrame) -> float:
    """Latest candle high-low range."""

    if df is None or df.empty:
        return 0.0

    try:
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0

    if not np.isfinite(high) or not np.isfinite(low):
        return 0.0

    return max(high - low, 0.0)


def candle_body(df: pd.DataFrame) -> float:
    """Latest candle absolute body size."""

    if df is None or df.empty or "open" not in df.columns:
        return 0.0

    try:
        open_price = float(df["open"].iloc[-1])
        close_price = float(df["close"].iloc[-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0

    if not np.isfinite(open_price) or not np.isfinite(close_price):
        return 0.0

    return abs(close_price - open_price)


def range_atr_ratio(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> float:
    """Latest candle range divided by current ATR."""

    current_atr = get_current_atr(df, period)

    if current_atr <= 0:
        return 0.0

    return float(candle_range(df) / current_atr)


def body_atr_ratio(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> float:
    """Latest candle body divided by current ATR."""

    current_atr = get_current_atr(df, period)

    if current_atr <= 0:
        return 0.0

    return float(candle_body(df) / current_atr)


# ---------------------------------------------------------------------------
# SPIKE / ABNORMAL VOLATILITY
# ---------------------------------------------------------------------------

def detect_volatility_spike(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    range_threshold: float = SPIKE_RANGE_ATR,
    body_threshold: float = SPIKE_BODY_ATR,
) -> bool:
    """
    Detect an abnormal single-candle volatility spike.

    A spike requires either:
    - candle range >= configured ATR multiple
    - candle body >= configured ATR multiple
    """

    if not ABNORMAL_VOLATILITY_ENABLED:
        return False

    if df is None or df.empty:
        return False

    current_atr = get_current_atr(df, period)

    if current_atr <= 0:
        return False

    current_range = candle_range(df)
    current_body = candle_body(df)

    return (
        current_range >= current_atr * float(range_threshold)
        or current_body >= current_atr * float(body_threshold)
    )


def is_extreme_candle(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    range_threshold: float = VOLATILITY_RANGE_ATR_EXTREME,
    body_threshold: float = VOLATILITY_BODY_ATR_EXTREME,
) -> bool:
    """Detect an extreme candle relative to ATR."""

    if df is None or df.empty:
        return False

    current_atr = get_current_atr(df, period)

    if current_atr <= 0:
        return False

    return (
        candle_range(df) >= current_atr * float(range_threshold)
        or candle_body(df) >= current_atr * float(body_threshold)
    )


# ---------------------------------------------------------------------------
# REGIME CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_volatility(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> VolatilityAnalysis:
    """
    Classify current volatility.

    Regime logic:

    LOW:
        ATR expansion ratio below normal expansion zone.

    NORMAL:
        Normal ATR conditions.

    HIGH:
        ATR expansion / candle geometry indicates elevated volatility.

    EXTREME:
        ATR expansion or candle spike reaches configured extreme level.

    Important:
        Volatility regime is a risk state, not a trade direction.
    """

    cleaned = _clean_ohlc(df)

    minimum_bars = max(
        period + 10,
        VOLATILITY_BASELINE_PERIOD + period,
    )

    if len(cleaned) < minimum_bars:
        return VolatilityAnalysis(
            atr=0.0,
            atr_pct=0.0,
            regime=VolatilityRegime.NORMAL,
            description="Insufficient data for volatility classification",
            data_sufficient=False,
            risk_allowed=False,
            block_new_signal=True,
            warning="INSUFFICIENT_VOLATILITY_DATA",
        )

    current_atr = get_current_atr(
        cleaned,
        period,
    )

    if current_atr <= 0:
        return VolatilityAnalysis(
            atr=0.0,
            atr_pct=0.0,
            regime=VolatilityRegime.NORMAL,
            description="Invalid ATR",
            data_sufficient=False,
            risk_allowed=False,
            block_new_signal=True,
            warning="INVALID_ATR",
        )

    try:
        close = float(cleaned["close"].iloc[-1])
    except (IndexError, TypeError, ValueError):
        close = 0.0

    if close <= 0:
        return VolatilityAnalysis(
            atr=0.0,
            atr_pct=0.0,
            regime=VolatilityRegime.NORMAL,
            description="Invalid close price",
            data_sufficient=False,
            risk_allowed=False,
            block_new_signal=True,
            warning="INVALID_CLOSE",
        )

    current_atr_pct = (
        current_atr / close
    ) * 100.0

    baseline = atr_baseline(
        cleaned,
        period,
        VOLATILITY_BASELINE_PERIOD,
    )

    expansion_ratio = (
        current_atr / baseline
        if baseline > 0
        else 1.0
    )

    range_ratio = range_atr_ratio(
        cleaned,
        period,
    )

    body_ratio = body_atr_ratio(
        cleaned,
        period,
    )

    expanding = (
        expansion_ratio
        >= float(VOLATILITY_EXPANSION_THRESHOLD)
    )

    spike = detect_volatility_spike(
        cleaned,
        period,
    )

    extreme_candle = is_extreme_candle(
        cleaned,
        period,
    )

    extreme_atr = (
        expansion_ratio
        >= float(VOLATILITY_EXTREME_MULTIPLIER)
    )

    warning_atr = (
        expansion_ratio
        >= float(VOLATILITY_WARNING_MULTIPLIER)
    )

    is_extreme = (
        extreme_atr
        or extreme_candle
        or spike
    )

    if is_extreme:
        regime = VolatilityRegime.EXTREME

    elif warning_atr or expanding:
        regime = VolatilityRegime.HIGH

    elif expansion_ratio < 0.75:
        regime = VolatilityRegime.LOW

    else:
        regime = VolatilityRegime.NORMAL

    abnormal = (
        spike
        or extreme_candle
        or expansion_ratio
        >= float(VOLATILITY_WARNING_MULTIPLIER)
    )

    block_new_signal = (
        ABNORMAL_VOLATILITY_ENABLED
        and ABNORMAL_VOLATILITY_BLOCK_EXTREME
        and is_extreme
    )

    risk_allowed = not block_new_signal

    warning_parts = []

    if expanding:
        warning_parts.append("ATR_EXPANSION")

    if spike:
        warning_parts.append("VOLATILITY_SPIKE")

    if extreme_candle:
        warning_parts.append("EXTREME_CANDLE")

    if block_new_signal:
        warning_parts.append("NEW_SIGNAL_BLOCKED")

    warning = (
        "|".join(warning_parts)
        if warning_parts
        else None
    )

    description = (
        f"ATR={current_atr:.6f} "
        f"({current_atr_pct:.2f}%) | "
        f"Regime={regime.value} | "
        f"Baseline={baseline:.6f} | "
        f"Expansion={expansion_ratio:.2f}x | "
        f"Range/ATR={range_ratio:.2f} | "
        f"Body/ATR={body_ratio:.2f}"
    )

    if warning:
        description += f" | Warning={warning}"

    return VolatilityAnalysis(
        atr=current_atr,
        atr_pct=current_atr_pct,
        regime=regime,
        description=description,
        atr_baseline=baseline,
        atr_expansion_ratio=expansion_ratio,
        range_atr_ratio=range_ratio,
        body_atr_ratio=body_ratio,
        is_expanding=expanding,
        is_abnormal=abnormal,
        is_spike=spike,
        is_extreme=is_extreme,
        data_sufficient=True,
        closed_candle_confirmed=True,
        risk_allowed=risk_allowed,
        block_new_signal=block_new_signal,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# HARD VOLATILITY GATES
# ---------------------------------------------------------------------------

def volatility_allows_signal(
    analysis: VolatilityAnalysis,
) -> bool:
    """
    Final hard volatility gate.

    A quality score must never override this.
    """

    if analysis is None:
        return False

    if not analysis.data_sufficient:
        return False

    if analysis.block_new_signal:
        return False

    if not analysis.risk_allowed:
        return False

    return True


def is_abnormal_volatility(
    analysis: VolatilityAnalysis,
) -> bool:
    """Return True when volatility is abnormal."""

    return bool(
        analysis is not None
        and analysis.is_abnormal
    )


def is_extreme_volatility(
    analysis: VolatilityAnalysis,
) -> bool:
    """Return True when volatility is extreme."""

    return bool(
        analysis is not None
        and analysis.is_extreme
    )


# ---------------------------------------------------------------------------
# COMPATIBILITY HELPERS
# ---------------------------------------------------------------------------

def volatility_summary(
    analysis: VolatilityAnalysis,
) -> str:
    """Compact summary for logs / Telegram / debugging."""

    if analysis is None:
        return "Volatility: unavailable"

    return (
        f"{analysis.regime.value} | "
        f"ATR {analysis.atr:.6f} "
        f"({analysis.atr_pct:.2f}%) | "
        f"Expansion {analysis.atr_expansion_ratio:.2f}x | "
        f"Range/ATR {analysis.range_atr_ratio:.2f} | "
        f"Body/ATR {analysis.body_atr_ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# MODULE SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Lightweight sanity test.
    np.random.seed(42)

    size = 300

    base = 100.0 + np.cumsum(
        np.random.normal(0, 0.5, size)
    )

    high = base + np.random.uniform(0.1, 0.8, size)
    low = base - np.random.uniform(0.1, 0.8, size)

    close = base
    open_price = (
        close
        + np.random.normal(0, 0.2, size)
    )

    volume = np.random.uniform(
        100,
        1000,
        size,
    )

    test_df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    result = classify_volatility(test_df)

    print("=== VOLATILITY ENGINE TEST ===")
    print(result.description)
    print(
        f"Regime: {result.regime.value}"
    )
    print(
        f"Signal Allowed: "
        f"{volatility_allows_signal(result)}"
    )