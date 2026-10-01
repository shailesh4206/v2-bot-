"""
swings.py — Institutional swing / market-structure engine.

Responsibilities:
- Strict no-look-ahead swing detection
- Confirmed pivot detection
- ATR-normalized swing significance
- Equal High / Equal Low detection
- HH / HL / LH / LL structure labels
- Structure sequence validation
- Market structure bias

Design principles:
- A swing is confirmed only after right_bars candles have closed.
- No future candle is used to make an already-confirmed decision.
- Swing significance is normalized using ATR.
- Equal levels use both percentage and ATR tolerance.
- Structure labels are assigned only between swings of the same type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from indicators.volatility import atr as calc_atr
from config.strategy_config import (
    SWING_LEFT_BARS_4H,
    SWING_RIGHT_BARS_4H,
    SWING_LEFT_BARS_1H,
    SWING_RIGHT_BARS_1H,
    SWING_LEFT_BARS_15M,
    SWING_RIGHT_BARS_15M,
    MIN_SWING_ATR_MULTIPLIER,
    EQUAL_LEVEL_TOLERANCE_PCT,
    EQUAL_LEVEL_TOLERANCE_ATR,
    STRUCTURE_SEQUENCE_REQUIRED,
)


# ============================================================================
# ENUMS
# ============================================================================

class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class StructureLabel(str, Enum):
    HH = "HH"   # Higher High
    HL = "HL"   # Higher Low
    LH = "LH"   # Lower High
    LL = "LL"   # Lower Low
    EH = "EH"   # Equal High
    EL = "EL"   # Equal Low


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class SwingPoint:
    swing_type: SwingType
    price: float
    candle_idx: int
    timestamp: pd.Timestamp

    label: Optional[StructureLabel] = None
    confirmed: bool = True

    # Institutional metadata
    atr_value: float = 0.0
    significance_atr: float = 0.0
    confirmation_idx: int = -1
    confirmation_timestamp: Optional[pd.Timestamp] = None
    equal_to_previous: bool = False


# ============================================================================
# TIMEFRAME CONFIGURATION
# ============================================================================

def _get_pivot_params(
    timeframe: str,
) -> Tuple[int, int]:
    """
    Return left/right pivot confirmation bars.

    Only supported strategy timeframes are allowed.
    """

    timeframe = str(timeframe).lower().strip()

    params = {
        "4h": (
            SWING_LEFT_BARS_4H,
            SWING_RIGHT_BARS_4H,
        ),
        "1h": (
            SWING_LEFT_BARS_1H,
            SWING_RIGHT_BARS_1H,
        ),
        "15m": (
            SWING_LEFT_BARS_15M,
            SWING_RIGHT_BARS_15M,
        ),
    }

    if timeframe not in params:
        raise ValueError(
            f"Unsupported swing timeframe: {timeframe}. "
            f"Allowed: 4h, 1h, 15m"
        )

    return params[timeframe]


# ============================================================================
# DATA VALIDATION
# ============================================================================

def _validate_dataframe(
    df: pd.DataFrame,
) -> bool:
    """Validate minimum OHLC dataframe requirements."""

    if df is None or df.empty:
        return False

    required = {"high", "low", "close"}

    if not required.issubset(df.columns):
        return False

    try:
        values = (
            df[["high", "low", "close"]]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(values).all():
            return False

        if (values[:, 0] < values[:, 1]).any():
            return False

    except (TypeError, ValueError):
        return False

    return True


def _clean_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return clean numeric OHLC data."""

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
# EQUAL LEVEL DETECTION
# ============================================================================

def _price_tolerance(
    price: float,
    atr_value: float = 0.0,
    tolerance_pct: float = EQUAL_LEVEL_TOLERANCE_PCT,
    tolerance_atr: float = EQUAL_LEVEL_TOLERANCE_ATR,
) -> float:
    """
    Dynamic price tolerance.

    Uses the larger of:
    - percentage tolerance
    - ATR tolerance
    """

    if price <= 0:
        return 0.0

    pct_tolerance = (
        price
        * float(tolerance_pct)
        / 100.0
    )

    atr_tolerance = (
        max(float(atr_value), 0.0)
        * float(tolerance_atr)
    )

    return max(
        pct_tolerance,
        atr_tolerance,
    )


def are_equal_levels(
    price_a: float,
    price_b: float,
    atr_value: float = 0.0,
    tolerance_pct: float = EQUAL_LEVEL_TOLERANCE_PCT,
    tolerance_atr: float = EQUAL_LEVEL_TOLERANCE_ATR,
) -> bool:
    """Return True if two prices are within dynamic equal-level tolerance."""

    if price_a <= 0 or price_b <= 0:
        return False

    reference = (
        price_a + price_b
    ) / 2.0

    tolerance = _price_tolerance(
        reference,
        atr_value,
        tolerance_pct,
        tolerance_atr,
    )

    return abs(price_a - price_b) <= tolerance


# ============================================================================
# SWING SIGNIFICANCE
# ============================================================================

def _swing_significance(
    price: float,
    reference_price: float,
    atr_value: float,
) -> float:
    """
    Return distance from reference price in ATR units.
    """

    if atr_value <= 0:
        return 0.0

    return abs(
        price - reference_price
    ) / atr_value


def _passes_swing_size_filter(
    swing_price: float,
    reference_price: float,
    atr_value: float,
    min_atr_mult: float,
) -> bool:
    """
    Determine whether a swing is sufficiently significant.

    If ATR is unavailable, the pivot is not rejected solely because
    ATR could not be calculated.
    """

    if atr_value <= 0:
        return True

    distance = abs(
        swing_price - reference_price
    )

    return (
        distance
        >= float(min_atr_mult) * atr_value
    )


# ============================================================================
# SWING HIGH DETECTION
# ============================================================================

def detect_swing_highs(
    df: pd.DataFrame,
    timeframe: str = "15m",
    left_bars: Optional[int] = None,
    right_bars: Optional[int] = None,
    min_atr_mult: float = MIN_SWING_ATR_MULTIPLIER,
) -> List[SwingPoint]:
    """
    Detect confirmed swing highs.

    No-look-ahead guarantee:

    A pivot at index i is only returned when:
        i + right_bars < len(df)

    Therefore all confirmation candles are already available.

    The latest unconfirmed candles are never treated as confirmed swings.
    """

    cleaned = _clean_dataframe(df)

    if not _validate_dataframe(cleaned):
        return []

    default_left, default_right = _get_pivot_params(
        timeframe
    )

    left = (
        default_left
        if left_bars is None
        else max(int(left_bars), 1)
    )

    right = (
        default_right
        if right_bars is None
        else max(int(right_bars), 1)
    )

    if len(cleaned) < left + right + 1:
        return []

    atr_series = calc_atr(cleaned)

    highs = cleaned["high"].to_numpy(
        dtype=float
    )

    lows = cleaned["low"].to_numpy(
        dtype=float
    )

    swings: List[SwingPoint] = []

    for i in range(
        left,
        len(cleaned) - right,
    ):
        window = highs[
            i - left:
            i + right + 1
        ]

        current_high = highs[i]

        # Strict unique pivot.
        if not np.isclose(
            current_high,
            np.max(window),
            rtol=0.0,
            atol=0.0,
        ):
            continue

        if np.count_nonzero(
            window == current_high
        ) != 1:
            continue

        atr_value = 0.0

        if i < len(atr_series):
            raw_atr = atr_series.iloc[i]

            if (
                pd.notna(raw_atr)
                and np.isfinite(raw_atr)
            ):
                atr_value = float(raw_atr)

        # Reference = lowest low in the confirmed left structure.
        reference_low = float(
            np.min(
                lows[
                    i - left:
                    i
                ]
            )
        )

        if not _passes_swing_size_filter(
            current_high,
            reference_low,
            atr_value,
            min_atr_mult,
        ):
            continue

        significance = _swing_significance(
            current_high,
            reference_low,
            atr_value,
        )

        confirmation_idx = i + right

        swings.append(
            SwingPoint(
                swing_type=SwingType.HIGH,
                price=float(current_high),
                candle_idx=i,
                timestamp=cleaned.index[i],
                label=None,
                confirmed=True,
                atr_value=atr_value,
                significance_atr=significance,
                confirmation_idx=confirmation_idx,
                confirmation_timestamp=cleaned.index[
                    confirmation_idx
                ],
            )
        )

    return swings


# ============================================================================
# SWING LOW DETECTION
# ============================================================================

def detect_swing_lows(
    df: pd.DataFrame,
    timeframe: str = "15m",
    left_bars: Optional[int] = None,
    right_bars: Optional[int] = None,
    min_atr_mult: float = MIN_SWING_ATR_MULTIPLIER,
) -> List[SwingPoint]:
    """
    Detect confirmed swing lows.

    No-look-ahead guarantee is identical to swing highs.
    """

    cleaned = _clean_dataframe(df)

    if not _validate_dataframe(cleaned):
        return []

    default_left, default_right = _get_pivot_params(
        timeframe
    )

    left = (
        default_left
        if left_bars is None
        else max(int(left_bars), 1)
    )

    right = (
        default_right
        if right_bars is None
        else max(int(right_bars), 1)
    )

    if len(cleaned) < left + right + 1:
        return []

    atr_series = calc_atr(cleaned)

    lows = cleaned["low"].to_numpy(
        dtype=float
    )

    highs = cleaned["high"].to_numpy(
        dtype=float
    )

    swings: List[SwingPoint] = []

    for i in range(
        left,
        len(cleaned) - right,
    ):
        window = lows[
            i - left:
            i + right + 1
        ]

        current_low = lows[i]

        if not np.isclose(
            current_low,
            np.min(window),
            rtol=0.0,
            atol=0.0,
        ):
            continue

        if np.count_nonzero(
            window == current_low
        ) != 1:
            continue

        atr_value = 0.0

        if i < len(atr_series):
            raw_atr = atr_series.iloc[i]

            if (
                pd.notna(raw_atr)
                and np.isfinite(raw_atr)
            ):
                atr_value = float(raw_atr)

        reference_high = float(
            np.max(
                highs[
                    i - left:
                    i
                ]
            )
        )

        if not _passes_swing_size_filter(
            current_low,
            reference_high,
            atr_value,
            min_atr_mult,
        ):
            continue

        significance = _swing_significance(
            current_low,
            reference_high,
            atr_value,
        )

        confirmation_idx = i + right

        swings.append(
            SwingPoint(
                swing_type=SwingType.LOW,
                price=float(current_low),
                candle_idx=i,
                timestamp=cleaned.index[i],
                label=None,
                confirmed=True,
                atr_value=atr_value,
                significance_atr=significance,
                confirmation_idx=confirmation_idx,
                confirmation_timestamp=cleaned.index[
                    confirmation_idx
                ],
            )
        )

    return swings


# ============================================================================
# EQUAL HIGH / LOW ANNOTATION
# ============================================================================

def _annotate_equal_levels(
    swings: List[SwingPoint],
) -> List[SwingPoint]:
    """
    Mark equal consecutive highs/lows.

    Equal levels are metadata only; they are not automatically
    treated as liquidity pools here.
    """

    if len(swings) < 2:
        return swings

    for i in range(1, len(swings)):
        previous = swings[i - 1]
        current = swings[i]

        if previous.swing_type != current.swing_type:
            continue

        atr_reference = max(
            previous.atr_value,
            current.atr_value,
        )

        if are_equal_levels(
            previous.price,
            current.price,
            atr_reference,
        ):
            current.equal_to_previous = True

            if current.swing_type == SwingType.HIGH:
                current.label = StructureLabel.EH
            else:
                current.label = StructureLabel.EL

    return swings


# ============================================================================
# MARKET STRUCTURE LABELING
# ============================================================================

def _label_swings(
    swings: List[SwingPoint],
    tolerance_pct: float = EQUAL_LEVEL_TOLERANCE_PCT,
    tolerance_atr: float = EQUAL_LEVEL_TOLERANCE_ATR,
) -> List[SwingPoint]:
    """
    Label same-type swings.

    High sequence:
        HH / LH / EH

    Low sequence:
        HL / LL / EL
    """

    if not swings:
        return []

    labeled = list(swings)

    previous: Optional[SwingPoint] = None

    for current in labeled:
        if previous is None:
            previous = current
            continue

        # Defensive: only compare same swing type.
        if current.swing_type != previous.swing_type:
            continue

        atr_reference = max(
            current.atr_value,
            previous.atr_value,
        )

        tolerance = _price_tolerance(
            (
                current.price
                + previous.price
            ) / 2.0,
            atr_reference,
            tolerance_pct,
            tolerance_atr,
        )

        difference = (
            current.price
            - previous.price
        )

        if abs(difference) <= tolerance:
            if current.swing_type == SwingType.HIGH:
                current.label = StructureLabel.EH
            else:
                current.label = StructureLabel.EL

            current.equal_to_previous = True

        elif current.swing_type == SwingType.HIGH:
            if difference > 0:
                current.label = StructureLabel.HH
            else:
                current.label = StructureLabel.LH

        else:
            if difference > 0:
                current.label = StructureLabel.HL
            else:
                current.label = StructureLabel.LL

        previous = current

    return labeled


def label_market_structure(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    equal_tolerance_pct: float = EQUAL_LEVEL_TOLERANCE_PCT,
    equal_tolerance_atr: float = EQUAL_LEVEL_TOLERANCE_ATR,
) -> Tuple[
    List[SwingPoint],
    List[SwingPoint],
]:
    """
    Assign HH/HL/LH/LL/EH/EL labels.

    Returns:
        labeled_highs, labeled_lows
    """

    labeled_highs = _label_swings(
        swing_highs,
        equal_tolerance_pct,
        equal_tolerance_atr,
    )

    labeled_lows = _label_swings(
        swing_lows,
        equal_tolerance_pct,
        equal_tolerance_atr,
    )

    return (
        labeled_highs,
        labeled_lows,
    )


# ============================================================================
# STRUCTURE SEQUENCE
# ============================================================================

def _last_label(
    swings: List[SwingPoint],
) -> Optional[StructureLabel]:
    """Return the latest valid structure label."""

    for swing in reversed(swings):
        if swing.label is not None:
            return swing.label

    return None


def has_bullish_structure_sequence(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
) -> bool:
    """
    Confirm a basic bullish structure sequence.

    Preferred sequence:
        HH + HL

    This prevents one isolated HH/HL from being treated as a full
    bullish structure.
    """

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False

    high_labels = [
        swing.label
        for swing in swing_highs[-3:]
    ]

    low_labels = [
        swing.label
        for swing in swing_lows[-3:]
    ]

    has_hh = (
        StructureLabel.HH
        in high_labels
    )

    has_hl = (
        StructureLabel.HL
        in low_labels
    )

    return bool(has_hh and has_hl)


def has_bearish_structure_sequence(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
) -> bool:
    """
    Confirm a basic bearish structure sequence.

    Preferred sequence:
        LH + LL
    """

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False

    high_labels = [
        swing.label
        for swing in swing_highs[-3:]
    ]

    low_labels = [
        swing.label
        for swing in swing_lows[-3:]
    ]

    has_lh = (
        StructureLabel.LH
        in high_labels
    )

    has_ll = (
        StructureLabel.LL
        in low_labels
    )

    return bool(has_lh and has_ll)


# ============================================================================
# MARKET STRUCTURE BIAS
# ============================================================================

def get_market_structure_bias(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
) -> str:
    """
    Determine market structure bias.

    Returns:
        BULLISH
        BEARISH
        NEUTRAL

    If STRUCTURE_SEQUENCE_REQUIRED is enabled, an isolated HH or LL
    is not sufficient to establish directional structure.
    """

    if not swing_highs and not swing_lows:
        return "NEUTRAL"

    bullish_score = 0
    bearish_score = 0

    recent_highs = (
        swing_highs[-3:]
        if len(swing_highs) >= 3
        else swing_highs
    )

    recent_lows = (
        swing_lows[-3:]
        if len(swing_lows) >= 3
        else swing_lows
    )

    for swing in recent_highs:
        if swing.label == StructureLabel.HH:
            bullish_score += 1

        elif swing.label == StructureLabel.LH:
            bearish_score += 1

    for swing in recent_lows:
        if swing.label == StructureLabel.HL:
            bullish_score += 1

        elif swing.label == StructureLabel.LL:
            bearish_score += 1

    if STRUCTURE_SEQUENCE_REQUIRED:
        bullish_sequence = (
            has_bullish_structure_sequence(
                swing_highs,
                swing_lows,
            )
        )

        bearish_sequence = (
            has_bearish_structure_sequence(
                swing_highs,
                swing_lows,
            )
        )

        if bullish_sequence and not bearish_sequence:
            return "BULLISH"

        if bearish_sequence and not bullish_sequence:
            return "BEARISH"

        if bullish_sequence and bearish_sequence:
            # Conflicting structure remains neutral.
            return "NEUTRAL"

        return "NEUTRAL"

    if bullish_score > bearish_score:
        return "BULLISH"

    if bearish_score > bullish_score:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================================
# STRUCTURE HELPERS
# ============================================================================

def get_latest_swing_high(
    swing_highs: List[SwingPoint],
) -> Optional[SwingPoint]:
    """Return latest confirmed swing high."""

    return (
        swing_highs[-1]
        if swing_highs
        else None
    )


def get_latest_swing_low(
    swing_lows: List[SwingPoint],
) -> Optional[SwingPoint]:
    """Return latest confirmed swing low."""

    return (
        swing_lows[-1]
        if swing_lows
        else None
    )


def get_latest_structure_level(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
) -> Optional[SwingPoint]:
    """Return whichever confirmed swing occurred most recently."""

    candidates = []

    if swing_highs:
        candidates.append(
            swing_highs[-1]
        )

    if swing_lows:
        candidates.append(
            swing_lows[-1]
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda swing: swing.candle_idx,
    )


def structure_summary(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
) -> str:
    """Compact structure summary for logs/debugging."""

    bias = get_market_structure_bias(
        swing_highs,
        swing_lows,
    )

    high_label = (
        swing_highs[-1].label.value
        if swing_highs
        and swing_highs[-1].label
        else "-"
    )

    low_label = (
        swing_lows[-1].label.value
        if swing_lows
        and swing_lows[-1].label
        else "-"
    )

    return (
        f"Bias={bias} | "
        f"LastHigh={high_label} | "
        f"LastLow={low_label} | "
        f"Highs={len(swing_highs)} | "
        f"Lows={len(swing_lows)}"
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

    print("=== SWING ENGINE TEST ===")
    print(
        f"Swing Highs: {len(highs)}"
    )
    print(
        f"Swing Lows: {len(lows)}"
    )
    print(
        structure_summary(
            highs,
            lows,
        )
    )