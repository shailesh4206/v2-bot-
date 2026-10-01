"""
market_structure/choch.py
Institutional Change of Character (CHoCH) Detection Engine.

Bullish CHoCH:
    Bearish structure
        ↓
    Lower High (LH)
        ↓
    Closed candle breaks above LH
        ↓
    Bullish CHoCH

Bearish CHoCH:
    Bullish structure
        ↓
    Higher Low (HL)
        ↓
    Closed candle breaks below HL
        ↓
    Bearish CHoCH

Institutional rules:
- Closed-candle confirmation only.
- CHoCH must break a meaningful structure level.
- Established opposing structure is required.
- Optional chronology gate ensures CHoCH happens AFTER liquidity sweep.
- No future candle beyond supplied dataframe is used.
- BOS remains a separate confirmation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from market_structure.swings import (
    SwingPoint,
    StructureLabel,
)

from indicators.volatility import atr as calc_atr

from config.strategy_config import (
    MIN_SWING_ATR_MULTIPLIER,
)


# ============================================================================
# ENUM
# ============================================================================

class CHoCHDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class CHoCHEvent:
    direction: CHoCHDirection

    # Broken structure
    broken_level: float
    broken_swing: SwingPoint

    # Confirmation candle
    break_candle_idx: int
    break_timestamp: pd.Timestamp
    break_close: float

    # Structure context
    prior_structure: str

    # Institutional diagnostics
    break_distance: float = 0.0
    break_distance_atr: float = 0.0
    close_confirmed: bool = True


# ============================================================================
# DATA VALIDATION
# ============================================================================

def _valid_dataframe(df: pd.DataFrame) -> bool:
    """Validate required OHLC columns."""

    if df is None or df.empty:
        return False

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    return required.issubset(df.columns)


# ============================================================================
# ATR
# ============================================================================

def _atr_series(df: pd.DataFrame) -> pd.Series:
    """Safely calculate ATR."""

    try:
        values = calc_atr(df)

        if isinstance(values, pd.Series):
            return pd.to_numeric(
                values,
                errors="coerce",
            )

        return pd.Series(
            values,
            index=df.index,
            dtype="float64",
        )

    except Exception:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )


# ============================================================================
# STRUCTURE VALIDATION
# ============================================================================

def _structure_before(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    swing_idx: int,
    direction: CHoCHDirection,
) -> tuple[bool, str]:
    """
    Verify that meaningful opposing structure existed
    before the CHoCH level.

    Bullish:
        LH required
        LL strengthens confirmation

    Bearish:
        HL required
        HH strengthens confirmation
    """

    if direction == CHoCHDirection.BULLISH:

        previous_highs = [
            sh
            for sh in swing_highs
            if sh.candle_idx < swing_idx
        ]

        previous_lows = [
            sl
            for sl in swing_lows
            if sl.candle_idx < swing_idx
        ]

        has_lh = any(
            sh.label == StructureLabel.LH
            for sh in previous_highs[-4:]
        )

        has_ll = any(
            sl.label == StructureLabel.LL
            for sl in previous_lows[-4:]
        )

        if has_lh and has_ll:
            return True, "LH + LL sequence"

        if has_lh:
            return True, "LH structure"

        return False, "No established bearish structure"

    # ------------------------------------------------------------------------
    # BEARISH CHoCH
    # ------------------------------------------------------------------------

    previous_highs = [
        sh
        for sh in swing_highs
        if sh.candle_idx < swing_idx
    ]

    previous_lows = [
        sl
        for sl in swing_lows
        if sl.candle_idx < swing_idx
    ]

    has_hh = any(
        sh.label == StructureLabel.HH
        for sh in previous_highs[-4:]
    )

    has_hl = any(
        sl.label == StructureLabel.HL
        for sl in previous_lows[-4:]
    )

    if has_hh and has_hl:
        return True, "HH + HL sequence"

    if has_hl:
        return True, "HL structure"

    return False, "No established bullish structure"


# ============================================================================
# MEANINGFUL BREAK
# ============================================================================

def _is_meaningful_break(
    break_distance: float,
    atr_value: float,
) -> bool:
    """
    Prevent tiny structure breaks caused by market noise.
    """

    if (
        not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        return False

    distance_atr = (
        abs(break_distance)
        / atr_value
    )

    return (
        distance_atr
        >= MIN_SWING_ATR_MULTIPLIER
    )


# ============================================================================
# CHoCH DETECTOR
# ============================================================================

def detect_choch(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    lookback: int = 5,
    after_candle_idx: Optional[int] = None,
) -> List[CHoCHEvent]:
    """
    Detect confirmed CHoCH events.

    Parameters
    ----------
    df:
        OHLC dataframe containing CLOSED candles only.

    swing_highs:
        Confirmed swing highs.

    swing_lows:
        Confirmed swing lows.

    lookback:
        Number of recent swing points to inspect.

    after_candle_idx:
        Optional chronology gate.

        If supplied:

            CHoCH break candle > after_candle_idx

        This is critical for:

            Liquidity Sweep
                    ↓
                  CHoCH
                    ↓
                   BOS
                    ↓
                 Retest
    """

    events: List[CHoCHEvent] = []

    if not _valid_dataframe(df):
        return events

    if not swing_highs and not swing_lows:
        return events

    if len(df) < 20:
        return events

    closes = pd.to_numeric(
        df["close"],
        errors="coerce",
    ).to_numpy()

    if np.isnan(closes).all():
        return events

    atr_values = _atr_series(df)

    lookback = max(
        1,
        int(lookback),
    )

    # ------------------------------------------------------------------------
    # Chronology
    # ------------------------------------------------------------------------

    minimum_break_idx = 0

    if after_candle_idx is not None:
        minimum_break_idx = (
            int(after_candle_idx) + 1
        )

    # ========================================================================
    # BULLISH CHoCH
    # ========================================================================

    recent_highs = (
        swing_highs[-lookback:]
        if len(swing_highs) > lookback
        else swing_highs
    )

    lh_swings = [
        sh
        for sh in recent_highs
        if sh.label == StructureLabel.LH
    ]

    lh_swings = sorted(
        lh_swings,
        key=lambda x: x.candle_idx,
        reverse=True,
    )

    used_breaks = set()

    for lh in lh_swings:

        level = float(lh.price)

        if (
            not np.isfinite(level)
            or level <= 0
        ):
            continue

        structure_ok, prior_structure = (
            _structure_before(
                swing_highs,
                swing_lows,
                lh.candle_idx,
                CHoCHDirection.BULLISH,
            )
        )

        if not structure_ok:
            continue

        start_idx = max(
            lh.candle_idx + 1,
            minimum_break_idx,
        )

        if start_idx >= len(df):
            continue

        for i in range(
            start_idx,
            len(df),
        ):

            close = closes[i]

            if not np.isfinite(close):
                continue

            # Closed candle must close above LH.
            if close <= level:
                continue

            atr_value = (
                float(atr_values.iloc[i])
                if pd.notna(
                    atr_values.iloc[i]
                )
                else 0.0
            )

            break_distance = (
                close - level
            )

            if not _is_meaningful_break(
                break_distance,
                atr_value,
            ):
                continue

            break_distance_atr = (
                break_distance / atr_value
                if atr_value > 0
                else 0.0
            )

            key = (
                round(level, 8),
                i,
                "BULLISH",
            )

            if key in used_breaks:
                continue

            used_breaks.add(key)

            events.append(
                CHoCHEvent(
                    direction=CHoCHDirection.BULLISH,
                    broken_level=level,
                    broken_swing=lh,
                    break_candle_idx=i,
                    break_timestamp=df.index[i],
                    break_close=float(close),
                    prior_structure=prior_structure,
                    break_distance=float(
                        break_distance
                    ),
                    break_distance_atr=float(
                        break_distance_atr
                    ),
                    close_confirmed=True,
                )
            )

            break

    # ========================================================================
    # BEARISH CHoCH
    # ========================================================================

    recent_lows = (
        swing_lows[-lookback:]
        if len(swing_lows) > lookback
        else swing_lows
    )

    hl_swings = [
        sl
        for sl in recent_lows
        if sl.label == StructureLabel.HL
    ]

    hl_swings = sorted(
        hl_swings,
        key=lambda x: x.candle_idx,
        reverse=True,
    )

    used_breaks = set()

    for hl in hl_swings:

        level = float(hl.price)

        if (
            not np.isfinite(level)
            or level <= 0
        ):
            continue

        structure_ok, prior_structure = (
            _structure_before(
                swing_highs,
                swing_lows,
                hl.candle_idx,
                CHoCHDirection.BEARISH,
            )
        )

        if not structure_ok:
            continue

        start_idx = max(
            hl.candle_idx + 1,
            minimum_break_idx,
        )

        if start_idx >= len(df):
            continue

        for i in range(
            start_idx,
            len(df),
        ):

            close = closes[i]

            if not np.isfinite(close):
                continue

            # Closed candle must close below HL.
            if close >= level:
                continue

            atr_value = (
                float(atr_values.iloc[i])
                if pd.notna(
                    atr_values.iloc[i]
                )
                else 0.0
            )

            break_distance = (
                level - close
            )

            if not _is_meaningful_break(
                break_distance,
                atr_value,
            ):
                continue

            break_distance_atr = (
                break_distance / atr_value
                if atr_value > 0
                else 0.0
            )

            key = (
                round(level, 8),
                i,
                "BEARISH",
            )

            if key in used_breaks:
                continue

            used_breaks.add(key)

            events.append(
                CHoCHEvent(
                    direction=CHoCHDirection.BEARISH,
                    broken_level=level,
                    broken_swing=hl,
                    break_candle_idx=i,
                    break_timestamp=df.index[i],
                    break_close=float(close),
                    prior_structure=prior_structure,
                    break_distance=float(
                        break_distance
                    ),
                    break_distance_atr=float(
                        break_distance_atr
                    ),
                    close_confirmed=True,
                )
            )

            break

    # ------------------------------------------------------------------------
    # Chronological order
    # ------------------------------------------------------------------------

    events.sort(
        key=lambda event: (
            event.break_candle_idx,
            event.break_timestamp,
        )
    )

    return events


# ============================================================================
# LATEST CHoCH
# ============================================================================

def get_latest_choch(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    direction: Optional[CHoCHDirection] = None,
    lookback: int = 5,
    after_candle_idx: Optional[int] = None,
) -> Optional[CHoCHEvent]:
    """
    Return the latest confirmed CHoCH.

    after_candle_idx can be used to enforce:

        CHoCH AFTER SWEEP
    """

    events = detect_choch(
        df=df,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        lookback=lookback,
        after_candle_idx=after_candle_idx,
    )

    if direction is not None:
        events = [
            event
            for event in events
            if event.direction == direction
        ]

    if not events:
        return None

    return events[-1]


# ============================================================================
# RECENT CHoCH
# ============================================================================

def has_recent_choch(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    direction: CHoCHDirection,
    lookback_candles: int = 15,
    after_candle_idx: Optional[int] = None,
) -> bool:
    """
    Check whether a valid CHoCH occurred recently.

    If after_candle_idx is supplied, the CHoCH must occur
    strictly AFTER that candle.
    """

    events = detect_choch(
        df=df,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        lookback=5,
        after_candle_idx=after_candle_idx,
    )

    events = [
        event
        for event in events
        if event.direction == direction
    ]

    if not events:
        return False

    latest = events[-1]

    candles_since = (
        len(df) - 1
        - latest.break_candle_idx
    )

    if candles_since < 0:
        return False

    return (
        candles_since
        <= int(lookback_candles)
    )


# ============================================================================
# SELF TEST
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CHoCH ENGINE SELF-TEST")
    print("=" * 70)

    print(
        "Bullish direction:",
        CHoCHDirection.BULLISH.value,
    )

    print(
        "Bearish direction:",
        CHoCHDirection.BEARISH.value,
    )

    print(
        "Minimum swing ATR:",
        MIN_SWING_ATR_MULTIPLIER,
    )

    print(
        "Closed-candle confirmation: True"
    )

    print(
        "Sweep chronology support: True"
    )

    print(
        "ATR structure validation: True"
    )

    print(
        "✓ Institutional CHoCH engine loaded successfully."
    )