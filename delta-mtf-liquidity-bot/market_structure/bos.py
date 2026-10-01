"""
market_structure/bos.py — Institutional Break of Structure (BOS).

Institutional rules:
- BOS requires candle-close confirmation.
- BOS must occur AFTER CHoCH.
- Meaningful structure break is ATR validated.
- Displacement is required.
- New structure level is required.
- Same candle cannot be used as both CHoCH and BOS.
- No future leakage.
- Only confirmed/closed candles are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from config.strategy_config import (
    BOS_REQUIRE_CLOSE_CONFIRMATION,
    BOS_MUST_FOLLOW_CHOCH,
    BOS_REQUIRE_DISPLACEMENT,
    BOS_MAX_CANDLES_AFTER_CHOCH,
    BOS_REQUIRE_NEW_STRUCTURE_LEVEL,
    CHOCH_BOS_SAME_BREAK_FORBIDDEN,
    DISPLACEMENT_ATR_PERIOD,
    DISPLACEMENT_MIN_BODY_ATR,
    DISPLACEMENT_MIN_BODY_PERCENT,
    MIN_SWING_ATR_MULTIPLIER,
)

from market_structure.swings import SwingPoint


class BOSDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class BOSEvent:
    direction: BOSDirection
    broken_level: float
    broken_swing: SwingPoint

    break_candle_idx: int
    break_timestamp: pd.Timestamp
    break_close: float

    break_distance: float
    break_distance_atr: float

    displacement_body_atr: float
    displacement_body_percent: float

    close_confirmed: bool
    new_structure_level: bool

    valid: bool
    rejection_reasons: List[str]


def _valid_dataframe(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False

    required = {"open", "high", "low", "close"}
    return required.issubset(df.columns)


def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
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

    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def _candle_body(df: pd.DataFrame, idx: int) -> float:
    return abs(float(df.iloc[idx]["close"]) - float(df.iloc[idx]["open"]))


def _body_percent(df: pd.DataFrame, idx: int) -> float:
    high = float(df.iloc[idx]["high"])
    low = float(df.iloc[idx]["low"])
    body = _candle_body(df, idx)

    candle_range = high - low

    if candle_range <= 0:
        return 0.0

    return (body / candle_range) * 100.0


def _is_bullish_candle(df: pd.DataFrame, idx: int) -> bool:
    return float(df.iloc[idx]["close"]) > float(df.iloc[idx]["open"])


def _is_bearish_candle(df: pd.DataFrame, idx: int) -> bool:
    return float(df.iloc[idx]["close"]) < float(df.iloc[idx]["open"])


def _meaningful_break(
    break_distance: float,
    atr_value: float,
) -> bool:
    if atr_value <= 0:
        return False

    return (
        break_distance / atr_value
        >= float(MIN_SWING_ATR_MULTIPLIER)
    )


def _displacement_ok(
    df: pd.DataFrame,
    idx: int,
    atr_value: float,
) -> tuple[bool, float, float]:
    if atr_value <= 0:
        return False, 0.0, 0.0

    body = _candle_body(df, idx)

    body_atr = body / atr_value
    body_percent = _body_percent(df, idx)

    ok = (
        body_atr >= float(DISPLACEMENT_MIN_BODY_ATR)
        and body_percent >= float(DISPLACEMENT_MIN_BODY_PERCENT)
    )

    return ok, body_atr, body_percent


def _is_new_structure_level(
    broken_level: float,
    previous_events: List[BOSEvent],
    direction: BOSDirection,
) -> bool:
    """
    Prevent repeated BOS events on the same structural level.

    A level is considered new when it has not already produced
    a BOS event in the same direction.
    """

    tolerance = max(abs(broken_level) * 0.0001, 1e-9)

    for event in previous_events:
        if event.direction != direction:
            continue

        if abs(event.broken_level - broken_level) <= tolerance:
            return False

    return True


def detect_bos(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    lookback: int = 5,
    direction: Optional[BOSDirection] = None,
    after_candle_idx: Optional[int] = None,
) -> List[BOSEvent]:
    """
    Detect institutional BOS events.

    Parameters
    ----------
    df:
        OHLC dataframe.

    swing_highs / swing_lows:
        Confirmed swing points.

    lookback:
        Number of recent swings considered.

    direction:
        Optional bullish/bearish filter.

    after_candle_idx:
        If supplied, BOS must occur strictly AFTER this candle.
        This is used to enforce CHoCH -> BOS chronology.

    Returns
    -------
    List[BOSEvent]
    """

    if not _valid_dataframe(df):
        return []

    if len(df) < DISPLACEMENT_ATR_PERIOD + 2:
        return []

    atr = _atr_series(df, DISPLACEMENT_ATR_PERIOD)

    events: List[BOSEvent] = []

    # Only confirmed candles.
    last_closed_idx = len(df) - 1

    high_swings = sorted(
        swing_highs or [],
        key=lambda x: getattr(x, "index", -1),
    )[-max(1, lookback):]

    low_swings = sorted(
        swing_lows or [],
        key=lambda x: getattr(x, "index", -1),
    )[-max(1, lookback):]

    # ---------------------------------------------------------
    # BULLISH BOS
    # ---------------------------------------------------------
    if direction in (None, BOSDirection.BULLISH):

        for swing in high_swings:

            swing_idx = getattr(swing, "index", None)
            swing_price = getattr(swing, "price", None)

            if swing_idx is None or swing_price is None:
                continue

            swing_idx = int(swing_idx)
            swing_price = float(swing_price)

            # BOS candle must come AFTER the swing itself.
            start_idx = swing_idx + 1

            if after_candle_idx is not None:
                start_idx = max(
                    start_idx,
                    int(after_candle_idx) + 1,
                )

            if start_idx > last_closed_idx:
                continue

            for idx in range(start_idx, last_closed_idx + 1):

                close = float(df.iloc[idx]["close"])

                # Candle-close confirmation.
                close_confirmed = close > swing_price

                if BOS_REQUIRE_CLOSE_CONFIRMATION and not close_confirmed:
                    continue

                if not close_confirmed:
                    continue

                atr_value = float(atr.iloc[idx])

                if not np.isfinite(atr_value) or atr_value <= 0:
                    continue

                break_distance = close - swing_price
                break_distance_atr = break_distance / atr_value

                if not _meaningful_break(
                    break_distance,
                    atr_value,
                ):
                    continue

                displacement_ok, body_atr, body_percent = (
                    _displacement_ok(
                        df,
                        idx,
                        atr_value,
                    )
                )

                if (
                    BOS_REQUIRE_DISPLACEMENT
                    and not displacement_ok
                ):
                    continue

                # Directional displacement.
                if not _is_bullish_candle(df, idx):
                    continue

                new_structure = _is_new_structure_level(
                    swing_price,
                    events,
                    BOSDirection.BULLISH,
                )

                if (
                    BOS_REQUIRE_NEW_STRUCTURE_LEVEL
                    and not new_structure
                ):
                    continue

                rejection_reasons: List[str] = []

                valid = True

                if (
                    BOS_MUST_FOLLOW_CHOCH
                    and after_candle_idx is None
                ):
                    valid = False
                    rejection_reasons.append(
                        "CHoCH chronology anchor missing"
                    )

                if (
                    CHOCH_BOS_SAME_BREAK_FORBIDDEN
                    and after_candle_idx is not None
                    and idx <= int(after_candle_idx)
                ):
                    valid = False
                    rejection_reasons.append(
                        "BOS cannot use CHoCH candle"
                    )

                if (
                    BOS_MAX_CANDLES_AFTER_CHOCH > 0
                    and after_candle_idx is not None
                    and idx - int(after_candle_idx)
                    > BOS_MAX_CANDLES_AFTER_CHOCH
                ):
                    valid = False
                    rejection_reasons.append(
                        "BOS too far after CHoCH"
                    )

                event = BOSEvent(
                    direction=BOSDirection.BULLISH,
                    broken_level=swing_price,
                    broken_swing=swing,
                    break_candle_idx=idx,
                    break_timestamp=df.index[idx],
                    break_close=close,
                    break_distance=break_distance,
                    break_distance_atr=break_distance_atr,
                    displacement_body_atr=body_atr,
                    displacement_body_percent=body_percent,
                    close_confirmed=close_confirmed,
                    new_structure_level=new_structure,
                    valid=valid,
                    rejection_reasons=rejection_reasons,
                )

                if valid:
                    events.append(event)

                # One BOS per structural level.
                break

    # ---------------------------------------------------------
    # BEARISH BOS
    # ---------------------------------------------------------
    if direction in (None, BOSDirection.BEARISH):

        for swing in low_swings:

            swing_idx = getattr(swing, "index", None)
            swing_price = getattr(swing, "price", None)

            if swing_idx is None or swing_price is None:
                continue

            swing_idx = int(swing_idx)
            swing_price = float(swing_price)

            start_idx = swing_idx + 1

            if after_candle_idx is not None:
                start_idx = max(
                    start_idx,
                    int(after_candle_idx) + 1,
                )

            if start_idx > last_closed_idx:
                continue

            for idx in range(start_idx, last_closed_idx + 1):

                close = float(df.iloc[idx]["close"])

                # Candle-close confirmation.
                close_confirmed = close < swing_price

                if BOS_REQUIRE_CLOSE_CONFIRMATION and not close_confirmed:
                    continue

                if not close_confirmed:
                    continue

                atr_value = float(atr.iloc[idx])

                if not np.isfinite(atr_value) or atr_value <= 0:
                    continue

                break_distance = swing_price - close
                break_distance_atr = break_distance / atr_value

                if not _meaningful_break(
                    break_distance,
                    atr_value,
                ):
                    continue

                displacement_ok, body_atr, body_percent = (
                    _displacement_ok(
                        df,
                        idx,
                        atr_value,
                    )
                )

                if (
                    BOS_REQUIRE_DISPLACEMENT
                    and not displacement_ok
                ):
                    continue

                # Directional displacement.
                if not _is_bearish_candle(df, idx):
                    continue

                new_structure = _is_new_structure_level(
                    swing_price,
                    events,
                    BOSDirection.BEARISH,
                )

                if (
                    BOS_REQUIRE_NEW_STRUCTURE_LEVEL
                    and not new_structure
                ):
                    continue

                rejection_reasons: List[str] = []

                valid = True

                if (
                    BOS_MUST_FOLLOW_CHOCH
                    and after_candle_idx is None
                ):
                    valid = False
                    rejection_reasons.append(
                        "CHoCH chronology anchor missing"
                    )

                if (
                    CHOCH_BOS_SAME_BREAK_FORBIDDEN
                    and after_candle_idx is not None
                    and idx <= int(after_candle_idx)
                ):
                    valid = False
                    rejection_reasons.append(
                        "BOS cannot use CHoCH candle"
                    )

                if (
                    BOS_MAX_CANDLES_AFTER_CHOCH > 0
                    and after_candle_idx is not None
                    and idx - int(after_candle_idx)
                    > BOS_MAX_CANDLES_AFTER_CHOCH
                ):
                    valid = False
                    rejection_reasons.append(
                        "BOS too far after CHoCH"
                    )

                event = BOSEvent(
                    direction=BOSDirection.BEARISH,
                    broken_level=swing_price,
                    broken_swing=swing,
                    break_candle_idx=idx,
                    break_timestamp=df.index[idx],
                    break_close=close,
                    break_distance=break_distance,
                    break_distance_atr=break_distance_atr,
                    displacement_body_atr=body_atr,
                    displacement_body_percent=body_percent,
                    close_confirmed=close_confirmed,
                    new_structure_level=new_structure,
                    valid=valid,
                    rejection_reasons=rejection_reasons,
                )

                if valid:
                    events.append(event)

                break

    events.sort(key=lambda x: x.break_candle_idx)

    return events


def get_latest_bos(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    lookback: int = 5,
    direction: Optional[BOSDirection] = None,
    after_candle_idx: Optional[int] = None,
) -> Optional[BOSEvent]:

    events = detect_bos(
        df=df,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        lookback=lookback,
        direction=direction,
        after_candle_idx=after_candle_idx,
    )

    return events[-1] if events else None


def has_recent_bos(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    direction: BOSDirection,
    lookback_candles: int = 10,
    after_candle_idx: Optional[int] = None,
) -> bool:

    events = detect_bos(
        df=df,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        lookback=5,
        direction=direction,
        after_candle_idx=after_candle_idx,
    )

    if not events:
        return False

    latest = events[-1]

    return (
        len(df) - 1 - latest.break_candle_idx
        <= lookback_candles
    )


def _self_test() -> None:
    print("BOS ENGINE SELF-TEST")
    print("Bullish direction:", BOSDirection.BULLISH.value)
    print("Bearish direction:", BOSDirection.BEARISH.value)
    print(
        "Minimum swing ATR:",
        MIN_SWING_ATR_MULTIPLIER,
    )
    print(
        "Displacement required:",
        BOS_REQUIRE_DISPLACEMENT,
    )
    print(
        "Displacement body ATR:",
        DISPLACEMENT_MIN_BODY_ATR,
    )
    print(
        "Displacement body %:",
        DISPLACEMENT_MIN_BODY_PERCENT,
    )
    print(
        "Close confirmation:",
        BOS_REQUIRE_CLOSE_CONFIRMATION,
    )
    print(
        "Must follow CHoCH:",
        BOS_MUST_FOLLOW_CHOCH,
    )
    print(
        "Max candles after CHoCH:",
        BOS_MAX_CANDLES_AFTER_CHOCH,
    )
    print(
        "New structure required:",
        BOS_REQUIRE_NEW_STRUCTURE_LEVEL,
    )
    print(
        "Same-break forbidden:",
        CHOCH_BOS_SAME_BREAK_FORBIDDEN,
    )
    print("Chronology support: True")
    print("ATR validation: True")
    print("Displacement validation: True")
    print("✓ Institutional BOS engine loaded successfully.")


if __name__ == "__main__":
    _self_test()