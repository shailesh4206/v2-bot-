"""
bos.py — Break of Structure (BOS) detection.

Bullish BOS: Price closes ABOVE a meaningful previous swing high.
Bearish BOS: Price closes BELOW a meaningful previous swing low.

Rules:
- Confirmation requires a CANDLE CLOSE (not a wick breach).
- Only meaningful swings qualify (filtered by swing detection).
- No look-ahead: only use data available at time of detection.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from market_structure.swings import SwingPoint, SwingType


class BOSDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class BOSEvent:
    direction:     BOSDirection
    broken_level:  float        # The swing high/low that was broken
    broken_swing:  SwingPoint   # The swing point that was broken
    break_candle_idx: int       # Index of the candle that confirmed the BOS
    break_timestamp:  pd.Timestamp
    break_close:   float        # Close price that broke the level


def detect_bos(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
    lookback:    int = 5,       # Only look at the last N swings
) -> List[BOSEvent]:
    """
    Detect all BOS events in the DataFrame.

    Uses confirmed candle closes only. No wick-based BOS.

    Args:
        df:           OHLCV DataFrame
        swing_highs:  List of detected swing highs
        swing_lows:   List of detected swing lows
        lookback:     Number of recent swings to check against

    Returns:
        List of BOSEvent objects, in chronological order
    """
    bos_events: List[BOSEvent] = []

    closes = df["close"].values

    # Bullish BOS: close breaks above recent swing high
    relevant_highs = swing_highs[-lookback:] if len(swing_highs) > lookback else swing_highs
    for sh in relevant_highs:
        level = sh.price
        start_search = sh.candle_idx + 1
        for i in range(start_search, len(df)):
            if closes[i] > level:
                bos_events.append(BOSEvent(
                    direction=BOSDirection.BULLISH,
                    broken_level=level,
                    broken_swing=sh,
                    break_candle_idx=i,
                    break_timestamp=df.index[i],
                    break_close=float(closes[i]),
                ))
                break  # Only the first break per swing level

    # Bearish BOS: close breaks below recent swing low
    relevant_lows = swing_lows[-lookback:] if len(swing_lows) > lookback else swing_lows
    for sl in relevant_lows:
        level = sl.price
        start_search = sl.candle_idx + 1
        for i in range(start_search, len(df)):
            if closes[i] < level:
                bos_events.append(BOSEvent(
                    direction=BOSDirection.BEARISH,
                    broken_level=level,
                    broken_swing=sl,
                    break_candle_idx=i,
                    break_timestamp=df.index[i],
                    break_close=float(closes[i]),
                ))
                break

    # Sort by break candle index
    bos_events.sort(key=lambda e: e.break_candle_idx)
    return bos_events


def get_latest_bos(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
    lookback: int = 5,
    direction: Optional[BOSDirection] = None,
) -> Optional[BOSEvent]:
    """
    Return the most recent BOS event.

    Args:
        direction: If set, only return BOS of that direction.
    """
    events = detect_bos(df, swing_highs, swing_lows, lookback)
    if direction:
        events = [e for e in events if e.direction == direction]
    return events[-1] if events else None


def has_recent_bos(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
    direction: BOSDirection,
    lookback_candles: int = 10,
) -> bool:
    """
    Check if a BOS occurred within the last N candles.

    Args:
        lookback_candles: Only consider BOS events in the last N candles.
    """
    events = detect_bos(df, swing_highs, swing_lows, lookback=5)
    events = [e for e in events if e.direction == direction]
    if not events:
        return False
    latest = events[-1]
    return (len(df) - 1 - latest.break_candle_idx) <= lookback_candles
