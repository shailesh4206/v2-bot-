"""
choch.py — Change of Character (CHoCH) detection.

Bullish CHoCH: After a local downtrend (LH sequence), price breaks a meaningful 
               previous Lower High — signaling a potential structure reversal.

Bearish CHoCH: After a local uptrend (HL sequence), price breaks a meaningful
               previous Higher Low — signaling a potential structure reversal.

CHoCH is the first sign of structure change. BOS confirms it.

Rules:
- Must follow established local structure (not random swing breaks).
- Candle CLOSE required for confirmation.
- ATR-based minimum swing size prevents noise triggers.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from market_structure.swings import SwingPoint, SwingType, StructureLabel
from indicators.volatility import atr as calc_atr
from config.strategy_config import MIN_SWING_ATR_MULTIPLIER


class CHoCHDirection(str, Enum):
    BULLISH = "BULLISH"   # Potential bottom — bearish → bullish
    BEARISH = "BEARISH"   # Potential top   — bullish → bearish


@dataclass
class CHoCHEvent:
    direction:         CHoCHDirection
    broken_level:      float        # The LH (for bullish) or HL (for bearish) that was broken
    broken_swing:      SwingPoint
    break_candle_idx:  int
    break_timestamp:   pd.Timestamp
    break_close:       float
    prior_structure:   str          # Description of structure before CHoCH


def detect_choch(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
    lookback: int = 5,
) -> List[CHoCHEvent]:
    """
    Detect CHoCH events in the DataFrame.

    Bullish CHoCH: We need at least one LH (Lower High) in recent highs.
                   Price closes above that LH level → Bullish CHoCH.

    Bearish CHoCH: We need at least one HL (Higher Low) in recent lows.
                   Price closes below that HL level → Bearish CHoCH.

    Returns:
        List of CHoCHEvent, chronological order
    """
    events: List[CHoCHEvent] = []
    closes = df["close"].values

    # ── Bullish CHoCH: break of a LH (Lower High)
    recent_highs = swing_highs[-lookback:] if len(swing_highs) > lookback else swing_highs
    lh_swings = [sh for sh in recent_highs if sh.label == StructureLabel.LH]

    for lh in lh_swings:
        level = lh.price
        start = lh.candle_idx + 1
        for i in range(start, len(df)):
            if closes[i] > level:
                # Verify there was bearish structure before this point
                prior_lows = [sl for sl in swing_lows if sl.candle_idx < lh.candle_idx]
                prior_str = "LH confirmed"
                if any(sl.label == StructureLabel.LL for sl in prior_lows[-3:]):
                    prior_str = "LH + LL sequence"
                events.append(CHoCHEvent(
                    direction=CHoCHDirection.BULLISH,
                    broken_level=level,
                    broken_swing=lh,
                    break_candle_idx=i,
                    break_timestamp=df.index[i],
                    break_close=float(closes[i]),
                    prior_structure=prior_str,
                ))
                break

    # ── Bearish CHoCH: break of a HL (Higher Low)
    recent_lows = swing_lows[-lookback:] if len(swing_lows) > lookback else swing_lows
    hl_swings = [sl for sl in recent_lows if sl.label == StructureLabel.HL]

    for hl in hl_swings:
        level = hl.price
        start = hl.candle_idx + 1
        for i in range(start, len(df)):
            if closes[i] < level:
                prior_highs = [sh for sh in swing_highs if sh.candle_idx < hl.candle_idx]
                prior_str = "HL confirmed"
                if any(sh.label == StructureLabel.HH for sh in prior_highs[-3:]):
                    prior_str = "HH + HL sequence"
                events.append(CHoCHEvent(
                    direction=CHoCHDirection.BEARISH,
                    broken_level=level,
                    broken_swing=hl,
                    break_candle_idx=i,
                    break_timestamp=df.index[i],
                    break_close=float(closes[i]),
                    prior_structure=prior_str,
                ))
                break

    events.sort(key=lambda e: e.break_candle_idx)
    return events


def get_latest_choch(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
    direction: Optional[CHoCHDirection] = None,
    lookback: int = 5,
) -> Optional[CHoCHEvent]:
    """Return the most recent CHoCH event, optionally filtered by direction."""
    events = detect_choch(df, swing_highs, swing_lows, lookback)
    if direction:
        events = [e for e in events if e.direction == direction]
    return events[-1] if events else None


def has_recent_choch(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
    direction: CHoCHDirection,
    lookback_candles: int = 15,
) -> bool:
    """Check if a CHoCH of the given direction occurred within last N candles."""
    events = detect_choch(df, swing_highs, swing_lows)
    events = [e for e in events if e.direction == direction]
    if not events:
        return False
    latest = events[-1]
    return (len(df) - 1 - latest.break_candle_idx) <= lookback_candles
