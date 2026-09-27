"""
swings.py — Swing high/low detection with strict no-look-ahead guarantee.

Method: Pivot detection requiring left_bars confirmed candles BEFORE
and right_bars confirmed candles AFTER the potential pivot.

At index i:
  - Swing High if df["high"][i] == max(df["high"][i-left : i+right+1])
  - Only marked when candle at i+right has already CLOSED

This ensures the swing is never marked using future information.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from indicators.volatility import atr as calc_atr
from config.strategy_config import (
    SWING_LEFT_BARS_4H, SWING_RIGHT_BARS_4H,
    SWING_LEFT_BARS_1H, SWING_RIGHT_BARS_1H,
    SWING_LEFT_BARS_15M, SWING_RIGHT_BARS_15M,
    MIN_SWING_ATR_MULTIPLIER,
)


class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW  = "LOW"


class StructureLabel(str, Enum):
    HH = "HH"   # Higher High
    HL = "HL"   # Higher Low
    LH = "LH"   # Lower High
    LL = "LL"   # Lower Low
    EH = "EH"   # Equal High
    EL = "EL"   # Equal Low


@dataclass
class SwingPoint:
    swing_type:  SwingType
    price:       float
    candle_idx:  int
    timestamp:   pd.Timestamp
    label:       Optional[StructureLabel] = None
    confirmed:   bool = True


def _get_pivot_params(timeframe: str):
    params = {
        "4h":  (SWING_LEFT_BARS_4H,  SWING_RIGHT_BARS_4H),
        "1h":  (SWING_LEFT_BARS_1H,  SWING_RIGHT_BARS_1H),
        "15m": (SWING_LEFT_BARS_15M, SWING_RIGHT_BARS_15M),
    }
    return params.get(timeframe, (3, 2))


def detect_swing_highs(
    df: pd.DataFrame,
    timeframe: str = "15m",
    left_bars: Optional[int] = None,
    right_bars: Optional[int] = None,
    min_atr_mult: float = MIN_SWING_ATR_MULTIPLIER,
) -> List[SwingPoint]:
    """
    Detect confirmed swing highs with no look-ahead bias.

    A swing high at index i is ONLY recorded when we have reached index i+right_bars,
    meaning all right_bars confirmation candles have fully closed.

    Args:
        df:          OHLCV DataFrame
        timeframe:   "4h", "1h", "15m"
        left_bars:   Override pivot left bars
        right_bars:  Override pivot right bars
        min_atr_mult: Minimum swing size as ATR multiplier

    Returns:
        List of SwingPoint objects sorted by candle_idx
    """
    l, r = _get_pivot_params(timeframe)
    left_bars  = left_bars  or l
    right_bars = right_bars or r

    if len(df) < left_bars + right_bars + 1:
        return []

    atr_series = calc_atr(df)
    highs      = df["high"].values
    swings     = []

    # Iterate up to len-right_bars so we never peek into unconfirmed candles
    for i in range(left_bars, len(df) - right_bars):
        window = highs[i - left_bars : i + right_bars + 1]
        if highs[i] == window.max() and list(window).count(highs[i]) == 1:
            # Check minimum size vs. ATR
            atr_val = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0
            if atr_val > 0:
                left_low  = df["low"].iloc[i - left_bars : i].min()
                swing_size = highs[i] - left_low
                if swing_size < min_atr_mult * atr_val:
                    continue  # Too small — filter out

            swings.append(SwingPoint(
                swing_type=SwingType.HIGH,
                price=float(highs[i]),
                candle_idx=i,
                timestamp=df.index[i],
            ))

    return swings


def detect_swing_lows(
    df: pd.DataFrame,
    timeframe: str = "15m",
    left_bars: Optional[int] = None,
    right_bars: Optional[int] = None,
    min_atr_mult: float = MIN_SWING_ATR_MULTIPLIER,
) -> List[SwingPoint]:
    """
    Detect confirmed swing lows with no look-ahead bias.
    """
    l, r = _get_pivot_params(timeframe)
    left_bars  = left_bars  or l
    right_bars = right_bars or r

    if len(df) < left_bars + right_bars + 1:
        return []

    atr_series = calc_atr(df)
    lows       = df["low"].values
    swings     = []

    for i in range(left_bars, len(df) - right_bars):
        window = lows[i - left_bars : i + right_bars + 1]
        if lows[i] == window.min() and list(window).count(lows[i]) == 1:
            atr_val = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0
            if atr_val > 0:
                right_high = df["high"].iloc[i : i + right_bars + 1].max()
                swing_size = right_high - lows[i]
                if swing_size < min_atr_mult * atr_val:
                    continue

            swings.append(SwingPoint(
                swing_type=SwingType.LOW,
                price=float(lows[i]),
                candle_idx=i,
                timestamp=df.index[i],
            ))

    return swings


def label_market_structure(
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
    equal_tolerance_pct: float = 0.15,
) -> tuple:
    """
    Assign HH/HL/LH/LL labels to swing points.

    Returns:
        (labeled_highs, labeled_lows)
    """
    labeled_highs = _label_swings(swing_highs, equal_tolerance_pct)
    labeled_lows  = _label_swings(swing_lows,  equal_tolerance_pct)
    return labeled_highs, labeled_lows


def _label_swings(swings: List[SwingPoint], tol_pct: float = 0.15) -> List[SwingPoint]:
    if len(swings) < 2:
        return swings

    labeled = [swings[0]]
    for i in range(1, len(swings)):
        prev = labeled[-1]
        curr = swings[i]
        tol  = prev.price * (tol_pct / 100)

        if curr.swing_type == SwingType.HIGH:
            if curr.price > prev.price + tol:
                curr.label = StructureLabel.HH
            elif curr.price < prev.price - tol:
                curr.label = StructureLabel.LH
            else:
                curr.label = StructureLabel.EH
        else:  # LOW
            if curr.price > prev.price + tol:
                curr.label = StructureLabel.HL
            elif curr.price < prev.price - tol:
                curr.label = StructureLabel.LL
            else:
                curr.label = StructureLabel.EL

        labeled.append(curr)

    return labeled


def get_market_structure_bias(
    swing_highs: List[SwingPoint],
    swing_lows:  List[SwingPoint],
) -> str:
    """
    Determine overall structure bias based on last 2 swings each.

    Returns: "BULLISH", "BEARISH", "NEUTRAL"
    """
    recent_highs = swing_highs[-2:] if len(swing_highs) >= 2 else swing_highs
    recent_lows  = swing_lows[-2:]  if len(swing_lows)  >= 2 else swing_lows

    bullish_score = 0
    bearish_score = 0

    for sh in recent_highs:
        if sh.label == StructureLabel.HH:
            bullish_score += 1
        elif sh.label == StructureLabel.LH:
            bearish_score += 1

    for sl in recent_lows:
        if sl.label == StructureLabel.HL:
            bullish_score += 1
        elif sl.label == StructureLabel.LL:
            bearish_score += 1

    if bullish_score > bearish_score:
        return "BULLISH"
    elif bearish_score > bullish_score:
        return "BEARISH"
    return "NEUTRAL"
