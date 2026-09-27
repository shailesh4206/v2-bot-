"""
sweep.py — Liquidity Sweep Detection.

Bullish Sweep (Sell-Side Liquidity Swept):
  1. Price approaches meaningful sell-side liquidity
  2. Price trades BELOW the liquidity level (wick)
  3. Price closes BACK ABOVE the level (rejection)
  4. Bullish structural confirmation follows

Bearish Sweep (Buy-Side Liquidity Swept):
  1. Price approaches meaningful buy-side liquidity
  2. Price trades ABOVE the liquidity level (wick)
  3. Price closes BACK BELOW the level (rejection)
  4. Bearish structural confirmation follows

All detection is close-based for confirmation.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from liquidity.pools import LiquidityPool, LiquiditySide
from indicators.volatility import atr as calc_atr


class SweepDirection(str, Enum):
    BULLISH = "BULLISH"   # Sell-side swept → expect price to rise
    BEARISH = "BEARISH"   # Buy-side swept  → expect price to fall


@dataclass
class SweepEvent:
    direction:         SweepDirection
    pool:              LiquidityPool
    sweep_candle_idx:  int
    sweep_timestamp:   pd.Timestamp
    sweep_low:         float          # How far below (bullish) / above (bearish) price swept
    close_price:       float          # Candle close price (back above/below level)
    wick_size:         float          # Size of the rejection wick
    body_size:         float          # Size of the candle body
    wick_to_body_ratio: float         # Wick/body ratio (higher = stronger rejection)
    sweep_volume:      float          # Volume at sweep candle
    avg_volume:        float          # 20-period average volume at time of sweep
    rejection_strength: str           # "STRONG", "MODERATE", "WEAK"


def detect_sweeps(
    df: pd.DataFrame,
    pools: List[LiquidityPool],
    lookback: int = 10,
) -> List[SweepEvent]:
    """
    Detect liquidity sweep events across all pools.

    Args:
        df:       OHLCV DataFrame (confirmed candles only)
        pools:    List of liquidity pools to check
        lookback: Only check the last N candles for sweep activity

    Returns:
        List of SweepEvent objects, most recent first
    """
    sweeps: List[SweepEvent] = []
    if df.empty or len(df) < 5:
        return sweeps

    # Calculate 20-period volume average
    vol_avg = df["volume"].rolling(20).mean()
    atr_series = calc_atr(df)

    # Check only recent candles
    start_idx = max(0, len(df) - lookback)

    for i in range(start_idx, len(df)):
        candle = df.iloc[i]
        o  = candle["open"]
        h  = candle["high"]
        l  = candle["low"]
        c  = candle["close"]
        v  = candle["volume"]
        av = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) else 0.0

        for pool in pools:
            level = pool.level

            # ── Bullish Sweep (sell-side liquidity swept)
            if pool.side == LiquiditySide.SELL_SIDE:
                swept  = l < level              # Wick traded below level
                closed = c > level              # Closed back above
                if swept and closed:
                    wick_size  = level - l
                    body_size  = abs(c - o)
                    ratio      = wick_size / body_size if body_size > 0 else 99.0
                    strength   = _rejection_strength(ratio)

                    sweeps.append(SweepEvent(
                        direction=SweepDirection.BULLISH,
                        pool=pool,
                        sweep_candle_idx=i,
                        sweep_timestamp=df.index[i],
                        sweep_low=l,
                        close_price=c,
                        wick_size=wick_size,
                        body_size=body_size,
                        wick_to_body_ratio=ratio,
                        sweep_volume=v,
                        avg_volume=av,
                        rejection_strength=strength,
                    ))

            # ── Bearish Sweep (buy-side liquidity swept)
            elif pool.side == LiquiditySide.BUY_SIDE:
                swept  = h > level              # Wick traded above level
                closed = c < level              # Closed back below
                if swept and closed:
                    wick_size  = h - level
                    body_size  = abs(c - o)
                    ratio      = wick_size / body_size if body_size > 0 else 99.0
                    strength   = _rejection_strength(ratio)

                    sweeps.append(SweepEvent(
                        direction=SweepDirection.BEARISH,
                        pool=pool,
                        sweep_candle_idx=i,
                        sweep_timestamp=df.index[i],
                        sweep_low=h,
                        close_price=c,
                        wick_size=wick_size,
                        body_size=body_size,
                        wick_to_body_ratio=ratio,
                        sweep_volume=v,
                        avg_volume=av,
                        rejection_strength=strength,
                    ))

    # Sort by candle index (most recent last)
    sweeps.sort(key=lambda s: s.sweep_candle_idx)
    return sweeps


def get_latest_sweep(
    df: pd.DataFrame,
    pools: List[LiquidityPool],
    direction: Optional[SweepDirection] = None,
    lookback: int = 10,
) -> Optional[SweepEvent]:
    """Return the most recent sweep event, optionally filtered by direction."""
    sweeps = detect_sweeps(df, pools, lookback)
    if direction:
        sweeps = [s for s in sweeps if s.direction == direction]
    return sweeps[-1] if sweeps else None


def _rejection_strength(wick_to_body_ratio: float) -> str:
    """Classify rejection strength from wick-to-body ratio."""
    if wick_to_body_ratio >= 2.0:
        return "STRONG"
    elif wick_to_body_ratio >= 1.0:
        return "MODERATE"
    else:
        return "WEAK"
