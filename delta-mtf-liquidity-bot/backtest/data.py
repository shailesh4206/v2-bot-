"""
backtest/data.py — Historical data loader for backtesting.

Fetches and caches multi-timeframe candle data for replay.
Only supports 4H, 1H, 15M. NO 5M.
"""

import os
import pickle
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Tuple

from market_data.candles import fetch_candles
from config.settings import TIMEFRAMES

logger = logging.getLogger(__name__)
CACHE_DIR = "logs/backtest_cache"


def load_backtest_data(
    symbol:     str,
    start_date: str,
    end_date:   str,
    use_cache:  bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load 4H + 1H + 15M historical data for backtesting.

    Args:
        symbol:     e.g., "BTCUSD"
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"
        use_cache:  If True, saves/loads from local pickle cache

    Returns:
        Dict {"4h": df_4h, "1h": df_1h, "15m": df_15m}
    """
    cache_key = f"{symbol}_{start_date}_{end_date}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if use_cache and os.path.exists(cache_path):
        logger.info(f"Loading cached backtest data: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    logger.info(f"Fetching backtest data for {symbol} {start_date} → {end_date}")

    data = {}
    for tf in ["4h", "1h", "15m"]:
        logger.info(f"  Fetching {tf} candles …")
        df = fetch_candles(symbol, tf, start=start_ts, end=end_ts)
        data[tf] = df
        logger.info(f"  {tf}: {len(df)} candles")

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Backtest data cached: {cache_path}")

    return data


def align_mtf_at_index(
    data:        Dict[str, pd.DataFrame],
    m15_idx:     int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    At a given 15M candle index, return the aligned 4H and 1H DataFrames
    containing ONLY candles that were closed AT OR BEFORE that 15M candle.

    This is the core no-look-ahead alignment function for backtesting.

    Args:
        data:    {"4h": df, "1h": df, "15m": df}
        m15_idx: Current index in the 15M DataFrame

    Returns:
        (df_4h_aligned, df_1h_aligned, df_15m_slice)
    """
    df_15m = data["15m"]
    df_1h  = data["1h"]
    df_4h  = data["4h"]

    current_ts = df_15m.index[m15_idx]

    # Slice 4H and 1H to only include candles closed before the current 15M close
    df_4h_aligned  = df_4h[df_4h.index <= current_ts]
    df_1h_aligned  = df_1h[df_1h.index <= current_ts]
    df_15m_slice   = df_15m.iloc[:m15_idx + 1]

    return df_4h_aligned, df_1h_aligned, df_15m_slice
