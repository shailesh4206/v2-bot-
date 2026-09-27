"""
candles.py — Candle fetching and DataFrame construction.

Supports: 4H, 1H, 15M (NO 5M).
Includes pagination for long historical ranges.
Local in-memory cache to reduce API calls during repeated analysis.
"""

import time
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict

from market_data.delta_client import DeltaClient
from config.settings import CANDLES_LIMIT, TIMEFRAMES

logger = logging.getLogger(__name__)

# Resolution in seconds (for pagination math)
RESOLUTION_SECONDS: Dict[str, int] = {
    "4h":  4 * 3600,
    "1h":  3600,
    "15m": 900,
}

# In-memory cache: (symbol, resolution, end_ts) → DataFrame
_CANDLE_CACHE: Dict[str, pd.DataFrame] = {}


def resolution_to_seconds(resolution: str) -> int:
    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported resolution '{resolution}'. Use: {list(RESOLUTION_SECONDS.keys())}")
    return RESOLUTION_SECONDS[resolution]


def fetch_candles(
    symbol: str,
    resolution: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    limit: Optional[int] = None,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a symbol/resolution.

    Args:
        symbol:     e.g., "BTCUSD"
        resolution: "4h", "1h", "15m"
        start:      Unix timestamp (seconds). If None, derived from limit.
        end:        Unix timestamp (seconds). If None, uses now.
        limit:      Number of candles to fetch (if start not given).
        use_cache:  If True, returns cached data if available.

    Returns:
        pd.DataFrame with columns: [time, open, high, low, close, volume]
                                   index = datetime (UTC)
    """
    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported resolution: '{resolution}'. Valid: {list(RESOLUTION_SECONDS.keys())}")

    res_seconds = resolution_to_seconds(resolution)

    if end is None:
        end = int(time.time())

    if start is None:
        n = limit or CANDLES_LIMIT.get(resolution, 200)
        start = end - (n * res_seconds)

    cache_key = f"{symbol}|{resolution}|{end}"
    if use_cache and cache_key in _CANDLE_CACHE:
        logger.debug(f"Cache hit: {cache_key}")
        return _CANDLE_CACHE[cache_key]

    client = DeltaClient()
    all_candles = []

    # Paginate: Delta may limit a single request to ~500–1000 candles
    CHUNK_SIZE = 500
    chunk_seconds = CHUNK_SIZE * res_seconds

    current_start = start
    while current_start < end:
        current_end = min(current_start + chunk_seconds, end)
        raw = client.get_candles(symbol, resolution, current_start, current_end)
        if raw:
            all_candles.extend(raw)
        current_start = current_end + res_seconds
        if current_start >= end:
            break

    if not all_candles:
        logger.warning(f"No candles returned for {symbol} {resolution}")
        return pd.DataFrame()

    df = _to_dataframe(all_candles)
    df = df[df.index >= pd.Timestamp(start, unit="s", tz="UTC")]
    df = df[df.index <= pd.Timestamp(end, unit="s", tz="UTC")]
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    if use_cache:
        _CANDLE_CACHE[cache_key] = df

    logger.info(f"Fetched {len(df)} candles for {symbol} {resolution}")
    return df


def fetch_latest_candles(
    symbol: str,
    resolution: str,
    n: int = 200,
) -> pd.DataFrame:
    """Convenience: fetch the most recent N confirmed candles."""
    res_seconds = resolution_to_seconds(resolution)
    now = int(time.time())
    # Step back 1 candle to avoid the currently open (unconfirmed) candle
    end = now - res_seconds
    start = end - (n * res_seconds)
    return fetch_candles(symbol, resolution, start=start, end=end)


def _to_dataframe(raw: list) -> pd.DataFrame:
    """Convert raw API response list to a clean DataFrame."""
    records = []
    for c in raw:
        if isinstance(c, dict):
            records.append({
                "time":   c.get("time") or c.get("t"),
                "open":   float(c.get("open")  or c.get("o") or 0),
                "high":   float(c.get("high")  or c.get("h") or 0),
                "low":    float(c.get("low")   or c.get("l") or 0),
                "close":  float(c.get("close") or c.get("c") or 0),
                "volume": float(c.get("volume")or c.get("v") or 0),
            })
        elif isinstance(c, (list, tuple)) and len(c) >= 6:
            records.append({
                "time":   c[0],
                "open":   float(c[1]),
                "high":   float(c[2]),
                "low":    float(c[3]),
                "close":  float(c[4]),
                "volume": float(c[5]),
            })

    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def get_previous_day_levels(
    symbol: str,
    resolution: str = "1h",
) -> Dict[str, float]:
    """
    Calculate Previous Day High and Previous Day Low
    using confirmed closed candles.

    Returns dict with 'PDH' and 'PDL'.
    """
    df = fetch_latest_candles(symbol, "1h", n=48)
    if df.empty:
        return {}

    now_utc = datetime.now(timezone.utc)
    yesterday = (now_utc - timedelta(days=1)).date()

    mask = df.index.date == yesterday
    prev_day = df[mask]

    if prev_day.empty:
        return {}

    return {
        "PDH": float(prev_day["high"].max()),
        "PDL": float(prev_day["low"].min()),
    }


def clear_cache():
    """Clear in-memory candle cache."""
    global _CANDLE_CACHE
    _CANDLE_CACHE.clear()
    logger.debug("Candle cache cleared")
