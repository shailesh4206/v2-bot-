"""
instruments.py — Instrument listing and liquidity filtering.

Fetches available Delta Exchange India instruments and applies
configurable liquidity filters before the bot monitors a symbol.

Signal-only bot.
No capital, balance, or position-sizing logic.
"""

import logging
from typing import List, Dict, Optional, Any

from market_data.delta_client import DeltaClient
from config.settings import (
    MONITORED_SYMBOLS,
    MIN_24H_VOLUME_USD,
    MAX_SPREAD_PERCENT,
)

logger = logging.getLogger(__name__)


def get_all_instruments() -> List[Dict]:
    """Fetch all available Delta Exchange instruments."""
    client = DeltaClient()
    instruments = client.get_instruments()

    if not instruments:
        logger.warning("No instruments returned from Delta Exchange")
        return []

    if not isinstance(instruments, list):
        logger.error(
            f"Unexpected instruments response type: {type(instruments).__name__}"
        )
        return []

    return instruments


def _normalize_ticker(ticker_data: Any, symbol: str) -> Optional[Dict]:
    """
    Normalize Delta ticker response.

    Delta may return:
      - a dict
      - a list containing ticker dict(s)

    Returns one ticker dictionary for the requested symbol.
    """

    if isinstance(ticker_data, dict):
        return ticker_data

    if isinstance(ticker_data, list):

        if not ticker_data:
            return None

        # Try to find the requested symbol.
        for item in ticker_data:
            if not isinstance(item, dict):
                continue

            item_symbol = (
                item.get("symbol")
                or item.get("ticker_symbol")
                or item.get("product_symbol")
            )

            if item_symbol == symbol:
                return item

        # If only one ticker was returned, safely use it.
        if len(ticker_data) == 1 and isinstance(ticker_data[0], dict):
            return ticker_data[0]

    logger.warning(
        f"Unexpected ticker response for {symbol}: "
        f"{type(ticker_data).__name__}"
    )

    return None


def _extract_number(data: Dict, *keys, default=0.0) -> float:
    """Extract the first valid numeric field from a dictionary."""

    for key in keys:
        value = data.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return default


def _normalize_orderbook(orderbook_data: Any) -> Optional[Dict]:
    """
    Normalize Delta orderbook response.

    Expected normalized format:

    {
        "buy": [...],
        "sell": [...]
    }
    """

    if not isinstance(orderbook_data, dict):
        logger.warning(
            f"Unexpected orderbook response type: "
            f"{type(orderbook_data).__name__}"
        )
        return None

    return orderbook_data


def _extract_price(level: Any) -> float:
    """Extract price from list/tuple/dict orderbook level."""

    if isinstance(level, (list, tuple)):
        if not level:
            return 0.0

        try:
            return float(level[0])
        except (TypeError, ValueError):
            return 0.0

    if isinstance(level, dict):
        value = (
            level.get("price")
            or level.get("limit_price")
            or level.get("p")
        )

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def get_liquid_symbols(
    candidate_symbols: Optional[List[str]] = None,
    min_volume_usd: float = MIN_24H_VOLUME_USD,
    max_spread_pct: float = MAX_SPREAD_PERCENT,
) -> List[str]:
    """
    Filter candidate symbols using volume and spread.

    If market-data validation fails, the symbol is NOT automatically
    approved. This prevents bad API responses from bypassing filters.
    """

    if candidate_symbols is None:
        candidate_symbols = MONITORED_SYMBOLS

    client = DeltaClient()
    liquid = []

    for symbol in candidate_symbols:

        try:

            # --------------------------------------------------
            # TICKER
            # --------------------------------------------------

            raw_ticker = client.get_ticker(symbol)
            ticker = _normalize_ticker(raw_ticker, symbol)

            if not ticker:
                logger.warning(
                    f"{symbol}: ticker unavailable — filtered out"
                )
                continue

            # --------------------------------------------------
            # VOLUME
            # --------------------------------------------------

            volume_24h = _extract_number(
                ticker,
                "volume",
                "volume_24h",
                "turnover",
                "turnover_24h",
                default=0.0,
            )

            close_price = _extract_number(
                ticker,
                "close",
                "last_price",
                "mark_price",
                "price",
                default=0.0,
            )

            # Some Delta ticker fields may already represent USD
            # turnover/volume. Prefer turnover if available.
            turnover_24h = _extract_number(
                ticker,
                "turnover",
                "turnover_24h",
                "volume_usd",
                default=0.0,
            )

            if turnover_24h > 0:
                volume_usd = turnover_24h
            else:
                volume_usd = volume_24h * close_price

            if volume_usd < min_volume_usd:

                logger.info(
                    f"{symbol}: volume ${volume_usd:,.0f} "
                    f"< minimum ${min_volume_usd:,.0f} — filtered out"
                )

                continue

            # --------------------------------------------------
            # ORDERBOOK / SPREAD
            # --------------------------------------------------

            raw_orderbook = client.get_orderbook(
                symbol,
                depth=1,
            )

            orderbook = _normalize_orderbook(raw_orderbook)

            if not orderbook:
                logger.warning(
                    f"{symbol}: orderbook unavailable — "
                    f"filtered out"
                )
                continue

            bids = (
                orderbook.get("buy")
                or orderbook.get("bids")
                or []
            )

            asks = (
                orderbook.get("sell")
                or orderbook.get("asks")
                or []
            )

            if not bids or not asks:
                logger.warning(
                    f"{symbol}: empty orderbook — filtered out"
                )
                continue

            best_bid = _extract_price(bids[0])
            best_ask = _extract_price(asks[0])

            if best_bid <= 0 or best_ask <= 0:

                logger.warning(
                    f"{symbol}: invalid bid/ask — filtered out"
                )

                continue

            spread_pct = (
                (best_ask - best_bid)
                / best_bid
                * 100
            )

            if spread_pct > max_spread_pct:

                logger.info(
                    f"{symbol}: spread {spread_pct:.3f}% "
                    f"> max {max_spread_pct:.3f}% — filtered out"
                )

                continue

            # --------------------------------------------------
            # PASSED
            # --------------------------------------------------

            liquid.append(symbol)

            logger.info(
                f"{symbol}: passed liquidity filter "
                f"(volume ${volume_usd:,.0f}, "
                f"spread {spread_pct:.3f}%)"
            )

        except Exception as e:

            logger.error(
                f"Error checking liquidity for {symbol}: {e}",
                exc_info=True,
            )

            # IMPORTANT:
            # Do NOT automatically approve a symbol when
            # liquidity validation fails.

    return liquid


def get_symbol_info(symbol: str) -> Optional[Dict]:
    """Get metadata for a specific symbol."""

    instruments = get_all_instruments()

    for inst in instruments:

        if not isinstance(inst, dict):
            continue

        if (
            inst.get("symbol") == symbol
            or inst.get("ticker_symbol") == symbol
        ):
            return inst

    return None