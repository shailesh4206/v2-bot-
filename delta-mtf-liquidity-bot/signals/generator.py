"""
signals/generator.py — Signal generation orchestrator.

Fetches market data for all monitored symbols, runs both
intraday and swing pipelines, and returns validated setup results.

Runs on every confirmed 15M candle close (scheduled by main.py).
"""

import logging
from typing import List, Optional
from datetime import datetime, timezone

from market_data.candles import fetch_latest_candles, get_previous_day_levels
from market_data.instruments import get_liquid_symbols
from strategy.mtf import analyze_mtf
from strategy.intraday import run_intraday_pipeline, SetupResult
from strategy.swing import run_swing_pipeline
from indicators.trend import classify_trend
from config.settings import (
    MONITORED_SYMBOLS, BTC_SYMBOL,
    INTRADAY_ENABLED, SWING_ENABLED,
    CANDLES_LIMIT,
)

logger = logging.getLogger(__name__)


def generate_signals(
    use_liquidity_filter: bool = True,
) -> List[SetupResult]:
    """
    Main signal generation function.

    1. Get liquid symbols
    2. Fetch BTC context
    3. For each symbol:
       a. Fetch 4H, 1H, 15M data
       b. Run MTF analysis
       c. Run intraday pipeline (if enabled)
       d. Run swing pipeline (if enabled)
    4. Return all valid, sendable setups

    Returns:
        List of SetupResult objects that passed all filters
    """
    logger.info(f"Signal generation started at {datetime.now(timezone.utc).isoformat()}")
    all_results: List[SetupResult] = []

    # ── Get monitored symbols (with optional liquidity filter) ────────────
    if use_liquidity_filter:
        symbols = get_liquid_symbols(MONITORED_SYMBOLS)
        if not symbols:
            logger.warning("No liquid symbols found — using configured defaults")
            symbols = MONITORED_SYMBOLS
    else:
        symbols = MONITORED_SYMBOLS

    # ── Fetch BTC context for correlation filter ───────────────────────────
    btc_h4_trend = None
    btc_h1_trend = None

    if BTC_SYMBOL in symbols or True:  # Always fetch BTC for altcoin correlation
        try:
            btc_4h = fetch_latest_candles(BTC_SYMBOL, "4h", n=CANDLES_LIMIT["4h"])
            btc_1h = fetch_latest_candles(BTC_SYMBOL, "1h", n=CANDLES_LIMIT["1h"])
            if not btc_4h.empty:
                btc_h4_trend = classify_trend(btc_4h, "4h")
            if not btc_1h.empty:
                btc_h1_trend = classify_trend(btc_1h, "1h")
            logger.info(f"BTC context: 4H={btc_h4_trend.bias.value if btc_h4_trend else '?'} "
                       f"1H={btc_h1_trend.bias.value if btc_h1_trend else '?'}")
        except Exception as e:
            logger.warning(f"Could not fetch BTC context: {e}")

    # ── Process each symbol ────────────────────────────────────────────────
    for symbol in symbols:
        try:
            results = _process_symbol(
                symbol=symbol,
                btc_h4_trend=btc_h4_trend,
                btc_h1_trend=btc_h1_trend,
            )
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    logger.info(f"Signal generation complete. Valid signals: {len(all_results)}")
    return all_results


def _process_symbol(
    symbol:       str,
    btc_h4_trend,
    btc_h1_trend,
) -> List[SetupResult]:
    """Fetch data and run pipelines for a single symbol."""
    logger.info(f"Processing {symbol} …")

    # ── Fetch candle data ─────────────────────────────────────────────────
    try:
        df_4h  = fetch_latest_candles(symbol, "4h",  n=CANDLES_LIMIT["4h"])
        df_1h  = fetch_latest_candles(symbol, "1h",  n=CANDLES_LIMIT["1h"])
        df_15m = fetch_latest_candles(symbol, "15m", n=CANDLES_LIMIT["15m"])
    except Exception as e:
        logger.error(f"Failed to fetch candles for {symbol}: {e}")
        return []

    if df_4h.empty or df_1h.empty or df_15m.empty:
        logger.warning(f"{symbol}: One or more timeframes returned empty data")
        return []

    # ── Previous Day High/Low ─────────────────────────────────────────────
    pdlevels = get_previous_day_levels(symbol)
    pdh = pdlevels.get("PDH")
    pdl = pdlevels.get("PDL")

    # ── Multi-Timeframe Analysis ──────────────────────────────────────────
    mtf_ctx = analyze_mtf(symbol, df_4h, df_1h, df_15m, pdh=pdh, pdl=pdl)
    logger.info(
        f"{symbol} MTF: 4H={mtf_ctx.h4_trend.bias.value} "
        f"1H={mtf_ctx.h1_trend.bias.value} "
        f"Status={mtf_ctx.status.value}"
    )

    results: List[SetupResult] = []

    # ── Intraday Pipeline ─────────────────────────────────────────────────
    if INTRADAY_ENABLED:
        try:
            intraday = run_intraday_pipeline(
                symbol=symbol,
                df_4h=df_4h, df_1h=df_1h, df_15m=df_15m,
                mtf_ctx=mtf_ctx,
                btc_h4_trend=btc_h4_trend,
                btc_h1_trend=btc_h1_trend,
                pdh=pdh, pdl=pdl,
            )
            results.extend(intraday)
            if intraday:
                logger.info(f"{symbol}: {len(intraday)} INTRADAY signal(s) generated")
        except Exception as e:
            logger.error(f"{symbol} intraday pipeline error: {e}", exc_info=True)

    # ── Swing Pipeline ────────────────────────────────────────────────────
    if SWING_ENABLED:
        try:
            swing = run_swing_pipeline(
                symbol=symbol,
                df_4h=df_4h, df_1h=df_1h, df_15m=df_15m,
                mtf_ctx=mtf_ctx,
                btc_h4_trend=btc_h4_trend,
                btc_h1_trend=btc_h1_trend,
                pdh=pdh, pdl=pdl,
            )
            results.extend(swing)
            if swing:
                logger.info(f"{symbol}: {len(swing)} SWING signal(s) generated")
        except Exception as e:
            logger.error(f"{symbol} swing pipeline error: {e}", exc_info=True)

    return results
