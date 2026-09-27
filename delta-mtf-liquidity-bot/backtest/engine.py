"""
backtest/engine.py — Backtesting engine.

Replays 4H + 1H + 15M candle data chronologically.
Runs identical strategy logic to live mode (same modules).

Critical rules:
- NO look-ahead: swing detection only uses candles closed before current index
- NO 5M data anywhere
- Fees and slippage applied to every trade
- Development/OOS split enforced
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import pandas as pd

from backtest.data import load_backtest_data, align_mtf_at_index
from strategy.mtf import analyze_mtf
from strategy.intraday import run_intraday_pipeline
from strategy.swing import run_swing_pipeline
from config.settings import (
    STRATEGY_VERSION, INTRADAY_ENABLED, SWING_ENABLED,
    BACKTEST_FEE_PERCENT, BACKTEST_SLIPPAGE_PERCENT, BACKTEST_DEV_SPLIT,
)
from config.strategy_config import INTRADAY_EXPIRY_MINUTES, SWING_EXPIRY_HOURS

logger = logging.getLogger(__name__)

MIN_CANDLES_REQUIRED = 210  # Need enough for EMA200 + ATR warmup


def run_backtest(
    symbol:     str,
    start_date: str,
    end_date:   str,
    use_cache:  bool = True,
    dev_split:  float = BACKTEST_DEV_SPLIT,
    fee_pct:    float = BACKTEST_FEE_PERCENT,
    slippage_pct: float = BACKTEST_SLIPPAGE_PERCENT,
) -> Dict:
    """
    Run a full backtest for a symbol over a date range.

    Args:
        symbol:     e.g., "BTCUSD"
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"
        dev_split:  Fraction for development (70% default)
        fee_pct:    Fee % per side
        slippage_pct: Slippage % per side

    Returns:
        Dict with all backtest results
    """
    run_id = f"{symbol}_{start_date}_{end_date}_{uuid.uuid4().hex[:6]}"
    logger.info(f"Backtest started: {run_id}")

    data = load_backtest_data(symbol, start_date, end_date, use_cache=use_cache)

    df_15m = data.get("15m", pd.DataFrame())
    if df_15m.empty or len(df_15m) < MIN_CANDLES_REQUIRED:
        logger.error(f"Insufficient 15M data for backtest: {len(df_15m)} candles")
        return {"error": "Insufficient data"}

    # ── Dev/OOS split ──────────────────────────────────────────────────────
    split_idx = int(len(df_15m) * dev_split)
    split_ts  = df_15m.index[split_idx]
    logger.info(f"Dev/OOS split at index {split_idx} ({split_ts.date()})")

    trades: List[dict] = []
    pending_signals: List[dict] = []  # Signals waiting for entry/outcome

    # ── Main replay loop ───────────────────────────────────────────────────
    WARMUP = MIN_CANDLES_REQUIRED
    for i in range(WARMUP, len(df_15m)):
        current_ts = df_15m.index[i]
        is_oos     = i >= split_idx

        # Align all timeframes to the current 15M candle (no look-ahead)
        df_4h, df_1h, df_15m_slice = align_mtf_at_index(data, i)

        if len(df_4h) < 50 or len(df_1h) < 50:
            continue  # Not enough higher-TF data yet

        # ── Check pending signals for this candle ──────────────────────────
        current_high  = float(df_15m.iloc[i]["high"])
        current_low   = float(df_15m.iloc[i]["low"])
        current_close = float(df_15m.iloc[i]["close"])

        _process_pending(
            pending=pending_signals,
            completed=trades,
            current_ts=current_ts,
            high=current_high,
            low=current_low,
            close=current_close,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
        )

        # ── Generate new signals ───────────────────────────────────────────
        try:
            mtf_ctx = analyze_mtf(symbol, df_4h, df_1h, df_15m_slice)

            new_setups = []
            if INTRADAY_ENABLED:
                new_setups += run_intraday_pipeline(
                    symbol=symbol, df_4h=df_4h, df_1h=df_1h, df_15m=df_15m_slice,
                    mtf_ctx=mtf_ctx,
                )
            if SWING_ENABLED:
                new_setups += run_swing_pipeline(
                    symbol=symbol, df_4h=df_4h, df_1h=df_1h, df_15m=df_15m_slice,
                    mtf_ctx=mtf_ctx,
                )

            for setup in new_setups:
                if not setup.valid:
                    continue

                # Expiry time
                if setup.mode == "INTRADAY":
                    expiry = current_ts + timedelta(minutes=INTRADAY_EXPIRY_MINUTES)
                else:
                    expiry = current_ts + timedelta(hours=SWING_EXPIRY_HOURS)

                pending_signals.append({
                    "signal_id":   str(uuid.uuid4())[:8].upper(),
                    "symbol":      symbol,
                    "direction":   setup.direction,
                    "mode":        setup.mode,
                    "timeframe":   setup.timeframe,
                    "quality":     setup.quality.value,
                    "sweep_score": setup.sweep_score,
                    "entry":       setup.entry,
                    "sl":          setup.sl,
                    "tp":          setup.tp,
                    "rr":          setup.rr,
                    "liquidity_type": setup.liquidity_type,
                    "signal_ts":   current_ts,
                    "expiry_ts":   expiry,
                    "state":       "PENDING",
                    "entry_ts":    None,
                    "split_type":  "OUT_OF_SAMPLE" if is_oos else "IN_SAMPLE",
                })

        except Exception as e:
            logger.debug(f"Signal generation error at index {i}: {e}")
            continue

    # Mark remaining pending signals as expired
    for p in pending_signals:
        if p["state"] != "DONE":
            p["outcome"] = "EXPIRED"
            p["r_multiple"] = 0.0
            p["net_r"] = 0.0
            trades.append(p)

    logger.info(f"Backtest complete: {len(trades)} trades")

    from backtest.metrics import compute_metrics
    results = compute_metrics(trades)
    results["run_id"]   = run_id
    results["symbol"]   = symbol
    results["start"]    = start_date
    results["end"]      = end_date
    results["strategy"] = STRATEGY_VERSION
    return results


def _process_pending(
    pending, completed, current_ts, high, low, close,
    fee_pct, slippage_pct
):
    """Check pending signals against current candle's price action."""
    still_pending = []
    risk_per_entry = fee_pct / 100 + slippage_pct / 100  # Cost as fraction

    for sig in pending:
        if sig["state"] == "DONE":
            continue

        entry  = sig["entry"]
        sl_p   = sig["sl"]
        tp_p   = sig["tp"]
        expiry = sig["expiry_ts"]

        # Expiry
        if current_ts > expiry and sig["state"] == "PENDING":
            sig["outcome"]    = "EXPIRED"
            sig["r_multiple"] = 0.0
            sig["net_r"]      = 0.0
            sig["state"]      = "DONE"
            completed.append(sig)
            continue

        # Entry check
        if sig["state"] == "PENDING":
            if sig["direction"] == "LONG" and low <= entry:
                sig["state"]    = "ACTIVE"
                sig["entry_ts"] = current_ts
            elif sig["direction"] == "SHORT" and high >= entry:
                sig["state"]    = "ACTIVE"
                sig["entry_ts"] = current_ts

        if sig["state"] != "ACTIVE":
            still_pending.append(sig)
            continue

        # TP/SL check
        risk = abs(entry - sl_p)

        if sig["direction"] == "LONG":
            tp_hit = high >= tp_p
            sl_hit = low  <= sl_p
        else:
            tp_hit = low  <= tp_p
            sl_hit = high >= sl_p

        cost_r = (risk_per_entry * 2 * entry) / risk if risk > 0 else 0

        if tp_hit:
            raw_r = abs(tp_p - entry) / risk if risk > 0 else 0
            sig["outcome"]    = "TP_HIT"
            sig["exit_price"] = tp_p
            sig["exit_ts"]    = current_ts
            sig["r_multiple"] = round(raw_r, 2)
            sig["fees_r"]     = round(cost_r, 3)
            sig["net_r"]      = round(raw_r - cost_r, 3)
            sig["state"]      = "DONE"
            completed.append(sig)
        elif sl_hit:
            sig["outcome"]    = "SL_HIT"
            sig["exit_price"] = sl_p
            sig["exit_ts"]    = current_ts
            sig["r_multiple"] = -1.0
            sig["fees_r"]     = round(cost_r, 3)
            sig["net_r"]      = round(-1.0 - cost_r, 3)
            sig["state"]      = "DONE"
            completed.append(sig)
        else:
            still_pending.append(sig)

    pending.clear()
    pending.extend(still_pending)
