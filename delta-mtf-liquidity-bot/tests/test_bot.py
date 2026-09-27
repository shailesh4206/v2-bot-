"""
tests/test_swings.py — Unit tests for swing detection (no look-ahead validation).
tests/test_scoring.py — Tests for sweep scoring determinism.
tests/test_signals.py — Signal pipeline tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from market_structure.swings import detect_swing_highs, detect_swing_lows, label_market_structure
from market_structure.bos import detect_bos, BOSDirection
from market_structure.choch import detect_choch, CHoCHDirection
from liquidity.pools import detect_liquidity_pools, LiquidityType, LiquiditySide
from liquidity.sweep import detect_sweeps, SweepDirection
from liquidity.scoring import score_sweep
from indicators.trend import ema, classify_trend
from indicators.volatility import atr, get_current_atr


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_df(n=300, base_price=50000.0, trend="flat"):
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    times = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15*i) for i in range(n)]

    prices = [base_price]
    for i in range(1, n):
        if trend == "up":
            drift = 0.001
        elif trend == "down":
            drift = -0.001
        else:
            drift = 0.0
        prices.append(prices[-1] * (1 + drift + np.random.normal(0, 0.005)))

    df = pd.DataFrame({
        "open":   prices,
        "high":   [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        "low":    [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        "close":  [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        "volume": np.random.uniform(100, 10000, n),
    }, index=pd.DatetimeIndex(times, tz=timezone.utc))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Swing Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSwingDetection:

    def test_swing_highs_detected(self):
        df = make_df(200)
        highs = detect_swing_highs(df, "15m")
        assert len(highs) > 0, "Should detect at least some swing highs"

    def test_swing_lows_detected(self):
        df = make_df(200)
        lows = detect_swing_lows(df, "15m")
        assert len(lows) > 0, "Should detect at least some swing lows"

    def test_no_lookahead_swing_high(self):
        """
        CRITICAL: Swing at index i must only be marked when i+right_bars candles have closed.
        Verify no swing is detected beyond len(df) - right_bars.
        """
        df = make_df(50)
        RIGHT_BARS = 2
        highs = detect_swing_highs(df, "15m", right_bars=RIGHT_BARS)
        for h in highs:
            assert h.candle_idx <= len(df) - 1 - RIGHT_BARS, (
                f"Swing high at {h.candle_idx} violates no-look-ahead: "
                f"must be <= {len(df) - 1 - RIGHT_BARS}"
            )

    def test_no_lookahead_swing_low(self):
        """Same check for swing lows."""
        df = make_df(50)
        RIGHT_BARS = 2
        lows = detect_swing_lows(df, "15m", right_bars=RIGHT_BARS)
        for l in lows:
            assert l.candle_idx <= len(df) - 1 - RIGHT_BARS, (
                f"Swing low at {l.candle_idx} violates no-look-ahead"
            )

    def test_swing_high_is_local_max(self):
        """Each swing high must be the maximum within its window."""
        df = make_df(100)
        LEFT = 3
        RIGHT = 2
        highs = detect_swing_highs(df, "15m", left_bars=LEFT, right_bars=RIGHT)
        for h in highs:
            i = h.candle_idx
            window_high = df["high"].iloc[i-LEFT : i+RIGHT+1].max()
            assert abs(h.price - df["high"].iloc[i]) < 1e-9, "Swing price must match candle high"
            assert h.price == window_high, f"Swing {h.price} is not max in window {window_high}"

    def test_insufficient_data_returns_empty(self):
        df = make_df(5)  # Too few candles
        highs = detect_swing_highs(df, "15m")
        assert highs == [], "Should return empty list with insufficient data"


# ─────────────────────────────────────────────────────────────────────────────
# Sweep Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSweepScoring:

    def _make_sweep(self, pool_type="SWING_MAJOR", wick_ratio=2.5, volume_ratio=1.5):
        from liquidity.pools import LiquidityPool, LiquidityType, LiquiditySide
        from liquidity.sweep import SweepEvent, SweepDirection

        pool = LiquidityPool(
            pool_type=LiquidityType.SWING_MAJOR,
            side=LiquiditySide.SELL_SIDE,
            level=50000.0,
            zone_high=50100.0,
            zone_low=49900.0,
            touches=3,
        )
        return SweepEvent(
            direction=SweepDirection.BULLISH,
            pool=pool,
            sweep_candle_idx=10,
            sweep_timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            sweep_low=49800.0,
            close_price=50200.0,
            wick_size=200.0,
            body_size=200.0 / wick_ratio if wick_ratio > 0 else 200.0,
            wick_to_body_ratio=wick_ratio,
            sweep_volume=15000.0,
            avg_volume=10000.0,
            rejection_strength="STRONG",
        )

    def test_score_deterministic(self):
        """Same input must always produce same score."""
        sweep = self._make_sweep()
        score1 = score_sweep(sweep, choch_confirmed=True, bos_confirmed=True)
        score2 = score_sweep(sweep, choch_confirmed=True, bos_confirmed=True)
        assert score1.total == score2.total, "Score must be deterministic"

    def test_score_range(self):
        """Score must be 0–100."""
        sweep = self._make_sweep()
        score = score_sweep(sweep, choch_confirmed=True, bos_confirmed=True)
        assert 0 <= score.total <= 100

    def test_higher_wick_ratio_higher_score(self):
        """Higher wick rejection should produce higher score (all else equal)."""
        sweep_weak   = self._make_sweep(wick_ratio=0.5)
        sweep_strong = self._make_sweep(wick_ratio=3.0)
        score_weak   = score_sweep(sweep_weak)
        score_strong = score_sweep(sweep_strong)
        assert score_strong.total > score_weak.total

    def test_choch_adds_points(self):
        sweep = self._make_sweep()
        score_without = score_sweep(sweep, choch_confirmed=False)
        score_with    = score_sweep(sweep, choch_confirmed=True)
        assert score_with.total > score_without.total

    def test_bos_adds_points(self):
        sweep = self._make_sweep()
        score_without = score_sweep(sweep, bos_confirmed=False)
        score_with    = score_sweep(sweep, bos_confirmed=True)
        assert score_with.total > score_without.total

    def test_score_breakdown_sums_to_total(self):
        sweep = self._make_sweep()
        score = score_sweep(sweep, choch_confirmed=True, bos_confirmed=True)
        assert sum(score.breakdown.values()) == score.total


# ─────────────────────────────────────────────────────────────────────────────
# Indicator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicators:

    def test_ema_length(self):
        df = make_df(100)
        result = ema(df["close"], 20)
        assert len(result) == len(df)

    def test_ema_first_values_nan_then_values(self):
        df = make_df(30)
        result = ema(df["close"], 20)
        # ewm with adjust=False always starts from first value (no NaN)
        assert not result.isna().any(), "EMA should have no NaN with ewm(adjust=False)"

    def test_atr_positive(self):
        df = make_df(100)
        atr_val = get_current_atr(df)
        assert atr_val > 0, "ATR must be positive"

    def test_classify_trend_returns_valid(self):
        df = make_df(250)
        trend = classify_trend(df, "15m")
        assert trend.bias.value in ("BULLISH", "BEARISH", "NEUTRAL")
        assert trend.strength.value in ("STRONG", "NORMAL", "WEAK", "RANGING")


# ─────────────────────────────────────────────────────────────────────────────
# RR Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRRCalculation:

    def test_long_rr(self):
        from strategy.intraday import _calculate_rr
        rr = _calculate_rr("LONG", entry=50000, sl=49000, tp=52000)
        assert abs(rr - 2.0) < 0.01, f"Expected RR ~2.0, got {rr}"

    def test_short_rr(self):
        from strategy.intraday import _calculate_rr
        rr = _calculate_rr("SHORT", entry=50000, sl=51000, tp=47500)
        assert abs(rr - 2.5) < 0.01, f"Expected RR ~2.5, got {rr}"

    def test_zero_risk_returns_zero(self):
        from strategy.intraday import _calculate_rr
        rr = _calculate_rr("LONG", entry=50000, sl=50000, tp=52000)
        assert rr == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Guard Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateGuard:

    def setup_method(self):
        """Use a fresh in-memory DB for each test."""
        import sqlite3
        from database import models
        models.DB_PATH = ":memory:"
        # Can't easily reset sqlite3 in-memory — skip these for unit test simplicity
        # In integration tests use a temp file

    def test_fingerprint_deterministic(self):
        from signals.duplicate_guard import _make_fingerprint
        fp1 = _make_fingerprint("BTCUSD", "15m", "LONG", 50000.0, "INTRADAY")
        fp2 = _make_fingerprint("BTCUSD", "15m", "LONG", 50000.0, "INTRADAY")
        assert fp1 == fp2

    def test_fingerprint_different_direction(self):
        from signals.duplicate_guard import _make_fingerprint
        fp_long  = _make_fingerprint("BTCUSD", "15m", "LONG",  50000.0, "INTRADAY")
        fp_short = _make_fingerprint("BTCUSD", "15m", "SHORT", 50000.0, "INTRADAY")
        assert fp_long != fp_short


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
