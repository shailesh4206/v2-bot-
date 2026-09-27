"""
strategy_config.py — All strategy parameters for the MTF Liquidity Signal Bot.

Every threshold is configurable here.
Changes should only be made after human review and approval.
Strategy version history is tracked in the database.
"""

# ─────────────────────────────────────────────
# STRATEGY VERSION (mirrors settings.py)
# ─────────────────────────────────────────────
STRATEGY_VERSION = "v1.0"
STRATEGY_DATE    = "2026-09-27"

# ─────────────────────────────────────────────
# LIQUIDITY SWEEP SCORING
# ─────────────────────────────────────────────
MIN_SWEEP_SCORE = 70          # Minimum score (0–100) to consider a valid sweep

# Scoring weights (must sum to 100)
SWEEP_SCORE_WEIGHTS = {
    "pool_significance":  20,  # Type of liquidity pool (PDH/PDL = high, minor swing = low)
    "pool_touches":       10,  # Number of previous touches at the level
    "htf_4h_confluence":  15,  # Level visible/relevant on 4H
    "htf_1h_confluence":  10,  # Level visible/relevant on 1H
    "wick_rejection":     15,  # Wick/body ratio of sweep candle
    "volume_at_sweep":    10,  # Sweep candle volume vs average
    "atr_context":         5,  # Not overextended (ATR-based)
    "choch_confirmed":    10,  # CHoCH followed the sweep
    "bos_confirmed":       5,  # BOS followed the sweep
}

# Pool significance scores (0–20)
POOL_SIGNIFICANCE_SCORES = {
    "PDH":           20,   # Previous Day High
    "PDL":           20,   # Previous Day Low
    "EQUAL_HIGHS":   17,
    "EQUAL_LOWS":    17,
    "SWING_MAJOR":   15,   # Major multi-candle swing high/low
    "SWING_MINOR":    8,   # Minor swing high/low
    "SESSION_HIGH":  13,
    "SESSION_LOW":   13,
    "RESISTANCE":    12,
    "SUPPORT":       12,
}

# ─────────────────────────────────────────────
# RISK / REWARD
# ─────────────────────────────────────────────
MIN_RR_INTRADAY = 2.0    # Minimum 1:2 RR for intraday signals
MIN_RR_SWING    = 2.5    # Minimum 1:2.5 RR for swing signals

# ─────────────────────────────────────────────
# STOP LOSS
# ─────────────────────────────────────────────
ATR_BUFFER_MULTIPLIER = 0.2   # SL buffer = 0.2 × ATR14

# ─────────────────────────────────────────────
# VOLUME CONFIRMATION
# ─────────────────────────────────────────────
VOLUME_AVG_PERIOD         = 20     # Rolling average period
VOLUME_CONFIRM_MULTIPLIER = 1.0    # Confirmation candle volume > 1.0 × 20-period avg
VOLUME_STRICT_MODE        = True   # If False, volume filter is advisory only

# ─────────────────────────────────────────────
# TREND / ADX
# ─────────────────────────────────────────────
ADX_PERIOD             = 14
ADX_STRONG_TREND_THRESHOLD = 25   # ADX > 25 = trending
ADX_WEAK_TREND_THRESHOLD   = 15   # ADX < 15 = ranging/neutral

EMA_FAST_15M = 20
EMA_SLOW_15M = 50
EMA_FAST_1H  = 50
EMA_SLOW_1H  = 200
EMA_FAST_4H  = 50
EMA_SLOW_4H  = 200

# ─────────────────────────────────────────────
# SWING DETECTION
# Pivot method: left_bars candles before + right_bars confirmed after.
# right_bars MUST be confirmed (no look-ahead).
# ─────────────────────────────────────────────
SWING_LEFT_BARS_4H  = 3
SWING_RIGHT_BARS_4H = 2

SWING_LEFT_BARS_1H  = 3
SWING_RIGHT_BARS_1H = 2

SWING_LEFT_BARS_15M = 3
SWING_RIGHT_BARS_15M = 2

# Minimum swing size as ATR multiplier (prevents tiny fluctuations)
MIN_SWING_ATR_MULTIPLIER = 0.3

# ─────────────────────────────────────────────
# EQUAL HIGH/LOW DETECTION
# Two levels within this % of each other = equal
# ─────────────────────────────────────────────
EQUAL_LEVEL_TOLERANCE_PERCENT = 0.15   # 0.15%

# ─────────────────────────────────────────────
# MTF CONFLICT FILTER
# ─────────────────────────────────────────────
MTF_CONFLICT_REJECT = True   # True = reject on major MTF conflict

# Conflict classification thresholds
# Example: 4H strongly bearish (ADX > 30 + price below EMA200) + 1H bullish = MAJOR CONFLICT
CONFLICT_STRONG_ADX_THRESHOLD = 30   # ADX above this = "strong" trend for conflict purposes

# ─────────────────────────────────────────────
# BTC CORRELATION FILTER
# Applied to non-BTC coins only
# ─────────────────────────────────────────────
BTC_CORRELATION_FILTER_ENABLED = True

# If BTC 1H/4H is strongly bearish, block or downgrade ALT LONG signals
BTC_BLOCK_ALT_LONG_ON_STRONG_BEAR   = True
BTC_DOWNGRADE_ALT_LONG_ON_BEAR      = True   # If block=False, downgrade quality instead

# If BTC 1H/4H is strongly bullish, block or downgrade ALT SHORT signals
BTC_BLOCK_ALT_SHORT_ON_STRONG_BULL  = True
BTC_DOWNGRADE_ALT_SHORT_ON_BULL     = True

# ─────────────────────────────────────────────
# COUNTERTREND
# ─────────────────────────────────────────────
COUNTERTREND_ALLOWED = False   # Do NOT aggressively countertrade strong 4H trend

# ─────────────────────────────────────────────
# SIGNAL QUALITY GATES
# ─────────────────────────────────────────────
SEND_QUALITY_A_PLUS = True
SEND_QUALITY_A      = True
SEND_QUALITY_B      = False   # B-quality signals are tracked but NOT sent to Telegram
SEND_REJECTED       = False

# A+ requirements (all must pass)
A_PLUS_REQUIRE_MTF_ALIGNED    = True
A_PLUS_REQUIRE_RETEST         = True
A_PLUS_REQUIRE_VOLUME_CONFIRM = True
A_PLUS_MIN_SWEEP_SCORE        = 80
A_PLUS_MIN_RR_INTRADAY        = 2.5
A_PLUS_MIN_RR_SWING           = 3.0
A_PLUS_NO_MTF_CONFLICT        = True

# A requirements (relaxed from A+)
A_REQUIRE_RETEST              = False   # Retest preferred but not mandatory
A_REQUIRE_VOLUME_CONFIRM      = True
A_MIN_SWEEP_SCORE             = 70
A_MIN_RR_INTRADAY             = 2.0
A_MIN_RR_SWING                = 2.5

# ─────────────────────────────────────────────
# SIGNAL EXPIRY
# ─────────────────────────────────────────────
INTRADAY_EXPIRY_MINUTES = 30    # 15M signal expires after 30 min if entry not reached
SWING_EXPIRY_HOURS      = 12    # 1H swing signal expires after 12 hours

# ─────────────────────────────────────────────
# CHASE PROTECTION
# Reject entries if price is too far from the setup structure
# ─────────────────────────────────────────────
CHASE_PROTECTION_ATR_MULTIPLIER = 1.5   # If entry is > 1.5×ATR from structure, reject

# ─────────────────────────────────────────────
# LEARNING / DATA SUFFICIENCY
# ─────────────────────────────────────────────
MIN_SAMPLE_SIZE_FOR_RECOMMENDATION = 30   # Minimum completed trades before suggesting changes
MIN_SAMPLE_PERIOD_DAYS             = 14   # Minimum 2 weeks of data

# ─────────────────────────────────────────────
# RETEST CONFIRMATION
# After BOS, wait for price to return toward broken structure
# ─────────────────────────────────────────────
RETEST_TOLERANCE_ATR_MULTIPLIER = 0.5   # Retest counts if price comes within 0.5×ATR of level
RETEST_REQUIRED_INTRADAY        = False  # Preferred but configurable
RETEST_REQUIRED_SWING           = False  # Preferred but configurable
