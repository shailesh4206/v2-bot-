"""
strategy_config.py
Institution-Grade Strategy Configuration
Delta MTF Liquidity Signal Bot

IMPORTANT
---------
- Strategy parameters only.
- No order execution parameters are stored here.
- Hard gates must never be bypassed by scoring.
- Changes require human review + backtest + OOS + paper validation.

Strategy hierarchy
------------------
4H Macro Regime
    ↓
1H Structure
    ↓
Liquidity Pool
    ↓
Liquidity Sweep
    ↓
Displacement
    ↓
CHoCH
    ↓
BOS
    ↓
Retest / Acceptance
    ↓
15M Execution
    ↓
Volatility / RVOL / BTC Regime
    ↓
RR / Risk Geometry
    ↓
Quality Engine
    ↓
Final Validator
"""


# ============================================================
# STRATEGY VERSION
# ============================================================

STRATEGY_VERSION = "v2.0-institutional"
STRATEGY_DATE = "2026-10-01"


# ============================================================
# SUPPORTED MARKET / TIMEFRAMES
# ============================================================

SUPPORTED_TIMEFRAMES = ("4h", "1h", "15m")

MACRO_TIMEFRAME = "4h"
STRUCTURE_TIMEFRAME = "1h"
EXECUTION_TIMEFRAME = "15m"

FORBIDDEN_TIMEFRAMES = ("5m",)


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

MIN_SWEEP_SCORE = 70

SCORE_CANNOT_OVERRIDE_HARD_GATES = True


# ------------------------------------------------------------
# Sweep scoring
# ------------------------------------------------------------

SWEEP_SCORE_WEIGHTS = {
    "pool_significance": 20,
    "pool_touches": 10,
    "htf_4h_confluence": 10,
    "htf_1h_confluence": 10,
    "wick_rejection": 10,
    "volume_at_sweep": 10,
    "atr_context": 5,
    "displacement": 10,
    "choch_confirmed": 10,
    "bos_confirmed": 5,
}

SWEEP_SCORE_TOTAL = sum(SWEEP_SCORE_WEIGHTS.values())


POOL_SIGNIFICANCE_SCORES = {
    "PDH": 20,
    "PDL": 20,
    "EQUAL_HIGHS": 17,
    "EQUAL_LOWS": 17,
    "SWING_MAJOR": 15,
    "SWING_MINOR": 8,
    "SESSION_HIGH": 13,
    "SESSION_LOW": 13,
    "RESISTANCE": 12,
    "SUPPORT": 12,
}


# ------------------------------------------------------------
# Sweep hard gates
# ------------------------------------------------------------

SWEEP_REQUIRE_RECLAIM = True
SWEEP_REQUIRE_REJECTION = True
SWEEP_REQUIRE_MIN_DEPTH = True

# Compatibility aliases used by liquidity.sweep.py
SWEEP_RECLAIM_REQUIRED = SWEEP_REQUIRE_RECLAIM
SWEEP_REJECTION_REQUIRED = SWEEP_REQUIRE_REJECTION
SWEEP_MIN_DEPTH_REQUIRED = SWEEP_REQUIRE_MIN_DEPTH

SWEEP_MIN_DEPTH_ATR = 0.05
SWEEP_MAX_DEPTH_ATR = 1.00

SWEEP_MIN_WICK_TO_BODY = 1.0
SWEEP_MIN_BODY_ATR = 0.05

# Compatibility alias
SWEEP_MIN_WICK_BODY_RATIO = SWEEP_MIN_WICK_TO_BODY

SWEEP_REJECT_EXTREME_VOLATILITY = True


# ============================================================
# DISPLACEMENT
# ============================================================

DISPLACEMENT_REQUIRED = True

DISPLACEMENT_ATR_PERIOD = 14

DISPLACEMENT_MIN_BODY_ATR = 0.60

# Compatibility alias used by liquidity.sweep.py
DISPLACEMENT_ATR_MULTIPLIER = DISPLACEMENT_MIN_BODY_ATR

DISPLACEMENT_MIN_BODY_PERCENT = 55.0

# Compatibility alias
DISPLACEMENT_BODY_PERCENT = DISPLACEMENT_MIN_BODY_PERCENT

DISPLACEMENT_REQUIRE_VOLUME = False

# Compatibility alias
DISPLACEMENT_VOLUME_CONFIRM = DISPLACEMENT_REQUIRE_VOLUME

DISPLACEMENT_MIN_RVOL = 1.20

DISPLACEMENT_MAX_CANDLES_AFTER_SWEEP = 4


# ============================================================
# MARKET STRUCTURE — CHoCH
# ============================================================

CHOCH_REQUIRED = True

CHOCH_MUST_FOLLOW_SWEEP = True

CHOCH_BOS_SAME_BREAK_FORBIDDEN = True

CHOCH_REQUIRE_CLOSE_CONFIRMATION = True

CHOCH_REQUIRE_DISPLACEMENT = True

CHOCH_MAX_CANDLES_AFTER_SWEEP = 8


# ============================================================
# MARKET STRUCTURE — BOS
# ============================================================

BOS_REQUIRED = True

BOS_REQUIRE_CLOSE_CONFIRMATION = True
BOS_CLOSE_CONFIRMATION = BOS_REQUIRE_CLOSE_CONFIRMATION
BOS_MUST_FOLLOW_CHOCH = True

BOS_REQUIRE_DISPLACEMENT = True

BOS_MAX_CANDLES_AFTER_CHOCH = 8

BOS_REQUIRE_NEW_STRUCTURE_LEVEL = True


# ============================================================
# RETEST / ACCEPTANCE
# ============================================================

RETEST_REQUIRED_INTRADAY = True
RETEST_REQUIRED_SWING = True

RETEST_TOLERANCE_ATR_MULTIPLIER = 0.35

RETEST_REQUIRE_TOUCH = True

RETEST_MAX_CANDLES_AFTER_BOS_INTRADAY = 6
RETEST_MAX_CANDLES_AFTER_BOS_SWING = 12

RETEST_REQUIRE_REJECTION_OR_ACCEPTANCE = True

RETEST_REQUIRE_CLOSE_CONFIRMATION = True

RETEST_INVALIDATION_ATR = 0.30


# ============================================================
# RISK / REWARD
# ============================================================

MIN_RR_INTRADAY = 2.0
MIN_RR_SWING = 2.5

A_PLUS_MIN_RR_INTRADAY = 2.5
A_PLUS_MIN_RR_SWING = 3.0

MAX_REASONABLE_RR = 8.0


# ============================================================
# STOP LOSS
# ============================================================

ATR_BUFFER_MULTIPLIER = 0.20

MIN_SL_ATR_MULTIPLIER = 0.35

MAX_SL_ATR_MULTIPLIER_INTRADAY = 3.0
MAX_SL_ATR_MULTIPLIER_SWING = 5.0

SL_REQUIRE_STRUCTURE_INVALIDATION = True


# ============================================================
# TAKE PROFIT
# ============================================================

TP1_ENABLED = True
TP2_ENABLED = True

TP1_R_MULTIPLE = 1.5
TP2_MIN_R_MULTIPLE = 2.0

TP_REQUIRE_LIQUIDITY_TARGET = True


# ============================================================
# VOLUME / RVOL
# ============================================================

VOLUME_AVG_PERIOD = 20
VOLUME_MEDIAN_PERIOD = 20

VOLUME_CONFIRM_MULTIPLIER = 1.0

VOLUME_STRICT_MODE = True

RVOL_CONFIRM_MIN = 1.00
RVOL_STRONG_MIN = 1.50
RVOL_EXTREME_MIN = 2.50

EXTREME_VOLUME_IS_NOT_A_TRIGGER = True

VOLUME_DIRECTION_CONFIRMATION = True


# ============================================================
# TREND / ADX / DI
# ============================================================

ADX_PERIOD = 14

ADX_STRONG_TREND_THRESHOLD = 25
ADX_WEAK_TREND_THRESHOLD = 15
ADX_EXTREME_TREND_THRESHOLD = 35

DI_CONFIRMATION_ENABLED = True
DI_MIN_SEPARATION = 3.0


# ============================================================
# EMA SETTINGS
# ============================================================

EMA_FAST_15M = 20
EMA_SLOW_15M = 50

EMA_FAST_1H = 50
EMA_SLOW_1H = 200

EMA_FAST_4H = 50
EMA_SLOW_4H = 200

EMA_SLOPE_LOOKBACK = 5
EMA_SLOPE_MIN_PERCENT = 0.02


# ============================================================
# SWING DETECTION
# ============================================================

SWING_LEFT_BARS_4H = 3
SWING_RIGHT_BARS_4H = 2

SWING_LEFT_BARS_1H = 3
SWING_RIGHT_BARS_1H = 2

SWING_LEFT_BARS_15M = 3
SWING_RIGHT_BARS_15M = 2

MIN_SWING_ATR_MULTIPLIER = 0.30

STRUCTURE_SEQUENCE_REQUIRED = True

EQUAL_LEVEL_TOLERANCE_PERCENT = 0.15
EQUAL_LEVEL_TOLERANCE_ATR = 0.15

# Compatibility alias
EQUAL_LEVEL_TOLERANCE_PCT = EQUAL_LEVEL_TOLERANCE_PERCENT


# ============================================================
# MTF STRUCTURE
# ============================================================

STRICT_MTF_HIERARCHY = True

MTF_4H_MACRO_REQUIRED = True
MTF_1H_STRUCTURE_REQUIRED = True
MTF_15M_EXECUTION_REQUIRED = True

MTF_NEUTRAL_IS_NOT_ALIGNED = True

MTF_CONFLICT_REJECT = True

CONFLICT_STRONG_ADX_THRESHOLD = 30

COUNTERTREND_ALLOWED = False


# ============================================================
# BTC REGIME / CORRELATION
# ============================================================

BTC_CORRELATION_FILTER_ENABLED = True

BTC_REGIME_TIMEFRAMES = ("4h", "1h")

BTC_BLOCK_ALT_LONG_ON_STRONG_BEAR = True
BTC_DOWNGRADE_ALT_LONG_ON_BEAR = True

BTC_BLOCK_ALT_SHORT_ON_STRONG_BULL = True
BTC_DOWNGRADE_ALT_SHORT_ON_BULL = True

BTC_STRONG_TREND_ADX = 30

BTC_EXTREME_VOLATILITY_BLOCK_ALTS = True


# ============================================================
# VOLATILITY ENGINE
# ============================================================

ATR_PERIOD = 14

ATR_BASELINE_PERIOD = 50

# Compatibility alias
VOLATILITY_BASELINE_PERIOD = ATR_BASELINE_PERIOD

ATR_EXPANSION_LOOKBACK = 20

# Compatibility alias
VOLATILITY_EXPANSION_THRESHOLD = ATR_EXPANSION_LOOKBACK

ATR_EXPANSION_WARNING_RATIO = 1.50
ATR_EXPANSION_EXTREME_RATIO = 2.00

# Compatibility aliases
VOLATILITY_WARNING_MULTIPLIER = ATR_EXPANSION_WARNING_RATIO
VOLATILITY_EXTREME_MULTIPLIER = ATR_EXPANSION_EXTREME_RATIO


# ------------------------------------------------------------
# Candle range / ATR
# ------------------------------------------------------------

RANGE_ATR_HIGH = 2.50
RANGE_ATR_EXTREME = 3.50

# Compatibility aliases
VOLATILITY_RANGE_ATR_HIGH = RANGE_ATR_HIGH
VOLATILITY_RANGE_ATR_EXTREME = RANGE_ATR_EXTREME


# ------------------------------------------------------------
# Candle body / ATR
# ------------------------------------------------------------

BODY_ATR_EXTREME = 3.00

# Compatibility alias
VOLATILITY_BODY_ATR_EXTREME = BODY_ATR_EXTREME


# ------------------------------------------------------------
# Abnormal volatility
# ------------------------------------------------------------

ABNORMAL_VOLATILITY_FILTER_ENABLED = True

EXTREME_VOLATILITY_BLOCK_NEW_SIGNALS = True

# Compatibility aliases
ABNORMAL_VOLATILITY_ENABLED = ABNORMAL_VOLATILITY_FILTER_ENABLED
ABNORMAL_VOLATILITY_BLOCK_EXTREME = (
    EXTREME_VOLATILITY_BLOCK_NEW_SIGNALS
)


# ------------------------------------------------------------
# Spike detector
# ------------------------------------------------------------

SPIKE_DETECTOR_ENABLED = True

SPIKE_RANGE_ATR = 3.0
SPIKE_BODY_ATR = 2.0

SPIKE_COOLDOWN_CANDLES_15M = 2
SPIKE_COOLDOWN_CANDLES_1H = 1

# Compatibility aliases
VOLATILITY_SPIKE_DETECTOR_ENABLED = SPIKE_DETECTOR_ENABLED
VOLATILITY_SPIKE_RANGE_ATR = SPIKE_RANGE_ATR
VOLATILITY_SPIKE_BODY_ATR = SPIKE_BODY_ATR


# ============================================================
# CHASE PROTECTION
# ============================================================

CHASE_PROTECTION_ENABLED = True

CHASE_PROTECTION_ATR_MULTIPLIER = 1.5

CHASE_PROTECTION_15M_ATR = 1.25

ENTRY_MAX_DISTANCE_FROM_RETEST_ATR = 0.75


# ============================================================
# ENTRY VALIDATION
# ============================================================

ENTRY_REQUIRE_CLOSED_CANDLE = True

ENTRY_USE_CONFIRMATION_CLOSE = True

ENTRY_REJECT_IF_PRICE_ALREADY_EXTENDED = True

ENTRY_MAX_DISTANCE_FROM_BOS_ATR = 1.25

ENTRY_MAX_CONFIRMATION_EXTENSION_ATR = 1.00


# ============================================================
# SIGNAL QUALITY
# ============================================================

SEND_QUALITY_A_PLUS = True
SEND_QUALITY_A = True
SEND_QUALITY_B = False
SEND_REJECTED = False


# ------------------------------------------------------------
# A+ requirements
# ------------------------------------------------------------

A_PLUS_REQUIRE_MTF_ALIGNED = True
A_PLUS_REQUIRE_RETEST = True
A_PLUS_REQUIRE_VOLUME_CONFIRM = True

A_PLUS_MIN_SWEEP_SCORE = 80

A_PLUS_MIN_RR_INTRADAY = 2.5
A_PLUS_MIN_RR_SWING = 3.0

A_PLUS_NO_MTF_CONFLICT = True

A_PLUS_REQUIRE_DISPLACEMENT = True
A_PLUS_REQUIRE_CHOCH = True
A_PLUS_REQUIRE_BOS = True


# ------------------------------------------------------------
# A requirements
# ------------------------------------------------------------

A_REQUIRE_RETEST = True
A_REQUIRE_VOLUME_CONFIRM = True

A_MIN_SWEEP_SCORE = 70

A_MIN_RR_INTRADAY = 2.0
A_MIN_RR_SWING = 2.5

A_REQUIRE_DISPLACEMENT = True
A_REQUIRE_CHOCH = True
A_REQUIRE_BOS = True


# ------------------------------------------------------------
# B quality
# ------------------------------------------------------------

B_TRACK_FOR_ANALYSIS = True
B_SEND_TO_TELEGRAM = False


# ============================================================
# SIGNAL EXPIRY
# ============================================================

INTRADAY_EXPIRY_MINUTES = 30
SWING_EXPIRY_HOURS = 12

EXPIRY_FROM_SETUP_TIMESTAMP = True

INVALIDATE_ON_STRUCTURE_BREAK = True
INVALIDATE_ON_OPPOSITE_CHOCH = True
INVALIDATE_ON_BOS_FAILURE = True


# ============================================================
# DUPLICATE / COOLDOWN
# ============================================================

DUPLICATE_GUARD_ENABLED = True

DUPLICATE_INCLUDE_SETUP_CANDLE = True
DUPLICATE_INCLUDE_SWEEP_CANDLE = True
DUPLICATE_INCLUDE_LIQUIDITY_CLUSTER = True
DUPLICATE_INCLUDE_BOS = True
DUPLICATE_INCLUDE_STRATEGY_VERSION = True

INTRADAY_COOLDOWN_MINUTES = 30
SWING_COOLDOWN_HOURS = 12

SAME_SYMBOL_DIRECTION_COOLDOWN = True
SAME_LIQUIDITY_CLUSTER_COOLDOWN = True


# ============================================================
# CORRELATED EXPOSURE / SIGNAL CLUSTERING
# ============================================================

CORRELATED_EXPOSURE_FILTER_ENABLED = True

CORRELATED_SYMBOL_GROUPS = (
    ("BTCUSD", "ETHUSD", "SOLUSD"),
)

MAX_ACTIVE_SIGNALS_PER_CORRELATED_GROUP = 1

BLOCK_CORRELATED_ACTIVE_SIGNAL = True


# ============================================================
# LIQUIDITY POOL QUALITY
# ============================================================

LIQUIDITY_FRESHNESS_ENABLED = True

LIQUIDITY_MAX_AGE_CANDLES_1H = 100
LIQUIDITY_MAX_AGE_CANDLES_4H = 100

EQUAL_LEVEL_MIN_TOUCHES = 2

LIQUIDITY_TOUCH_SATURATION = 5


# ------------------------------------------------------------
# Compatibility aliases
# ------------------------------------------------------------

MIN_EQUAL_LEVEL_TOUCHES = EQUAL_LEVEL_MIN_TOUCHES

LIQUIDITY_FRESHNESS_BARS = LIQUIDITY_MAX_AGE_CANDLES_1H

LIQUIDITY_MAX_AGE_BARS = LIQUIDITY_MAX_AGE_CANDLES_1H

LIQUIDITY_SATURATION_TOUCHES = LIQUIDITY_TOUCH_SATURATION


# ============================================================
# DATA QUALITY
# ============================================================

DATA_QUALITY_GATE_ENABLED = True

MIN_CANDLES_4H = 250
MIN_CANDLES_1H = 250
MIN_CANDLES_15M = 250

REQUIRE_CONTINUOUS_CANDLES = True

REJECT_MISSING_CANDLES = True
REJECT_INVALID_OHLC = True
REJECT_FUTURE_CANDLES = True

REQUIRE_CONFIRMED_CLOSED_CANDLES = True


# ============================================================
# LEARNING / DATA SUFFICIENCY
# ============================================================

MIN_SAMPLE_SIZE_FOR_RECOMMENDATION = 30
MIN_SAMPLE_PERIOD_DAYS = 14

AUTO_APPLY_LEARNING_CHANGES = False

REQUIRE_HUMAN_APPROVAL_FOR_STRATEGY_CHANGE = True
REQUIRE_BACKTEST_BEFORE_STRATEGY_CHANGE = True
REQUIRE_OOS_BEFORE_STRATEGY_CHANGE = True
REQUIRE_PAPER_TEST_BEFORE_LIVE = True


# ============================================================
# MFE / MAE / OUTCOME TRACKING
# ============================================================

TRACK_MFE = True
TRACK_MAE = True
TRACK_MAX_R = True
TRACK_HOLD_TIME = True

TRACK_ENTRY_SLIPPAGE = True
TRACK_EXIT_SLIPPAGE = True

TRACK_CANDLE_PATH = True


# ============================================================
# PARTIAL TARGET / BREAKEVEN TRACKING
# ============================================================

BREAKEVEN_ENABLED = True

BREAKEVEN_AFTER_TP1 = True

BREAKEVEN_BUFFER_ATR = 0.05

TRAILING_ENABLED = False
TRAILING_AFTER_TP2 = False


# ============================================================
# EXECUTION SAFETY
# ============================================================

SIGNAL_ONLY_MODE = True

AUTO_ORDER_EXECUTION_ENABLED = False

REQUIRE_HUMAN_EXECUTION = True


# ============================================================
# SCORING / HARD-GATE POLICY
# ============================================================

HARD_GATES = (
    "data_quality",
    "closed_candle",
    "mtf_alignment",
    "liquidity_sweep",
    "displacement",
    "choch",
    "bos",
    "retest_or_acceptance",
    "volatility",
    "entry_geometry",
    "sl_geometry",
    "minimum_rr",
    "duplicate_guard",
)

SCORE_IS_SECONDARY_TO_HARD_GATES = True


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_strategy_config() -> None:
    """Validate configuration at startup."""

    # --------------------------------------------------------
    # Sweep
    # --------------------------------------------------------

    if not 0 <= MIN_SWEEP_SCORE <= 100:
        raise ValueError(
            "MIN_SWEEP_SCORE must be between 0 and 100"
        )

    if SWEEP_SCORE_TOTAL != 100:
        raise ValueError(
            f"SWEEP_SCORE_WEIGHTS must sum to 100, "
            f"got {SWEEP_SCORE_TOTAL}"
        )

    if SWEEP_MIN_DEPTH_ATR < 0:
        raise ValueError(
            "SWEEP_MIN_DEPTH_ATR cannot be negative"
        )

    if SWEEP_MAX_DEPTH_ATR <= SWEEP_MIN_DEPTH_ATR:
        raise ValueError(
            "SWEEP_MAX_DEPTH_ATR must be greater than "
            "SWEEP_MIN_DEPTH_ATR"
        )

    if SWEEP_MIN_WICK_TO_BODY < 0:
        raise ValueError(
            "SWEEP_MIN_WICK_TO_BODY cannot be negative"
        )

    # --------------------------------------------------------
    # Displacement
    # --------------------------------------------------------

    if DISPLACEMENT_MIN_BODY_ATR <= 0:
        raise ValueError(
            "DISPLACEMENT_MIN_BODY_ATR must be positive"
        )

    if DISPLACEMENT_ATR_MULTIPLIER <= 0:
        raise ValueError(
            "DISPLACEMENT_ATR_MULTIPLIER must be positive"
        )

    if not 0 < DISPLACEMENT_MIN_BODY_PERCENT <= 100:
        raise ValueError(
            "DISPLACEMENT_MIN_BODY_PERCENT must be between "
            "0 and 100"
        )

    if DISPLACEMENT_MAX_CANDLES_AFTER_SWEEP < 1:
        raise ValueError(
            "DISPLACEMENT_MAX_CANDLES_AFTER_SWEEP must be >= 1"
        )

    # --------------------------------------------------------
    # RR
    # --------------------------------------------------------

    if MIN_RR_INTRADAY <= 0 or MIN_RR_SWING <= 0:
        raise ValueError(
            "Minimum RR values must be positive"
        )

    if MIN_RR_SWING < MIN_RR_INTRADAY:
        raise ValueError(
            "Swing minimum RR should not be lower "
            "than intraday minimum RR"
        )

    if MAX_REASONABLE_RR < MIN_RR_INTRADAY:
        raise ValueError(
            "MAX_REASONABLE_RR must be >= minimum intraday RR"
        )

    # --------------------------------------------------------
    # Stop loss
    # --------------------------------------------------------

    if ATR_BUFFER_MULTIPLIER <= 0:
        raise ValueError(
            "ATR_BUFFER_MULTIPLIER must be positive"
        )

    if MIN_SL_ATR_MULTIPLIER <= 0:
        raise ValueError(
            "MIN_SL_ATR_MULTIPLIER must be positive"
        )

    if MAX_SL_ATR_MULTIPLIER_INTRADAY < MIN_SL_ATR_MULTIPLIER:
        raise ValueError(
            "Intraday maximum SL must be >= minimum SL"
        )

    if MAX_SL_ATR_MULTIPLIER_SWING < MIN_SL_ATR_MULTIPLIER:
        raise ValueError(
            "Swing maximum SL must be >= minimum SL"
        )

    # --------------------------------------------------------
    # Take profit
    # --------------------------------------------------------

    if TP1_R_MULTIPLE <= 0:
        raise ValueError(
            "TP1_R_MULTIPLE must be positive"
        )

    if TP2_MIN_R_MULTIPLE <= TP1_R_MULTIPLE:
        raise ValueError(
            "TP2_MIN_R_MULTIPLE must be greater "
            "than TP1_R_MULTIPLE"
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if VOLUME_AVG_PERIOD < 5:
        raise ValueError(
            "VOLUME_AVG_PERIOD is too small"
        )

    if VOLUME_MEDIAN_PERIOD < 5:
        raise ValueError(
            "VOLUME_MEDIAN_PERIOD is too small"
        )

    if RVOL_CONFIRM_MIN <= 0:
        raise ValueError(
            "RVOL_CONFIRM_MIN must be positive"
        )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    if ADX_PERIOD < 5:
        raise ValueError(
            "ADX_PERIOD is too small"
        )

    if ADX_WEAK_TREND_THRESHOLD < 0:
        raise ValueError(
            "ADX_WEAK_TREND_THRESHOLD cannot be negative"
        )

    if ADX_STRONG_TREND_THRESHOLD < ADX_WEAK_TREND_THRESHOLD:
        raise ValueError(
            "ADX_STRONG_TREND_THRESHOLD must be >= "
            "ADX_WEAK_TREND_THRESHOLD"
        )

    if ADX_EXTREME_TREND_THRESHOLD < ADX_STRONG_TREND_THRESHOLD:
        raise ValueError(
            "ADX_EXTREME_TREND_THRESHOLD must be >= "
            "ADX_STRONG_TREND_THRESHOLD"
        )

    # --------------------------------------------------------
    # Retest
    # --------------------------------------------------------

    if RETEST_TOLERANCE_ATR_MULTIPLIER <= 0:
        raise ValueError(
            "RETEST_TOLERANCE_ATR_MULTIPLIER must be positive"
        )

    if RETEST_MAX_CANDLES_AFTER_BOS_INTRADAY < 1:
        raise ValueError(
            "Intraday retest window must be >= 1"
        )

    if RETEST_MAX_CANDLES_AFTER_BOS_SWING < 1:
        raise ValueError(
            "Swing retest window must be >= 1"
        )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    if ATR_PERIOD < 2:
        raise ValueError(
            "ATR_PERIOD must be >= 2"
        )

    if ATR_BASELINE_PERIOD < ATR_PERIOD:
        raise ValueError(
            "ATR_BASELINE_PERIOD should be >= ATR_PERIOD"
        )

    if ATR_EXPANSION_LOOKBACK < 5:
        raise ValueError(
            "ATR_EXPANSION_LOOKBACK is too small"
        )

    if ATR_EXPANSION_WARNING_RATIO <= 0:
        raise ValueError(
            "ATR_EXPANSION_WARNING_RATIO must be positive"
        )

    if ATR_EXPANSION_EXTREME_RATIO <= ATR_EXPANSION_WARNING_RATIO:
        raise ValueError(
            "ATR_EXPANSION_EXTREME_RATIO must be greater "
            "than warning ratio"
        )

    if RANGE_ATR_EXTREME < RANGE_ATR_HIGH:
        raise ValueError(
            "RANGE_ATR_EXTREME must be >= RANGE_ATR_HIGH"
        )

    if BODY_ATR_EXTREME <= 0:
        raise ValueError(
            "BODY_ATR_EXTREME must be positive"
        )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    if MIN_CANDLES_4H < 100:
        raise ValueError(
            "MIN_CANDLES_4H is too small"
        )

    if MIN_CANDLES_1H < 100:
        raise ValueError(
            "MIN_CANDLES_1H is too small"
        )

    if MIN_CANDLES_15M < 100:
        raise ValueError(
            "MIN_CANDLES_15M is too small"
        )

    # --------------------------------------------------------
    # Timeframes
    # --------------------------------------------------------

    if not SUPPORTED_TIMEFRAMES:
        raise ValueError(
            "SUPPORTED_TIMEFRAMES cannot be empty"
        )

    if "5m" in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            "5m is explicitly forbidden"
        )

    if MACRO_TIMEFRAME not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            "MACRO_TIMEFRAME must be supported"
        )

    if STRUCTURE_TIMEFRAME not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            "STRUCTURE_TIMEFRAME must be supported"
        )

    if EXECUTION_TIMEFRAME not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            "EXECUTION_TIMEFRAME must be supported"
        )

    # --------------------------------------------------------
    # Learning safety
    # --------------------------------------------------------

    if AUTO_APPLY_LEARNING_CHANGES:
        raise ValueError(
            "AUTO_APPLY_LEARNING_CHANGES must remain False"
        )

    # --------------------------------------------------------
    # Execution safety
    # --------------------------------------------------------

    if AUTO_ORDER_EXECUTION_ENABLED:
        raise ValueError(
            "AUTO_ORDER_EXECUTION_ENABLED must remain False"
        )

    if not SIGNAL_ONLY_MODE:
        raise ValueError(
            "SIGNAL_ONLY_MODE must remain True"
        )

    if not REQUIRE_HUMAN_EXECUTION:
        raise ValueError(
            "REQUIRE_HUMAN_EXECUTION must remain True"
        )

    # --------------------------------------------------------
    # Hard-gate safety
    # --------------------------------------------------------

    if not SCORE_CANNOT_OVERRIDE_HARD_GATES:
        raise ValueError(
            "Score must never override hard gates"
        )

    if not SCORE_IS_SECONDARY_TO_HARD_GATES:
        raise ValueError(
            "Score must remain secondary to hard gates"
        )

    # --------------------------------------------------------
    # Correlated exposure
    # --------------------------------------------------------

    if MAX_ACTIVE_SIGNALS_PER_CORRELATED_GROUP < 1:
        raise ValueError(
            "MAX_ACTIVE_SIGNALS_PER_CORRELATED_GROUP must be >= 1"
        )

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    if EQUAL_LEVEL_MIN_TOUCHES < 1:
        raise ValueError(
            "EQUAL_LEVEL_MIN_TOUCHES must be >= 1"
        )

    if LIQUIDITY_TOUCH_SATURATION < EQUAL_LEVEL_MIN_TOUCHES:
        raise ValueError(
            "LIQUIDITY_TOUCH_SATURATION must be >= "
            "EQUAL_LEVEL_MIN_TOUCHES"
        )

    # --------------------------------------------------------
    # Institutional safety
    # --------------------------------------------------------

    if not DISPLACEMENT_REQUIRED:
        raise ValueError(
            "Institutional mode requires displacement"
        )

    if not CHOCH_REQUIRED:
        raise ValueError(
            "Institutional mode requires CHoCH"
        )

    if not BOS_REQUIRED:
        raise ValueError(
            "Institutional mode requires BOS"
        )

    if not RETEST_REQUIRED_INTRADAY:
        raise ValueError(
            "Institutional intraday mode requires retest"
        )

    if not RETEST_REQUIRED_SWING:
        raise ValueError(
            "Institutional swing mode requires retest"
        )


# ============================================================
# RUN VALIDATION ON IMPORT
# ============================================================

validate_strategy_config()


# ============================================================
# OPTIONAL SELF-TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("INSTITUTIONAL STRATEGY CONFIG")
    print("=" * 60)

    print(f"Strategy Version : {STRATEGY_VERSION}")
    print(f"Strategy Date    : {STRATEGY_DATE}")

    print(f"Timeframes       : {SUPPORTED_TIMEFRAMES}")

    print(f"Macro TF         : {MACRO_TIMEFRAME}")
    print(f"Structure TF     : {STRUCTURE_TIMEFRAME}")
    print(f"Execution TF     : {EXECUTION_TIMEFRAME}")

    print(f"Min Sweep Score  : {MIN_SWEEP_SCORE}")

    print(f"Min RR Intraday  : {MIN_RR_INTRADAY}")
    print(f"Min RR Swing     : {MIN_RR_SWING}")

    print(f"ATR Period       : {ATR_PERIOD}")
    print(f"ATR Baseline     : {ATR_BASELINE_PERIOD}")

    print(f"Vol Baseline     : {VOLATILITY_BASELINE_PERIOD}")

    print(f"ADX Period       : {ADX_PERIOD}")

    print(f"Signal Only      : {SIGNAL_ONLY_MODE}")
    print(f"Auto Execution   : {AUTO_ORDER_EXECUTION_ENABLED}")

    print("-" * 60)

    print(f"Sweep Reclaim     : {SWEEP_RECLAIM_REQUIRED}")
    print(f"Sweep Rejection   : {SWEEP_REJECTION_REQUIRED}")
    print(f"Sweep Min Depth   : {SWEEP_MIN_DEPTH_REQUIRED}")

    print(f"Displacement      : {DISPLACEMENT_REQUIRED}")
    print(f"Displacement ATR  : {DISPLACEMENT_ATR_MULTIPLIER}")

    print(f"CHoCH             : {CHOCH_REQUIRED}")
    print(f"BOS               : {BOS_REQUIRED}")

    print(f"Retest Intraday   : {RETEST_REQUIRED_INTRADAY}")
    print(f"Retest Swing      : {RETEST_REQUIRED_SWING}")

    print(f"Spike Detector    : {SPIKE_DETECTOR_ENABLED}")

    print("=" * 60)
    print("CONFIG VALIDATION: PASS")
    print("=" * 60)