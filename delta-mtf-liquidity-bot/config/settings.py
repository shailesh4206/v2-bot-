"""
settings.py — Global bot configuration for Delta Exchange India MTF Liquidity Signal Bot.

ALL parameters are configurable here.
NO capital, balance, or position-size logic anywhere.
NO 5M timeframe anywhere.
Signal-only bot — does NOT place orders.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# DELTA EXCHANGE API
# ─────────────────────────────────────────────
USE_TESTNET = False

DELTA_BASE_URL_LIVE = "https://api.india.delta.exchange"
DELTA_BASE_URL_TEST = "https://cdn-ind.testnet.deltaex.org"

DELTA_BASE_URL = DELTA_BASE_URL_TEST if USE_TESTNET else DELTA_BASE_URL_LIVE

DELTA_API_KEY    = os.getenv("DELTA_API_KEY", "")
DELTA_API_SECRET = os.getenv("DELTA_API_SECRET", "")

# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ─────────────────────────────────────────────
# TIMEFRAMES — 4H + 1H + 15M ONLY. NO 5M.
# ─────────────────────────────────────────────
TIMEFRAMES = {
    "4h":  "4h",
    "1h":  "1h",
    "15m": "15m",
}

# Candles to fetch for live analysis (most recent N candles per TF)
CANDLES_LIMIT = {
    "4h":  200,
    "1h":  200,
    "15m": 200,
}

# ─────────────────────────────────────────────
# MONITORED SYMBOLS
# Delta Exchange India perpetual contract symbols
# Add more symbols here — no code changes needed.
# ─────────────────────────────────────────────
MONITORED_SYMBOLS = [
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
]

# BTC symbol — used for correlation filter
BTC_SYMBOL = "BTCUSD"

# ─────────────────────────────────────────────
# STRATEGY VERSION
# ─────────────────────────────────────────────
STRATEGY_VERSION = "v1.0"

# ─────────────────────────────────────────────
# STRATEGY MODES
# ─────────────────────────────────────────────
STRATEGY_MODES = ["INTRADAY", "SWING"]

INTRADAY_ENABLED = True
SWING_ENABLED    = True

# ─────────────────────────────────────────────
# OPERATIONAL MODES
# ─────────────────────────────────────────────
PAPER_MODE       = False   # True = use real data, no orders, track paper results
LIVE_SIGNAL_MODE = True    # True = monitor live market, send Telegram signals

# ─────────────────────────────────────────────
# SCAN INTERVAL
# Bot rescans on each confirmed 15M candle close.
# Polling interval in seconds (slightly after the 15m boundary).
# ─────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 300  # 5 minutes

# ─────────────────────────────────────────────
# WEEKLY REPORT SCHEDULE
# Day: 0=Monday, hour/minute in IST
# ─────────────────────────────────────────────
WEEKLY_REPORT_DAY    = 0   # Monday
WEEKLY_REPORT_HOUR   = 9
WEEKLY_REPORT_MINUTE = 0

# ─────────────────────────────────────────────
# LIQUIDITY FILTER (minimum requirements)
# ─────────────────────────────────────────────
MIN_24H_VOLUME_USD   = 1_000_000   # $1M minimum daily volume
MAX_SPREAD_PERCENT   = 0.1          # 0.1% max spread

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_DIR    = "logs"
LOG_FILE   = "logs/bot.log"

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DATABASE_URL = "sqlite:///delta_bot.db"

# ─────────────────────────────────────────────
# BACKTEST DEFAULTS
# ─────────────────────────────────────────────
BACKTEST_START         = "2024-01-01"
BACKTEST_END           = "2024-12-31"
BACKTEST_DEV_SPLIT     = 0.70   # 70% development, 30% out-of-sample
BACKTEST_FEE_PERCENT   = 0.05   # 0.05% per side (maker fee)
BACKTEST_SLIPPAGE_PERCENT = 0.02  # 0.02% slippage

# ─────────────────────────────────────────────
# RATE LIMITING
# Delta Exchange: candle endpoint weight = 3
# ─────────────────────────────────────────────
API_REQUESTS_PER_MINUTE = 20
API_RETRY_ATTEMPTS      = 3
API_RETRY_BACKOFF       = 2.0   # seconds, doubles each retry
