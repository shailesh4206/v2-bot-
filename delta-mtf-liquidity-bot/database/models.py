"""
database/models.py — SQLite schema for the Delta MTF Liquidity Signal Bot.

Tables:
  - signals          : All generated signals with full metadata
  - signal_outcomes  : Outcome tracking (entry, TP, SL, expiry)
  - strategy_versions: Version changelog
  - weekly_reports   : Stored weekly report text
  - learning_observations: Detected performance patterns
  - recommendations  : Strategy improvement suggestions
  - backtest_results : Backtesting run results

No capital-dependent fields anywhere.
"""

import sqlite3
import os
from config.settings import DATABASE_URL

# Strip "sqlite:///" prefix for raw sqlite3
DB_PATH = DATABASE_URL.replace("sqlite:///", "")


def get_connection():
    """Get a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    return conn


def initialize_database():
    """Create all tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ── signals ──────────────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id           TEXT    UNIQUE NOT NULL,
        symbol              TEXT    NOT NULL,
        direction           TEXT    NOT NULL,   -- LONG / SHORT
        mode                TEXT    NOT NULL,   -- INTRADAY / SWING
        timeframe           TEXT    NOT NULL,   -- 15m / 1h
        strategy_version    TEXT    NOT NULL,

        -- Price levels (no capital fields)
        entry               REAL,
        sl                  REAL,
        tp                  REAL,
        rr                  REAL,

        -- Signal quality
        quality             TEXT,              -- A+ / A / B / REJECTED
        sweep_score         INTEGER,

        -- Liquidity
        liquidity_type      TEXT,
        liquidity_level     REAL,

        -- Context
        h4_trend            TEXT,
        h1_trend            TEXT,
        m15_setup           TEXT,
        choch_confirmed     INTEGER,           -- 0/1
        bos_confirmed       INTEGER,
        retest_confirmed    INTEGER,
        volume_confirmed    INTEGER,
        btc_correlation     TEXT,
        mtf_status          TEXT,

        -- Lifecycle
        state               TEXT    DEFAULT 'DETECTED',
        signal_timestamp    TEXT    NOT NULL,
        expiry_timestamp    TEXT,
        rejection_reason    TEXT,

        -- Paper mode flag
        is_paper            INTEGER DEFAULT 0
    )
    """)

    # ── signal_outcomes ──────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS signal_outcomes (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id           TEXT    NOT NULL REFERENCES signals(signal_id),
        outcome             TEXT,              -- TP_HIT / SL_HIT / EXPIRED / ACTIVE
        entry_timestamp     TEXT,
        exit_timestamp      TEXT,
        exit_price          REAL,
        r_multiple          REAL,              -- +2.4 for TP at 2.4R, -1.0 for SL
        notes               TEXT
    )
    """)

    # ── strategy_versions ─────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_versions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        version          TEXT    NOT NULL,
        date_applied     TEXT    NOT NULL,
        change_summary   TEXT,
        reason           TEXT,
        backtest_result  TEXT,
        oos_result       TEXT,
        paper_result     TEXT,
        approval_status  TEXT    DEFAULT 'PENDING_REVIEW',
        approved_by      TEXT
    )
    """)

    # ── weekly_reports ─────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weekly_reports (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        week_start       TEXT    NOT NULL,
        week_end         TEXT    NOT NULL,
        report_text      TEXT,
        generated_at     TEXT,
        sent_to_telegram INTEGER DEFAULT 0
    )
    """)

    # ── learning_observations ─────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS learning_observations (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        metric           TEXT    NOT NULL,
        segment          TEXT,
        value            REAL,
        sample_size      INTEGER,
        confidence_level TEXT,
        observation_text TEXT,
        created_at       TEXT
    )
    """)

    # ── recommendations ──────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        rec_id              TEXT    UNIQUE NOT NULL,
        current_rule        TEXT,
        observation         TEXT,
        evidence            TEXT,
        proposed_change     TEXT,
        expected_benefit    TEXT,
        possible_downside   TEXT,
        trades_analyzed     INTEGER,
        confidence          TEXT,
        backtest_required   INTEGER DEFAULT 1,
        paper_test_required INTEGER DEFAULT 1,
        status              TEXT    DEFAULT 'PENDING_REVIEW',
        created_at          TEXT,
        updated_at          TEXT
    )
    """)

    # ── backtest_results ──────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS backtest_results (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           TEXT    UNIQUE NOT NULL,
        strategy_version TEXT,
        symbol           TEXT,
        start_date       TEXT,
        end_date         TEXT,
        split_type       TEXT,   -- IN_SAMPLE / OUT_OF_SAMPLE / FULL
        total_signals    INTEGER,
        valid_signals    INTEGER,
        total_entries    INTEGER,
        tp_hits          INTEGER,
        sl_hits          INTEGER,
        expired          INTEGER,
        win_rate         REAL,
        total_r          REAL,
        profit_factor    REAL,
        max_drawdown_r   REAL,
        max_consec_losses INTEGER,
        avg_rr           REAL,
        fee_r            REAL,
        slippage_r       REAL,
        net_r            REAL,
        equity_curve_json TEXT,
        created_at       TEXT
    )
    """)

    # ── duplicate_guard ──────────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS duplicate_guard (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        fingerprint      TEXT    UNIQUE NOT NULL,
        signal_id        TEXT,
        created_at       TEXT,
        expires_at       TEXT
    )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {DB_PATH}")
