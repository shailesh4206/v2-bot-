"""
database/journal.py — CRUD operations for the signal trade journal.

All operations go through this module.
No capital-dependent fields stored or returned.
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from database.models import get_connection
from config.settings import STRATEGY_VERSION
from config.strategy_config import INTRADAY_EXPIRY_MINUTES, SWING_EXPIRY_HOURS

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────
# Signal CRUD
# ─────────────────────────────────────────────

def save_signal(setup, signal_id: Optional[str] = None) -> str:
    """
    Persist a new signal to the database.

    Args:
        setup:     SetupResult from intraday or swing pipeline
        signal_id: Pre-generated signal ID (optional)

    Returns:
        signal_id string
    """
    if signal_id is None:
        signal_id = str(uuid.uuid4())[:8].upper()

    now = _now_iso()

    # Calculate expiry
    if setup.mode == "INTRADAY":
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=INTRADAY_EXPIRY_MINUTES)).isoformat()
    else:
        expiry = (datetime.now(timezone.utc) + timedelta(hours=SWING_EXPIRY_HOURS)).isoformat()

    conn = get_connection()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO signals (
                signal_id, symbol, direction, mode, timeframe, strategy_version,
                entry, sl, tp, rr, quality, sweep_score,
                liquidity_type, liquidity_level,
                h4_trend, h1_trend, m15_setup,
                choch_confirmed, bos_confirmed, retest_confirmed, volume_confirmed,
                mtf_status, state, signal_timestamp, expiry_timestamp,
                rejection_reason, is_paper
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
        """, (
            signal_id, setup.symbol, setup.direction, setup.mode, setup.timeframe, STRATEGY_VERSION,
            setup.entry, setup.sl, setup.tp, setup.rr,
            setup.quality.value, setup.sweep_score,
            setup.liquidity_type, setup.liquidity_level,
            setup.h4_summary, setup.h1_summary, setup.m15_summary,
            int(setup.choch_confirmed), int(setup.bos_confirmed),
            int(setup.retest_confirmed), int(setup.volume_confirmed),
            "ALIGNED", "SIGNAL_SENT", now, expiry,
            setup.rejection_reason or "", 0,
        ))
        conn.commit()
        logger.info(f"Signal saved: {signal_id} {setup.symbol} {setup.direction} {setup.mode}")
    except Exception as e:
        logger.error(f"Error saving signal {signal_id}: {e}")
    finally:
        conn.close()

    return signal_id


def update_signal_state(signal_id: str, new_state: str):
    """Update the lifecycle state of a signal."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE signals SET state = ? WHERE signal_id = ?",
            (new_state, signal_id)
        )
        conn.commit()
    finally:
        conn.close()


def record_outcome(
    signal_id:       str,
    outcome:         str,       # TP_HIT / SL_HIT / EXPIRED
    exit_price:      float,
    entry_timestamp: Optional[str] = None,
    exit_timestamp:  Optional[str] = None,
    r_multiple:      Optional[float] = None,
    notes:           str = "",
):
    """Record the trade outcome for a signal."""
    if exit_timestamp is None:
        exit_timestamp = _now_iso()

    conn = get_connection()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO signal_outcomes
            (signal_id, outcome, entry_timestamp, exit_timestamp, exit_price, r_multiple, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (signal_id, outcome, entry_timestamp, exit_timestamp, exit_price, r_multiple, notes))

        conn.execute("UPDATE signals SET state = ? WHERE signal_id = ?", (outcome, signal_id))
        conn.commit()
        logger.info(f"Outcome recorded: {signal_id} → {outcome} @ {exit_price} (R: {r_multiple})")
    finally:
        conn.close()


def get_signal(signal_id: str) -> Optional[Dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM signals WHERE signal_id = ?", (signal_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_active_signals() -> List[Dict]:
    """Get signals that are SIGNAL_SENT or ENTRY_REACHED and not yet resolved."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT s.*, o.outcome, o.entry_timestamp, o.exit_timestamp, o.exit_price, o.r_multiple
            FROM signals s
            LEFT JOIN signal_outcomes o ON s.signal_id = o.signal_id
            WHERE s.state IN ('SIGNAL_SENT', 'ENTRY_REACHED', 'ACTIVE')
            AND s.is_paper = 0
            ORDER BY s.signal_timestamp DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_completed_signals(
    start: Optional[str] = None,
    end:   Optional[str] = None,
    is_paper: bool = False,
) -> List[Dict]:
    """Get signals with resolved outcomes (TP_HIT, SL_HIT, EXPIRED)."""
    conn = get_connection()
    try:
        query = """
            SELECT s.*, o.outcome, o.entry_timestamp, o.exit_timestamp,
                   o.exit_price, o.r_multiple
            FROM signals s
            JOIN signal_outcomes o ON s.signal_id = o.signal_id
            WHERE o.outcome IN ('TP_HIT', 'SL_HIT', 'EXPIRED')
            AND s.is_paper = ?
        """
        params = [int(is_paper)]

        if start:
            query += " AND s.signal_timestamp >= ?"
            params.append(start)
        if end:
            query += " AND s.signal_timestamp <= ?"
            params.append(end)

        query += " ORDER BY s.signal_timestamp ASC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_weekly_signals(week_start: str, week_end: str, is_paper: bool = False) -> List[Dict]:
    """Get all signals (including sent/expired) for a given week."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT s.*, o.outcome, o.r_multiple
            FROM signals s
            LEFT JOIN signal_outcomes o ON s.signal_id = o.signal_id
            WHERE s.signal_timestamp >= ? AND s.signal_timestamp <= ?
            AND s.is_paper = ?
            ORDER BY s.signal_timestamp ASC
        """, (week_start, week_end, int(is_paper))).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_recent_signals(limit: int = 10, is_paper: bool = False) -> List[Dict]:
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT s.*, o.outcome, o.r_multiple
            FROM signals s
            LEFT JOIN signal_outcomes o ON s.signal_id = o.signal_id
            WHERE s.is_paper = ?
            ORDER BY s.signal_timestamp DESC
            LIMIT ?
        """, (int(is_paper), limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_weekly_report(week_start: str, week_end: str, report_text: str):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO weekly_reports (week_start, week_end, report_text, generated_at, sent_to_telegram)
            VALUES (?, ?, ?, ?, 0)
        """, (week_start, week_end, report_text, _now_iso()))
        conn.commit()
    finally:
        conn.close()


def get_latest_weekly_report() -> Optional[Dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM weekly_reports ORDER BY generated_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def save_recommendation(rec: Dict) -> str:
    rec_id = str(uuid.uuid4())[:12].upper()
    conn = get_connection()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO recommendations
            (rec_id, current_rule, observation, evidence, proposed_change,
             expected_benefit, possible_downside, trades_analyzed, confidence,
             backtest_required, paper_test_required, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_REVIEW', ?, ?)
        """, (
            rec_id,
            rec.get("current_rule", ""),
            rec.get("observation", ""),
            rec.get("evidence", ""),
            rec.get("proposed_change", ""),
            rec.get("expected_benefit", ""),
            rec.get("possible_downside", ""),
            rec.get("trades_analyzed", 0),
            rec.get("confidence", "LOW"),
            int(rec.get("backtest_required", True)),
            int(rec.get("paper_test_required", True)),
            _now_iso(), _now_iso(),
        ))
        conn.commit()
    finally:
        conn.close()
    return rec_id


def get_pending_recommendations() -> List[Dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM recommendations WHERE status = 'PENDING_REVIEW' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_recommendation_status(rec_id: str, status: str):
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE recommendations SET status = ?, updated_at = ? WHERE rec_id = ?",
            (status, _now_iso(), rec_id)
        )
        conn.commit()
    finally:
        conn.close()


def save_learning_observation(obs: Dict):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO learning_observations
            (metric, segment, value, sample_size, confidence_level, observation_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            obs.get("metric"), obs.get("segment"), obs.get("value"),
            obs.get("sample_size"), obs.get("confidence_level"),
            obs.get("observation_text"), _now_iso()
        ))
        conn.commit()
    finally:
        conn.close()


def get_recent_observations(limit: int = 20) -> List[Dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM learning_observations ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
