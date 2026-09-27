"""
learning/versioning.py — Strategy version management with human approval workflow.

Version lifecycle:
  PENDING_REVIEW → APPROVED_FOR_BACKTEST → BACKTEST_PASSED/FAILED
  → APPROVED_FOR_PAPER_TEST → PAPER_TEST_PASSED/FAILED
  → APPROVED_FOR_LIVE_VERSION

The bot NEVER advances a version automatically.
All state transitions require human action.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List
from database.models import get_connection
from config.settings import STRATEGY_VERSION

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_current_version() -> str:
    return STRATEGY_VERSION


def record_version(
    version:        str,
    change_summary: str,
    reason:         str,
    approved_by:    str = "HUMAN",
) -> int:
    """Record a new strategy version entry in the database."""
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO strategy_versions
            (version, date_applied, change_summary, reason, approval_status, approved_by)
            VALUES (?, ?, ?, ?, 'PENDING_REVIEW', ?)
        """, (version, _now_iso(), change_summary, reason, approved_by))
        conn.commit()
        vid = cursor.lastrowid
        logger.info(f"Version {version} recorded (ID={vid})")
        return vid
    finally:
        conn.close()


def update_version_status(version: str, status: str, result_notes: str = ""):
    """Update the approval status of a version."""
    valid_statuses = [
        "PENDING_REVIEW",
        "APPROVED_FOR_BACKTEST",
        "REJECTED",
        "BACKTEST_PASSED",
        "BACKTEST_FAILED",
        "APPROVED_FOR_PAPER_TEST",
        "PAPER_TEST_PASSED",
        "PAPER_TEST_FAILED",
        "APPROVED_FOR_LIVE_VERSION",
    ]
    if status not in valid_statuses:
        logger.error(f"Invalid version status: {status}")
        return

    conn = get_connection()
    try:
        field_map = {
            "BACKTEST_PASSED": "backtest_result",
            "BACKTEST_FAILED": "backtest_result",
            "PAPER_TEST_PASSED": "paper_result",
            "PAPER_TEST_FAILED": "paper_result",
        }
        field = field_map.get(status)
        if field and result_notes:
            conn.execute(
                f"UPDATE strategy_versions SET approval_status = ?, {field} = ? WHERE version = ?",
                (status, result_notes, version)
            )
        else:
            conn.execute(
                "UPDATE strategy_versions SET approval_status = ? WHERE version = ?",
                (status, version)
            )
        conn.commit()
        logger.info(f"Version {version} status updated to {status}")
    finally:
        conn.close()


def get_version_history() -> List[Dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM strategy_versions ORDER BY date_applied DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def format_version_history() -> str:
    history = get_version_history()
    if not history:
        return f"Current version: {STRATEGY_VERSION}\nNo version history yet."

    lines = [f"📦 STRATEGY VERSION HISTORY\n\nCurrent: {STRATEGY_VERSION}\n"]
    for v in history[:10]:
        lines.append(
            f"v{v['version']} ({v['date_applied'][:10]})\n"
            f"  Status: {v['approval_status']}\n"
            f"  Change: {v['change_summary']}\n"
        )
    lines.append("\n⚠️ All version changes require human approval.")
    return "\n".join(lines)
