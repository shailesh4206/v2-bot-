"""
learning/recommendations.py — Strategy improvement recommendation engine.

Generates structured recommendations for human review.
NEVER applies changes automatically.
Human must approve through the versioning system.

Each recommendation includes:
1. Current rule
2. Observed problem
3. Evidence
4. Proposed change
5. Expected benefit
6. Possible downside
7. Trades analyzed
8. Confidence
9. Backtest required?
10. Paper test required?
"""

import logging
from typing import List, Dict
from learning.analyzer import analyze_performance
from database.journal import save_recommendation, get_completed_signals
from config.strategy_config import (
    MIN_SAMPLE_SIZE_FOR_RECOMMENDATION,
    MIN_SAMPLE_PERIOD_DAYS,
    MIN_SWEEP_SCORE,
    MIN_RR_INTRADAY,
    MIN_RR_SWING,
)
from config.settings import STRATEGY_VERSION

logger = logging.getLogger(__name__)


def generate_recommendations() -> List[Dict]:
    """
    Analyze performance data and generate strategy improvement recommendations.

    Returns list of recommendation dicts (also saved to database).
    DOES NOT apply any changes.
    """
    analysis = analyze_performance()
    n = analysis.get("sample_size", 0)

    if not analysis.get("sufficient_data", False):
        logger.info(f"Insufficient data for recommendations: {n} signals (need {MIN_SAMPLE_SIZE_FOR_RECOMMENDATION})")
        return []

    recommendations = []

    # ── 1. Sweep Score Threshold ───────────────────────────────────────────
    ss_data = analysis.get("by_sweep_score", {})
    low_bucket  = ss_data.get("70-74", {})
    high_bucket = ss_data.get("80-84", {})

    if (
        low_bucket.get("count", 0) >= 10 and high_bucket.get("count", 0) >= 10
        and low_bucket.get("win_rate", 100) < high_bucket.get("win_rate", 0) - 10
    ):
        rec = {
            "current_rule":      f"MIN_SWEEP_SCORE = {MIN_SWEEP_SCORE}",
            "observation":       f"Sweep scores 70–74 had {low_bucket['win_rate']:.1f}% win rate vs {high_bucket.get('win_rate',0):.1f}% for 80–84",
            "evidence":          f"Low bucket: {low_bucket['count']} trades | High bucket: {high_bucket['count']} trades",
            "proposed_change":   "Raise MIN_SWEEP_SCORE from 70 to 75",
            "expected_benefit":  "Fewer low-quality signals, potentially higher win rate",
            "possible_downside": "Fewer total signals, some valid setups may be missed",
            "trades_analyzed":   n,
            "confidence":        "MEDIUM" if n >= 50 else "LOW",
            "backtest_required": True,
            "paper_test_required": True,
        }
        recommendations.append(rec)
        logger.info("Recommendation generated: Raise sweep score threshold")

    # ── 2. Direction Imbalance ─────────────────────────────────────────────
    dir_data = analysis.get("by_direction", {})
    long_wr   = dir_data.get("LONG",  {}).get("win_rate", 50)
    short_wr  = dir_data.get("SHORT", {}).get("win_rate", 50)
    long_n    = dir_data.get("LONG",  {}).get("count", 0)
    short_n   = dir_data.get("SHORT", {}).get("count", 0)

    if long_n >= 15 and short_n >= 15 and abs(long_wr - short_wr) > 20:
        weaker    = "SHORT" if short_wr < long_wr else "LONG"
        stronger  = "LONG"  if weaker == "SHORT"  else "SHORT"
        weaker_wr = short_wr if weaker == "SHORT" else long_wr
        rec = {
            "current_rule":      f"Both LONG and SHORT enabled equally",
            "observation":       f"{weaker} signals had {weaker_wr:.1f}% win rate vs {stronger} signals",
            "evidence":          f"LONG: {long_n} trades ({long_wr:.1f}% WR) | SHORT: {short_n} trades ({short_wr:.1f}% WR)",
            "proposed_change":   f"Consider adding additional filter for {weaker} signals or raising quality threshold",
            "expected_benefit":  "Reduce losing trades in weaker direction",
            "possible_downside": "May miss reversals in weaker direction",
            "trades_analyzed":   n,
            "confidence":        "MEDIUM",
            "backtest_required": True,
            "paper_test_required": True,
        }
        recommendations.append(rec)

    # ── 3. Volume Confirmation Value ───────────────────────────────────────
    vol_data = analysis.get("by_volume", {})
    vol_true  = vol_data.get("volume_confirmed_TRUE",  {})
    vol_false = vol_data.get("volume_confirmed_FALSE", {})

    if (
        vol_true.get("count", 0) >= 10 and vol_false.get("count", 0) >= 10
        and vol_true.get("win_rate", 0) > vol_false.get("win_rate", 0) + 10
    ):
        rec = {
            "current_rule":      "VOLUME_STRICT_MODE = True (volume confirmation required)",
            "observation":       f"Volume-confirmed signals: {vol_true['win_rate']:.1f}% WR | Non-confirmed: {vol_false['win_rate']:.1f}% WR",
            "evidence":          f"{vol_true['count']} confirmed | {vol_false['count']} unconfirmed trades",
            "proposed_change":   "Maintain strict volume confirmation — data supports current rule",
            "expected_benefit":  "Existing rule appears effective",
            "possible_downside": "N/A",
            "trades_analyzed":   n,
            "confidence":        "MEDIUM",
            "backtest_required": False,
            "paper_test_required": False,
        }
        recommendations.append(rec)

    # ── 4. Retest Performance ───────────────────────────────────────────────
    ret_data  = analysis.get("by_retest", {})
    ret_true  = ret_data.get("retest_confirmed_TRUE",  {})
    ret_false = ret_data.get("retest_confirmed_FALSE", {})

    if (
        ret_true.get("count", 0) >= 10 and ret_false.get("count", 0) >= 10
        and ret_true.get("win_rate", 0) > ret_false.get("win_rate", 0) + 15
    ):
        rec = {
            "current_rule":      "RETEST_REQUIRED_INTRADAY = False (preferred but not required)",
            "observation":       f"Retested setups: {ret_true['win_rate']:.1f}% WR | No retest: {ret_false['win_rate']:.1f}% WR",
            "evidence":          f"{ret_true['count']} with retest | {ret_false['count']} without",
            "proposed_change":   "Consider setting RETEST_REQUIRED_INTRADAY = True to filter non-retested setups",
            "expected_benefit":  "Higher quality entries with price confirmation",
            "possible_downside": "Fewer signals, some valid fast-moves may be missed",
            "trades_analyzed":   n,
            "confidence":        "MEDIUM",
            "backtest_required": True,
            "paper_test_required": True,
        }
        recommendations.append(rec)

    # ── Save all to DB ─────────────────────────────────────────────────────
    saved_ids = []
    for rec in recommendations:
        rec_id = save_recommendation(rec)
        saved_ids.append(rec_id)
        logger.info(f"Recommendation saved: {rec_id}")

    logger.info(f"Generated {len(recommendations)} recommendations from {n} trades")
    return recommendations
