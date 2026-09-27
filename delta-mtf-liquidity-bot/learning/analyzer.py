"""
learning/analyzer.py — Performance analysis and pattern detection.

Analyzes completed signal data to identify:
- Best/worst coins, timeframes, directions
- Sweep score performance
- Liquidity type performance
- CHoCH, BOS, retest, volume performance
- MTF alignment effects
- BTC correlation effects

All analysis is statistical and deterministic.
NO causal claims from simple correlation.
Results feed into recommendations.py.
"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict

from database.journal import get_completed_signals
from config.strategy_config import MIN_SAMPLE_SIZE_FOR_RECOMMENDATION

logger = logging.getLogger(__name__)


def analyze_performance(
    start:     Optional[str] = None,
    end:       Optional[str] = None,
    is_paper:  bool = False,
) -> Dict:
    """
    Full performance analysis of completed signals.

    Returns a comprehensive dict of breakdowns.
    """
    signals = get_completed_signals(start=start, end=end, is_paper=is_paper)
    n = len(signals)

    if n == 0:
        return {"error": "No completed signals to analyze", "sample_size": 0}

    logger.info(f"Analyzing {n} completed signals …")

    return {
        "sample_size":    n,
        "overall":        _calc_overall(signals),
        "by_symbol":      _breakdown(signals, "symbol"),
        "by_mode":        _breakdown(signals, "mode"),
        "by_direction":   _breakdown(signals, "direction"),
        "by_quality":     _breakdown(signals, "quality"),
        "by_liquidity":   _breakdown(signals, "liquidity_type"),
        "by_choch":       _breakdown_bool(signals, "choch_confirmed"),
        "by_bos":         _breakdown_bool(signals, "bos_confirmed"),
        "by_retest":      _breakdown_bool(signals, "retest_confirmed"),
        "by_volume":      _breakdown_bool(signals, "volume_confirmed"),
        "by_sweep_score": _sweep_score_breakdown(signals),
        "by_mtf_status":  _breakdown(signals, "mtf_status"),
        "sufficient_data": n >= MIN_SAMPLE_SIZE_FOR_RECOMMENDATION,
    }


def _calc_overall(signals: List[Dict]) -> Dict:
    tp = [s for s in signals if s.get("outcome") == "TP_HIT"]
    sl = [s for s in signals if s.get("outcome") == "SL_HIT"]
    ex = [s for s in signals if s.get("outcome") == "EXPIRED"]

    entries = len(tp) + len(sl)
    r_vals  = [s["r_multiple"] for s in signals if s.get("r_multiple") is not None]

    win_rate = len(tp) / entries * 100 if entries > 0 else 0.0
    total_r  = sum(r_vals)
    avg_r    = total_r / len(r_vals) if r_vals else 0.0
    avg_rr   = _avg_rr(signals)

    pos_r = sum(r for r in r_vals if r > 0)
    neg_r = abs(sum(r for r in r_vals if r < 0))
    pf    = pos_r / neg_r if neg_r > 0 else float("inf")

    max_drawdown = _max_drawdown(r_vals)
    max_consec_l = _max_consecutive(signals, "SL_HIT")
    max_consec_w = _max_consecutive(signals, "TP_HIT")

    return {
        "total":          len(signals),
        "entries":        entries,
        "tp_hits":        len(tp),
        "sl_hits":        len(sl),
        "expired":        len(ex),
        "win_rate":       round(win_rate, 2),
        "total_r":        round(total_r, 2),
        "avg_r":          round(avg_r, 2),
        "avg_rr":         round(avg_rr, 2),
        "profit_factor":  round(pf, 2) if pf != float("inf") else None,
        "max_drawdown_r": round(max_drawdown, 2),
        "max_consec_losses": max_consec_l,
        "max_consec_wins":   max_consec_w,
    }


def _breakdown(signals: List[Dict], field: str) -> Dict:
    """Group signals by a field and compute stats per group."""
    groups: Dict[str, List] = defaultdict(list)
    for s in signals:
        key = str(s.get(field, "UNKNOWN"))
        groups[key].append(s)

    result = {}
    for key, group in groups.items():
        tp = sum(1 for s in group if s.get("outcome") == "TP_HIT")
        sl = sum(1 for s in group if s.get("outcome") == "SL_HIT")
        entries = tp + sl
        r_vals = [s["r_multiple"] for s in group if s.get("r_multiple") is not None]
        total_r = sum(r_vals)
        win_rate = tp / entries * 100 if entries > 0 else 0.0
        result[key] = {
            "count":    len(group),
            "tp":       tp,
            "sl":       sl,
            "win_rate": round(win_rate, 2),
            "total_r":  round(total_r, 2),
            "avg_r":    round(total_r / len(r_vals), 2) if r_vals else 0.0,
        }
    return result


def _breakdown_bool(signals: List[Dict], field: str) -> Dict:
    """Group signals by a boolean field (0/1)."""
    true_group  = [s for s in signals if s.get(field) == 1 or s.get(field) is True]
    false_group = [s for s in signals if s.get(field) == 0 or s.get(field) is False]

    def stats(group):
        tp = sum(1 for s in group if s.get("outcome") == "TP_HIT")
        sl = sum(1 for s in group if s.get("outcome") == "SL_HIT")
        entries = tp + sl
        r_vals  = [s["r_multiple"] for s in group if s.get("r_multiple") is not None]
        return {
            "count":    len(group),
            "tp":       tp,
            "sl":       sl,
            "win_rate": round(tp / entries * 100, 2) if entries > 0 else 0.0,
            "total_r":  round(sum(r_vals), 2),
        }

    return {
        f"{field}_TRUE":  stats(true_group),
        f"{field}_FALSE": stats(false_group),
    }


def _sweep_score_breakdown(signals: List[Dict]) -> Dict:
    """Break down performance by sweep score ranges."""
    buckets = {"70-74": [], "75-79": [], "80-84": [], "85-89": [], "90+": []}
    for s in signals:
        score = s.get("sweep_score", 0) or 0
        if score >= 90:
            buckets["90+"].append(s)
        elif score >= 85:
            buckets["85-89"].append(s)
        elif score >= 80:
            buckets["80-84"].append(s)
        elif score >= 75:
            buckets["75-79"].append(s)
        else:
            buckets["70-74"].append(s)

    result = {}
    for label, group in buckets.items():
        if not group:
            continue
        tp = sum(1 for s in group if s.get("outcome") == "TP_HIT")
        sl = sum(1 for s in group if s.get("outcome") == "SL_HIT")
        entries = tp + sl
        r_vals = [s["r_multiple"] for s in group if s.get("r_multiple") is not None]
        result[label] = {
            "count":    len(group),
            "win_rate": round(tp / entries * 100, 2) if entries > 0 else 0.0,
            "total_r":  round(sum(r_vals), 2),
        }
    return result


def _avg_rr(signals: List[Dict]) -> float:
    rr_vals = [s.get("rr") for s in signals if s.get("rr") is not None and s.get("rr", 0) > 0]
    return sum(rr_vals) / len(rr_vals) if rr_vals else 0.0


def _max_drawdown(r_vals: List[float]) -> float:
    """Maximum consecutive drawdown in R."""
    if not r_vals:
        return 0.0
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for r in r_vals:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _max_consecutive(signals: List[Dict], outcome: str) -> int:
    """Maximum consecutive occurrences of an outcome."""
    max_c = 0
    cur_c = 0
    for s in signals:
        if s.get("outcome") == outcome:
            cur_c += 1
            max_c = max(max_c, cur_c)
        else:
            cur_c = 0
    return max_c


def generate_observations(analysis: Dict) -> List[str]:
    """
    Generate human-readable observations from the analysis.
    Careful to not claim causation from correlation.
    """
    obs = []
    n = analysis.get("sample_size", 0)

    if n < MIN_SAMPLE_SIZE_FOR_RECOMMENDATION:
        obs.append(f"⚠️ Only {n} completed trades analyzed — insufficient for reliable conclusions (need ≥{MIN_SAMPLE_SIZE_FOR_RECOMMENDATION})")
        return obs

    overall = analysis.get("overall", {})
    obs.append(
        f"Win rate: {overall.get('win_rate',0):.1f}% over {overall.get('entries',0)} entries "
        f"| Total R: {overall.get('total_r',0):+.2f} | Profit Factor: {overall.get('profit_factor',0):.2f}"
    )

    # Sweep score observations
    ss_data = analysis.get("by_sweep_score", {})
    if "90+" in ss_data and "70-74" in ss_data:
        high_wr = ss_data["90+"].get("win_rate", 0)
        low_wr  = ss_data["70-74"].get("win_rate", 0)
        if high_wr > low_wr + 10:
            obs.append(
                f"Sweep score 90+ had {high_wr:.1f}% win rate vs {low_wr:.1f}% for 70-74 "
                f"(correlation, not necessarily causation — sample sizes vary)"
            )

    # Direction comparison
    dir_data = analysis.get("by_direction", {})
    long_wr  = dir_data.get("LONG",  {}).get("win_rate", 0)
    short_wr = dir_data.get("SHORT", {}).get("win_rate", 0)
    if abs(long_wr - short_wr) > 15 and min(
        dir_data.get("LONG",  {}).get("tp",0) + dir_data.get("LONG",  {}).get("sl",0),
        dir_data.get("SHORT", {}).get("tp",0) + dir_data.get("SHORT", {}).get("sl",0)
    ) >= 10:
        obs.append(
            f"LONGs: {long_wr:.1f}% win rate | SHORTs: {short_wr:.1f}% win rate "
            f"(15%+ difference observed)"
        )

    return obs[:7]
