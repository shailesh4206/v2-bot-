"""
backtest/metrics.py — Backtest performance metrics calculation.

All metrics are R-multiple based (capital-independent).
Generates equity curve data, drawdown curve, and breakdowns by:
- Month, symbol, mode (intraday/swing), direction, sweep score, quality
"""

import json
import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """
    Compute all backtest metrics from a list of completed trade dicts.

    Returns a comprehensive metrics dict.
    """
    if not trades:
        return {"error": "No trades to analyze"}

    completed = [t for t in trades if t.get("outcome") in ("TP_HIT", "SL_HIT", "EXPIRED")]
    entries   = [t for t in completed if t.get("outcome") in ("TP_HIT", "SL_HIT")]
    tp_hits   = [t for t in completed if t.get("outcome") == "TP_HIT"]
    sl_hits   = [t for t in completed if t.get("outcome") == "SL_HIT"]
    expired   = [t for t in completed if t.get("outcome") == "EXPIRED"]

    r_vals    = [t.get("r_multiple", 0) for t in entries]
    net_r_vals = [t.get("net_r", t.get("r_multiple", 0)) for t in entries]

    total_r  = sum(r_vals)
    net_r    = sum(net_r_vals)
    avg_r    = total_r / len(r_vals) if r_vals else 0
    win_rate = len(tp_hits) / len(entries) * 100 if entries else 0

    pos_r = sum(r for r in r_vals if r > 0)
    neg_r = abs(sum(r for r in r_vals if r < 0))
    pf    = round(pos_r / neg_r, 3) if neg_r > 0 else None

    # Expectancy = avg win * win_rate - avg loss * loss_rate
    avg_win  = sum(r for r in r_vals if r > 0) / len(tp_hits) if tp_hits else 0
    avg_loss = abs(sum(r for r in r_vals if r < 0)) / len(sl_hits) if sl_hits else 0
    wr_frac  = win_rate / 100
    expectancy = (wr_frac * avg_win) - ((1 - wr_frac) * avg_loss)

    fee_r   = sum(t.get("fees_r", 0) for t in entries)
    slip_r  = sum(t.get("slippage_r", 0) for t in entries)

    max_dd  = _max_drawdown(net_r_vals)
    max_cl  = _max_consecutive(completed, "SL_HIT")
    max_cw  = _max_consecutive(completed, "TP_HIT")

    avg_rr  = sum(t.get("rr", 0) for t in trades if t.get("rr")) / len(trades) if trades else 0

    eq_curve, dd_curve = _equity_drawdown_curves(net_r_vals)

    metrics = {
        "total_signals":      len(trades),
        "valid_signals":      len([t for t in trades if t.get("quality") in ("A+","A")]),
        "a_plus_signals":     len([t for t in trades if t.get("quality") == "A+"]),
        "a_signals":          len([t for t in trades if t.get("quality") == "A"]),
        "total_entries":      len(entries),
        "tp_hits":            len(tp_hits),
        "sl_hits":            len(sl_hits),
        "expired":            len(expired),
        "win_rate":           round(win_rate, 2),
        "loss_rate":          round(100 - win_rate, 2),
        "total_r":            round(total_r, 2),
        "net_r":              round(net_r, 2),
        "avg_r":              round(avg_r, 2),
        "avg_rr":             round(avg_rr, 2),
        "profit_factor":      pf,
        "expectancy":         round(expectancy, 3),
        "max_drawdown_r":     round(max_dd, 2),
        "max_consec_losses":  max_cl,
        "max_consec_wins":    max_cw,
        "total_fees_r":       round(fee_r, 3),
        "total_slippage_r":   round(slip_r, 3),
        "equity_curve":       eq_curve,
        "drawdown_curve":     dd_curve,

        # Breakdowns
        "by_mode":      _breakdown(trades, "mode"),
        "by_direction": _breakdown(trades, "direction"),
        "by_quality":   _breakdown(trades, "quality"),
        "by_symbol":    _breakdown(trades, "symbol"),
        "by_sweep_score": _sweep_breakdown(trades),
        "by_month":     _monthly_breakdown(trades),
        "by_split":     _breakdown(trades, "split_type"),

        # Liquidity metrics
        "liquidity_metrics": _liquidity_metrics(trades),
    }

    return metrics


def _breakdown(trades, field):
    groups = defaultdict(list)
    for t in trades:
        groups[str(t.get(field, "UNKNOWN"))].append(t)
    result = {}
    for key, group in groups.items():
        tp    = sum(1 for t in group if t.get("outcome") == "TP_HIT")
        sl    = sum(1 for t in group if t.get("outcome") == "SL_HIT")
        ex    = sum(1 for t in group if t.get("outcome") == "EXPIRED")
        ent   = tp + sl
        r_v   = [t.get("r_multiple", 0) for t in group if t.get("outcome") in ("TP_HIT","SL_HIT")]
        total = sum(r_v)
        result[key] = {
            "count":    len(group),
            "entries":  ent,
            "tp":       tp,
            "sl":       sl,
            "expired":  ex,
            "win_rate": round(tp / ent * 100, 2) if ent > 0 else 0.0,
            "total_r":  round(total, 2),
        }
    return result


def _sweep_breakdown(trades):
    buckets = {"70-74": [], "75-79": [], "80-84": [], "85-89": [], "90+": []}
    for t in trades:
        sc = t.get("sweep_score", 0) or 0
        if sc >= 90:   buckets["90+"].append(t)
        elif sc >= 85: buckets["85-89"].append(t)
        elif sc >= 80: buckets["80-84"].append(t)
        elif sc >= 75: buckets["75-79"].append(t)
        else:          buckets["70-74"].append(t)
    return {k: _brief_stats(v) for k, v in buckets.items() if v}


def _monthly_breakdown(trades):
    groups = defaultdict(list)
    for t in trades:
        ts = t.get("signal_ts")
        if ts:
            try:
                month = str(ts)[:7]  # "YYYY-MM"
            except Exception:
                month = "UNKNOWN"
        else:
            month = "UNKNOWN"
        groups[month].append(t)
    return {k: _brief_stats(v) for k, v in sorted(groups.items())}


def _liquidity_metrics(trades):
    total   = len(trades)
    tp      = [t for t in trades if t.get("outcome") == "TP_HIT"]
    by_type = defaultdict(list)
    for t in trades:
        by_type[t.get("liquidity_type", "UNKNOWN")].append(t)
    return {
        "total_sweeps":          total,
        "successful_sweeps":     len(tp),
        "sweep_success_rate":    round(len(tp) / total * 100, 2) if total > 0 else 0,
        "by_liquidity_type":     {k: _brief_stats(v) for k, v in by_type.items()},
    }


def _brief_stats(group):
    tp   = sum(1 for t in group if t.get("outcome") == "TP_HIT")
    sl   = sum(1 for t in group if t.get("outcome") == "SL_HIT")
    ent  = tp + sl
    r_v  = [t.get("r_multiple", 0) for t in group if t.get("outcome") in ("TP_HIT","SL_HIT")]
    return {
        "count":    len(group),
        "win_rate": round(tp / ent * 100, 2) if ent > 0 else 0.0,
        "total_r":  round(sum(r_v), 2),
    }


def _equity_drawdown_curves(net_r_vals):
    eq = []
    dd = []
    equity = 0.0
    peak   = 0.0
    for r in net_r_vals:
        equity += r
        if equity > peak:
            peak = equity
        eq.append(round(equity, 3))
        dd.append(round(peak - equity, 3))
    return eq, dd


def _max_drawdown(r_vals):
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


def _max_consecutive(trades, outcome):
    max_c = cur_c = 0
    for t in trades:
        if t.get("outcome") == outcome:
            cur_c += 1
            max_c = max(max_c, cur_c)
        else:
            cur_c = 0
    return max_c


def print_backtest_report(metrics: Dict):
    """Print a formatted backtest report to the console."""
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS")
    print("=" * 60)
    print(f"Symbol:       {metrics.get('symbol', '?')}")
    print(f"Period:       {metrics.get('start', '?')} → {metrics.get('end', '?')}")
    print(f"Strategy:     {metrics.get('strategy', '?')}")
    print("-" * 60)
    print(f"Total Signals: {metrics['total_signals']}")
    print(f"Valid Signals: {metrics['valid_signals']}")
    print(f"Entries:       {metrics['total_entries']}")
    print(f"TP Hits:       {metrics['tp_hits']}")
    print(f"SL Hits:       {metrics['sl_hits']}")
    print(f"Expired:       {metrics['expired']}")
    print("-" * 60)
    print(f"Win Rate:      {metrics['win_rate']:.2f}%")
    print(f"Total R:       {metrics['total_r']:+.2f}")
    print(f"Net R:         {metrics['net_r']:+.2f} (after fees/slippage)")
    print(f"Avg R:         {metrics['avg_r']:+.2f}")
    print(f"Avg RR:        1:{metrics['avg_rr']:.2f}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Expectancy:    {metrics['expectancy']:+.3f}R")
    print(f"Max Drawdown:  {metrics['max_drawdown_r']:.2f}R")
    print(f"Max Consec L:  {metrics['max_consec_losses']}")
    print(f"Fees:          {metrics['total_fees_r']:.3f}R")
    print("=" * 60)

    # By split
    by_split = metrics.get("by_split", {})
    print("\n📊 IN-SAMPLE vs OUT-OF-SAMPLE")
    for split, stats in by_split.items():
        print(f"  {split}: {stats['tp']}W / {stats['sl']}L | WR: {stats['win_rate']:.1f}% | R: {stats['total_r']:+.2f}")
    print("=" * 60)
