"""
scoring.py — Deterministic Liquidity Sweep Scoring (0–100).

The score is transparent, rule-based, and fully deterministic.
No AI prediction. No randomness. Same input = same output always.

Scoring breakdown (maximum 100 points):
  pool_significance:  20  — Type of liquidity pool
  pool_touches:       10  — Number of previous touches
  htf_4h_confluence:  15  — Level visible/relevant on 4H
  htf_1h_confluence:  10  — Level visible/relevant on 1H
  wick_rejection:     15  — Wick/body rejection ratio
  volume_at_sweep:    10  — Sweep candle volume vs average
  atr_context:         5  — Not overextended (ATR-based)
  choch_confirmed:    10  — CHoCH followed the sweep
  bos_confirmed:       5  — BOS followed the sweep
                    ─────
  TOTAL:             100
"""

from dataclasses import dataclass
from typing import Optional, List
from liquidity.pools import LiquidityPool, LiquidityType
from liquidity.sweep import SweepEvent
from config.strategy_config import (
    SWEEP_SCORE_WEIGHTS,
    POOL_SIGNIFICANCE_SCORES,
    MIN_SWEEP_SCORE,
)


@dataclass
class SweepScore:
    total:              int
    breakdown:          dict      # Component scores
    is_valid:           bool      # total >= MIN_SWEEP_SCORE
    rejection_note:     str       # Human-readable reason if rejected


def score_sweep(
    sweep: SweepEvent,
    h4_pools: Optional[List[LiquidityPool]] = None,
    h1_pools: Optional[List[LiquidityPool]] = None,
    choch_confirmed: bool = False,
    bos_confirmed: bool = False,
    atr_overextended: bool = False,
    min_score: int = MIN_SWEEP_SCORE,
) -> SweepScore:
    """
    Score a liquidity sweep event on a 0–100 scale.

    Args:
        sweep:             The detected SweepEvent
        h4_pools:          Liquidity pools detected on 4H (for confluence check)
        h1_pools:          Liquidity pools detected on 1H (for confluence check)
        choch_confirmed:   True if CHoCH occurred after the sweep
        bos_confirmed:     True if BOS occurred after the sweep
        atr_overextended:  True if price was overextended beyond ATR norm
        min_score:         Minimum threshold to mark as valid

    Returns:
        SweepScore with total and component breakdown
    """
    breakdown = {}
    W = SWEEP_SCORE_WEIGHTS

    # ── 1. Pool Significance (max 20) ──────────────────
    sig = POOL_SIGNIFICANCE_SCORES.get(sweep.pool.pool_type.value, 8)
    breakdown["pool_significance"] = sig

    # ── 2. Pool Touches (max 10) ────────────────────────
    touches = sweep.pool.touches
    if touches >= 4:
        touch_score = 10
    elif touches == 3:
        touch_score = 8
    elif touches == 2:
        touch_score = 6
    else:
        touch_score = 3
    breakdown["pool_touches"] = touch_score

    # ── 3. 4H Confluence (max 15) ──────────────────────
    h4_conf = _check_htf_confluence(sweep.pool.level, h4_pools or [], tolerance_pct=0.5)
    breakdown["htf_4h_confluence"] = 15 if h4_conf else 0

    # ── 4. 1H Confluence (max 10) ──────────────────────
    h1_conf = _check_htf_confluence(sweep.pool.level, h1_pools or [], tolerance_pct=0.3)
    breakdown["htf_1h_confluence"] = 10 if h1_conf else 0

    # ── 5. Wick Rejection Ratio (max 15) ───────────────
    ratio = sweep.wick_to_body_ratio
    if ratio >= 3.0:
        wick_score = 15
    elif ratio >= 2.0:
        wick_score = 12
    elif ratio >= 1.0:
        wick_score = 8
    elif ratio >= 0.5:
        wick_score = 4
    else:
        wick_score = 0
    breakdown["wick_rejection"] = wick_score

    # ── 6. Volume at Sweep (max 10) ─────────────────────
    if sweep.avg_volume > 0:
        vol_ratio = sweep.sweep_volume / sweep.avg_volume
    else:
        vol_ratio = 1.0

    if vol_ratio >= 2.0:
        vol_score = 10
    elif vol_ratio >= 1.5:
        vol_score = 8
    elif vol_ratio >= 1.0:
        vol_score = 5
    else:
        vol_score = 2
    breakdown["volume_at_sweep"] = vol_score

    # ── 7. ATR Context (max 5) ──────────────────────────
    # Full points if NOT overextended
    breakdown["atr_context"] = 0 if atr_overextended else 5

    # ── 8. CHoCH Confirmed (max 10) ─────────────────────
    breakdown["choch_confirmed"] = 10 if choch_confirmed else 0

    # ── 9. BOS Confirmed (max 5) ────────────────────────
    breakdown["bos_confirmed"] = 5 if bos_confirmed else 0

    # ── Total ────────────────────────────────────────────
    total = sum(breakdown.values())
    total = min(100, total)  # Cap at 100

    is_valid = total >= min_score
    rejection_note = (
        "" if is_valid
        else f"Score {total}/100 < minimum {min_score}. Weak factors: "
             + ", ".join(k for k, v in breakdown.items() if v == 0)
    )

    return SweepScore(
        total=total,
        breakdown=breakdown,
        is_valid=is_valid,
        rejection_note=rejection_note,
    )


def _check_htf_confluence(
    level: float,
    htf_pools: List[LiquidityPool],
    tolerance_pct: float = 0.5,
) -> bool:
    """
    Check if the level aligns with any higher-timeframe liquidity pool.

    Returns True if within tolerance_pct% of any HTF pool.
    """
    for pool in htf_pools:
        dist_pct = abs(pool.level - level) / max(pool.level, 1e-9) * 100
        if dist_pct <= tolerance_pct:
            return True
    return False


def format_score_breakdown(score: SweepScore) -> str:
    """Format score breakdown as a human-readable string."""
    lines = [f"🎯 Sweep Score: {score.total}/100"]
    for k, v in score.breakdown.items():
        max_v = SWEEP_SCORE_WEIGHTS.get(k, "?")
        bar   = "█" * v + "░" * (max_v - v) if isinstance(max_v, int) else ""
        lines.append(f"  {k:25s}: {v:2d}/{max_v}")
    if not score.is_valid:
        lines.append(f"❌ {score.rejection_note}")
    else:
        lines.append(f"✅ Valid (≥ {MIN_SWEEP_SCORE})")
    return "\n".join(lines)
