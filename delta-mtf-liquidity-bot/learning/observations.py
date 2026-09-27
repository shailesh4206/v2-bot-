"""
learning/observations.py — Store and retrieve learning observations.
learning/versioning.py — Strategy version management with human approval.
"""

# ─────────────────────────────────────────────────────────────────────────────
# observations.py content (merged for brevity)
# ─────────────────────────────────────────────────────────────────────────────

from database.journal import save_learning_observation, get_recent_observations
from learning.analyzer import generate_observations, analyze_performance
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def refresh_observations():
    """Run analysis and save fresh observations to the database."""
    analysis = analyze_performance()
    if analysis.get("error"):
        logger.info(f"Cannot generate observations: {analysis['error']}")
        return []

    obs_texts = generate_observations(analysis)
    n = analysis.get("sample_size", 0)

    saved = []
    for text in obs_texts:
        obs = {
            "metric":           "overall",
            "segment":          "all_signals",
            "value":            None,
            "sample_size":      n,
            "confidence_level": "LOW" if n < 30 else ("MEDIUM" if n < 100 else "HIGH"),
            "observation_text": text,
        }
        save_learning_observation(obs)
        saved.append(text)

    logger.info(f"Saved {len(saved)} observations")
    return saved
