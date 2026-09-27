"""
paper.py — Paper trading CLI entry point.

Usage:
    python paper.py

Runs the paper trading engine using real live market data
but WITHOUT placing any orders.
"""

import sys
import signal
import logging

from database.models import initialize_database
from paper.engine import PaperTradingEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/paper.log", encoding="utf-8"),
    ],
)


def handle_shutdown(signum, frame):
    print("\n📄 Paper trading stopped.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    import os
    os.makedirs("logs", exist_ok=True)

    initialize_database()

    print("\n📄 PAPER TRADING MODE")
    print("   Uses REAL market data")
    print("   Places ZERO orders")
    print("   ⚠️  SIGNAL-ONLY — Bot does NOT trade\n")

    engine = PaperTradingEngine()
    engine.start()
