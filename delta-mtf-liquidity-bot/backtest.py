"""
backtest.py — Backtest CLI entry point.

Usage:
    python backtest.py --symbol BTCUSD --start 2024-01-01 --end 2024-12-31
    python backtest.py --symbol ETHUSD --start 2024-06-01 --end 2024-12-31 --no-cache
"""

import argparse
import sys
from database.models import initialize_database
from backtest.engine import run_backtest
from backtest.metrics import print_backtest_report
from config.settings import BACKTEST_START, BACKTEST_END, MONITORED_SYMBOLS


def main():
    parser = argparse.ArgumentParser(
        description="Delta MTF Liquidity Bot — Backtester (4H + 1H + 15M only, NO 5M)"
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSD",
        choices=MONITORED_SYMBOLS + ["BTCUSD", "ETHUSD", "SOLUSD"],
        help="Symbol to backtest",
    )
    parser.add_argument("--start", default=BACKTEST_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=BACKTEST_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch (ignore cache)")
    parser.add_argument(
        "--dev-split",
        type=float,
        default=0.70,
        help="Development/OOS split ratio (default: 0.70)",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"DELTA MTF LIQUIDITY BOT — BACKTESTER")
    print(f"Symbol:   {args.symbol}")
    print(f"Period:   {args.start} → {args.end}")
    print(f"Dev/OOS:  {args.dev_split*100:.0f}% / {(1-args.dev_split)*100:.0f}%")
    print(f"Timeframes: 4H | 1H | 15M (NO 5M)")
    print(f"{'='*60}\n")

    initialize_database()

    results = run_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        use_cache=not args.no_cache,
        dev_split=args.dev_split,
    )

    if "error" in results:
        print(f"❌ Backtest failed: {results['error']}")
        sys.exit(1)

    print_backtest_report(results)
    print("\n✅ Backtest complete.")
    print("⚠️  Review results before any strategy modification.")
    print("   Human approval required for all changes.")


if __name__ == "__main__":
    main()
