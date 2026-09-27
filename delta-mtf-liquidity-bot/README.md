# Delta Exchange India — MTF Liquidity Sweep + CHoCH + BOS Signal Bot

> **SIGNAL-ONLY BOT. Does NOT place orders. No capital logic. 4H + 1H + 15M only.**

---

## 🚨 Critical Rules

| Rule | Status |
|---|---|
| Auto-trading | ❌ NEVER |
| Capital / position size | ❌ NOT CALCULATED |
| 5M timeframe | ❌ COMPLETELY REMOVED |
| Auto-strategy changes | ❌ NEVER |
| Human approval for changes | ✅ REQUIRED |

---

## Architecture

```
delta-mtf-liquidity-bot/
├── config/              — Settings & strategy config
├── market_data/         — Delta Exchange API client, candles, WebSocket
├── indicators/          — EMA, ATR, ADX, Volume
├── market_structure/    — Swings (no look-ahead), BOS, CHoCH
├── liquidity/           — Pool detection, sweep detection, scoring (0–100)
├── strategy/            — MTF alignment, intraday pipeline, swing pipeline, filters
├── signals/             — Generator, validator, lifecycle, duplicate guard
├── telegram/            — Bot, commands, message formatting
├── database/            — SQLite schema, trade journal
├── learning/            — Performance analyzer, observations, recommendations, versioning
├── weekly/              — Weekly report generator
├── backtest/            — Engine (no look-ahead), data loader, metrics
├── paper/               — Paper trading engine
└── tests/               — Unit tests
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
cp .env.example .env
# Edit .env with your Delta API keys and Telegram credentials
```

### 3. Run in Live Signal Mode
```bash
python main.py
```

### 4. Run in Paper Trading Mode
```bash
python paper.py
```

### 5. Run Backtester
```bash
python backtest.py --symbol BTCUSD --start 2024-01-01 --end 2024-12-31
```

---

## Strategy Overview

### Timeframes
| Timeframe | Purpose |
|---|---|
| **4H** | Macro context: EMA50/200, ADX, swing highs/lows, support/resistance |
| **1H** | Trend & structure: HH/HL/LH/LL, BOS, CHoCH, previous day levels |
| **15M** | Entry: liquidity sweep, CHoCH, BOS, retest, volume confirmation |

### Signal Pipeline (Intraday)
```
4H Macro → 1H Structure → 15M Liquidity Pool → 15M Sweep → CHoCH → BOS → Retest → Volume → RR → Signal
```

### Signal Pipeline (Swing)
```
4H Macro → 1H Structure → 1H Liquidity → 1H Sweep → CHoCH → BOS → (Optional 15M Refinement) → RR → Signal
```

---

## Sweep Scoring (0–100)

| Factor | Max Points |
|---|---|
| Pool significance (PDH/PDL/Equal H-L/Swing) | 20 |
| Previous touches at level | 10 |
| 4H confluence | 15 |
| 1H confluence | 10 |
| Wick rejection ratio | 15 |
| Volume at sweep | 10 |
| ATR context (not overextended) | 5 |
| CHoCH confirmed | 10 |
| BOS confirmed | 5 |

**Default minimum: 70/100**

---

## Signal Quality

| Grade | Sent to Telegram | Description |
|---|---|---|
| **A+** | ✅ | All conditions ideal |
| **A** | ✅ | Strong setup, relaxed criteria |
| **B** | ❌ | Valid but below quality bar |
| **REJECTED** | ❌ | Fails strategy filters |

---

## Telegram Commands

| Command | Description |
|---|---|
| `/status` | Bot health + last scan time |
| `/signals` | Recent signals (24h) |
| `/active` | Currently active signals |
| `/history` | Completed signals with outcomes |
| `/weekly` | Latest weekly report |
| `/performance` | Performance summary |
| `/learning` | Detected patterns |
| `/recommendations` | Pending strategy improvements |
| `/version` | Strategy version history |

---

## Signal Format (Telegram)
```
🚨 DELTA SIGNAL
₿ BTC
📊 INTRADAY | 15M
📈 Direction: LONG
🎯 Entry: 65,000.00
🛑 SL:    64,000.00
💰 TP:    67,500.00
📐 RR: 1:2.5
...
⚠️ MANUAL EXECUTION ONLY
Bot does NOT place orders.
```

---

## Self-Learning System

The bot analyzes completed trade data and generates:
- **Observations**: What patterns appear in the data
- **Recommendations**: Specific parameter change suggestions with evidence
- **Version proposals**: What v1.1 should look like

**All recommendations require human approval. No automatic changes.**

Approval workflow:
```
PENDING_REVIEW → APPROVED_FOR_BACKTEST → BACKTEST_PASSED
→ APPROVED_FOR_PAPER_TEST → PAPER_TEST_PASSED → APPROVED_FOR_LIVE_VERSION
```

---

## Database (SQLite)

All data stored in `delta_bot.db`:
- `signals` — All signals (no capital fields)
- `signal_outcomes` — TP/SL/EXPIRED results + R-multiples
- `strategy_versions` — Version changelog
- `weekly_reports` — Stored reports
- `learning_observations` — Detected patterns
- `recommendations` — Improvement suggestions + approval status
- `backtest_results` — Per-run backtest metrics

---

## Backtesting

```bash
# Default
python backtest.py --symbol BTCUSD

# Custom period
python backtest.py --symbol ETHUSD --start 2024-01-01 --end 2024-06-30

# Force re-fetch (ignore cache)
python backtest.py --symbol SOLUSD --no-cache
```

Features:
- 70/30 dev/OOS split
- Fees and slippage applied
- R-multiple based (capital-independent)
- No look-ahead bias
- Same logic as live mode

---

## Configuration

### `config/settings.py`
- `USE_TESTNET` — Use Delta testnet
- `MONITORED_SYMBOLS` — Coins to watch
- `PAPER_MODE` / `LIVE_SIGNAL_MODE`

### `config/strategy_config.py`
- `MIN_SWEEP_SCORE` — Minimum sweep quality (default: 70)
- `MIN_RR_INTRADAY` / `MIN_RR_SWING`
- `ATR_BUFFER_MULTIPLIER` — SL buffer
- `COUNTERTREND_ALLOWED` — False by default
- All thresholds and parameters

---

## Disclaimer

This bot is for **informational and educational purposes only**.

- It does NOT place orders
- It does NOT manage capital
- All trading decisions are made manually by the user
- Past performance does not guarantee future results
- Cryptocurrency trading carries significant risk

**⚠️ MANUAL EXECUTION ONLY — BOT DOES NOT PLACE ORDERS.**
