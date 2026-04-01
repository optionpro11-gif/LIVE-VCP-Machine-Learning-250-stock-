# 📊 Multi-Asset Swing Strategy Playbook
## Mayya Capital Partners — Comprehensive Reference

**Instruments:** 14 across 4 asset classes
**Timeframes:** Daily (D1) + 4-Hour (H4)
**Based on:** Walk-forward validated 30M research adapted to swing

---

## 🎯 The 14-Instrument Universe

### Forex (5)

| Pair | Primary Strategy | Direction | Sharpe | Key Edge |
|------|-----------------|----------:|-------:|----------|
| **XAUUSD** ← #1 | Momentum | LONG | 9.56 | Any positive momentum works |
| **USDCHF** ← #2 | VWAP MR Short | SHORT | 9.11 | Hidden gem — sell overbought rallies |
| **GBPUSD** ← #3 | Momentum (0.10%) | LONG | 7.62 | Needs stronger filter |
| **EURUSD** | Momentum + MR | BOTH | 5.86 | Hybrid — momentum + buy dips |
| **USDJPY** | EMA Trend | LONG | 1.61 | Small edge, diversification only |

### Indices (2)

| Index | Primary Strategy | Direction | Sharpe | Key Edge |
|-------|-----------------|----------:|-------:|----------|
| **S&P 500** | Momentum + BB | LONG | 3.70 | ⚠️ Kurtosis 150 → reduce size 40% |
| **NASDAQ** | ORB + EMA Cross | LONG | 3.35 | Breakout-driven |

### Commodities (3)

| Commodity | Primary Strategy | Direction | Key Edge |
|-----------|-----------------|----------:|----------|
| **Gold** | Momentum + Trend | LONG | #1 edge in entire universe |
| **Silver** | HA + PSAR | LONG | +243% PSAR in 4H study |
| **Crude Oil** | HA + RSI Divergence | BOTH | +53% RSI Div in HA study |

### Crypto (3)

| Crypto | Primary Strategy | Direction | Key Edge |
|--------|-----------------|----------:|----------|
| **BTC** | Momentum + HA Strong | LONG | +500% HA Strong in study |
| **ETH** | HA Strong + BB Bounce | LONG | Strong HA continuation |
| **SOL** | Momentum + EMA 9/21 | LONG | Highest vol — smallest size |

### 🇮🇳 India Nifty 250 VCP (High Velocity)
| Setup | Stop Loss | Target | Hold Time | CAGR |
|:------|:---:|:---:|:---:|:---|
| **Hyper-Velocity** | **5.0%** | **10.0%** | **12 Days** | **203%** |
| **Steady Swing** | 7.0% | 15.0% | 21 Days | 42% |

---

## 📐 Confluence Scoring System

Every signal gets a **0-100 score**. Higher = more confirmations = stronger trade.

| Score Range | Tier | Action | Position Size |
|:-----------:|:----:|:-------|:-------------:|
| 85-100 | 🔥 MAX | Rare — take aggressively | Max allocation |
| 70-84 | 💪 FULL | Strong confluence — full size | 100% |
| 50-69 | ✅ NORMAL | Good entry — standard size | 70% |
| 30-49 | 👀 WATCH | Watchlist — wait for more | 50% or skip |
| 0-29 | — IDLE | No signal | — |

### Momentum Score Components (max 100)

| Factor | Points | Condition |
|--------|-------:|-----------|
| Previous candle bullish | +30 | Close > Open on prev bar |
| Momentum > threshold | +10 | Per-instrument threshold |
| Above EMA(20) | +10 | Price > EMA 20 |
| Above EMA(50) | +10 | Price > EMA 50 |
| RSI not overbought | +10 | RSI < 70 |
| Volume surge | +10 | Vol > 20-day average |
| ATR expanding | +10 | Volatility breakout |
| Higher low | +10 | Structure confirmation |

### Mean Reversion Score (max 100)

| Factor | Points | Condition |
|--------|-------:|-----------|
| VWAP Z extreme | +30 | Z below/above threshold |
| Z very deep | +10 | 0.5 SD beyond threshold |
| RSI extreme | +25 | RSI at threshold |
| RSI very extreme | +10 | 5 pts beyond threshold |
| BB breach | +15 | Outside Bollinger Band |
| RSI divergence | +10 | Price-RSI divergence |

---

## 🛡️ Risk Management Rules

### Per-Trade Risk Caps

| Asset Class | Risk/Trade | ATR SL Mult | Position Scale |
|-------------|----------:|:-----------:|:--------------:|
| Forex | 0.50% | 2.0x | 0.5-1.0x |
| Indices | 0.40-0.50% | 2.0-2.5x | 0.6-0.7x |
| Commodities | 0.50% | 2.5x | 0.6-1.0x |
| Crypto | 0.30% | 3.0-3.5x | 0.3-0.4x |

### Portfolio Limits

- **Max total open risk:** 5.0%
- **Max per asset class:** 2.0%
- **Max correlated trades:** 3
- **Min reward:risk ratio:** 1.5:1
- **Default hold period:** 5 trading days

### Correlation Groups (don't double up)

| Group | Instruments |
|-------|------------|
| USD pairs | EURUSD, GBPUSD, USDCHF, USDCAD, USDJPY |
| US Indices | NASDAQ, S&P 500 |
| Precious metals | Gold, Silver |
| Crypto | BTC, ETH, SOL |
| Risk-on | NASDAQ, BTC, ETH, SOL |

---

## 📅 Daily Workflow

### End-of-Day (Post US Close)

1. **Run scanner:** `python swing_telegram_scanner.py`
2. **Check Telegram** for actionable signals (score ≥ 50)
3. **Review TradingView** — apply Pine overlay to flagged instruments
4. **Place orders** for next day:
   - Limit orders at entry price
   - Stop loss at the ATR-based level
   - Take profit at target
5. **Log trades** in your journal

### Weekly Review (Sunday)

1. **Run backtest update:** `python swing_strategy_master.py --backtest --period 1y`
2. **Review report:** `python swing_strategy_master.py --report`
3. **Check correlation exposure** — are you overweight in any group?
4. **Adjust position sizes** based on recent performance

---

## 🖥️ File Map

| File | Purpose |
|------|---------|
| `swing_strategy_config.py` | All configs, tickers, scoring weights, risk rules |
| `swing_strategy_master.py` | Main engine: scan, backtest, report |
| `swing_telegram_scanner.py` | Telegram alert sender |
| `pine_scripts/Swing_Multi_Asset_Overlay.pine` | TradingView overlay |

### Usage

```bash
# Scan all 14 instruments now
python swing_strategy_master.py --scan

# Scan specific instrument
python swing_strategy_master.py --scan --instrument XAUUSD,BTC

# Backtest all instruments (5 years)
python swing_strategy_master.py --backtest

# Backtest specific pair (2 years)
python swing_strategy_master.py --backtest --instrument EURUSD --period 2y

# Generate performance report (after backtest)
python swing_strategy_master.py --report

# Send scan results to Telegram
python swing_telegram_scanner.py

# Test Telegram connectivity
python swing_telegram_scanner.py --test
```

---

## ⚠️ Critical Rules

1. **NEVER short Gold, BTC, ETH** — all shorts fail historically
2. **S&P 500 has extreme kurtosis (150.5)** — use 60% of normal position size
3. **USDCHF short is the #2 edge** — don't skip it just because it's counter-intuitive
4. **Crypto uses 0.3% risk** — not 0.5% like forex. Volatility is 3-5x higher
5. **Don't take score < 30 signals** — no confluence = no edge
6. **Max 3 correlated trades** — 5 long USD trades = disaster if USD reverses

---

*Mayya Capital Partners — Research Division*
*"Velocity is the multiplier of edge. Turnover is the multiplier of wealth."*
