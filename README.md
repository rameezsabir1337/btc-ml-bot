# BTC/USDT 15m/4h ML Trading Bot + Dashboard

Machine-learning signal generation for BTCUSDT on the 15-minute timeframe (default;
4h swing mode also supported) with rolling walk-forward testing and grid search over
model families, hyperparameters, and label designs — plus a live web dashboard.

## Quick start

```bash
cd bot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# fetch data + run the full grid + walk-forward evaluation (default 15m)
.venv/bin/python main.py --days 300 --max-configs 200

# 4h swing mode (long-only ensemble, the currently deployed configuration)
.venv/bin/python main.py --tf 4h --days 365 --max-configs 300 --long-only --ensemble 5 --combine vote

# run a tiny smoke grid (fast, for testing)
.venv/bin/python main.py --days 100 --max-configs 8 --quick

# dashboard (reads artifacts/best)
.venv/bin/python -m uvicorn dashboard.api:app --port 8050
# open http://localhost:8050
```

## What it does

1. **Data** (`data.py`) — `ccxt.binance` `fetch_ohlcv`, resumable parquet cache that
   grows backward/forward, candle-quality cleaning (spike filter, quote-consistency,
   illiquid-bar flag).
2. **Features** (`features.py`) — ~66 core features: returns/z-scores, SMA/EMA offsets,
   rolling vol, max/min returns, RSI, MACD, ATR%, Bollinger position, volume z/ratio,
   dollar volume, candle body/wick shape, VWAP distance, pivot levels, session (time-of-day)
   cyclics, ADX/directional strength, up/down-streak, and higher-TF (15m/1h) trend context.
   All causal (no lookahead). In swing modes (4h) perp funding-rate (8h) AND open-interest
   (1h) features are attached as regime signal.
3. **Labels** (`labels.py`) — five grid-selectable label designs: `binary_min_move`,
   `triple_barrier`, `multi_class`, `regression`, and `triple_atr` (vol-adaptive barriers
   scaled by ATR instead of fixed %). Labels are leakage-safe (only used in training).
4. **Backtest** (`backtest.py`) — realistic, path-dependent and honest: 0.05% USDT-M perp
   taker + slippage, **real 8h funding charged while in-position**, fees scaled by the
   notional of each position change, **signal-only exits**
   (position held until the signal flattens — no TP/SL brackets), **vol-target sizing**,
   a **drawdown brake**, and optional **trend / funding-regime gates** (filter longs out of
   a down-trending higher-TF regime or a crowded-long funding regime). Risk metrics (Sharpe,
   Sortino, MaxDD, profit factor, win rate).
5. **Walk-forward** (`walkforward.py`) — expanding train windows with an embargo gap
   and label-horizon purging so test bars can never leak into training.
6. **Grid search** (`grid.py` + `search_space.py`) — searches over LightGBM / XGBoost /
   ExtraTrees × per-model hyperparameters × label method/params × confidence margin ×
   vol-target × dd-brake, ranked by a risk-adjusted objective on inner validation with a
   **consensus gate** (configs must be positive on most folds to win), then the winner is
   evaluated honestly on untouched out-of-sample folds. `--ensemble 5 --combine vote`
   runs a Sharpe-weighted majority vote over 5 diverse configs — the deployed mode, since
   ensembling fixes per-fold robustness (6/6 OOS folds positive vs 2/6 for any single).
7. **Dashboard** (`dashboard/`) — FastAPI + ECharts SPA: candlestick chart with
   indicator overlays and buy/sell markers, backtest equity vs buy&hold, grid results,
   per-fold walk-forward equity curves, and a live-streaming panel that recomputes the
   ensemble signal on new candles with the deployed models. A **paper ledger** records
   every live tick and marks forward equity (perp taker fees included) under `/api/paper`.

## Usage notes

- Data is public; no Binance API keys required for backtesting.
- `config.py` holds every knob (costs, walk-forward shape, grid budget, live polling).
- Grid size scales with `GRID.max_configs`; the search space lives in
  `search_space.py`. Wide margins → fewer trades → less fee bleed.
- The dashboard is read-only over `artifacts/best/`; re-run `main.py` to refresh it.

## Honest expectations

5m/15m crypto is very noisy. Fees dominate most strategies. The tool is built to find
whatever risk-adjusted edge survives honest out-of-sample testing — and to clearly report
when none does. The currently deployed 4h long-only ensemble shows 4.27 pooled OOS Sharpe
with 6/6 positive folds at 0.05% taker (was 2.73 / 2-of-6 at 0.1% spot taker pre-improvements);
costs are model inputs, and narrower fees (volume tier, maker fills) lift results further.
Tune costs to your actual Binance tier in `config.py` before trusting any number.
