# BTC/USDT ML Trading Bot + Dashboard

Machine-learning signal research for **BTC/USDT** with rolling walk-forward
validation, grid search over model families / hyperparameters / label designs,
a realistic cost-aware backtest, and a live FastAPI + ECharts dashboard.

> **Research / paper-trading only.** This project does **not** place orders and
> needs **no exchange API keys**. All data is public Binance market data fetched
> through `ccxt`; the dashboard's "live" panel recomputes signals and tracks a
> local paper ledger.

---

## Requirements

- Python **3.10+** (tested on 3.13)
- No API keys or exchange account required
- ~2 GB free disk for the virtualenv, candle cache, and artifacts

## Setup

```bash
git clone <your-repo-url> bot
cd bot

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The bot reads all configuration from `config.py` (not `config.json`, which is a
local editor config and is gitignored). Cost and runtime knobs can be overridden
with environment variables — see [Environment variables](#environment-variables).

---

## Quick start

### 1. Fetch data only

```bash
# ~3 years of daily bars (used by the current deployed artifact)
python main.py --tf 1d --days 1095 --fetch-only

# or 4h / 15m
python main.py --tf 4h --days 365  --fetch-only
python main.py --tf 15m --days 300 --fetch-only
```

Candles are cached under `data/cache/` and refreshed incrementally. The cache is
gitignored.

### 2. Smoke test (fast, low config count)

```bash
python main.py --days 100 --max-configs 8 --quick
```

> `main.py` always writes its result to `artifacts/best/`. Back that directory up
> first if you want to keep the currently deployed artifact:
> `cp -r artifacts/best artifacts/best_backup` (backups are gitignored).

### 3. Reproduce the deployed-style 1d long-only ensemble

```bash
BOT_TAKER_FEE=0.0010 \
python main.py --tf 1d --days 1095 --long-only \
  --ensemble 5 --combine vote --vote 0.65 --max-configs 300
```

### 4. Run the dashboard

```bash
python -m uvicorn dashboard.api:app --host 127.0.0.1 --port 8050
```

Open <http://127.0.0.1:8050>.

To run it detached (survives the shell), use:

```bash
python launch_dash.py
# logs: data/dash.log (override with DASH_LOG=/path/to/file)
# host/port: DASH_HOST / DASH_PORT env vars
```

The dashboard is read-only over `artifacts/best/`; re-run `main.py` (or
`promote_vote.py`) to refresh the artifact. On startup it refits the deployed
ensemble members on the stored history plus fresh candles, then polls for new
bars.

---

## CLI reference

### `main.py`

| Flag | Default | Description |
|---|---|---|
| `--days N` | `365` | History window to fetch |
| `--tf {15m,4h,1d}` | `15m` | Bar timeframe (sets folds, horizons, annualisation) |
| `--long-only` | off | Never take shorts |
| `--no-gates` | off | Disable trend + funding regime gates |
| `--confidence-sizing` | off | Scale size by signal strength |
| `--force-trend-gate COL` | none | Restrict grid to one trend-gate column |
| `--ensemble N` | `0` | Ensemble the top-N diverse configs |
| `--combine {vote,blend}` | `vote` | Ensemble combiner |
| `--vote FLOAT` | `0.5` | Vote threshold (agreement required to go long) |
| `--max-configs N` | `300` | Grid budget |
| `--n-jobs N` | `-1` | Parallel workers |
| `--quick` | off | Tiny smoke grid |
| `--show-grid` | off | Print the top-10 ranked configs |
| `--fetch-only` | off | Download/clean data and exit |
| `--force-refresh` | off | Ignore the cache and refetch |

### Dashboard endpoints

| Endpoint | Contents |
|---|---|
| `GET /` | Dashboard SPA |
| `GET /api/health` | Artifact/live health, retrain cadence |
| `GET /api/ohlcv?limit=2000` | Candles |
| `GET /api/indicators?limit=2000` | Indicator overlays |
| `GET /api/predictions?limit=2000` | OOS score / position |
| `GET /api/backtest?limit=5000` | Pooled metrics, folds, equity vs buy & hold |
| `GET /api/grid` | Best config + grid summary |
| `GET /api/walkforward` | Per-fold results |
| `GET /api/live` | Latest live signal |
| `POST /api/live/retrain` | Force a model refit |
| `GET /api/paper` | Paper-ledger equity snapshot |

---

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `BOT_TAKER_FEE` | `0.0010` | Per-side taker fee (0.10% = Binance VIP0 spot) |
| `BOT_USE_FUNDING` | `0` | `1` charges perp funding in PnL (spot mode leaves it off) |
| `BOT_RETRAIN_SECONDS` | `604800` | Live refit cadence (7 days) |
| `BOT_OBJ_DD` | `0.5` | Drawdown penalty weight in the grid objective |
| `BOT_OBJ_WIN` | `0.0` | Win-rate weight in the grid objective |
| `DASH_HOST` / `DASH_PORT` | `0.0.0.0` / `8050` | `launch_dash.py` bind address |
| `DASH_LOG` | `data/dash.log` | `launch_dash.py` log file |

Set them inline or export them before running:

```bash
export BOT_TAKER_FEE=0.0005
export BOT_USE_FUNDING=1
python main.py --tf 4h --days 365 --long-only --ensemble 5 --combine vote
```

---

## Research tools

These scripts reuse the saved artifact and/or the frozen evaluation dataset:

```bash
# Retrain-cadence / model-staleness study
python cadence.py --artifact best --stride 1 2 3 6 --cadence 7 14 28

# Vote-threshold sweep without re-running the grid
PYTHONPATH=$PWD python sweep_vote.py --artifact best --vote 0.5 0.6 0.65 0.7

# Promote an artifact to a vote/regime ensemble and write artifacts/best
PYTHONPATH=$PWD python promote_vote.py \
  --artifact best_spot_1d_win12 --combine regime \
  --soft-low 0.25 --soft-mid 0.50 --backup

# Re-evaluate the deployed-style fleet on the frozen dataset
python eval_ensemble.py --name best

# Search alternative member fleets
python search_members.py --top 15

# Option-B regime-adaptive prototype
python proto_optionb.py
```

`eval_ensemble.py` and `search_members.py` expect a local `frozen_eval/` dataset,
which is gitignored and must be generated locally.

---

## Project layout

```text
config.py             central settings (timeframes, costs, walk-forward, grid)
main.py               CLI: fetch -> features -> grid -> walk-forward -> report
data.py               ccxt fetch + resumable cache + candle cleaning
features.py           causal feature engineering (including funding/OI context)
labels.py             label designs (binary, triple barrier, multi-class, ...)
models.py             LightGBM / XGBoost / ExtraTrees wrappers
backtest.py           cost-aware, path-dependent backtest + risk metrics
walkforward.py        expanding-window folds with embargo/purge
grid.py               grid search, ranking, ensemble walk-forward
ensemble.py           weighted vote / regime / blend combiners
report.py             artifact persistence + console report
promote_vote.py       choose vote/regime policy and write artifacts/best
cadence.py            retrain-cadence experiments
sweep_vote.py         vote-threshold sweep on a saved artifact
eval_ensemble.py      reproducible frozen-dataset ensemble evaluation
search_members.py     member-fleet search
proto_optionb.py      regime-adaptive prototype
launch_dash.py        detached dashboard launcher
dashboard/api.py      FastAPI app + live signal + paper ledger
dashboard/static/     dashboard SPA (HTML/JS/ECharts)
artifacts/best/       tracked, ready-to-serve artifact used by the dashboard
data/                 fetched candles, caches, paper ledger (gitignored)
```

### Artifacts

`artifacts/best/` contains the artifact the dashboard serves:

- `report.json` — config, ensemble members, pooled/per-fold metrics, equity curve
- `ohlcv.parquet`, `features.parquet`, `predictions.parquet`

Re-running `main.py` or `promote_vote.py` overwrites this directory.

---

## Current artifact snapshot

The tracked `artifacts/best/report.json` was generated **2026-09-27** on daily
bars from **2023-08-19 → 2026-08-17**:

| Metric | Value |
|---|---|
| Pooled OOS Sharpe | 2.44 |
| Total return | +176.8% |
| Max drawdown | −8.9% |
| Win rate | 55.0% |
| Trades | 31 |
| Positive folds | 5 / 5 |

Per-fold Sharpe: 2.91, 4.26, 0.51, 0.36, 2.96.

> **Cost caveat:** `report.json` does not persist the fee assumption used for the
> run. The current default is `BOT_TAKER_FEE=0.0010` (0.10%/side spot) with
> `BOT_USE_FUNDING=0`. Re-run with explicit cost settings before comparing
> numbers or drawing live conclusions.

---

## Notes

- Market data is public and requires no credentials.
- Fees, slippage, and optional perp funding are modelled in `backtest.py`;
  adjust them to your exchange tier before trusting any performance number.
- Crypto edges are thin and regime-dependent. Treat this as a research framework,
  not investment advice.
