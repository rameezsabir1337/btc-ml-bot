"""Central configuration for the 15m/4h BTCUSDT ML trading bot."""
from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
CACHE_DIR = DATA_DIR / "cache"

for d in (DATA_DIR, ARTIFACT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Market / data
# ---------------------------------------------------------------------------
SYMBOL = "BTC/USDT"
PERP_SYMBOL = "BTC/USDT:USDT"   # perp for funding-rate / open-interest aux data
TIMEFRAME = "15m"
EXCHANGE = "binance"
# ~1 year of 15m bars = ~35k bars by default.
HISTORY_DAYS = 365
OHLCV_LIMIT = 1000          # per Binance fetch_ohlcv request
REQUIRE_MOST_RECENT = True  # if True, bail when the last candle is stale
MAX_STALE_AGE = 600         # seconds a live candle may lag before considered stale

# Timeframe / trading-mode state. `configure_timeframe()` sets the module
# state; the dashboard reads BARS_PER_YEAR / TIMEFRAME / LONG_ONLY to annualise
# and to fetch the correct live timeframe.
#   * 15m -> intraday (default); supports 1h/4h HTF trend context
#   * 4h  -> swing regime: multi-day holds so the ~24bps round-trip cost
#            amortises over large moves
#   * 1d  -> slow regime: weekly-hold labels so the spot 24bps round-trip
#            amortises over the largest moves (spot-fee viability path)
# (the old 5m mode has been removed)
SWING = True
LONG_ONLY = False
CONFIDENCE_SIZING = False       # scale position size by signal strength vs margin
GATES_ON = True               # apply trend (htf slope) + funding regime gates
GATE_CONFIRM_BARS = 2          # require the htf-slope gate to hold this many bars
                               # before flipping exposure (whipsaw hysteresis)
FORCE_TREND_GATE = None       # if set, grid only searches this trend gate (e.g. "htf1440_slope")
SWING_HTF = [60, 240]          # 1h / 4h HTF trend context (15m default mode)
MODE_HORIZONS = [24, 48, 96]   # label horizons (in bars) for the active mode
BARS_PER_YEAR_15M = 365 * 24 * 60 // 15  # 35040
BARS_PER_YEAR_4H = 365 * 24 // 4         # 2190
BARS_PER_YEAR_1D = 365                   # 1 bar/day
BARS_PER_YEAR = BARS_PER_YEAR_15M

# anti-churn / sizing knobs applied to every backtest position map
MIN_HOLD_BARS = 0     # 0 = off; hold a position >= N bars
VOL_TARGET = 0.0      # 0 = off; size positions toward target realized vol


def configure_timeframe(tf: str, long_only: bool = False) -> None:
    """Switch module state (and WF/GRID bounds) to a different bar regime."""
    global SWING, LONG_ONLY, SWING_HTF, TIMEFRAME, BARS_PER_YEAR, MODE_HORIZONS, MIN_HOLD_BARS, VOL_TARGET
    TIMEFRAME = tf
    LONG_ONLY = long_only
    SWING = True
    if tf == "15m":
        BARS_PER_YEAR = BARS_PER_YEAR_15M
        SWING_HTF = [60, 240]            # 1h / 4h trend context for the gate
        MODE_HORIZONS = [24, 48, 96]     # 6h / 12h / 24h labels
    elif tf == "4h":
        BARS_PER_YEAR = BARS_PER_YEAR_4H
        SWING_HTF = [1440]               # 1d trend context for the gate
        MODE_HORIZONS = [30, 60, 90]     # 5d / 10d / 15d labels
    elif tf == "1d":
        BARS_PER_YEAR = BARS_PER_YEAR_1D
        SWING_HTF = [10080]              # 1w trend context for the gate
        MODE_HORIZONS = [7, 14, 30]      # 1w / 2w / 1m labels
    else:
        raise ValueError(f"Unsupported timeframe {tf}")

    # scale walk-forward / grid bounds to bar density (~7d min test, ~20d min train)
    bars_per_day = BARS_PER_YEAR // 365
    WF.n_folds = 6
    WF.train_frac = 0.55
    WF.embargo_bars = 12
    WF.min_test_bars = max(40, int(7 * bars_per_day))
    WF.min_train_bars = max(150, int(20 * bars_per_day))
    # swing trades are fewer; don't reject sparse configs
    GRID.min_config_trades = max(8, int(1 * bars_per_day))
    GRID.score_subsample = min(30000, max(1500, int(60 * bars_per_day)))
    GRID.score_fold_cap = WF.n_folds   # rank on ALL folds: selection noise is the enemy
    GRID.min_positive_folds = max(2, min(3, WF.n_folds // 2))
    # Spot search (0.10% taker) can bias the grid toward high win-rate + low DD
    # with BOT_OBJ_DD / BOT_OBJ_WIN. Defaults keep the original Sharpe objective
    # (win-rate weight must be high to matter: win_rate spans ~0.4-0.6 while
    # Sharpe swings by units).
    GRID.objective_weight_drawdown = float(os.environ.get("BOT_OBJ_DD", 0.5))
    GRID.objective_weight_winrate = float(os.environ.get("BOT_OBJ_WIN", 0.0))

    # harden long-only swing trades: minimum 12h hold, vol-scaled sizing
    MIN_HOLD_BARS = 0
    VOL_TARGET = 0.0
    if tf == "4h":
        MIN_HOLD_BARS = 3     # 3 x 4h = 12h minimum hold
        VOL_TARGET = 0.0      # leave off until we tune it explicitly

# ---------------------------------------------------------------------------
# Costs. Default = Binance VIP0 spot: taker AND maker are both 0.10%. The
# perp-tuned 4h ensemble assumed 0.05% taker and degrades to negative Sharpe at
# 0.10%, so the spot artifact must come from a fresh grid search. Override with
# BOT_TAKER_FEE (e.g. 0.00075 for the BNB-discounted tier) and BOT_USE_FUNDING.
# ---------------------------------------------------------------------------
TAKER_FEE = float(os.environ.get("BOT_TAKER_FEE", 0.0010))   # per side
SLIPPAGE_BPS = 2.0          # per-side slippage in basis points
# Spot pays no funding. The perp funding/OI series remain USED AS FEATURES
# (crowding signals) but are not charged in PnL unless BOT_USE_FUNDING=1.
USE_FUNDING = os.environ.get("BOT_USE_FUNDING", "0") == "1"
FUNDING_RATE = 0.0001       # fallback per-8h funding rate when no funding series loaded

# Entry filter: only go long/short when predicted probability gap is above this.
MIN_SIGNAL_MARGIN = 0.0     # 0.00 = trade every model decision; tune in grid


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# Rolling windows used for SMA/EMA/vol/RSI-like indicators.
ROLL_WINDOWS = [6, 12, 24, 48, 96]   # bars @15m -> 1.5h,3h,6h,12h,24h
HIGHER_TIMEFRAMES = [15, 60]          # context frames in minutes for trend features
FEATURE_MIN_BARS = 150                # bars required before features are trusted


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    taker_fee: float = TAKER_FEE
    slippage_bps: float = SLIPPAGE_BPS
    use_funding: bool = USE_FUNDING
    funding_rate: float = FUNDING_RATE
    initial_capital: float = 10_000.0
    position_pct: float = 1.0          # fraction of equity per trade
    min_signal_margin: float = MIN_SIGNAL_MARGIN
    max_holding_bars: int = 0          # 0 = no cap
    min_hold_bars: int = 0             # 0 = off; hold a position >= N bars (anti-churn)
    vol_target: float = 0.0            # 0 = off; size positions to target realized vol
    bars_per_year: int = 0             # 0 = use config.BARS_PER_YEAR (timeframe-aware)
    # --- price/execution arrays aligned to the input bar grid (optional) ---
    high: np.ndarray | None = None     # kept for compatibility; no TP/SL exits simulated
    low: np.ndarray | None = None      # kept for compatibility; no TP/SL exits simulated
    funding: np.ndarray | None = None  # per-bar funding rate to charge while in-position
    # --- risk knobs ---
    dd_brake: float = 0.0              # 0 = off; cut exposure when running drawdown < -dd_brake
    dd_brake_scale: float = 0.3        # exposure multiplier while the brake is active
    dd_brake_recover: float = 0.6      # recover (re-arm) when drawdown returns above -dd_brake*hysteresis


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardConfig:
    n_folds: int = 6
    train_frac: float = 0.55           # initial train share of total length (expanding)
    embargo_bars: int = 12             # skip this many bars between train end and test
    purge_bars: int = 0                # extra purge at label boundary (usually 0; labels handle it)
    min_test_bars: int = 2000
    min_train_bars: int = 5000


WF = WalkForwardConfig()


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
@dataclass
class GridConfig:
    max_configs: int = 300             # cap on total grid configs evaluated
    n_jobs: int = -1                   # parallel workers (-1 = all cores)
    seed: int = 42
    objective_weight_drawdown: float = 0.5   # Sharpe - w*MaxDD composite
    objective_weight_winrate: float = 0.0    # + w*(win_rate - 0.5) in the objective
    min_config_trades: int = 100       # discard configs with too few trades
    cv_folds: int = 1                  # inner validation per config (1 = last-20% split)
    score_subsample: int = 30000       # cap train rows while scoring configs (speed)
    score_fold_cap: int = 3            # only first N walk-forward folds used for ranking
    min_positive_folds: int = 3        # consensus gate: configs with fewer positive folds
                                       # (by rank-objective) are deprioritized in ranking
    n_estimators_min: int = 100        # models need enough trees to be meaningful


GRID = GridConfig()


# ---------------------------------------------------------------------------
# Live streaming
# ---------------------------------------------------------------------------
LIVE_POLL_SECONDS = 5
LIVE_TAIL_BARS = 500          # recent window kept for live recompute
LIVE_RETRAIN_SECONDS = int(os.environ.get("BOT_RETRAIN_SECONDS", 7 * 86400))  # refit cadence