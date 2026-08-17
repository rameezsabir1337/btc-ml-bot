"""Realistic vectorized backtester: signals -> position -> returns -> metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from config import BacktestConfig, WF

# NB: annualization uses config.BARS_PER_YEAR at CALL time (timeframe-aware).
# Do not bind it at import time (configure_timeframe can change it later).


def to_positions(raw_pred: np.ndarray, margin: float) -> np.ndarray:
    """Map model output to target positions in {-1, 0, 1}.
    - regression: threshold by margin (min_signal_margin)
    - binary / multiclass: use as-is (already in {-1,0,1})
    """
    arr = np.asarray(raw_pred, dtype=float)
    lo, hi = arr.min(), arr.max()
    if lo >= -1.05 and hi <= 1.05 and set(np.unique(arr)).issubset({-1.0, 0.0, 1.0}):
        return arr  # already class-labeled
    # continuous scores -> threshold by margin around zero
    out = np.where(arr > margin, 1.0, np.where(arr < -margin, -1.0, 0.0))
    if config.CONFIDENCE_SIZING and lo != hi:
        # size the position by how far the score is past the margin (0..1),
        # so a weak long is barely invested and a strong bearish (short) score
        # drives the position to 0 -- short side informs exposure without trading.
        span = max(hi, -lo) - margin
        strength = np.clip((np.abs(arr) - margin) / span if span > 0 else 0.0, 0.0, 1.0)
        out = np.sign(out) * strength
    if config.LONG_ONLY:
        out[out < 0] = 0.0
    return out


def _cost_units(pos: np.ndarray) -> np.ndarray:
    """Notional traded per bar: |change in position|. A flat round trip (-1..1 or
    0..1) costs 1 unit per side; a sign reversal -1->+1 costs 2; fractional
    positions (confidence-sizing) pay proportional to the size delta, matching
    how real perp fees scale with notional."""
    d = np.abs(np.diff(np.asarray(pos, dtype=float), prepend=0.0))
    return np.minimum(d, 4.0)


def _min_hold(pos: np.ndarray, n: int) -> np.ndarray:
    """Extend every position segment to at least `n` bars (anti-churn)."""
    if n <= 1:
        return pos
    out = np.asarray(pos, dtype=float).copy()
    i = 0
    L = len(out)
    while i < L:
        if out[i] != 0:
            j = i
            sign = out[i]
            while j < L and out[j] != 0:
                j += 1
            for k in range(i, min(L, i + n)):
                out[k] = sign
            i = j
        else:
            i += 1
    return out


def _dd_brake_mult(returns: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    """Return a per-bar exposure multiplier that slashes size while the strategy is in
    a drawdown past dd_brake, re-arming after partial recovery (hysteresis)."""
    if cfg.dd_brake <= 0:
        return np.ones_like(returns)
    eq = cfg.initial_capital * np.cumprod(1 + np.nan_to_num(returns, nan=0.0))
    peak = np.maximum.accumulate(np.concatenate([[cfg.initial_capital], eq]))[1:]
    dd = eq / peak - 1
    active = np.zeros(len(returns), dtype=bool)
    armed = False
    for i, d in enumerate(dd):
        if d < -cfg.dd_brake:
            armed = True
        elif armed and d > -cfg.dd_brake * cfg.dd_brake_recover:
            armed = False
        active[i] = armed
    return np.where(active, cfg.dd_brake_scale, 1.0)


def backtest(signals: np.ndarray, close: np.ndarray, cfg: BacktestConfig = None,
             start_equity: float = None) -> dict:
    cfg = cfg or BacktestConfig()
    # start_equity lets fold windows chain into a single continuous equity curve
    # (fold N+1 begins at fold N's ending capital instead of resetting).
    start_eq = cfg.initial_capital if start_equity is None else float(start_equity)
    pos = to_positions(signals, cfg.min_signal_margin)
    pos = _min_hold(pos, cfg.min_hold_bars)
    # No TP/SL exits: positions are held until the signal flips/flatten (signal-only).
    n = len(close)
    bar_ret = close[1:] / close[:-1] - 1.0
    cost_per_unit = cfg.taker_fee + cfg.slippage_bps / 1e4
    cost = _cost_units(pos)[:-1] * cost_per_unit
    strat_ret = pos[:-1] * bar_ret - cost
    # realized perp funding paid while in-position. `funding` holds the PER-8H
    # funding rate ffill'd onto the bar grid, so scale it to this bar's share of
    # the 8h period (e.g. /2 for 4h bars, /96 for 15m) before charging.
    bps = cfg.bars_per_year or config.BARS_PER_YEAR
    if cfg.use_funding and cfg.funding is not None and len(cfg.funding) >= len(strat_ret):
        fr = np.nan_to_num(np.asarray(cfg.funding, dtype=float)[:-1], nan=0.0)
        seconds_per_bar = 365 * 24 * 3600 / bps
        per_bar = seconds_per_bar / (8 * 3600)
        strat_ret -= pos[:-1] * fr * per_bar
    # vol-target position sizing: scale exposure inversely to realized vol
    if cfg.vol_target > 0:
        rv = pd.Series(bar_ret).rolling(40, min_periods=10).std().values
        mult = np.clip(cfg.vol_target / rv, 0.1, 3.0)
        mult = np.nan_to_num(mult, nan=1.0)
        strat_ret *= mult
    strat_ret *= _dd_brake_mult(strat_ret, cfg)
    strat_ret = strat_ret * cfg.position_pct
    strat_ret = np.nan_to_num(strat_ret, nan=0.0)

    equity = start_eq * np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(np.concatenate([[start_eq], equity]))[1:]
    dd = equity / peak - 1
    max_dd = float(dd.min()) if len(dd) else 0.0

    bps = cfg.bars_per_year or config.BARS_PER_YEAR
    ann = bps / len(strat_ret) if len(strat_ret) else 0.0
    m = strat_ret.mean()
    sd = strat_ret.std()
    sharpe = (m * bps) / (sd * np.sqrt(bps)) if sd > 0 else 0.0
    downside = strat_ret[strat_ret < 0]
    sortino = (m * bps) / (downside.std() * np.sqrt(bps)) if len(downside) and downside.std() > 0 else 0.0

    wins = strat_ret[strat_ret > 0]
    losses = strat_ret[strat_ret < 0]
    # a trade = one position round-trip: count entries, plus each sign-reversal
    # (short->long or long->short closes one position and opens another).
    prev = np.concatenate([[0.0], pos[:-1]])[:-1]   # aligned to pos[:-1]
    entries = int(((pos[:-1] != 0) & (prev == 0)).sum())
    revs = int(((pos[:-1] != 0) & (prev != 0) &
                (np.sign(pos[:-1]) != np.sign(prev))).sum())
    num_trades = entries + revs
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    # win-rate = fraction of active (in-position) bars that end positive
    active = pos[:-1] != 0
    win_rate = float(round((strat_ret[active] > 0).mean(), 4)) if active.any() else 0.0
    total_return = float(equity[-1] / start_eq - 1) if len(equity) else 0.0

    return {
        "total_return": total_return,
        "annual_return": float(equity[-1] / start_eq) ** ann - 1 if equity[-1] > 0 else -1.0,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else None,
        "num_trades": num_trades,
        "bars": n,
        "equity": equity,
        "strat_ret": strat_ret,
        "positions": pos,
    }


def objective(metrics: dict, cfg) -> float:
    """Risk-adjusted objective = sharpe - w*|maxDD|; penalized if too few trades."""
    if metrics["num_trades"] < cfg.min_config_trades:
        return -1e9
    sharpe = metrics["sharpe"]
    dd_pen = cfg.objective_weight_drawdown * abs(metrics["max_drawdown"])
    return sharpe - dd_pen


def summarize(metrics_list: list[dict]) -> dict:
    """Aggregate a list of per-fold metric dicts into a pooled summary."""
    keys = ["sharpe", "sortino", "total_return", "max_drawdown", "win_rate", "num_trades"]
    pool = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if m[k] is not None]
        pool[f"{k}_mean"] = float(np.mean(vals)) if vals else 0.0
        pool[f"{k}_med"] = float(np.median(vals)) if vals else 0.0
    pool["num_folds"] = len(metrics_list)
    pool["folds_positive"] = int(sum(1 for m in metrics_list if m["sharpe"] > 0))
    return pool