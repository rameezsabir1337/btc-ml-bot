"""Label (target) generation. All label methods are leakage-safe: they are only
used for training, never fed as features at inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _forward_return(close: pd.Series, horizon: int) -> np.ndarray:
    return close.shift(-horizon).values / close.values - 1


def regression(close: pd.Series, horizon: int) -> np.ndarray:
    """Continuous forward return. Task: 'regression'."""
    return np.nan_to_num(_forward_return(close, horizon), nan=0.0)


def binary_min_move(close: pd.Series, horizon: int, threshold: float) -> np.ndarray:
    """+1 / -1 only when the forward move exceeds |threshold|, else 0 (skip).
    Task: 'binary' (train only on +/-1)."""
    fwd = _forward_return(close, horizon)
    lab = np.where(fwd > threshold, 1.0, np.where(fwd < -threshold, -1.0, 0.0))
    return np.nan_to_num(lab, nan=0.0)


def multi_class(close: pd.Series, horizon: int, up_thresh: float, dn_thresh: float) -> np.ndarray:
    """3-class {short=-1, flat=0, long=+1} with asymmetric thresholds.
    Task: 'multiclass'."""
    fwd = _forward_return(close, horizon)
    lab = np.where(fwd > up_thresh, 1.0, np.where(fwd < -dn_thresh, -1.0, 0.0))
    return np.nan_to_num(lab, nan=0.0)


def _atr_prices(high, low, close, win: int = 14) -> np.ndarray:
    H = np.asarray(high, dtype=float)
    L = np.asarray(low, dtype=float)
    C = np.asarray(close, dtype=float)
    pc = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc)))
    return pd.Series(tr).rolling(win, min_periods=5).mean().fillna(0.0).to_numpy()


def _barrier(high, low, close, horizon, tp, sl, min_final_frac: float = 0.0) -> np.ndarray:
    """Core triple-barrier on explicit tp/sl price arrays (aligned to each bar)."""
    H = high.astype(float)
    L = low.astype(float)
    C = close.astype(float)
    n = len(C)

    def padded(arr):
        p = np.full((n + horizon + 1,), np.nan)
        p[:n] = arr
        return p

    WH = sliding_window_view(padded(H), horizon)[:n]
    WL = sliding_window_view(padded(L), horizon)[:n]
    WC = sliding_window_view(padded(C), horizon)[:n]

    hit_tp = WH >= tp[:, None]
    hit_sl = WL <= sl[:, None]
    has_tp = hit_tp.any(axis=1)
    has_sl = hit_sl.any(axis=1)
    first_tp = np.argmax(hit_tp, axis=1)
    first_sl = np.argmax(hit_sl, axis=1)

    lab = np.zeros(n)
    both = has_tp & has_sl
    lab[both & (first_tp < first_sl)] = 1.0
    lab[both & (first_sl < first_tp)] = -1.0
    lab[both & (first_tp == first_sl)] = 0.0
    lab[has_tp & ~has_sl] = 1.0
    lab[has_sl & ~has_tp] = -1.0

    none = ~has_tp & ~has_sl
    final = WC[:, -1]
    move = final / C - 1
    win = np.abs(move) >= min_final_frac
    lab[none & win & (move > 0)] = 1.0
    lab[none & win & (move < 0)] = -1.0
    lab[none & ~win] = 0.0
    return lab


def triple_barrier(high: pd.Series, low: pd.Series, close: pd.Series,
                   horizon: int, tp_frac: float, sl_frac: float,
                   min_final_frac: float = 0.0) -> np.ndarray:
    """Path-dependent labeling. +1 if take-profit triggers first, -1 if stop-loss
    triggers first, else sign of the final move if above min_final_frac, else 0.
    Task: 'binary' (train only on +/-1)."""
    C = close.values.astype(float)
    return _barrier(high, low, close, horizon, C * (1 + tp_frac),
                    C * (1 - sl_frac), min_final_frac)


def triple_atr(high: pd.Series, low: pd.Series, close: pd.Series,
               horizon: int, tp_atr_mult: float, sl_atr_mult: float,
               min_final_frac: float = 0.0, atr_win: int = 14) -> np.ndarray:
    """Vol-adaptive triple barrier: TP/SL priced as multiples of realized ATR
    (`tp = close + tp_atr_mult * ATR`) so the label meaning is stationary across
    volatility regimes. Task: 'binary'."""
    atr = _atr_prices(high, low, close, atr_win)
    C = close.values.astype(float)
    return _barrier(high, low, close, horizon, C + tp_atr_mult * atr,
                    C - sl_atr_mult * atr, min_final_frac)


# dispatch --------------------------------------------------------------------
LABEL_METHODS = ("binary_min_move", "triple_barrier", "triple_atr", "multi_class", "regression")


def make_labels(df: pd.DataFrame, method: str, params: dict) -> tuple[np.ndarray, str]:
    """Return (labels aligned to df.index, task). df has open/high/low/close indexed by ts."""
    close = df["close"]
    if method == "binary_min_move":
        y = binary_min_move(close, params["horizon"], params["threshold"])
        return np_nan(y), "binary"
    if method == "triple_barrier":
        y = triple_barrier(df["high"], df["low"], close,
                           params["horizon"], params["tp_frac"],
                           params["sl_frac"], params.get("min_final_frac", 0.0))
        return np_nan(y), "binary"
    if method == "triple_atr":
        y = triple_atr(df["high"], df["low"], close,
                       params["horizon"], params["tp_atr_mult"],
                       params["sl_atr_mult"], params.get("min_final_frac", 0.0))
        return np_nan(y), "binary"
    if method == "multi_class":
        y = multi_class(close, params["horizon"], params["up_thresh"], params["dn_thresh"])
        return np_nan(y), "multiclass"
    if method == "regression":
        y = regression(close, params["horizon"])
        return np_nan(y), "regression"
    raise ValueError(f"Unknown label method {method}")


def np_nan(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0)


def entropy_information(nonzero_frac: float) -> float:
    """Fraction of nonzero (tradeable) labels in a sample selection."""
    return float(nonzero_frac)