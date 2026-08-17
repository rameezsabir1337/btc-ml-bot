"""Walk-forward grid search. Ranks configs on inner (purged) validation, then
evaluates the best config honestly on untouched out-of-sample test folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import GRID, WF, BacktestConfig, MIN_HOLD_BARS
from walkforward import expanding_folds, fold_indices
from backtest import to_positions, backtest, objective
from labels import make_labels
from models import make_model, fit_score
from search_space import build_configs


def _task(method: str) -> str:
    return "regression" if method == "regression" else \
        ("multiclass" if method == "multiclass" else "binary")


def _labels_for(df, c, X=None) -> np.ndarray:
    """Labels for a config, aligned to df.index (df is cleaned OHLCV).
    If X is given, reindex to X.index so feature/label rows stay in sync
    (feature_matrix trims the warmup window)."""
    y, _ = make_labels(df, c["label"]["method"], c["label"])
    if X is not None:
        s = pd.Series(y, index=df.index)
        return s.reindex(X.index).values
    return y


def _fit_predict(c, Xtr, ytr, Xte):
    task = _task(c["label"]["method"])
    if task == "binary":
        keep = ytr != 0
        Xtr, ytr = Xtr[keep], ytr[keep]
    if len(ytr) < GRID.n_estimators_min or len(np.unique(ytr)) < 2:
        return None, None, None
    model = make_model(c["model"], c["model_params"], task)
    scores = fit_score(model, Xtr, ytr, Xte, task)
    return model, scores, task


def _gated_positions(c, scores, Xte):
    """Map scores to positions, optionally gated by a higher-TF trend column:
    longs only when the gate trend is up, shorts only when it's down.
    A funding regime gate (funding_gate: funding-z threshold, 0=off) blocks
    longs when funding is crowded-positive and shorts when crowded-negative.
    The htf gate has hysteresis: it must be confirmed for `GATE_CONFIRM_BARS`
    consecutive bars before exposure flips (reduces whipsaw)."""
    import config as _cfg
    pos = to_positions(scores, c["margin"])
    gate = c.get("trend_gate", "none")
    if gate.startswith("htf") and gate.endswith("_slope") and gate in Xte.columns:
        g = pd.Series(Xte[gate].values).fillna(0.0)
        confirm = max(1, _cfg.GATE_CONFIRM_BARS)
        gd = np.sign(g)
        # require the gate to be CONFIRMED for `confirm` consecutive bars before
        # exposure is allowed: all last `confirm` bars strictly up = long ok.
        rmin = pd.Series(gd).rolling(confirm, min_periods=confirm).min()
        rmax = pd.Series(gd).rolling(confirm, min_periods=confirm).max()
        long_ok = rmin.fillna(0.0) > 0
        short_ok = rmax.fillna(0.0) < 0
        pos[(pos > 0) & ~long_ok] = 0.0
        pos[(pos < 0) & ~short_ok] = 0.0
    fz_th = float(c.get("funding_gate", 0.0) or 0.0)
    if fz_th > 0 and "funding_z" in Xte.columns:
        fz = Xte["funding_z"].values
        pos[(pos > 0) & (fz > fz_th)] = 0.0
        pos[(pos < 0) & (fz < -fz_th)] = 0.0
    return pos


def _bt_cfg(c, arrs, sl) -> BacktestConfig:
    """Build a BacktestConfig pulling the new risk knobs from the grid config
    + the X-aligned price/funding slices for the fold window."""
    import config as _cfg
    return BacktestConfig(
        min_signal_margin=c["margin"],
        min_hold_bars=MIN_HOLD_BARS,
        vol_target=float(c.get("vol_target", 0.0) or 0.0),
        dd_brake=float(c.get("dd_brake", 0.0) or 0.0),
        bars_per_year=_cfg.BARS_PER_YEAR,
        high=arrs["high"][sl],
        low=arrs["low"][sl],
        funding=arrs["funding"][sl] if arrs.get("funding") is not None else None,
    )


def _backtest_fold(c, X, df, arrs, fold) -> dict | None:
    tr, te = fold_indices(fold)
    y = _labels_for(df, c, X)
    Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
    model, scores, task = _fit_predict(c, Xtr, ytr, Xte)
    if model is None:
        return None
    pos = _gated_positions(c, scores, Xte)
    sl = slice(fold.test_start, fold.test_end)
    bt_cfg = _bt_cfg(c, arrs, sl)
    m = backtest(pos, arrs["close"][sl], bt_cfg)
    m["ts"] = X.index[te]
    m["scores"] = scores
    # m["positions"] (from backtest) is the FINAL post-exit position, matching returns
    return m


def score_config(c, X, df, arrs) -> float:
    """Median validation objective across (all) walk-forward folds, with a
    robustness gate: configs whose positive folds are below `GRID.min_positive_folds`
    are deeply penalized so a single lucky fold can't win the grid."""
    folds = expanding_folds(len(X), WF, horizon=c["label"]["horizon"])[:GRID.score_fold_cap]
    objs = []
    y = _labels_for(df, c, X)
    for fold in folds:
        tr, te = fold_indices(fold)
        vsplit = int(len(tr) * 0.8)
        vtr, vte = tr[:vsplit], tr[vsplit:]
        if len(vtr) < GRID.n_estimators_min or len(vte) < 2:
            continue
        if len(vtr) > GRID.score_subsample:
            rng = np.random.RandomState(GRID.seed)
            vtr = vtr[rng.choice(len(vtr), GRID.score_subsample, replace=False)]
        Xtr, ytr, Xv = X.iloc[vtr], y[vtr], X.iloc[vte]
        model, scores, task = _fit_predict(c, Xtr, ytr, Xv)
        if model is None:
            continue
        pos = _gated_positions(c, scores, Xv)
        sl = slice(vte[0], vte[-1] + 1)
        bt_cfg = _bt_cfg(c, arrs, sl)
        m = backtest(pos, arrs["close"][sl], bt_cfg)
        objs.append(objective(m, GRID))
    if not objs:
        return -1e9
    med = float(np.median(objs))
    if not np.isfinite(med):
        return -1e9
    n_pos = int(sum(1 for o in objs if o > 0))
    if n_pos < GRID.min_positive_folds:
        # robust-config penalty: keep *some* ordering among non-qualifiers by score
        return med - 10.0
    return med


def _grid_worker_init(tf: str, long_only: bool, confidence_sizing: bool = False) -> None:
    """joblib loky workers re-import `config` fresh; re-apply the active mode
    so fold/grid bounds and LONG_ONLY are correct in every subprocess."""
    from config import configure_timeframe
    configure_timeframe(tf, long_only=long_only)
    if confidence_sizing:
        import config as _c
        _c.CONFIDENCE_SIZING = True


def run_grid(X, df, arrs, configs=None, n_jobs=GRID.n_jobs):
    """Rank configs via parallel inner-CV scoring."""
    import config as _cfg
    configs = configs or build_configs()
    scores = Parallel(n_jobs=n_jobs,
                      initializer=_grid_worker_init,
                      initargs=(_cfg.TIMEFRAME, _cfg.LONG_ONLY, _cfg.CONFIDENCE_SIZING))(
        delayed(score_config)(c, X, df, arrs) for c in configs
    )
    return sorted(zip(configs, scores), key=lambda t: t[1], reverse=True)


def _trades_table(folds, fold_metrics, arrs):
    """Dashboard fold schema. `equity` already includes each fold's starting
    equity (from a rolled/continuous curve), so tiles chain across folds."""
    rows = []
    cap = BacktestConfig.initial_capital
    for f, m in zip(folds, fold_metrics):
        if m is None:
            continue
        sl = slice(f.test_start, f.test_end)
        close_fold = np.asarray(arrs["close"][sl], dtype=float)
        rows.append({
            "test_start": int(f.test_start),
            "test_end": int(f.test_end),
            "metrics": {k: m[k] for k in
                        ("sharpe", "sortino", "total_return", "max_drawdown",
                         "win_rate", "num_trades")},
            "ts": [int(ts.value // 10**6) for ts in m["ts"]],
            "equity": [round(float(x), 6) for x in m["equity"]],
            "buyhold": [round(float(cap * x / close_fold[0]), 6)
                        for x in close_fold],
        })
    return rows


def final_walkforward(best_c, X, df, arrs):
    """True out-of-sample evaluation of the chosen config across all folds,
    chained into a single continuous equity curve (no per-fold capital reset)."""
    import config as _cfg
    from ensemble import chain_folds
    folds = expanding_folds(len(X), WF, horizon=best_c["label"]["horizon"])
    fold_metrics = []
    for fold in folds:
        m = _backtest_fold(best_c, X, df, arrs, fold)
        if m is not None:
            fold_metrics.append(m)
    fold_metrics, rolled = chain_folds(fold_metrics, folds, arrs,
                                       _bt_cfg(best_c, arrs, slice(0, 0)),
                                       _cfg.BARS_PER_YEAR)
    trades_by_fold = _trades_table(folds, fold_metrics, arrs)
    return folds, fold_metrics, trades_by_fold, rolled


def diverse_subset(configs, n):
    """Pick the top-n configs by (already-sorted) score while forcing model-family
    diversity so the ensemble averages out per-model overfitting rather than
    doubling down on one algorithm."""
    chosen, seen = [], set()
    for c in configs:
        key = (c["model"], c["label"]["method"])
        if key in seen:
            continue
        if len(chosen) >= n:
            break
        seen.add(key)
        chosen.append(c)
    return chosen or configs[:1]


def ensemble_walkforward(top_configs, X, df, arrs, vote=0.5):
    """Out-of-sample evaluation of a majority-vote ensemble: per fold, fit each
    member on the train window, gate/clamp positions, average across members, then
    require >= `vote` agreement to go long (long-only members produce 0/1).
    Fold results are chained into one continuous equity curve."""
    import config as _cfg
    from ensemble import chain_folds
    horizon = top_configs[0]["label"]["horizon"]
    folds = expanding_folds(len(X), WF, horizon=horizon)
    fold_metrics = []
    for fold in folds:
        tr, te = fold_indices(fold)
        sl = slice(fold.test_start, fold.test_end)
        acc = np.zeros(len(te), dtype=float)
        valid = 0
        for c in top_configs:
            y = _labels_for(df, c, X)
            model, scores, task = _fit_predict(c, X.iloc[tr], y[tr], X.iloc[te])
            if model is None:
                continue
            acc += _gated_positions(c, scores, X.iloc[te])
            valid += 1
        if valid == 0:
            continue
        votes = acc / valid
        pos = (votes >= vote).astype(float)
        m = backtest(pos, arrs["close"][sl], _bt_cfg(top_configs[0], arrs, sl))
        m["ts"] = X.index[te]
        m["scores"] = votes
        # m["positions"] (from backtest) is the FINAL post-exit position
        fold_metrics.append(m)
    fold_metrics, rolled = chain_folds(fold_metrics, folds, arrs,
                                       _bt_cfg(top_configs[0], arrs, slice(0, 0)),
                                       _cfg.BARS_PER_YEAR)
    trades_by_fold = _trades_table(folds, fold_metrics, arrs)
    return folds, fold_metrics, trades_by_fold, rolled


def combined_walkforward(members, X, df, arrs, combine="vote", vote=0.5):
    """Out-of-sample evaluation of an ensemble, chained into ONE equity curve.
      'vote'  -> per-fold Sharpe-weighted majority vote over gated positions (V1)
      'blend' -> average member scores, threshold by the winner's margin, uses
                 the htf1440 trend gate (V2, long-only aware).
    Returns (folds, fold_metrics, trades_by_fold, rolled, weights) where
    `weights` = mean per-fold OOS Sharpe of each member (vote weights, clipped
    at 1e-3 so persistently-negative members are silenced)."""
    import config as _cfg
    import numpy as np
    from ensemble import (chain_folds, weighted_vote_walkforward,
                          blend_walkforward, _fold_sharpes)
    folds = expanding_folds(len(X), WF, horizon=members[0]["label"]["horizon"])
    if combine == "blend":
        fm = blend_walkforward(members, X, df, arrs, folds, margin=members[0]["margin"])
        weights = [1.0] * len(members)
    else:
        wgrid = np.asarray([_fold_sharpes(m, X, df, arrs, folds) for m in members])
        weights = np.clip(wgrid.mean(axis=1), 1e-3, None)
        fm = weighted_vote_walkforward(members, X, df, arrs, folds,
                                       weights=wgrid, vote=vote)
    fm, rolled = chain_folds(fm, folds, arrs,
                             _bt_cfg(members[0], arrs, slice(0, 0)),
                             _cfg.BARS_PER_YEAR)
    trades_by_fold = _trades_table(folds, fm, arrs)
    return folds, fm, trades_by_fold, rolled, [float(w) for w in weights]


def prepare_dataset(df):
    """Feature matrix + X-aligned price/funding arrays (close/high/low/funding)."""
    from config import SWING_HTF, SWING
    from features import feature_matrix, attach_funding_matrix, attach_oi_matrix
    X = feature_matrix(df, htf=SWING_HTF)
    if SWING:                      # swing modes get funding-rate + OI regime features
        X = attach_funding_matrix(df, X)
        X = attach_oi_matrix(df, X)
    base = df.reindex(X.index)
    arrs = {
        "close": base["close"].to_numpy(dtype=float),
        "high": base["high"].to_numpy(dtype=float),
        "low": base["low"].to_numpy(dtype=float),
        "funding": (X["funding_rate"].to_numpy(dtype=float)
                    if "funding_rate" in X.columns else None),
    }
    return X, arrs


__all__ = ["run_grid", "final_walkforward", "ensemble_walkforward",
           "diverse_subset", "prepare_dataset", "score_config"]