"""Retrain-cadence backtest: how much does deployed-model accuracy decay with staleness?

Two complementary views, both reusing the deployed 5-member ensemble + vote weights:

A. Stride decay sweep  -- keep the ~27d folds, refit the ensemble every `stride`
   folds and REUSE the fitted members on the folds in between. stride=1 is the
   deployed baseline (refit every fold); larger strides expose the cost of a
   stale model (max staleness = (stride-1) * ~27 days).

B. Cadence sweep       -- rebuild the walk-forward so each TEST window equals the
   retrain cadence period (7d / 14d / 28d), refitting every window. Directly
   answers "is weekly retraining materially better than monthly?".

Run:  .venv/bin/python cadence.py            (uses artifacts/best)
      .venv/bin/python cadence.py --artifact best_fee_05 --stride 1 2 3 6 --cadence 7 14 28
"""
from __future__ import annotations

import argparse

import numpy as np

import config
from walkforward import Fold, fold_indices
from grid import _labels_for, _fit_predict, _gated_positions, _bt_cfg, _trades_table
from ensemble import _bt, chain_folds, _fold_sharpes
from models import predict_score
from report import load_artifacts


def _deployed_setup(name="best"):
    """Reconstruct members / vote weights / X / arrs from the saved artifact."""
    data, X, df, _ = load_artifacts(name)
    report = data["report"]
    members = report["ensemble"]
    meta = report.get("meta", {}).get("ensemble_vote", {})
    vote = float(meta.get("vote", 0.5))
    base = df.reindex(X.index)
    arrs = {"close": base["close"].to_numpy(float),
            "high": base["high"].to_numpy(float),
            "low": base["low"].to_numpy(float)}
    arrs["funding"] = (X["funding_rate"].to_numpy(float)
                       if "funding_rate" in X.columns else None)
    return members, vote, X, df, arrs


def cadence_folds(n, test_len, horizon, cfg=None):
    """Expanding folds whose test windows are each ~`test_len` bars (a retrain
    cadence). Mirrors expanding_folds(): train [0, test_start-embargo-horizon],
    embargo gap, then the test window; expanding train across windows."""
    cfg = cfg or config.WF
    t0 = int(n * cfg.train_frac)
    folds = []
    test_start = t0 + cfg.embargo_bars
    while test_start + test_len <= n:
        raw_end = test_start - cfg.embargo_bars
        purged = max(0, raw_end - horizon)
        folds.append(Fold(0, purged, test_start, test_start + test_len,
                          train_raw_end=raw_end))
        test_start += test_len
        if len(folds) > 80:
            break
    return folds


def cadence_walkforward(members, X, df, arrs, folds, weights, vote, stride=1):
    """Weighted-vote ensemble with RETRAIN-ONLY-EVERY-`stride`-FOLDS. Members are
    fit fresh when fold_index % stride == 0, otherwise the previously fitted
    model is reused and only re-predicted/gated on the new test window."""
    fold_metrics = []
    fitted: dict[int, tuple] = {}
    weights = np.clip(weights, 1e-3, None)   # silence persistently-negative members
    for fi, fold in enumerate(folds):
        tr, te = fold_indices(fold)
        sl = slice(fold.test_start, fold.test_end)
        acc = np.zeros(len(te))
        wsum = 0.0
        ts = None
        for mi, m in enumerate(members):
            if fi % stride == 0:
                y = _labels_for(df, m, X)
                model, scores, task = _fit_predict(m, X.iloc[tr], y[tr], X.iloc[te])
                fitted[mi] = (model, task)
            else:
                model, task = fitted.get(mi, (None, None))
                scores = None if model is None else \
                    predict_score(model, X.iloc[te].values, task)
            if model is None:
                continue
            pos = _gated_positions(m, scores, X.iloc[te])
            acc += weights[mi, fi] * pos
            wsum += weights[mi, fi]
            ts = X.index[te]
        if wsum <= 0:
            continue
        votes = acc / wsum
        pos = (votes >= vote).astype(float)
        m = _bt(members, fold, pos, arrs, ts)
        m["scores"] = votes
        fold_metrics.append(m)
    return fold_metrics


def run(name, stride, vote, fold_override=None):
    """Run one ensemble walk-forward; return (folds, fold_metrics, rolled)."""
    config.configure_timeframe("4h", long_only=True)  # artifact mode
    members, vote_cfg, X, df, arrs = _deployed_setup(name)
    vote = vote or vote_cfg
    horizon = members[0]["label"]["horizon"]
    if fold_override is not None:
        folds = cadence_folds(len(X), fold_override, horizon)
    else:
        from walkforward import expanding_folds
        folds = expanding_folds(len(X), config.WF, horizon=horizon)
    wgrid = np.asarray([_fold_sharpes(m, X, df, arrs, folds) for m in members])
    fm = cadence_walkforward(members, X, df, arrs, folds, wgrid, vote, stride=stride)
    fm, rolled = chain_folds(fm, folds, arrs, _bt_cfg(members[0], arrs, slice(0, 0)),
                             config.BARS_PER_YEAR)
    return folds, fm, rolled


def row(label, rolled, fm):
    pos = np.asarray(rolled["positions"])
    pr = np.asarray(rolled["strat_ret"])
    n = min(len(pos), len(pr))
    pos, pr = pos[:n], pr[:n]
    prev = np.concatenate([[0.0], pos[:-1]])
    trades = int(((pos != 0) & (prev == 0)).sum() +
                 ((pos != 0) & (prev != 0) & (np.sign(pos) != np.sign(prev))).sum())
    win = float((pr[pos != 0] > 0).mean()) if (pos != 0).any() else 0.0
    pos_folds = sum(1 for m in fm if m["sharpe"] > 0)
    return (f"{label:<22} sharpe={rolled['sharpe']:6.2f} "
            f"sortino={rolled['sortino']:6.2f} "
            f"ret={rolled['equity'][-1] / config.BacktestConfig.initial_capital - 1:7.3f} "
            f"dd={rolled['max_drawdown']:6.3f} trades={trades:4d} "
            f"win={win:.3f} folds+={pos_folds}/{len(fm)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="best")
    ap.add_argument("--stride", nargs="+", type=int, default=[1, 2, 3, 6])
    ap.add_argument("--cadence", nargs="+", type=int, default=[7, 14, 28])
    ap.add_argument("--vote", type=float, default=None)
    args = ap.parse_args()

    config.configure_timeframe("4h", long_only=True)  # artifact mode
    bars_per_day = int(round(config.BARS_PER_YEAR / 365.0))

    print(f"=== A. Stride decay sweep (artifact={args.artifact}) ===")
    for stride in args.stride:
        folds, fm, rolled = run(args.artifact, stride, args.vote)
        d = int(round((folds[0].test_end - folds[0].test_start) / bars_per_day))
        print(f"  stride={stride} (refit every ~{stride * d}d)  "
              + row(f"stride {stride}", rolled, fm))

    print("\n=== B. Cadence sweep (refit every window, test window = cadence) ===")
    for cdays in args.cadence:
        folds, fm, rolled = run(args.artifact, 1, args.vote,
                                fold_override=cdays * bars_per_day)
        print("  " + row(f"cadence {cdays}d", rolled, fm))


if __name__ == "__main__":
    main()
