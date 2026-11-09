"""Vote-threshold sweep at spot economics on a saved artifact's ensemble members.

Reuses combined_walkforward (per-fold Sharpe-weighted majority vote) with varying
`vote` thresholds, without re-running the grid. Run:
    PYTHONPATH=$PWD .venv/bin/python sweep_vote.py --artifact best_spot_winrate_v05
"""
from __future__ import annotations

import argparse

import numpy as np

import config
from report import load_artifacts
from grid import combined_walkforward, prepare_dataset
from backtest import objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="best_spot_winrate_v05")
    ap.add_argument("--vote", nargs="+", type=float, default=[0.5, 0.55, 0.6, 0.65, 0.7])
    args = ap.parse_args()

    data, X, df, _ = load_artifacts(args.artifact)
    config.configure_timeframe(data["report"]["meta"].get("timeframe", "4h"),
                               long_only=data["report"]["meta"].get("long_only", True))
    members = data["report"]["ensemble"]
    base = df.reindex(X.index)
    arrs = {
        "close": base["close"].to_numpy(float),
        "high": base["high"].to_numpy(float),
        "low": base["low"].to_numpy(float),
        "funding": (X["funding_rate"].to_numpy(float)
                    if "funding_rate" in X.columns else None),
    }

    print(f"artifact={args.artifact}  members={len(members)}  "
          f"fee={config.TAKER_FEE}  use_funding={config.USE_FUNDING}")
    print("vote  sharpe  sortino  ret    dd      trades  win    folds+")
    for v in args.vote:
        folds, fm, tbt, rolled, _w = combined_walkforward(
            members, X, df, arrs, combine="vote", vote=v)
        if rolled is None:
            print(f"{v:4.2f}  no rolled curve")
            continue
        pos = np.asarray(rolled["positions"])
        pr = np.asarray(rolled["strat_ret"])
        N = len(pr)                       # strat_ret is one shorter than positions
        p = pos[:N + 1]                   # position earned into each bar
        prev = np.concatenate([[0.0], p[:N - 1]])
        trades = int(((p[1:N + 1] != 0) & (prev == 0)).sum())
        win = float(rolled["win_rate"])
        pos_folds = sum(1 for m in fm if m["sharpe"] > 0)
        print(f"{v:4.2f}  {rolled['sharpe']:6.2f}  {rolled['sortino']:7.2f}  "
              f"{rolled['equity'][-1]/config.BacktestConfig.initial_capital - 1:7.3f}  "
              f"{rolled['max_drawdown']:6.3f}  {trades:6d}  {win:5.3f}  {pos_folds}/{len(fm)}")


if __name__ == "__main__":
    main()
