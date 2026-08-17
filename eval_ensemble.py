"""Reproducible out-of-sample evaluation of the deployed-style V1 weighted-vote
ensemble on the frozen dataset, with correct horizon purging (no label leak)
and timeframe-aware annualization (4h bps), persisting honest artifacts for
the dashboard.

Usage:  .venv/bin/python eval_ensemble.py --name best
Quick:  .venv/bin/python eval_ensemble.py --name best2 --members 2

This is the persisted replacement for the ephemeral heredoc that produced
artifacts/best. Member specs below are the DEFAULTS; pass --list to print them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config import ARTIFACT_DIR
from walkforward import expanding_folds
from ensemble import weighted_vote_walkforward, blend_walkforward, chain_folds
from grid import _bt_cfg
from backtest import summarize
from report import build_report, print_report, save_artifacts, save_predictions

FROZEN = Path(__file__).resolve().parent / "frozen_eval"

# (horizon, threshold, margin) for each lgb binary_min_move member. Fleet picked
# by search_members.py --lofo (top fleet + modal LOFO holdout pick): all lgb,
# htf1440_slope gate, dd_brake 0.03, vol_target 0.1 (no TP/SL exits). Vote 0.5.
MEMBER_SPECS = [
    (60, 0.001, 0.0),
    (60, 0.001, 0.1),
    (90, 0.0005, 0.2),
]

MODEL_PARAMS = {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05,
                "min_child_samples": 50, "subsample": 0.9, "colsample_bytree": 0.8}


def make_member(spec):
    h, th, m = spec
    return {
        "model": "lgb",
        "model_params": dict(MODEL_PARAMS),
        "label": {"method": "binary_min_move", "horizon": int(h), "threshold": float(th)},
        "margin": float(m),
        "trend_gate": "htf1440_slope",
        "funding_gate": 0.0,
        "vol_target": 0.1, "dd_brake": 0.03,
    }


def load_frozen():
    X = pd.read_parquet(FROZEN / "features.parquet")
    X["ts"] = pd.to_datetime(X["ts"], unit="ms", utc=True)
    X = X.set_index("ts").sort_index()
    df = pd.read_parquet(FROZEN / "ohlcv.parquet")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    arrs = np.load(FROZEN / "arrs.npz")
    return X, df, arrs


def trades_table(folds, fold_metrics, arrs):
    """Rebuild the dashboard `folds` schema (metrics + ts + equity + buyhold).
    `m['equity']` already starts at the fold's starting capital, so tiles chain
    across folds on the continuous (rolled) curve."""
    cap = config.BacktestConfig.initial_capital
    rows = []
    for fold, m in zip(folds, fold_metrics):
        if m is None:
            continue
        sl = slice(fold.test_start, fold.test_end)
        close_fold = np.asarray(arrs["close"][sl], dtype=float)
        rows.append({
            "test_start": int(fold.test_start),
            "test_end": int(fold.test_end),
            "metrics": {k: m[k] for k in
                        ("sharpe", "sortino", "total_return", "max_drawdown",
                         "win_rate", "num_trades")},
            "ts": [int(ts.value // 10**6) for ts in m["ts"]],
            "equity": [round(float(x), 6) for x in m["equity"]],
            "buyhold": [round(float(cap * x / close_fold[0]), 6) for x in close_fold],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Reproducible V1 ensemble evaluation")
    ap.add_argument("--name", default="best", help="artifact dir under artifacts/")
    ap.add_argument("--vote", type=float, default=0.5)
    ap.add_argument("--members", type=int, default=len(MEMBER_SPECS),
                    help="use only the first N member specs")
    ap.add_argument("--blend", action="store_true",
                    help="also evaluate V2 score-blend (reference)")
    ap.add_argument("--list", action="store_true", help="print member specs and exit")
    args = ap.parse_args()

    if args.list:
        for i, spec in enumerate(MEMBER_SPECS):
            print(i, json.dumps(make_member(spec)))
        return

    config.configure_timeframe("4h", long_only=True)
    X, df, arrs = load_frozen()

    members = [make_member(s) for s in MEMBER_SPECS[:args.members]]
    horizon = max(m["label"]["horizon"] for m in members)
    folds = expanding_folds(len(X), config.WF, horizon=horizon)
    print(f"[eval] {len(members)} members, purge=horizon {horizon}, "
          f"folds={len(folds)} (4h bps={config.BARS_PER_YEAR})")

    fold_metrics = weighted_vote_walkforward(members, X, df, arrs, folds, vote=args.vote)
    fold_metrics, rolled = chain_folds(fold_metrics, folds, arrs,
                                       _bt_cfg(members[0], arrs, slice(0, 0)),
                                       config.BARS_PER_YEAR)
    trades_by_fold = trades_table(folds, fold_metrics, arrs)

    report = build_report(members[0], folds, fold_metrics, [], ensemble_members=members,
                          rolled_metrics=rolled, rolled_curve=rolled)
    report["meta"].update({"_runner": "eval_ensemble.py", "vote": args.vote,
                           "purge_horizon": horizon})
    print_report(report, trades_by_fold)
    save_artifacts(report, trades_by_fold, folds, df, X, name=args.name)
    save_predictions(fold_metrics, name=args.name)

    if args.blend:
        bm = blend_walkforward(members, X, df, arrs, folds, margin=0.0)
        print("\n[V2 blend reference]", summarize(bm))
    print(f"\nSaved honest artifacts to {ARTIFACT_DIR}/{args.name}/")


if __name__ == "__main__":
    main()