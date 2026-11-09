"""Promote a saved artifact to a specific ensemble combination: recompute the
walk-forward using the requested combination ('vote' or Option-B 'regime'),
rebuild the report (pooled metrics on the *continuous* chained curve + the
ensemble/regime meta needed by the live dashboard), and write to artifacts/best.

Usage:
    BOT_TAKER_FEE=0.0010 PYTHONPATH=$PWD .venv/bin/python promote_vote.py \\
        --artifact best_spot_1d_win12 --combine vote --vote 0.65
    BOT_TAKER_FEE=0.0010 PYTHONPATH=$PWD .venv/bin/python promote_vote.py \\
        --artifact best_spot_1d_win12 --combine regime --soft-low 0.25 --soft-mid 0.50
"""
from __future__ import annotations

import argparse
import shutil

import config
from grid import combined_walkforward
from report import build_report, load_artifacts, save_artifacts, save_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--vote", type=float, default=0.65)
    ap.add_argument("--combine", default="vote", choices=["vote", "regime"])
    ap.add_argument("--soft-low", type=float, default=0.25)
    ap.add_argument("--soft-mid", type=float, default=0.50)
    ap.add_argument("--weights", default="sharpe", choices=["sharpe", "equal"],
                    help="ensemble vote weights: per-fold OOS Sharpe (lookahead) "
                         "or a uniform vote (honest)")
    ap.add_argument("--policy", default=None, choices=["regime"],
                    help="apply the shared regime-adaptive entry policy "
                         "(per-regime vote threshold + entry filter)")
    ap.add_argument("--backup", action="store_true", help="back up the current best first")
    args = ap.parse_args()

    data, X, df, _ = load_artifacts(args.artifact)
    config.configure_timeframe(data["report"]["meta"].get("timeframe", "4h"),
                               long_only=data["report"]["meta"].get("long_only", True))
    members = data["report"]["ensemble"]
    best_c = members[0]
    base = df.reindex(X.index)
    arrs = {
        "close": base["close"].to_numpy(float),
        "high": base["high"].to_numpy(float),
        "low": base["low"].to_numpy(float),
        "funding": (X["funding_rate"].to_numpy(float)
                    if "funding_rate" in X.columns else None),
    }

    regime = None
    if args.combine == "regime":
        regime = {"soft_low": args.soft_low, "soft_mid": args.soft_mid,
                  "weights_equal": args.weights == "equal"}
        if args.policy:
            from ensemble import ENTRY_POLICY
            regime["entry_policy"] = ENTRY_POLICY
    print(f"promoting {args.artifact} -> combine={args.combine} vote={args.vote} "
          f"weights={args.weights} policy={args.policy} regime={regime}  "
          f"tf={config.TIMEFRAME}  fee={config.TAKER_FEE}  "
          f"funding={config.USE_FUNDING}")
    folds, fm, tbt, rolled, weights = combined_walkforward(
        members, X, df, arrs, combine=args.combine, vote=args.vote, regime=regime)
    if rolled is None:
        raise RuntimeError("no rolled curve produced")

    report = build_report(
        best_c, folds, fm, [(c, 0.0) for c in members],
        ensemble_members=members, ensemble_weights=weights, ensemble_vote=args.vote,
        rolled_metrics=rolled, rolled_curve=rolled)
    if args.combine == "regime":
        rg = rolled.get("_regime", {}) or {}
        report["meta"]["regime"] = {
            "gate_col": "htf10080_slope",
            "soft_low": args.soft_low,
            "soft_mid": args.soft_mid,
            "confirm": int(getattr(config, "GATE_CONFIRM_BARS", 2)),
            "rows": rg.get("rows") or [],
            "entry_policy": rg.get("entry_policy"),
            "weights_equal": bool(rg.get("weights_equal", False)),
        }
    print(f"  pooled: sharpe={rolled['sharpe']:.2f}  ret={rolled['total_return']:.3f}  "
          f"dd={rolled['max_drawdown']:.3f}  win={rolled['win_rate']:.3f}  "
          f"trades={rolled['num_trades']}  folds+={report['pooled_metrics']['folds_positive']}")

    if args.backup:
        shutil.rmtree("artifacts/best", ignore_errors=True)
    save_artifacts(report, tbt, folds, df, X)
    save_predictions(fm)
    print(f"  written to artifacts/best ({args.combine}, vote={args.vote})")


if __name__ == "__main__":
    main()