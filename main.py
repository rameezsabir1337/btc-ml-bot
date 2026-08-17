"""CLI orchestrator: fetch -> features -> grid search -> best config walk-forward -> report."""
from __future__ import annotations

import argparse
import time

import numpy as np

import config
from data import load_or_fetch, Fetcher
from grid import (run_grid, final_walkforward, prepare_dataset, ensemble_walkforward,
                  combined_walkforward, diverse_subset)
from report import build_report, print_report
import report as _report
from search_space import build_configs


def main():
    ap = argparse.ArgumentParser(description="BTCUSDT ML trading bot (walk-forward grid search)")
    ap.add_argument("--days", type=int, default=config.HISTORY_DAYS)
    ap.add_argument("--tf", type=str, default=config.TIMEFRAME,
                    choices=["15m", "4h"],
                    help="bar timeframe; tunes fold/grid bounds")
    ap.add_argument("--long-only", action="store_true",
                    help="never take short positions (clamp negatives to 0)")
    ap.add_argument("--no-gates", action="store_true",
                    help="disable trend + funding regime gates (position = score vs margin only)")
    ap.add_argument("--confidence-sizing", action="store_true",
                    help="scale long size by signal strength; strong bearish score goes flat")
    ap.add_argument("--force-trend-gate", type=str, default=None, metavar="COL",
                    help="restrict grid to this trend-gate column (e.g. htf1440_slope)")
    ap.add_argument("--ensemble", type=int, default=0, metavar="N",
                    help="ensemble over top-N diverse configs (default 0 = single best)")
    ap.add_argument("--combine", type=str, default="vote", choices=["vote", "blend"],
                    help="ensemble combiner: 'vote' = Sharpe-weighted majority vote, "
                         "'blend' = average scores w/ htf1440 trend gate (long-only)")
    ap.add_argument("--force-refresh", action="store_true")
    ap.add_argument("--max-configs", type=int, default=config.GRID.max_configs)
    ap.add_argument("--n-jobs", type=int, default=config.GRID.n_jobs)
    ap.add_argument("--quick", action="store_true",
                    help="run a tiny labeled smoke grid for testing")
    ap.add_argument("--show-grid", action="store_true", help="print top-N configs after ranking")
    ap.add_argument("--fetch-only", action="store_true", help="download + clean data and exit")
    args = ap.parse_args()

    t0 = time.time()
    config.configure_timeframe(args.tf, long_only=args.long_only)
    if args.no_gates:
        config.GATES_ON = False
    if args.confidence_sizing:
        config.CONFIDENCE_SIZING = True
    if args.force_trend_gate:
        config.FORCE_TREND_GATE = args.force_trend_gate
    config.GRID.max_configs = min(args.max_configs, 8) if args.quick else args.max_configs
    if args.quick:
        config.GRID.score_fold_cap = 1
        config.GRID.score_subsample = 15000

    print(f"[1/4] Fetching {args.days} days of {config.TIMEFRAME} {config.SYMBOL} "
          f"({'long-only' if config.LONG_ONLY else 'long+short'}) ...")
    df = load_or_fetch(days=args.days, force_refresh=args.force_refresh,
                       timeframe=config.TIMEFRAME)
    print(f"      {len(df):,} clean bars")
    if args.fetch_only:
        return

    print("[2/4] Building feature matrix ...")
    X, arrs = prepare_dataset(df)
    print(f"      {X.shape[0]:,} rows x {X.shape[1]} features")

    configs = build_configs()
    configs = configs[:args.max_configs]
    print(f"[3/4] Grid search over {len(configs)} configs ...")
    ranked = run_grid(X, df, arrs, configs=configs, n_jobs=args.n_jobs)
    best_c, best_score = ranked[0]
    print(f"      best score={best_score:.4f}  ({best_c['model']}/{best_c['label']['method']})")
    if args.show_grid:
        for c, s in ranked[:10]:
            print(f"      {s:8.3f}  {c['model']:<4} {c['label']['method']:<15} "
                  f"h={c['label']['horizon']} m={c['margin']} gate={c.get('trend_gate')}")

    print("[4/4] Final walk-forward evaluation (out-of-sample) ...")
    if args.ensemble > 1:
        members = diverse_subset([c for c, _ in ranked], args.ensemble)
        print(f"      ensemble of {len(members)} ({args.combine}): "
              + ", ".join(f"{c['model']}/{c['label']['method']}" for c in members))
        folds, fold_metrics, trades_by_fold, rolled, vet_weights = combined_walkforward(
            members, X, df, arrs, combine=args.combine)
        best_c = members[0]
        _vet_weights = vet_weights
    else:
        folds, fold_metrics, trades_by_fold, rolled = final_walkforward(best_c, X, df, arrs)
        _vet_weights = None

    report = build_report(best_c, folds, fold_metrics, ranked,
                          ensemble_members=(members if args.ensemble > 1 else None),
                          ensemble_weights=_vet_weights,
                          rolled_metrics=rolled,
                          rolled_curve=(rolled if rolled is not None else None))
    print_report(report, trades_by_fold)
    _report.save_artifacts(report, trades_by_fold, folds, df, X)
    _report.save_predictions(fold_metrics)
    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts written to {config.ARTIFACT_DIR}/best/")


if __name__ == "__main__":
    main()