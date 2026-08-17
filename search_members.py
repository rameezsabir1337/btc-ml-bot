"""Grid-search the V1 weighted-vote member fleet on the frozen 4h dataset.

Honest rules that inherit from the deployed evaluation:
  - purge horizon = max member horizon (folds built with horizon=90 -> same folds
    as eval_ensemble, no label leak)
  - 4h bars-per-year annualization, funding per-bar, honest num_trades
  - all members share the fixed risk layer (vol 0.1, dd_brake 0.03); no TP/SL exits

Method: fit every candidate member once per fold and cache its gated positions,
then evaluate any fleet combo as a Sharpe-weighted majority vote directly on the
cached positions (no refits). The baseline 5-member fleet is a subset of the
candidate pool, so its score here must reproduce Sharpe 2.13 / 4-of-6.

Selection note: choosing the "best" fleet on the same folds used for reporting
overfits those folds, so the script ranks by (all-folds positive first, then
mean fold Sharpe) and prints robustness across vote thresholds. Deploy only if
the winner beats baseline materially and is robust.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

import config
from walkforward import expanding_folds
from ensemble import _member_fold, _bt_cfg
from backtest import backtest
from eval_ensemble import load_frozen, MEMBER_SPECS

PRUNE_HORIZONS = (60, 90)
THRESHOLDS = (0.0005, 0.001, 0.002)
MARGINS = (0.0, 0.1, 0.2)
VOTES = (0.5, 0.6, 0.7)
SIZES = (3, 4, 5)
STRONG_MIN_POS = 4      # candidate must be positive on at least this many folds
MIN_AVG_TRADES = 2.0    # candidate must trade enough to matter


def make_member(spec):
    h, th, m = spec
    return {
        "model": "lgb",
        "model_params": {"n_estimators": 200, "num_leaves": 31,
                         "learning_rate": 0.05, "min_child_samples": 50,
                         "subsample": 0.9, "colsample_bytree": 0.8},
        "label": {"method": "binary_min_move", "horizon": int(h),
                  "threshold": float(th)},
        "margin": float(m),
        "trend_gate": "htf1440_slope",
        "funding_gate": 0.0,
        "vol_target": 0.1, "dd_brake": 0.03,
    }


def main():
    ap = argparse.ArgumentParser(description="Member-fleet grid search")
    ap.add_argument("--bind", type=int, nargs="+", default=list(range(18)),
                    help="restrict to candidate indices (--bind 0 3 7...)")
    ap.add_argument("--top", type=int, default=15, help="report top N fleets")
    ap.add_argument("--lofo", action="store_true",
                    help="leave-one-fold-out selection-honesty check")
    args = ap.parse_args()

    config.configure_timeframe("4h", long_only=True)
    X, df, arrs = load_frozen()

    pool = [spec for spec in itertools.product(PRUNE_HORIZONS, THRESHOLDS, MARGINS)]
    if args.bind != list(range(len(pool))):
        pool = [pool[i] for i in args.bind]
    members = [make_member(s) for s in pool]
    horizon = 90  # max member horizon in the restricted pool
    folds = expanding_folds(len(X), config.WF, horizon=horizon)
    print(f"[search] {len(pool)} candidates, purge={horizon}, "
          f"folds={len(folds)}, 4h bps={config.BARS_PER_YEAR}")

    # stage 1: fit once per candidate per fold, cache positions + fold sharpe
    P = {}  # (mi, fi) -> gated positions
    S = {}  # (mi, fi) -> OOS sharpe
    N = {}  # (mi, fi) -> num_trades
    stat = []
    for mi, m in enumerate(members):
        per_fold = []
        for fi, fold in enumerate(folds):
            r = _member_fold(m, X, df, arrs, fold)
            if r is None:
                P[(mi, fi)], S[(mi, fi)], N[(mi, fi)] = None, 0.0, 0
                continue
            _, pos, ts = r
            sl = slice(fold.test_start, fold.test_end)
            bm = backtest(pos, arrs["close"][sl], _bt_cfg(m, arrs, sl))
            P[(mi, fi)] = pos
            S[(mi, fi)] = float(bm["sharpe"]) if np.isfinite(bm["sharpe"]) else 0.0
            N[(mi, fi)] = int(bm["num_trades"])
            per_fold.append(S[(mi, fi)])
        mean = float(np.mean(per_fold)) if per_fold else 0.0
        stat.append({
            "spec": pool[mi], "mean": mean, "min": float(np.min(per_fold)) if per_fold else 0.0,
            "pos": int(np.sum(np.asarray(per_fold) > 0)) if per_fold else 0,
            "avg_trades": float(np.mean([N[(mi, fi)] for fi in range(len(folds))])) or 0.0,
        })

    full = pd.DataFrame(stat)
    print("\n== CANDIDATE POOL (18) sorted by mean fold Sharpe ==")
    with pd.option_context("display.width", 140):
        print(full.sort_values("mean", ascending=False).to_string(index=False))

    strong = full[(full.pos >= STRONG_MIN_POS) & (full.avg_trades >= MIN_AVG_TRADES)]
    print(f"\n== STRONG MEMBERS ({len(strong)} of {len(pool)}) ==")
    with pd.option_context("display.width", 140):
        print(strong.sort_values("mean", ascending=False).to_string(index=False))

    if len(strong) == 0:
        print("No strong members; widen grid.")
        return
    si = strong.index.tolist()

    # stage 2: evaluate fleet combos on cached positions (pure numpy, no refits)
    def fleet_score(combo, vote):
        per = []
        for fi in range(len(folds)):
            acc, wsum = np.zeros(fold_len(fi)), 0.0
            for mi in combo:
                pos = P.get((mi, fi))
                if pos is None:
                    continue
                w = max(S[(mi, fi)], 1e-3)
                acc += w * pos
                wsum += w
            if wsum <= 0:
                continue
            pos = (acc / wsum >= vote).astype(float)
            sl = slice(folds[fi].test_start, folds[fi].test_end)
            bm = backtest(pos, arrs["close"][sl], _bt_cfg(members[0], arrs, sl))
            per.append((float(bm["sharpe"]) if np.isfinite(bm["sharpe"]) else 0.0,
                        int(bm["num_trades"])))
        return per

    def fold_len(fi):
        f = folds[fi]
        return f.test_end - f.test_start

    rows = []
    for size in SIZES:
        for combo in itertools.combinations(si, size):
            votes = {v: fleet_score(combo, v) for v in VOTES}
            for v, per in votes.items():
                sharps = np.asarray([p[0] for p in per]) if per else np.asarray([])
                rows.append({
                    "fleet": "+".join(f"{pool[i][0]}/{pool[i][1]}/{pool[i][2]:g}"
                                      for i in combo),
                    "size": size, "vote": v,
                    "mean": float(sharps.mean()) if len(sharps) else 0.0,
                    "min": float(sharps.min()) if len(sharps) else 0.0,
                    "pos": int(np.sum(sharps > 0)),
                    "avg_trades": float(np.mean([p[1] for p in per])) if per else 0.0,
                })
    res = pd.DataFrame(rows)

    # baseline = deployed 5-member fleet (subset of pool)
    base_idx = [pool.index(s) for s in MEMBER_SPECS if s in pool]
    base_per = fleet_score(base_idx, 0.5)
    base_sharps = np.asarray([p[0] for p in base_per])
    print(f"\n== BASELINE (deployed 5-member, vote 0.5) ==")
    print(f"mean={base_sharps.mean():.2f} min={base_sharps.min():.2f} "
          f"pos={int(np.sum(base_sharps > 0))}/6 "
          f"avg_trades={np.mean([p[1] for p in base_per]):.1f} "
          f"per-fold={np.round(base_sharps, 2).tolist()}")

    res = res.sort_values(["pos", "mean"], ascending=[False, False]).reset_index(drop=True)
    print(f"\n== TOP {args.top} FLEETS (rank: folds-positive desc, mean Sharpe desc) ==")
    with pd.option_context("display.width", 200):
        print(res.head(args.top).to_string(index=False))

    # deployment recommendation
    six = res[res.pos == 6]
    if len(six):
        rec = six.sort_values("mean", ascending=False).iloc[0]
        specs = [f"{x[0]}/{x[1]}/{x[2]:g}" for x in pool]
        combo_idx = [specs.index(part) for part in rec["fleet"].split("+")]
        print("\nRECOMMEND:", rec["fleet"], f"mean={rec['mean']:.2f} "
              f"min={rec['min']:.2f} trades={rec['avg_trades']:.1f} vote={rec['vote']:g}")
        print("combo as pool indices:", combo_idx,
              "-> specs:", [pool[i] for i in combo_idx])
    else:
        best = res.iloc[0]
        print("\nNo 6/6 fleet; next best:", best["fleet"],
              f"mean={best['mean']:.2f} pos={best['pos']}/6")

    if args.lofo:
        _lofo(si, pool, members, folds, arrs, base_idx, P, S, SIZES, VOTES)


def _lofo(si, pool, members, folds, arrs, base_idx, P, S, SIZES, VOTES):
    """Selection-honesty: for each held-out fold, rank fleets on the other 5
    folds (pos desc, then mean Sharpe), take the top-1, evaluate it on the
    held-out fold. The 6 held-out Sharpes estimate what the whole
    search-then-deploy procedure delivers on unseen data."""
    print("\n== LOFO SELECTION-HONESTY CHECK ==")
    combos = []
    for size in SIZES:
        for combo in itertools.combinations(si, size):
            combos.append(combo)
    held = []

    def _score(combo, vote, folds_subset):
        per = []
        for fi in folds_subset:
            acc, wsum = np.zeros(folds[fi].test_end - folds[fi].test_start), 0.0
            for mi in combo:
                pos = P.get((mi, fi))
                if pos is None:
                    continue
                w = max(S[(mi, fi)], 1e-3)
                acc += w * pos
                wsum += w
            if wsum <= 0:
                continue
            pos = (acc / wsum >= vote).astype(float)
            sl = slice(folds[fi].test_start, folds[fi].test_end)
            bm = backtest(pos, arrs["close"][sl], _bt_cfg(members[0], arrs, sl))
            per.append(float(bm["sharpe"]) if np.isfinite(bm["sharpe"]) else 0.0)
        return np.asarray(per) if per else np.asarray([])

    for hf in range(len(folds)):
        rest = [f for f in range(len(folds)) if f != hf]
        best_combo, best_key = None, None
        for combo in combos:
            for vote in VOTES:
                sh = _score(combo, vote, rest)
                if len(sh) == 0:
                    continue
                key = (int(np.sum(sh > 0)), float(sh.mean()))
                if best_key is None or key > best_key:
                    best_key, best_combo = key, (combo, vote)
        combo, vote = best_combo
        sh = _score(combo, vote, [hf])
        held.append((sh[0] if len(sh) else 0.0,
                     "+".join(f"{pool[i][0]}/{pool[i][1]}/{pool[i][2]:g}"
                              for i in combo), vote))
    held_sh = np.asarray([h[0] for h in held])
    print(f"held-out fold Sharpes : {np.round(held_sh, 2).tolist()}")
    print(f"search LOFO           : mean={held_sh.mean():.2f} "
          f"pos={int(np.sum(held_sh > 0))}/6")
    base_sh = _score(base_idx, 0.5, list(range(len(folds))))
    print(f"baseline (fixed fleet): mean={base_sh.mean():.2f} "
          f"pos={int(np.sum(base_sh > 0))}/6")
    for i, h in enumerate(held):
        print(f"  holdout fold {i}: pick ({h[1]}) vote={h[2]:g} -> sharpe {h[0]:.2f}")


if __name__ == "__main__":
    main()