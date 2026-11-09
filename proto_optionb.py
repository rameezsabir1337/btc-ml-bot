"""Option B prototype: evaluate regime-adaptive ensemble + selective-entry overlay
on the saved 1d artifact members. Finds a combination strategy that captures the
fold-5 uptrend (BTC +30%) while preserving the fold 2/3 bear protection (-26%/-23%).

Strategies per fold:
  vote0.65   : current deployed (pooled Sharpe weights, fixed 0.65 vote)
  vote_soft  : per-fold Sharpe weights + graded vote threshold by weekly regime
  blend_soft : average member scores, graded margin by weekly regime
All long-only; confirmed-down weekly regime blocks longs entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from config import configure_timeframe, GATE_CONFIRM_BARS, WF
configure_timeframe("1d", long_only=True)

from report import load_artifacts
from ensemble import _member_fold, _fold_sharpes, chain_folds, _bt
from walkforward import expanding_folds
from grid import _bt_cfg


MEMBERS_ART = "best_spot_1d_win12"


def _prep():
    data, X, df, _ = load_artifacts(MEMBERS_ART)
    members = data["report"]["ensemble"]
    base = df.reindex(X.index)
    arrs = {
        "close": base["close"].to_numpy(float),
        "high": base["high"].to_numpy(float),
        "low": base["low"].to_numpy(float),
        "funding": (X["funding_rate"].to_numpy(float)
                    if "funding_rate" in X.columns else None),
    }
    return members, X, df, arrs


def _regime(gate, confirm):
    """Regime from weekly-slope gate: 1 strong-up, 0.5 weak-up/flat, 0 neutral, -1 down(block)."""
    gd = np.sign(np.asarray(gate, dtype=float))
    up = pd.Series(gd).rolling(confirm, min_periods=confirm).min().fillna(0.0)
    down = pd.Series(gd).rolling(confirm, min_periods=confirm).max().fillna(0.0)
    reg = np.full(len(gd), 0.0)
    reg[down < 0] = -1.0          # confirmed down -> block
    reg[(reg != -1) & (up > 0)] = 1.0     # strong up
    reg[(reg == 0) & (up >= 0)] = 0.5     # weak up / flat
    return reg


def _run(combine, gate_soft, members, X, df, arrs, folds):
    n_members = len(members)
    smat = [[_member_fold(m, X, df, arrs, f) for f in folds] for m in members]
    fold_metrics = []
    for fi, fold in enumerate(folds):
        sl = slice(fold.test_start, fold.test_end)
        ts = X.index[fold.test_start:fold.test_end]
        gate = X["htf10080_slope"].values[sl].astype(float)
        reg = _regime(gate, GATE_CONFIRM_BARS)

        wcol = np.clip([_fold_sharpes(m, X, df, arrs, [fold])[0]
                        for m in members], 0.0, None)
        if wcol.sum() <= 0:
            wcol = np.ones(n_members)
        w = wcol / wcol.sum()

        n = len(ts)
        if combine == "vote":
            thr = (np.where(reg >= 1.0, 0.30, np.where(reg >= 0.5, 0.50, 0.65))
                   if gate_soft else np.full(n, 0.65))
            votes = np.zeros(n)
            for mi in range(n_members):
                r = smat[mi][fi]
                if r is None:
                    continue
                scores = np.asarray(r[0], dtype=float)
                minlen = min(len(votes), len(scores))
                votes[:minlen] += w[mi] * scores[:minlen]
            pos = (votes >= thr).astype(float)
            scores_out = votes
        else:  # blend
            acc = np.zeros(n); cnt = 0
            for mi in range(n_members):
                r = smat[mi][fi]
                if r is None:
                    continue
                scores = np.asarray(r[0], dtype=float)
                minlen = min(len(acc), len(scores))
                acc[:minlen] += scores[:minlen]; cnt += 1
            avg = acc / max(cnt, 1)
            margin = members[0]["margin"]
            if gate_soft:
                thr = np.where(reg >= 1.0, margin, np.where(reg >= 0.5, margin + 0.05, margin + 0.10))
            else:
                thr = np.full(n, margin)
            pos = (avg >= thr).astype(float)
            scores_out = avg
        pos[reg < 0] = 0.0                    # bear protection
        m = _bt(members, fold, pos, arrs, ts)
        m["scores"] = scores_out
        fold_metrics.append(m)
    fm, rolled = chain_folds(fold_metrics, folds, arrs,
                             _bt_cfg(members[0], arrs, slice(0, 0)), config.BARS_PER_YEAR)
    return fm, rolled


def main():
    members, X, df, arrs = _prep()
    folds = expanding_folds(len(X), WF, horizon=7)
    print("folds:", [(f.test_start, f.test_end) for f in folds])
    for combine in ["vote", "blend"]:
        for gate_soft in [False, True]:
            fm, rolled = _run(combine, gate_soft, members, X, df, arrs, folds)
            per = [f"({m['num_trades']}tr w{m['win_rate']:.2f} sh{m['sharpe']:.2f})"
                   for m in fm] if fm else []
            print(f"{combine:6s} soft={int(gate_soft)}: pool={rolled['sharpe']:.2f} "
                  f"ret={rolled['total_return']:.3f} dd={rolled['max_drawdown']:.3f} "
                  f"win={rolled['win_rate']:.3f} trades={rolled['num_trades']}\n   folds={per}")


if __name__ == "__main__":
    main()