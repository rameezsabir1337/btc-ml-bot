"""Ensemble combination methods evaluated on frozen walk-forward folds.

V1 = weighted majority vote   : member gated positions weighted by per-fold
                                out-of-sample Sharpe (clip < 0 to small eps).
V2 = score blend              : average member scores, threshold by margin,
                                long-only, optional confidence sizing.
V4 = risk-off overlay         : per-fold binary classifier predicts an adverse
                                stretch; force positions flat when it fires.
All members use the same fixed risk layer (dd_brake, vol_target; no TP/SL exits)
via grid._bt_cfg.

chain_folds() re-links per-fold OOS positions into ONE continuous backtest so
equity compounds across fold boundaries (no capital reset per fold).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from walkforward import fold_indices
from grid import _labels_for, _fit_predict, _gated_positions, _bt_cfg
from backtest import backtest
from models import make_model


def _member_fold(c, X, df, arrs, fold):
    """Fit member `c` on the fold train window; return scores/gated pos/ts or None."""
    tr, te = fold_indices(fold)
    y = _labels_for(df, c, X)
    Xte = X.iloc[te]
    model, scores, task = _fit_predict(c, X.iloc[tr], y[tr], Xte)
    if model is None:
        return None
    pos = _gated_positions(c, scores, Xte)
    return scores, pos, X.index[te]


def _bt(members, fold, pos, arrs, close_ts=None):
    sl = slice(fold.test_start, fold.test_end)
    m = backtest(pos, arrs["close"][sl], _bt_cfg(members[0], arrs, sl))
    m["ts"] = close_ts
    # m["positions"] is backtest's FINAL positions (post min-hold / TP/SL exits),
    # matching strat_ret exactly -- do not overwrite with the pre-exit signal.
    return m


def _fold_sharpes(member, X, df, arrs, folds):
    """Standalone per-fold OOS Sharpe of a member (for vote weights)."""
    out = []
    for fold in folds:
        r = _member_fold(member, X, df, arrs, fold)
        if r is None:
            out.append(0.0)
            continue
        _, pos, ts = r
        sl = slice(fold.test_start, fold.test_end)
        m = backtest(pos, arrs["close"][sl], _bt_cfg(member, arrs, sl))
        out.append(float(m["sharpe"]) if np.isfinite(m["sharpe"]) else 0.0)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Regime-adaptive entry policy (shared by walk-forward + live dashboard)
# ---------------------------------------------------------------------------
# Per-regime vote threshold plus an entry condition evaluated on the bar.
# `cond` is a list of AND-ed groups; inside a group the clauses are OR-ed.
# Clauses are "col<op>value".
#   up   -> long only while BOTH the 12- and 6-bar EMA filters are up (thr 0.0)
#   flat -> any member long + positive 6-bar momentum             (thr 0.1)
#   down -> counter-trend bounce: >=3 members long + above 12-bar EMA (0.6)
# Equal member weights are used so no fold's own OOS Sharpe leaks into its
# own positions (the Sharpe-weighted vote was the source of the negative folds).
ENTRY_POLICY = {
    "up":   {"thr": 0.0, "cond": [["ema_off_12>0"], ["ema_off_6>0"]]},
    "flat": {"thr": 0.1, "cond": [["ret_6>0"]]},
    "down": {"thr": 0.6, "cond": [["ema_off_12>0"]]},
}


def cond_pass(cond, Xsl):
    """Evaluate a policy `cond` spec against feature frame `Xsl` -> bool array.
    Groups are AND-ed; clauses inside a group are OR-ed."""
    if not cond:
        return np.ones(len(Xsl), dtype=bool)
    out = np.ones(len(Xsl), dtype=bool)
    for group in cond:
        g = np.zeros(len(Xsl), dtype=bool)
        for clause in group:
            for op in (">=", "<=", "==", ">", "<"):
                i = clause.find(op)
                if i > 0:
                    col, val = clause[:i], float(clause[i + len(op):])
                    s = np.asarray(Xsl[col], dtype=float)
                    s = np.nan_to_num(s, nan=0.0)
                    if op == ">":
                        g |= s > val
                    elif op == ">=":
                        g |= s >= val
                    elif op == "<":
                        g |= s < val
                    elif op == "<=":
                        g |= s <= val
                    else:
                        g |= s == val
                    break
        out &= g
    return out


def apply_entry_policy(pos, masks, policy, Xsl):
    """Zero longs whose regime's entry condition fails on that bar."""
    pos = np.asarray(pos, dtype=float).copy()
    long_ = pos > 0
    if not long_.any():
        return pos
    for cls in ("up", "flat", "down"):
        spec = (policy or {}).get(cls)
        if not spec:
            continue
        sel = long_ & np.asarray(masks[cls], dtype=bool)
        if not sel.any():
            continue
        pos[sel & ~cond_pass(spec.get("cond"), Xsl)] = 0.0
    return pos

def _metrics_from_ret(ret, eq_start, eq_seg, pos, bps):
    """Per-fold metrics sliced from a continuous (rolled) equity curve.
    `pos` is the position series aligned 1:1 to `ret` (len(pos)==len(ret))."""
    L = len(ret)
    sd = ret.std()
    m = ret.mean()
    sharpe = (m * bps) / (sd * np.sqrt(bps)) if sd > 0 else 0.0
    dn = ret[ret < 0]
    sortino = (m * bps) / (dn.std() * np.sqrt(bps)) if len(dn) and dn.std() > 0 else 0.0
    eq_i = np.concatenate([[eq_start], eq_seg])
    peak_i = np.maximum.accumulate(eq_i)
    max_dd = float((eq_i / peak_i - 1).min()) if len(eq_i) else 0.0
    prev = np.concatenate([[0.0], pos[:-1]])
    entries = int(((pos != 0) & (prev == 0)).sum())
    revs = int(((pos != 0) & (prev != 0) &
                (np.sign(pos) != np.sign(prev))).sum())
    num_trades = entries + revs
    active = pos != 0
    win_rate = float(round((ret[active] > 0).mean(), 4)) if active.any() else 0.0
    total_return = float(eq_seg[-1] / eq_start - 1) if len(eq_seg) else 0.0
    return {
        "total_return": total_return,
        "annual_return": (float(eq_seg[-1] / eq_start) ** (bps / L) - 1
                          if L and eq_seg[-1] > 0 else -1.0),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "bars": L,
        "equity": eq_i,       # includes starting equity -> chains across folds
        "strat_ret": ret,
    }


def chain_folds(fold_metrics, folds, arrs, cfg, bps):
    """Chain per-fold OOS `fold_metrics` (each carrying gated `positions` + `ts`)
    into ONE continuous backtest: fold N+1 starts at fold N's ending equity.

    Returns (fold_metrics_rolled, rolled) where the per-fold dicts are sliced
    from the continuous curve and `rolled` holds the pooled continuous metrics
    plus full-period equity/ts.
    """
    import config as _cfg
    cap = _cfg.BacktestConfig.initial_capital
    pairs = [(f, m) for f, m in zip(folds, fold_metrics)
             if m is not None and m.get("positions") is not None]
    if not pairs:
        return [], None
    folds_u, fms = zip(*pairs)
    pos_cat = np.concatenate([m["positions"] for m in fms])
    sls = [slice(f.test_start, f.test_end) for f in folds_u]
    cfg = dataclasses.replace(
        cfg,
        high=np.concatenate([arrs["high"][sl] for sl in sls]),
        low=np.concatenate([arrs["low"][sl] for sl in sls]),
        funding=(np.concatenate([arrs["funding"][sl] for sl in sls])
                 if arrs.get("funding") is not None else None),
    )
    rolled = backtest(pos_cat, np.concatenate([arrs["close"][sl] for sl in sls]),
                      cfg, start_equity=cap)
    rolled["ts"] = np.concatenate([np.asarray(m["ts"]) for m in fms])
    rolled["positions"] = pos_cat
    if fms[0].get("scores") is not None:
        rolled["scores"] = np.concatenate([np.asarray(m["scores"]).ravel() for m in fms])

    out, off = [], 0
    for m in fms:
        L = len(m["positions"])
        if off == 0:
            # fold 0: no boundary; its returns are indices [0, L-1)
            a, b = 0, L - 1
            eq_start = cap
        else:
            # fold i>=1: include the boundary return (index off-1, the prior
            # fold's last position earning into this fold's first close) so the
            # equity curve shares an exact point with the previous fold.
            a, b = off - 1, off + L - 1
            eq_start = rolled["equity"][off - 2]
        ret = rolled["strat_ret"][a:b]
        eq_seg = rolled["equity"][a:b]
        per = _metrics_from_ret(ret, eq_start, eq_seg, pos_cat[a:b], bps)
        per["ts"] = m["ts"]
        per["positions"] = m["positions"]
        per["scores"] = np.asarray(m["scores"]).ravel() if m.get("scores") is not None else None
        out.append(per)
        off += L
    return out, rolled


def weighted_vote_walkforward(members, X, df, arrs, folds, weights=None, vote=0.5):
    """V1: per-fold weighted majority vote on gated positions."""
    if weights is None:
        weights = np.asarray([_fold_sharpes(m, X, df, arrs, folds) for m in members])
    weights = np.clip(weights, 1e-3, None)      # drop-negative members from voting
    fold_metrics = []
    for fi, fold in enumerate(folds):
        sl = slice(fold.test_start, fold.test_end)
        acc = np.zeros(fold.test_end - fold.test_start)
        wsum = 0.0
        ts = None
        for mi, m in enumerate(members):
            r = _member_fold(m, X, df, arrs, fold)
            if r is None:
                continue
            _, pos, ts = r
            acc += weights[mi, fi] * pos
            wsum += weights[mi, fi]
        if wsum <= 0:
            continue
        votes = acc / wsum
        pos = (votes >= vote).astype(float)
        m = _bt(members, fold, pos, arrs, ts)
        m["scores"] = votes
        fold_metrics.append(m)
    return fold_metrics


def regime_vote_walkforward(members, X, df, arrs, folds, weights=None, vote=0.5,
                            gate_col="htf10080_slope", confirm=2,
                            soft=(0.30, 0.50), return_table=False,
                            weights_equal=False, entry_policy=None):
    """V1b (Option B): per-fold Sharpe-weighted majority vote over member GATED
    positions, but the vote threshold adapts to the higher-TF regime so we add
    SELECTIVE long entries in confirmed up-trends (weak members barely count and
    can't clear a fixed 0.5-0.65 bar) while bear folds stay protected because the
    member trend-gates already force positions flat.

    Regime (from the weekly-slope gate, hysteresis = `confirm` bars):
      strongly up (confirmed >0)  -> low  threshold (aggressive selective longs)
      weak up / flat (>=0)        -> mid  threshold
      down (confirmed <0)         -> high threshold; members' own down-gate also
                                     blocks, so no longs in bear (folds 2/3)

    With `entry_policy` (see ENTRY_POLICY) the per-regime threshold AND the
    per-regime entry condition come from the policy, and `weights_equal` forces
    a uniform member vote so no fold's own OOS Sharpe leaks into its positions.
    """
    import pandas as pd
    if weights is None:
        weights = np.asarray([_fold_sharpes(m, X, df, arrs, folds) for m in members])
    if weights_equal:
        weights = np.ones_like(np.asarray(weights, dtype=float))
    else:
        weights = np.clip(weights, 1e-3, None)   # keep every member voting (V1 parity)
    low_th, mid_th = soft
    fold_metrics = []
    regime_rows = []   # (regime_class, fold_weights_row) for live weight selection
    for fi, fold in enumerate(folds):
        sl = slice(fold.test_start, fold.test_end)
        gate = pd.Series(X[gate_col].values[sl]).fillna(0.0)
        gd = np.sign(gate)
        up = gd.rolling(confirm, min_periods=confirm).min().fillna(0.0)     # >=0 weak up
        dn = gd.rolling(confirm, min_periods=confirm).max().fillna(0.0)     # <0 confirmed down
        strong_up = (up > 0).to_numpy()
        down = (dn < 0).to_numpy()
        # dominant regime over the fold window, for live weight selection
        n_up = int(strong_up.sum())
        n_dn = int(down.sum())
        n_flat = len(strong_up) - n_up - n_dn
        dom = "up" if n_up >= max(n_dn, n_flat) else ("down" if n_dn > n_flat else "flat")
        wrow = [float(weights[mi, fi]) for mi in range(len(members))]
        regime_rows.append((dom, wrow))
        thr = np.full(fold.test_end - fold.test_start, mid_th)
        thr[strong_up] = low_th
        thr[down] = vote  # lean conservative when confirmed down (belt & braces)
        if entry_policy:
            n = fold.test_end - fold.test_start
            thr = np.full(n, float(entry_policy.get("flat", {}).get("thr", 0.5)))
            thr[strong_up] = float(entry_policy.get("up", {}).get("thr", 0.5))
            thr[down] = float(entry_policy.get("down", {}).get("thr", 0.5))

        acc = np.zeros(fold.test_end - fold.test_start)
        wsum = 0.0
        ts = None
        for mi, m in enumerate(members):
            r = _member_fold(m, X, df, arrs, fold)
            if r is None:
                continue
            _, pos, ts = r
            acc += weights[mi, fi] * pos
            wsum += weights[mi, fi]
        if wsum <= 0:
            continue
        votes = acc / wsum
        pos = (votes >= thr).astype(float)
        if entry_policy:
            masks = {"up": strong_up, "down": down,
                     "flat": ~strong_up & ~down}
            pos = apply_entry_policy(pos, masks, entry_policy, X.iloc[sl])
        m = _bt(members, fold, pos, arrs, ts)
        m["scores"] = votes
        fold_metrics.append(m)
    if return_table:
        return fold_metrics, regime_rows
    return fold_metrics


def blend_walkforward(members, X, df, arrs, folds, margin=0.0,
                      confidence_sizing=True, gate_trend=True):
    """V2: average member scores -> positions, long-only, confidence-sized."""
    import config
    prev = config.CONFIDENCE_SIZING
    config.CONFIDENCE_SIZING = confidence_sizing
    from backtest import to_positions
    fold_metrics = []
    try:
        for fold in folds:
            sl = slice(fold.test_start, fold.test_end)
            acc = np.zeros(fold.test_end - fold.test_start)
            n = 0
            ts = None
            for m in members:
                r = _member_fold(m, X, df, arrs, fold)
                if r is None:
                    continue
                scores, _, ts = r
                acc += np.asarray(scores, dtype=float)
                n += 1
            if n == 0:
                continue
            avg = acc / n
            pos = to_positions(avg, margin)
            if gate_trend:
                gate = "htf1440_slope"
                if gate in X.columns:
                    g = X.iloc[fold.test_start:fold.test_end][gate].values
                    pos[pos > 0] = np.where(g[pos > 0] < 0, 0.0, pos[pos > 0])
            fold_metrics.append(_bt(members, fold, pos, arrs, ts))
    finally:
        config.CONFIDENCE_SIZING = prev
    return fold_metrics


# ---------------------------------------------------------------------------
# V4: regime risk-off overlay
# ---------------------------------------------------------------------------
RISK_FEATS = ["vol_z", "htf1440_slope", "funding_z", "ret_over_atr", "rsi", "atr_pct"]


def _adverse_label(close: np.ndarray, window: int = 30, threshold: float = -0.025) -> np.ndarray:
    """1 where the forward `window`-bar max drawdown from the current close is
    deeper than `threshold` (label only; leakage-safe, never a feature)."""
    n = len(close)
    lab = np.zeros(n)
    for t in range(n - window):
        worst = close[t + 1:t + window + 1].min() / close[t] - 1
        lab[t] = 1.0 if worst < threshold else 0.0
    return lab


def _risk_model():
    try:
        return make_model("lgb", {"n_estimators": 200, "num_leaves": 8,
                                  "learning_rate": 0.05, "min_child_samples": 50,
                                  "subsample": 0.8, "colsample_bytree": 0.8}, "binary")
    except Exception:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, C=0.5)


def apply_risk_off(pos, X, df, arrs, folds, prob_threshold=0.5, window=30, threshold=-0.025):
    """Per-fold: train a risk classifier on the fold train window, zero out test
    positions when it predicts an adverse stretch with prob > prob_threshold."""
    close = np.asarray(df["close"].values, dtype=float)
    lab = _adverse_label(close, window, threshold)
    out = np.asarray(pos, dtype=float).copy()
    te_all = np.arange(len(out)) >= 0
    for fold in folds:
        tr, te = fold_indices(fold)
        feats = [c for c in RISK_FEATS if c in X.columns]
        Xf = X[feats].values
        ytr = lab[tr]
        if len(np.unique(ytr)) < 2:
            continue
        mdl = _risk_model()
        mdl.fit(Xf[tr], ytr)
        p = mdl.predict_proba(Xf[te])[:, 1]
        mask = p > prob_threshold
        seg = slice(fold.test_start, fold.test_end)
        out[te[mask]] = 0.0
    return out


__all__ = ["weighted_vote_walkforward", "blend_walkforward",
           "apply_risk_off", "RISK_FEATS"]
