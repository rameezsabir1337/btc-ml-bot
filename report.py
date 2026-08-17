"""Report generation + artifact persistence for dashboard consumption."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

import config
from config import ARTIFACT_DIR
from backtest import summarize


def fold_table(trades_by_fold) -> pd.DataFrame:
    rows = []
    for f in trades_by_fold:
        m = f["metrics"]
        rows.append({
            "test_start": f["test_start"],
            "test_end": f["test_end"],
            **m,
        })
    return pd.DataFrame(rows)


def _rolled_pooled(rolled, fold_metrics) -> dict:
    """Pooled metrics measured on the single continuous (rolled) equity curve."""
    base = {"sharpe": rolled["sharpe"], "sortino": rolled["sortino"],
            "total_return": rolled["total_return"],
            "max_drawdown": rolled["max_drawdown"],
            "win_rate": rolled["win_rate"], "num_trades": rolled["num_trades"]}
    pool = {}
    for k, v in base.items():
        pool[f"{k}_mean"] = float(v)
        pool[f"{k}_med"] = float(v)
    pool["num_folds"] = len(fold_metrics)
    pool["folds_positive"] = int(sum(1 for m in fold_metrics if m["sharpe"] > 0))
    return pool


def build_report(best_c, folds, fold_metrics, ranked, ensemble_members=None,
                 rolled_metrics=None, rolled_curve=None,
                 ensemble_weights=None, ensemble_vote=0.5) -> dict:
    pooled = summarize(fold_metrics)
    if rolled_metrics is not None:
        pooled = _rolled_pooled(rolled_metrics, fold_metrics)
    report = {
        "best_config": {
            "model": best_c["model"],
            "model_params": best_c["model_params"],
            "label": best_c["label"],
            "margin": best_c["margin"],
            "trend_gate": best_c.get("trend_gate", "none"),
            "funding_gate": float(best_c.get("funding_gate", 0.0) or 0.0),
            "vol_target": float(best_c.get("vol_target", 0.0) or 0.0),
            "dd_brake": float(best_c.get("dd_brake", 0.0) or 0.0),
        },
        "ensemble": [{"model": c["model"], "model_params": c["model_params"],
                      "label": c["label"], "margin": c["margin"],
                      "trend_gate": c.get("trend_gate", "none"),
                      "funding_gate": float(c.get("funding_gate", 0.0) or 0.0),
                      "vol_target": float(c.get("vol_target", 0.0) or 0.0),
                      "dd_brake": float(c.get("dd_brake", 0.0) or 0.0)}
                     for c in (ensemble_members or [])],
        "pooled_metrics": pooled,
        "num_folds_evaluated": len(fold_metrics),
        "num_configs_ranked": len(ranked),
        "top_score": float(ranked[0][1]) if ranked else None,
        "bottom_score": float(ranked[-1][1]) if ranked else None,
    }
    report["meta"] = {
        "timeframe": config.TIMEFRAME,
        "long_only": config.LONG_ONLY,
        "bars_per_year": config.BARS_PER_YEAR,
        "min_hold_bars": config.MIN_HOLD_BARS,
        "vol_target": config.VOL_TARGET,
        "use_funding": config.USE_FUNDING,
        "rollover": rolled_metrics is not None,
    }
    if ensemble_members:
        report["meta"]["ensemble_vote"] = {
            "weights": [float(w) for w in (ensemble_weights or [])],
            "vote": float(ensemble_vote),
        }
    if rolled_curve is not None:
        # continuous equity + timestamps for the dashboard top chart (prepend the
        # starting capital so the curve includes its origin point). Timestamp.value
        # is always ns regardless of the index's stored resolution.
        ts = pd.DatetimeIndex(np.asarray(rolled_curve["ts"]))
        report["rolled"] = {
            "ts": [int(t.value // 10**6) for t in ts],
            "equity": [round(float(config.BacktestConfig.initial_capital), 6)] +
                      [round(float(v), 6) for v in rolled_curve["equity"]],
        }
    return report


def print_report(report: dict, trades_by_fold, verbose=True):
    bc = report["best_config"]
    print("\n=== BEST CONFIG ===")
    print(f"model      : {bc['model']}")
    print(f"label      : {bc['label']}")
    print(f"margin     : {bc['margin']}")
    print(f"exits      : none (signal-only, no TP/SL) "
          f"vol={bc.get('vol_target',0)} dd_brake={bc.get('dd_brake',0)}")
    print(f"gates      : trend={bc.get('trend_gate','none')} "
          f"funding_z={bc.get('funding_gate',0)}")
    print(f"\n=== POOLED (out-of-sample) ===")
    pm = report["pooled_metrics"]
    for k in ("sharpe_mean", "sortino_mean", "total_return_mean",
              "max_drawdown_mean", "win_rate_mean", "num_trades_mean", "folds_positive"):
        print(f"  {k:<20}: {pm.get(k):.4f}" if isinstance(pm.get(k), float)
              else f"  {k:<20}: {pm.get(k)}")
    print("\n=== PER-FOLD ===")
    for f in trades_by_fold:
        m = f["metrics"]
        print(f"  [{f['test_start']}:{f['test_end']}] "
              f"sharpe={m['sharpe']:.2f} dd={m['max_drawdown']:.2f} "
              f"ret={m['total_return']:.3f} trades={m['num_trades']} "
              f"win={m['win_rate']:.2f}")


def save_artifacts(report, trades_by_fold, folds, df, X, name="best"):
    """Persist JSON + parquet artifacts the dashboard reads."""
    (ARTIFACT_DIR / name).mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / name / "report.json", "w") as fh:
        json.dump({"report": report, "folds": trades_by_fold}, fh, indent=2, default=str)

    # full feature + close snapshot
    X.copy().reset_index().to_parquet(ARTIFACT_DIR / name / "features.parquet", index=False)
    df.reset_index().to_parquet(ARTIFACT_DIR / name / "ohlcv.parquet", index=False)


def save_predictions(fold_metrics, name="best"):
    """OOS model scores + positions, aggregated across folds, for the dashboard."""
    frames = [pd.DataFrame({
        "ts": m["ts"],
        "score": np.asarray(m["scores"]).ravel(),
        "position": np.asarray(m["positions"]).ravel(),
        "sharpe": m["sharpe"],
    }) for m in fold_metrics if m is not None]
    (ARTIFACT_DIR / name).mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames).to_parquet(ARTIFACT_DIR / name / "predictions.parquet", index=False)


def _restore_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stored ts (int ms or datetime) to a UTC datetime index."""
    if df["ts"].dtype.kind in "iu":
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


def load_artifacts(name="best"):
    with open(ARTIFACT_DIR / name / "report.json") as fh:
        data = json.load(fh)
    X = _restore_ts(pd.read_parquet(ARTIFACT_DIR / name / "features.parquet"))
    df = _restore_ts(pd.read_parquet(ARTIFACT_DIR / name / "ohlcv.parquet"))
    preds = None
    p = ARTIFACT_DIR / name / "predictions.parquet"
    if p.exists():
        preds = _restore_ts(pd.read_parquet(p))
    return data, X, df, preds