"""Model factory: LightGBM / XGBoost / ExtraTrees, task-aware fit & predict."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_class_weight

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None

MODELS = ("lgb", "xgb", "rf", "et")


def _weights(y: np.ndarray) -> dict | None:
    classes = np.unique(y[y != 0])
    if len(classes) < 2:
        return None
    try:
        w = compute_class_weight("balanced", classes=classes, y=y[y != 0])
        return {int(c): float(v) for c, v in zip(classes, w)}
    except ValueError:
        return None


def make_model(name: str, params: dict, task: str):
    """Return an unfitted sklearn-style model for the given task.
    n_jobs=1 internally: the grid is parallelised across processes, so per-model
    threading would thrash the CPU."""
    common = {"n_jobs": 1, "random_state": 42}
    if name == "lgb" and LGBMClassifier is not None:
        if task == "regression":
            return LGBMRegressor(**{**common, "verbosity": -1}, **params)
        return LGBMClassifier(**{**common, "verbosity": -1, "class_weight": "balanced"}, **params)
    if name == "xgb" and XGBClassifier is not None:
        if task == "regression":
            return XGBRegressor(**{**common, "verbosity": 0, "eval_metric": "rmse"}, **params)
        return XGBClassifier(**{**common, "verbosity": 0, "eval_metric": "logloss", "scale_pos_weight": None}, **params)
    if name == "et":
        if task == "regression":
            return ExtraTreesRegressor(**{**common}, **params)
        return ExtraTreesClassifier(**{**common, "class_weight": "balanced"}, **params)
    if name == "rf":
        if task == "regression":
            return RandomForestRegressor(**{**common}, **params)
        return RandomForestClassifier(**{**common, "class_weight": "balanced"}, **params)
    raise ValueError(f"Unknown model {name}")


def predict_score(model, X: np.ndarray, task: str) -> np.ndarray:
    """Continuous score roughly in [-1,1] (class tasks) or standardized return
    (regression, scaled by training-prediction std for interpretable margins)."""
    if task == "regression":
        raw = np.asarray(model.predict(X), dtype=float)
        scale = getattr(model, "score_scale_", None)
        return raw / scale if scale else raw
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if task == "multiclass":
            # encoded classes {0:short, 1:flat, 2:long}
            return p[:, 2] - p[:, 0]
        return 2 * p[:, 1] - 1                    # binary encoded {0,1} -> [-1,1]
    raw = np.asarray(model.predict(X), dtype=float)
    return raw


def encode_y(y, task: str) -> np.ndarray:
    """Map labels to the class encoding each model family accepts.
    binary: {+1,-1} -> {1,0}; multiclass: {-1,0,1} -> {0,1,2}; regression: unchanged."""
    y = np.asarray(y, dtype=float)
    if task == "binary":
        return (y == 1).astype(int)
    if task == "multiclass":
        return (y + 1).astype(int)
    return y


def fit_score(model, Xtr, ytr, Xte, task: str) -> np.ndarray:
    """Fit (with task-aware label encoding) and return prediction scores.
    For regression, scores are standardized by the training prediction std so
    confidence margins are interpretable across models."""
    y = encode_y(ytr, task)
    model.fit(Xtr.values, y)
    scores = predict_score(model, Xte.values, task)
    if task == "regression":
        tr_scores = predict_score(model, Xtr.values, task)
        scale = float(np.std(tr_scores)) or 1.0
        model.score_scale_ = scale
        scores = scores / scale
    return scores