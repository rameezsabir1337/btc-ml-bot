"""Defines the grid search space and builds candidate configs."""
from __future__ import annotations

import itertools
import random

import config as _cfg

# model hyperparameter ranges (per family)
MODEL_PARAMS = {
    "lgb": {
        "n_estimators": [100, 200, 400],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.02, 0.05, 0.1],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "xgb": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.02, 0.05, 0.1],
        "min_child_weight": [3, 7, 15],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "rf": {
        "n_estimators": [200, 400, 800],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [2, 5, 15],
    },
    "et": {
        "n_estimators": [200, 400, 800],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [2, 5, 15],
    },
}

# label methods and their param grid. `horizon` is filled from the *live*
# configured mode horizons (config.MODE_HORIZONS) at build time so the grid
# always searches the active timeframe's label horizons regardless of when
# this module was imported relative to configure_timeframe().
LABEL_SPACE = {
    "binary_min_move": {
        "horizon": None,                 # -> config.MODE_HORIZONS at build time
        "threshold": [0.0005, 0.001, 0.002, 0.003],
        # margins (min-confidence filter). class scores in [-1,1]; higher = fewer trades
        "margin": [0.0, 0.1, 0.2, 0.35, 0.5],
    },
    "triple_barrier": {
        "horizon": None,                 # -> config.MODE_HORIZONS at build time
        "tp_frac": [0.002, 0.004, 0.006],
        "sl_frac": [0.001, 0.002, 0.003],
        "margin": [0.0, 0.15, 0.3, 0.45],
    },
    "triple_atr": {
        "horizon": None,                 # -> config.MODE_HORIZONS at build time
        "tp_atr_mult": [1.0, 1.5, 2.0],  # TP = close + k*ATR (vol-adaptive)
        "sl_atr_mult": [0.5, 1.0, 1.5],  # SL = close - k*ATR
        "margin": [0.0, 0.15, 0.3, 0.45],
    },
    "regression": {
        "horizon": None,                 # -> config.MODE_HORIZONS at build time
        # margins on z-scaled model scores; higher = fewer trades
        "margin": [0.0, 0.5, 1.0, 1.5, 2.0],
    },
}

# optional higher-TF trend gate applied at position mapping.
# gates are derived from the ACTIVE mode's higher-timeframe contexts so only
# columns that actually exist in the feature matrix are offered (15m mode:
# 1h/4h; 4h mode: 1d). Computed lazily at build time because
# config.configure_timeframe() runs after module import.
def active_trend_gates() -> list[str]:
    if getattr(_cfg, "FORCE_TREND_GATE", None):
        return [_cfg.FORCE_TREND_GATE]
    htfs = [m for m in (_cfg.SWING_HTF if _cfg.SWING_HTF else _cfg.HIGHER_TIMEFRAMES) if m]
    return ["none"] + [f"htf{m}_slope" for m in htfs]

# (tp, sl) / (tp_atr, sl_atr) exit spaces removed: the bot is signal-only, no TP/SL.
VOL_TARGETS = [0.0, 0.10, 0.15, 0.20, 0.25]   # realized-vol target sizing
DD_BRAKES = [0.0, 0.03, 0.05, 0.08]            # exposure brake at running drawdown
FUNDING_GATES = [0.0, 1.5, 2.0, 3.0, 4.0]      # funding-z threshold; 0 = off
MIN_HOLDS = [0, 3, 5, 8]                       # min-hold bars (anti-churn, fee saver)


def build_configs(seed=None) -> list[dict]:
    """Expand model-family x per-model-params x label-space, attach a seeded random
    draw of execution/risk knobs (vol-target/dd-brake) to each, then cap to the
    grid budget with a seeded sample for reproducibility.

    Label horizons are resolved from the *current* config.MODE_HORIZONS so the
    search always matches the active timeframe."""
    rng = random.Random(_cfg.GRID.seed if seed is None else seed)
    horizons = list(_cfg.MODE_HORIZONS)
    configs: list[dict] = []
    for model in sorted(MODEL_PARAMS):
        mkeys = list(MODEL_PARAMS[model])
        mvals = [MODEL_PARAMS[model][k] for k in mkeys]
        for mcombo in itertools.product(*mvals):
            for label in LABEL_SPACE:
                lkeys = list(LABEL_SPACE[label])
                lvals = [horizons if k == "horizon" else LABEL_SPACE[label][k] for k in lkeys]
                for lcombo in itertools.product(*lvals):
                    params = dict(zip(mkeys, mcombo))
                    lp = dict(zip(lkeys, lcombo))
                    margin = lp.pop("margin")
                    for gate in (active_trend_gates() if _cfg.GATES_ON else ["none"]):
                        configs.append({
                            "model": model,
                            "model_params": params,
                            "label": {"method": label, **lp},
                            "margin": margin,
                            "trend_gate": gate,
                            "vol_target": rng.choice(VOL_TARGETS),
                            "dd_brake": rng.choice(DD_BRAKES),
                            "funding_gate": rng.choice(FUNDING_GATES if _cfg.GATES_ON else [0.0]),
                            "min_hold_bars": rng.choice(MIN_HOLDS),
                        })
    if len(configs) > _cfg.GRID.max_configs:
        configs = rng.sample(configs, _cfg.GRID.max_configs)
    return configs