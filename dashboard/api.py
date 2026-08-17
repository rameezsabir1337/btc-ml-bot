"""FastAPI dashboard serving run artifacts + live streaming signal."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from backtest import backtest
from data import Fetcher, clean_candles, Fetcher as _F
from features import (build_features, FEATURE_COLUMNS, FUNDING_COLUMNS,
                      OI_COLUMNS, attach_funding_matrix, attach_oi_matrix)
from models import make_model, fit_score, predict_score
from report import load_artifacts
from grid import _gated_positions

_to_ms = _F._ts_to_ms

app = FastAPI(title="BTCUSDT 15m/4h ML Bot Dashboard")

# ---------------------------------------------------------------------------
# load artifacts
# ---------------------------------------------------------------------------
ARTIFACT_NAME = "best"
_data, _X, _df, _preds = load_artifacts(ARTIFACT_NAME)
# adopt the mode the artifacts were produced in (4h / long-only / sizing knobs)
_meta = _data["report"].get("meta", {})
if not _meta or not _meta.get("timeframe"):
    # Fallback for artifacts saved before `meta` was persisted: infer the bar
    # regime from the spacing of the stored OHLCV index.
    _difs = _df.index.to_series().diff().dropna()
    if len(_difs):
        _med = _difs.median()
        _cand = {"4h": pd.Timedelta(hours=4), "15m": pd.Timedelta(minutes=15)}
        _tinf = min(_cand, key=lambda tf: abs((_cand[tf] - _med).total_seconds()))
        _meta = {"timeframe": _tinf, "long_only": True}
if _meta.get("timeframe"):
    config.configure_timeframe(_meta["timeframe"], long_only=bool(_meta.get("long_only", False)))
config.MIN_HOLD_BARS = int(_meta.get("min_hold_bars", 0))
config.VOL_TARGET = float(_meta.get("vol_target", 0.0))
best_c = _data["report"]["best_config"]
_bc = {"model": best_c["model"], "model_params": best_c["model_params"],
       "label": best_c["label"], "margin": best_c["margin"],
       "trend_gate": best_c.get("trend_gate", "none"),
       "funding_gate": float(best_c.get("funding_gate", 0.0) or 0.0)}
# ensemble deployment: when the artifact report carries members + vote weights,
# the live signal is a Sharpe-weighted majority vote over their gated positions.
_ENS = _data["report"].get("ensemble") or []
_ENS_VET = (_data["report"].get("meta") or {}).get("ensemble_vote") or {}
_ENSEMBLE = len(_ENS) >= 2 and bool(_ENS_VET.get("weights"))
if _ENSEMBLE:
    _ens_weights = [float(w) for w in _ENS_VET["weights"]]
    _ens_vote_th = float(_ENS_VET.get("vote", 0.5))
# columns the deployed model actually uses (includes perp funding + OI regime)
_MODEL_COLS = [c for c in (list(FEATURE_COLUMNS) + list(FUNDING_COLUMNS)
                           + list(OI_COLUMNS)) if c in _X.columns]


def _task(method: str) -> str:
    return "regression" if method == "regression" else \
        ("multiclass" if method == "multiclass" else "binary")


# equity curve from OOS predictions (annualised per active timeframe)
def _funding_arr():
    if "funding_rate" not in _X.columns:
        return None
    return (_X["funding_rate"].reindex(_df.index).ffill().fillna(0.0)
            .to_numpy(dtype=float))


def _equity():
    # prefer the persisted continuous (rolled-across-folds) equity curve so the
    # chart matches the pooled numbers exactly.
    if _data["report"].get("rolled"):
        r = _data["report"]["rolled"]
        eq = pd.Series(r["equity"], index=pd.to_datetime(r["ts"], unit="ms", utc=True))
        cap = config.BacktestConfig.initial_capital
        buyhold = (cap * _df["close"] / _df["close"].iloc[0]).reindex(eq.index).ffill()
        return eq, buyhold
    pos = pd.Series(0.0, index=_df.index)
    pos.update(_preds["position"])
    cfg = config.BacktestConfig(bars_per_year=config.BARS_PER_YEAR,
                                min_hold_bars=config.MIN_HOLD_BARS,
                                vol_target=float(best_c.get("vol_target", 0.0) or 0.0),
                                dd_brake=float(best_c.get("dd_brake", 0.0) or 0.0),
                                high=_df["high"].values, low=_df["low"].values,
                                funding=_funding_arr())
    m = backtest(pos.values, _df["close"].values, cfg)
    eq = pd.Series(m["equity"], index=_df.index[1:])
    cap = config.BacktestConfig.initial_capital
    buyhold = cap * (_df["close"] / _df["close"].iloc[0])
    return eq, buyhold


_EQUITY, _BUYHOLD = _equity()


def _indicator_tail(limit):
    X = _X.iloc[-limit:]
    cols = [c for c in ["rsi", "macd", "macd_sig", "atr_pct", "bb_pos", "vwap_off",
                        "htf15_ret", "htf60_ret"] if c in X.columns]
    out = X[cols].reset_index()
    out["ts"] = _to_ms(out["ts"])
    return out


# ---------------------------------------------------------------------------
# live streaming
# ---------------------------------------------------------------------------
def _fit_one(c, df, X):
    """Fit ONE deployed model for config `c` (single best or ensemble member).
    Features `X` are precomputed and index-aligned to `df`. Binary tasks train
    only on y!=0 rows (mirrors grid eval)."""
    task = _task(c["label"]["method"])
    from labels import make_labels
    from models import encode_y
    y = pd.Series(make_labels(df, c["label"]["method"], c["label"])[0], index=df.index)
    y = y.reindex(X.index).values
    model = make_model(c["model"], c["model_params"], task)
    if task == "binary":
        keep = y != 0
        Xtr, ytr = X[keep], y[keep]
    else:
        Xtr, ytr = X, y
    if len(ytr) < 2 or len(np.unique(ytr)) < 2:
        raise ValueError(f"fit aborted: too few rows for {c['model']}/{c['label']['method']}")
    model = model.fit(Xtr.values, encode_y(ytr, task))
    model.train_n = int(len(ytr))
    return model, task


def _live_gated(c, score, feats):
    """Live position (long-side) for config c on the last feature bar, with the
    same gates as grid evaluation: htf-slope trend gate with GATE_CONFIRM_BARS
    hysteresis + funding-z gate + long-only clamp."""
    import config as _cfg
    pos = 1 if score > c["margin"] else 0
    gate = c.get("trend_gate", "none")
    if gate.startswith("htf") and gate.endswith("_slope") and gate in feats.columns:
        g = pd.Series(feats[gate].values).fillna(0.0)
        confirm = max(1, _cfg.GATE_CONFIRM_BARS)
        rmin = pd.Series(g).rolling(confirm, min_periods=confirm).min()
        if pos > 0 and not (float(rmin.iloc[-1]) > 0 if len(rmin) else False):
            pos = 0
    if pos > 0:
        fz_th = float(c.get("funding_gate", 0.0) or 0.0)
        if fz_th > 0 and "funding_z" in feats.columns:
            if float(feats["funding_z"].fillna(0.0).iloc[-1]) > fz_th:
                pos = 0
        if config.LONG_ONLY:
            pos = max(pos, 0)
    return pos


class Live:
    def __init__(self):
        self.lock = threading.Lock()
        self.fetcher = Fetcher(timeframe=config.TIMEFRAME)
        self.last = None
        self.started = time.time()
        self.models = []
        self.task = None
        self.err = None
        self.fitting = False
        self.retraining = False
        self.last_retrain = None
        self.model_ts = None
        self.train_n = 0
        self.last_retrain_err = None

    def _fit_deployed(self):
        self.models = []
        self.train_n = 0
        for c in (_ENS if _ENSEMBLE else [_bc]):
            m, t = _fit_one(c, _df, _X)
            self.models.append((c, m, t))
            self.train_n = max(self.train_n, int(getattr(m, "train_n", 0)))
        self.task = self.models[0][2]
        self.err = None
        self.model_ts = time.time()
        self.last_retrain = time.time()

    def _retrain_deployed(self):
        """Refit every deployed model on the full known history (frozen artifact
        + freshly fetched bars). Mirrors the initial fit: same label method,
        same feature columns, binary trains only on y!=0 rows. Swaps the models
        atomically under the lock so prediction never sees half-built models."""
        try:
            if not self.models:
                return
            tail = self.fetcher.fetch_latest(limit=config.LIVE_TAIL_BARS)
            base = _df.reset_index().rename(columns={"index": "ts"})
            full = pd.concat([base, tail]).drop_duplicates("ts").sort_values("ts")
            full = clean_candles(full)
            feats = build_features(full, htf=config.SWING_HTF)
            feats = attach_funding_matrix(full, feats)
            feats = attach_oi_matrix(full, feats)
            for _c in _MODEL_COLS:
                if _c not in feats.columns:
                    feats[_c] = np.nan
            X = feats[_MODEL_COLS].reindex(full.index)
            # drop rows with NaN core features (indicator warmup) but KEEP rows
            # where only funding/OI features are NaN (live coverage is shorter
            # than candle history); LGBM ignores NaN.
            core = [c for c in _MODEL_COLS if c not in FUNDING_COLUMNS]
            ok = X[core].notna().all(axis=1)
            Xall = X[ok]
            new_models, n_tr = [], 0
            for c, _, _ in self.models:
                m, t = _fit_one(c, full, Xall)
                new_models.append((c, m, t))
                n_tr = max(n_tr, int(getattr(m, "train_n", 0)))
            if len(new_models) != len(self.models):
                raise ValueError("retrain aborted: member fit failed")
            now = time.time()
            with self.lock:
                self.models = new_models
                self.task = new_models[0][2]
                self.model_ts = now
                self.last_retrain = now
                self.train_n = int(n_tr)
                self.last_retrain_err = None
            print(f"[live] retrained {len(new_models)} model(s) "
                  f"/ {len(Xall.columns)} cols / {self.train_n} rows ts={now:.0f}")
        except Exception as e:
            self.last_retrain_err = str(e)
        finally:
            self.retraining = False

    def refresh(self):
        if not self.models:
            if self.fitting:          # already being fitted; skip this cycle
                return
            self.fitting = True
            try:
                self._fit_deployed()
            except Exception as e:
                self.err = str(e)
            finally:
                self.fitting = False
            if not self.models:
                return
        if not self.retraining and (time.time() - (self.last_retrain or 0)) >= config.LIVE_RETRAIN_SECONDS:
            self.retraining = True
            threading.Thread(target=self._retrain_deployed, daemon=True).start()
        try:
            tail = self.fetcher.fetch_latest(limit=300)
            base = _df.reset_index().rename(columns={"index": "ts"})
            full = pd.concat([base, tail]).drop_duplicates("ts").sort_values("ts")
            full = clean_candles(full)
            # only the recent window is needed for live features (warmup ~150 bars)
            feats = build_features(full.iloc[-2000:], htf=config.SWING_HTF)
            feats = attach_funding_matrix(full, feats)
            feats = attach_oi_matrix(full, feats)
            # ensure every deployed-model column exists (funding/HTF may not load
            # for a live window); unfilled -> NaN -> fillna(0) below.
            for _c in _MODEL_COLS:
                if _c not in feats.columns:
                    feats[_c] = np.nan
            last_row = feats.dropna(subset=_MODEL_COLS, how="all").iloc[-1:]
            row = last_row[_MODEL_COLS].fillna(0.0)
            ts = str(last_row.index[-1])
            # latest candle + indicators for the overlay
            c = full.iloc[-1]
            gate = _bc.get("trend_gate", "none")
            if _ENSEMBLE and len(self.models) > 1:
                votes, wsum = 0.0, 0.0
                for (mc, model, task), w in zip(self.models, _ens_weights):
                    sc = float(predict_score(model, row.values, task)[0])
                    votes += w * _live_gated(mc, sc, feats)
                    wsum += w
                vote = votes / max(wsum, 1e-9)
                pos = 1 if vote >= _ens_vote_th else 0
                score = vote
                model_name = f"ensemble-{len(self.models)}"
            else:
                score = float(predict_score(self.models[0][1], row.values, self.task)[0])
                pos = _live_gated(_bc, score, feats)
                model_name = _bc["model"]
            ind = {k: (None if pd.isna(last_row[k].iloc[-1]) else round(float(last_row[k].iloc[-1]), 6))
                   for k in ["rsi", "macd", "atr_pct", "bb_pos", "vwap_off", "htf15_ret", "htf60_ret"]
                   if k in last_row.columns}
            self.last = {
                "ts": ts,
                "close": float(c["close"]),
                "volume": float(c["volume"]),
                "score": round(score, 4),
                "position": int(pos),
                "model": model_name,
                "margin": _bc["margin"],
                "trend_gate": gate,
                "indicators": ind,
                "uptime_s": int(time.time() - self.started),
                "model_ts": self.model_ts,
                "retrained_at": self.last_retrain,
                "retraining": bool(self.retraining),
                "train_n": int(self.train_n),
                "last_retrain_err": self.last_retrain_err,
            }
            _paper.update(self.last)
        except Exception as e:
            self.err = str(e)


_live = Live()


# ---------------------------------------------------------------------------
# paper ledger: records every live tick and marks an equity curve assuming the
# signal is filled at the close it prints, paying perp taker TAKER_FEE on each
# |position change| (mirrors the backtest cost model). Persisted to CSV so the
# account survives dashboard restarts.
# ---------------------------------------------------------------------------
class PaperLedger:
    def __init__(self, path="data/paper_ledger.csv"):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._seed_equity = float(config.BacktestConfig.initial_capital)
        self._rows = []
        self._cap = 8000
        if self._path.exists():
            try:
                self._rows = (pd.read_csv(self._path, parse_dates=["ts"])
                              .to_dict("records"))
            except Exception:
                self._rows = []

    def update(self, last):
        if not last:
            return
        ts = last["ts"]
        with self._lock:
            if self._rows and self._rows[-1]["ts"] == ts:
                return
            pos = float(last["position"])
            close = float(last["close"])
            prev = self._rows[-1] if self._rows else None
            if prev is None:
                equity = self._seed_equity
                fee = 0.0
            else:
                ret = close / float(prev["close"]) - 1.0
                fee = config.TAKER_FEE * abs(pos - float(prev["position"]))
                equity = float(prev["equity"]) * (1.0 + float(prev["position"]) * ret - fee)
            self._rows.append({"ts": ts, "close": close, "position": pos,
                               "score": float(last["score"]), "fee": round(fee, 6),
                               "equity": round(equity, 4)})
            del self._rows[:max(0, len(self._rows) - self._cap)]

    def flush(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self._rows).to_csv(self._path, index=False)
        except Exception:
            pass

    def snapshot(self):
        with self._lock:
            rows = list(self._rows)
        if not rows:
            return {"live": False, "msg": "no paper rows yet"}
        eq = np.asarray([float(r["equity"]) for r in rows])
        eqs = (eq[1:] / np.abs(eq[:-1])) - 1.0
        sd = eqs.std()
        sharpe = float(eqs.mean() * config.BARS_PER_YEAR / (sd * np.sqrt(config.BARS_PER_YEAR))) \
            if len(eqs) > 2 and sd > 0 else 0.0
        poses = [float(r["position"]) for r in rows]
        trades = sum(1 for i in range(1, len(poses)) if poses[i] != poses[i - 1])
        return {
            "live": True,
            "equity": [{"ts": r["ts"], "equity": float(r["equity"])} for r in rows],
            "tail": rows[-120:],
            "initial_capital": self._seed_equity,
            "total_return": float(eq[-1] / self._seed_equity - 1),
            "sharpe": round(sharpe, 3),
            "trades": trades,
            "rows": len(rows),
            "last": rows[-1],
        }


_paper = PaperLedger()


def _poll():
    while True:
        time.sleep(config.LIVE_POLL_SECONDS)
        _live.refresh()


threading.Thread(target=_poll, daemon=True).start()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse("dashboard/static/index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "artifact": ARTIFACT_NAME, "live": _live.last is not None,
            "live_err": getattr(_live, "err", None)}


@app.get("/api/ohlcv")
def ohlcv(limit: int = Query(2000)):
    df = _df.iloc[-limit:].reset_index()
    df["ts"] = _to_ms(df["ts"])
    return df[["ts", "open", "high", "low", "close", "volume"]].to_dict("records")


@app.get("/api/indicators")
def indicators(limit: int = Query(2000)):
    return _indicator_tail(limit).to_dict("records")


@app.get("/api/predictions")
def predictions(limit: int = Query(2000)):
    p = _preds.iloc[-limit:].reset_index()
    p["ts"] = _to_ms(p["ts"])
    return p[["ts", "score", "position"]].to_dict("records")


@app.get("/api/backtest")
def backtest_endpoint(limit: int = Query(5000)):
    eq = _EQUITY.iloc[-limit:]
    bh = _BUYHOLD.iloc[-limit:]
    return {
        "pooled": _data["report"]["pooled_metrics"],
        "folds": _data["folds"],
        "equity": [{"ts": int(_to_ms(pd.Series([ts]))[0]), "equity": float(e)}
                   for ts, e in eq.items()],
        "buyhold": [{"ts": int(_to_ms(pd.Series([ts]))[0]), "equity": float(e)}
                    for ts, e in bh.items()],
    }


@app.get("/api/grid")
def grid():
    r = _data["report"]
    return {"best_config": r["best_config"], "pooled_metrics": r["pooled_metrics"],
            "num_configs_ranked": r["num_configs_ranked"],
            "top_score": r["top_score"], "bottom_score": r["bottom_score"]}


@app.get("/api/walkforward")
def walkforward():
    return _data["folds"]


@app.get("/api/paper")
def paper():
    return _paper.snapshot()


@app.get("/api/live")
def live():
    return _live.last or {"live": False, "msg": "no live tick yet"}


@app.post("/api/live/retrain")
def retrain():
    if _live.retraining:
        return {"ok": False, "msg": "retrain already in progress"}
    _live.retraining = True
    threading.Thread(target=_live._retrain_deployed, daemon=True).start()
    return {"ok": True, "msg": "retrain started"}


app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()