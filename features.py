"""Feature engineering for BTCUSDT. All features use only info available at time t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROLL_WINDOWS, HIGHER_TIMEFRAMES, FEATURE_MIN_BARS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=5).mean()
    sd = s.rolling(win, min_periods=5).std()
    return (s - m) / sd.replace(0, np.nan)


def _rsi(close: pd.Series, win: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(win).mean()
    loss = (-delta.clip(upper=0)).rolling(win).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, win: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - pc).abs(),
         (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(win).mean()


def _htf_context(df: pd.DataFrame, minutes_list) -> pd.DataFrame:
    """Higher-timeframe trend context resampled to the bar grid (no lookahead)."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    for m in minutes_list:
        r = close.resample(f"{m}min", label="right", closed="right")
        rc = r.last()
        h = pd.DataFrame({
            "ret": rc.pct_change(),
            "slope": rc / rc.rolling(min(20, max(4, m // 5))).mean() - 1,
        }).shift(1)                      # only completed higher-TF bars
        h = h.reindex(out.index, method="ffill")
        out[f"htf{m}_ret"] = h["ret"]
        out[f"htf{m}_slope"] = h["slope"]
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, windows=ROLL_WINDOWS,
                   htf=HIGHER_TIMEFRAMES, min_bars=FEATURE_MIN_BARS) -> pd.DataFrame:
    """Return a DataFrame of feature columns (indexed by bar-open timestamp).
    `htf=None` means the mode default (HIGHER_TIMEFRAMES); pass `[]` to disable
    the higher-TF context entirely (4h swing mode)."""
    if htf is None:
        htf = HIGHER_TIMEFRAMES
    if df.index.name != "ts":
        df = df.set_index("ts")
    df = df.sort_index()

    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    feats = pd.DataFrame(index=df.index)

    # -- price level / returns --------------------------------------------
    ret = close.pct_change()
    feats["ret"] = ret
    for w in windows:
        feats[f"ret_{w}"] = close.pct_change(w)
        feats[f"z_{w}"] = _zscore(close, w)
        feats[f"sma_off_{w}"] = close / close.rolling(w, min_periods=5).mean() - 1
        feats[f"ema_off_{w}"] = close / _ema(close, w) - 1
        feats[f"vol_{w}"] = ret.rolling(w, min_periods=5).std()
        feats[f"vol_z_{w}"] = _zscore(ret, w)
        feats[f"max_ret_{w}"] = ret.rolling(w, min_periods=5).max()
        feats[f"min_ret_{w}"] = ret.rolling(w, min_periods=5).min()

    # -- momentum ----------------------------------------------------------
    feats["rsi"] = _rsi(close)
    macd = _ema(close, 12) - _ema(close, 26)
    macd_sig = _ema(macd, 9)
    feats["macd"] = macd
    feats["macd_sig"] = macd_sig
    feats["macd_hist"] = macd - macd_sig

    # -- volatility / ATR ----------------------------------------------------
    atr = _atr(df)
    feats["atr"] = atr
    feats["atr_pct"] = atr / close.replace(0, np.nan)
    feats["ret_over_atr"] = ret / atr.replace(0, np.nan)
    feats["bb_pos"] = ((close - close.rolling(20).mean())
                       / (2 * close.rolling(20).std()).replace(0, np.nan))

    # -- volume --------------------------------------------------------------
    vol_med = vol.rolling(48, min_periods=10).median()
    feats["vol_z"] = _zscore(vol, 48)
    feats["vol_ratio"] = vol / vol_med.replace(0, np.nan)
    feats["dollar_vol"] = close * vol / (close * vol).rolling(48, min_periods=10).mean()

    # -- candle shape (signed body / wick structure) --------------------------
    rng = (high - low).replace(0, np.nan)
    feats["body_ratio"] = (close - df["open"]) / rng
    feats["upper_wick"] = (high - np.maximum(close, df["open"])) / rng
    feats["lower_wick"] = (np.minimum(close, df["open"]) - low) / rng
    feats["close_pos"] = (close - low) / rng

    # -- VWAP -----------------------------------------------------------------
    tp = (high + low + close) / 3
    cvol = (tp * vol).cumsum()
    cvols = vol.cumsum()
    feats["vwap"] = cvol / cvols.replace(0, np.nan)
    feats["vwap_off"] = close / feats["vwap"] - 1

    # -- pivot point / support/resistance -------------------------------------------
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low       # R1 resistance
    r2 = pivot + (high - low)    # R2 resistance
    r3 = high + 2 * (pivot - low) # R3 resistance
    s1 = (2 * pivot) - high       # S1 support
    s2 = pivot - (high - low)    # S2 support
    s3 = low - 2 * (high - pivot) # S3 support

    feats["pivot"] = pivot
    feats["r1"] = r1
    feats["r2"] = r2
    feats["r3"] = r3
    feats["s1"] = s1
    feats["s2"] = s2
    feats["s3"] = s3

    # -- time-of-day / session cyclics ----------------------------------------
    hour = df.index.hour + df.index.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    wd = df.index.dayofweek
    feats["weekday_sin"] = np.sin(2 * np.pi * wd / 7.0)
    feats["weekday_cos"] = np.cos(2 * np.pi * wd / 7.0)

    # -- ADX / directional strength ----------------------------------------------
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    trs = pd.Series(high - low).rolling(14, min_periods=5).sum()
    pdi = 100 * pd.Series(plus_dm).rolling(14, min_periods=5).sum() / trs.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm).rolling(14, min_periods=5).sum() / trs.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    feats["adx"] = dx.rolling(14, min_periods=5).mean()
    feats["pdi"] = pdi
    feats["mdi"] = mdi

    # -- streak (consecutive same-sign closes) -----------------------------------
    rsign = np.sign(ret.fillna(0.0)).astype(int)
    ids = pd.Series(rsign).cumsum() * 0 + (rsign != pd.Series(rsign).shift(1).fillna(0)).cumsum()
    cnt = pd.Series(np.ones(len(df.index)), index=df.index).groupby(ids.values).cumcount() + 1
    feats["streak"] = np.asarray(rsign, dtype=float) * cnt.values

# -- higher-TF context -----------------------------------------------------
    feats = pd.concat([feats, _htf_context(df, htf)], axis=1) if htf else feats

    # -- liquidity flag from data cleaning --------------------------------------
    if "illiquid" in df.columns:
        feats["illiquid"] = df["illiquid"]

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats


FEATURE_COLUMNS = [
    "ret", "rsi", "macd", "macd_sig", "macd_hist", "atr_pct", "ret_over_atr",
    "bb_pos", "vol_z", "vol_ratio", "body_ratio", "upper_wick", "lower_wick",
    "close_pos", "vwap_off",
    "pivot", "r1", "r2", "r3", "s1", "s2", "s3",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "adx", "pdi", "mdi", "streak",
    "htf15_ret", "htf15_slope", "htf60_ret", "htf60_slope",
    "htf240_ret", "htf240_slope", "htf1440_ret", "htf1440_slope",
] + [f"ret_{w}" for w in ROLL_WINDOWS] + [f"z_{w}" for w in ROLL_WINDOWS] \
    + [f"sma_off_{w}" for w in ROLL_WINDOWS] + [f"ema_off_{w}" for w in ROLL_WINDOWS] \
    + [f"vol_{w}" for w in ROLL_WINDOWS] + [f"vol_z_{w}" for w in ROLL_WINDOWS] \
    + [f"max_ret_{w}" for w in ROLL_WINDOWS] + [f"min_ret_{w}" for w in ROLL_WINDOWS]


def feature_matrix(df: pd.DataFrame, htf=None) -> pd.DataFrame:
    """Full feature matrix restricted to trusted (post warmup) rows.
    htf=None -> default higher-timeframe contexts (15m mode); [] -> 4h swing mode."""
    feats = build_features(df, htf=HIGHER_TIMEFRAMES if htf is None else htf)
    feats = feats[feats.index >= feats.index[0] + pd.Timedelta(minutes=90)]
    present = [c for c in FEATURE_COLUMNS if c in feats.columns]
    feats = feats.dropna(subset=present, how="all")
    return feats[present]


# ---------------------------------------------------------------------------
# funding-rate aux data (perpetual, 8h) — regime / positioning signal
# ---------------------------------------------------------------------------
FUNDING_COLUMNS = ["funding_rate", "funding_dev", "funding_z", "funding_abs"]


def add_funding(feats: pd.DataFrame, funding: pd.Series) -> pd.DataFrame:
    """Forward-fill perp funding (8h) onto the feature grid and build regime
    features. Rows before funding coverage stay NaN (dropped by caller)."""
    fr = funding.reindex(feats.index, method="ffill").astype(float)
    feats = feats.copy()
    ema = fr.ewm(span=20, min_periods=10).mean()
    sd = fr.ewm(span=20, min_periods=10).std()
    feats["funding_rate"] = fr
    feats["funding_dev"] = fr - ema
    feats["funding_z"] = (fr - ema) / sd.replace(0, np.nan)
    feats["funding_abs"] = fr.abs()
    return feats


def attach_funding_matrix(df, feats, days=None) -> pd.DataFrame:
    """Load funding covering df's span, attach as features, but do NOT truncate to
    bars where funding exists: candle history is far longer than Binance's funding
    window (~5-6 months), so older bars simply carry NaN funding features (models
    ignore NaN) and the backtest charges 0 funding there (negligible, ~0.02%/yr)."""
    from data import load_funding
    days = days or max(30, int((df.index[-1] - df.index[0]).total_seconds() / 86400))
    fund = load_funding(days=days)
    if not len(fund):
        return feats
    return add_funding(feats, fund)


# ---------------------------------------------------------------------------
# open-interest aux data (perp, 1h) — positioning / crowding signal
# ---------------------------------------------------------------------------
OI_COLUMNS = ["oi", "oi_change", "oi_z"]


def attach_oi_matrix(df, feats, days=None) -> pd.DataFrame:
    """Forward-fill perp open interest (1h) onto the feature grid (causal; the
    cached 1h series ends at the last COMPLETED hour so no lookahead). Older bars
    before OI coverage keep NaN (models ignore NaN). Returns feats unchanged if
    OI data is unavailable."""
    from data import load_open_interest
    days = days or max(30, int((df.index[-1] - df.index[0]).total_seconds() / 86400))
    oi = load_open_interest(days=days)
    if not len(oi):
        return feats
    o = oi.reindex(feats.index, method="ffill").astype(float)
    feats = feats.copy()
    feats["oi"] = o
    feats["oi_change"] = o.pct_change()
    rm = o.rolling(48, min_periods=10).mean()
    rs = o.rolling(48, min_periods=10).std()
    feats["oi_z"] = (o - rm) / rs.replace(0, np.nan)
    return feats