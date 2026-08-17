"""ccxt data fetching, resumable caching and candle-quality filtering."""
from __future__ import annotations

import time
import ccxt
import numpy as np
import pandas as pd

from config import (SYMBOL, PERP_SYMBOL, TIMEFRAME, EXCHANGE, HISTORY_DAYS, OHLCV_LIMIT,
                    DATA_DIR, CACHE_DIR, FEATURE_MIN_BARS)


class Fetcher:
    """Rate-limit aware Binance OHLCV fetcher with CSV/parquet caching."""

    def __init__(self, symbol=SYMBOL, timeframe=TIMEFRAME, exchange=EXCHANGE):
        self.symbol = symbol
        self.timeframe = timeframe
        ex_cls = getattr(ccxt, exchange)
        self.ex = ex_cls({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        self.cache_path = CACHE_DIR / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        # '15m' or '1h' -> interval in seconds
        self.interval_s = self.ex.parse_timeframe(timeframe)

    # -- ts helpers (unit-robust: pandas 2.3 gives datetime64[ms, UTC]) --------
    @staticmethod
    def _ms_to_ts(ms):
        return pd.to_datetime(ms, unit="ms", utc=True)

    @classmethod
    def _ts_to_ms(cls, col) -> pd.Series:
        ints = col.astype("int64")
        if "ns" in str(col.dtype):
            return ints // 10 ** 6
        return ints

    # -- raw fetch ----------------------------------------------------------
    def _fetch_range(self, since_ms: int, until_ms: int) -> list[list]:
        all_rows: list[list] = []
        cursor = since_ms
        while cursor < until_ms:
            batch = self.ex.fetch_ohlcv(self.symbol, self.timeframe, since=cursor, limit=OHLCV_LIMIT)
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= cursor:  # no progress -> avoid infinite loop
                break
            cursor = last_ts + self.interval_s * 1000
            time.sleep(self.ex.rateLimit / 1000 if self.ex.rateLimit else 0.2)
        return all_rows

    def fetch_history(self, days=HISTORY_DAYS, force_refresh=False) -> pd.DataFrame:
        """Fetch (or load cached) OHLCV history, growing the cache both forward
        and backward as needed; returns only the requested window."""
        until_ms = int(time.time() * 1000)
        since_ms = until_ms - int(days * 86400 * 1000)
        interval_ms = self.interval_s * 1000

        if not force_refresh and self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            df["ts"] = self._ms_to_ts(df["ts"])
            df = df.drop_duplicates("ts").sort_values("ts")
            cached_start = int(self._ts_to_ms(df["ts"]).min())
            cached_end = int(self._ts_to_ms(df["ts"]).max())
            pieces = [df]
            # backfill forward (new bars since last fetch)
            if cached_end < until_ms - interval_ms:
                rows = self._fetch_range(cached_end + interval_ms, until_ms)
                if rows:
                    pieces.append(self._rows_to_df(rows))
            # backfill backward (older bars now requested)
            if since_ms < cached_start:
                rows = self._fetch_range(since_ms, cached_start - interval_ms)
                if rows:
                    pieces.append(self._rows_to_df(rows))
            full = pd.concat(pieces).drop_duplicates("ts").sort_values("ts")
            self._save(full)                       # cache keeps the full span
            return full[(full["ts"] >= self._ms_to_ts(since_ms)) &
                        (full["ts"] <= self._ms_to_ts(until_ms))]

        rows = self._fetch_range(since_ms, until_ms)
        df = self._rows_to_df(rows)
        self._save(df)
        return df

    @classmethod
    def _rows_to_df(cls, rows: list[list]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = cls._ms_to_ts(df["ts"])
        return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    def _save(self, df: pd.DataFrame) -> None:
        out = df.copy()
        out["ts"] = self._ts_to_ms(out["ts"])
        out.to_parquet(self.cache_path, index=False)

    # -- live tail ----------------------------------------------------------
    def fetch_latest(self, limit=100) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        return self._rows_to_df(rows)


def clean_candles(df: pd.DataFrame, min_bars=FEATURE_MIN_BARS) -> pd.DataFrame:
    """Denoise / clean raw candles: dedupe, spike filter, illiquid bar filter."""
    df = df.copy().set_index("ts").sort_index()
    # numeric coercion, drop NaNs and bad rows
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[df["volume"] >= 0]
    # quote-consistency: high >= max(open,close), low <= min(open,close)
    df = df[(df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9) &
            (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9)]

    # spike filter: a move > ~5x rolling ATR spread is treated as a feed error
    spread = (df["high"] - df["low"]).fillna(0)
    atr = spread.rolling(48, min_periods=10).median()
    drop = (spread > atr * 5).where(atr > 0, False).fillna(False)
    # drop these *data* spikes (do not resample; just exclude the corrupted event)
    df = df[~drop]

    # illiquid bars: tiny volume relative to rolling median -> keep but it's fine;
    # we surface a liquidity flag for feature/model use rather than hard-drop.
    vol_med = df["volume"].rolling(min_bars, min_periods=10).median().replace(0, np.nan)
    df["illiquid"] = (df["volume"] < vol_med * 0.05).fillna(False).astype(int)

    if len(df) < min_bars:
        raise ValueError(f"Only {len(df)} clean bars after filtering; need >={min_bars}.")
    return df


def load_or_fetch(days=HISTORY_DAYS, force_refresh=False, timeframe=None) -> pd.DataFrame:
    """Convenience: fetch + clean in one call. timeframe overrides TIMEFRAME."""
    tf = timeframe or TIMEFRAME
    f = Fetcher(timeframe=tf)
    raw = f.fetch_history(days=days, force_refresh=force_refresh)
    return clean_candles(raw)


def load_funding(symbol=None, days=HISTORY_DAYS, force_refresh=False) -> pd.Series:
    """Perp funding-rate history (8h intervals) as a Series indexed by UTC ts.

    Binance's public funding endpoint grants ~5-6 months; we paginate backward
    with endTime until the requested window is covered. Cached under data/cache.
    """
    perp = symbol or PERP_SYMBOL
    cache = CACHE_DIR / "funding_8h.parquet"
    until_ms = int(time.time() * 1000)
    since_ms = until_ms - int(days * 86400 * 1000)

    df = None
    if not force_refresh and cache.exists():
        df = pd.read_parquet(cache)
        df["ts"] = Fetcher._ms_to_ts(df["ts"])
        df = df.drop_duplicates("ts").sort_values("ts").set_index("ts")
        df = df[df.index >= Fetcher._ms_to_ts(int(1.6e12))]   # drop garbage/stale ts

    if df is None or (not force_refresh and len(df) and int(df.index.max().value // 10 ** 6) < until_ms - 8 * 3600 * 1000):
        ex = ccxt.binance({"enableRateLimit": True})
        rows, end = [], until_ms
        for _ in range(60):                       # ~500 * 8h ≈ 166 days per call
            try:
                batch = ex.fetch_funding_rate_history(perp, limit=1000, params={"endTime": end})
            except Exception:
                break
            if not batch:
                break
            rows.extend(batch)
            oldest = min(r["timestamp"] for r in batch)
            if oldest <= since_ms or len(batch) < 1000:
                break
            end = oldest - 1
            time.sleep(ex.rateLimit / 1000 if ex.rateLimit else 0.2)
        if rows:
            fresh = pd.DataFrame(rows)
            fresh = fresh[["timestamp", "fundingRate"]].rename(columns={"timestamp": "ts"})
            fresh["ts"] = Fetcher._ms_to_ts(fresh["ts"])
            fresh = fresh.drop_duplicates("ts").sort_values("ts").set_index("ts")
            if df is not None and len(df):
                df = pd.concat([df, fresh]).drop_duplicates().sort_index()
            else:
                df = fresh
            out = df.copy().reset_index()
            out["ts"] = Fetcher._ts_to_ms(out["ts"])
            out.to_parquet(cache, index=False)

    if df is None or not len(df):
        return pd.Series(dtype=float)
    df = df[(df.index >= Fetcher._ms_to_ts(since_ms)) &
            (df.index <= Fetcher._ms_to_ts(until_ms))].sort_index()
    return pd.Series(df["fundingRate"].values.astype(float), index=df.index)


def load_open_interest(symbol=None, days=HISTORY_DAYS, force_refresh=False) -> pd.Series:
    """Perp open-interest history (1h) as a Series indexed by UTC ts.

    Binance's /fapi openInterestHist is paginated backward; cached under
    data/cache/oi_1h.parquet. Returns an empty Series if OI is unavailable
    (caller simply skips the OI features)."""
    perp = symbol or PERP_SYMBOL
    cache = CACHE_DIR / "oi_1h.parquet"
    until_ms = int(time.time() * 1000)
    since_ms = until_ms - int(days * 86400 * 1000)

    df = None
    if not force_refresh and cache.exists():
        df = pd.read_parquet(cache)
        df["ts"] = Fetcher._ms_to_ts(df["ts"])
        df = df.drop_duplicates("ts").sort_values("ts").set_index("ts")
        df = df[df.index >= Fetcher._ms_to_ts(int(1.6e12))]

    need = df is None or (not force_refresh and len(df) and
                          int(df.index.max().value // 10 ** 6) < until_ms - 3600 * 1000)
    if need:
        try:
            ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
            rows, end = [], until_ms
            for _ in range(60):
                try:
                    batch = ex.fetch_open_interest_history(perp, "1h", limit=500,
                                                           params={"endTime": end})
                except Exception:
                    batch = None
                if not batch:
                    break
                rows.extend(batch)
                oldest = min(r["timestamp"] for r in batch)
                if oldest <= since_ms or len(batch) < 500:
                    break
                end = oldest - 1
                time.sleep(ex.rateLimit / 1000 if ex.rateLimit else 0.2)
        except Exception as exc:  # OI unavailability must not break the pipeline
            print(f"  [OI] fetch failed ({exc}); continuing without OI features")
            rows = []
        if rows:
            raw = pd.DataFrame(rows)
            oi_col = next((c for c in ("openInterest", "sumOpenInterest", "baseVolume")
                           if c in raw.columns), None)
            if oi_col is not None:
                fresh = pd.DataFrame({
                    "ts": Fetcher._ms_to_ts(pd.to_numeric(raw["timestamp"], errors="coerce")),
                    "oi": pd.to_numeric(raw[oi_col], errors="coerce"),
                }).dropna()
                fresh = fresh.drop_duplicates("ts").sort_values("ts").set_index("ts")
                if df is not None and len(df):
                    df = pd.concat([df, fresh])
                else:
                    df = fresh
                df = df[~df.index.duplicated(keep="last")].sort_index()
                out = df.reset_index()
                out["ts"] = Fetcher._ts_to_ms(out["ts"])
                out.to_parquet(cache, index=False)

    if df is None or not len(df):
        return pd.Series(dtype=float)
    df = df[(df.index >= Fetcher._ms_to_ts(since_ms)) &
            (df.index <= Fetcher._ms_to_ts(until_ms))].sort_index()
    return pd.Series(df["oi"].values.astype(float), index=df.index)


if __name__ == "__main__":
    df = load_or_fetch()
    print(f"Loaded {len(df):,} clean {TIMEFRAME} bars ({SYMBOL})")
    print(df.tail())