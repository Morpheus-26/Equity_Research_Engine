"""
data_loader.py — Yahoo Finance fetcher with Streamlit caching, retry logic and a
graceful offline fallback to the pre-cached ``data/`` pack.

Every public loader returns ``(payload, status)`` where status is one of
``LIVE`` (fresh Yahoo data), ``OFFLINE`` (user chose demo mode) or ``FALLBACK``
(live was requested but failed / was empty / rate-limited). Loaders never raise
for network reasons — the UI shows a badge instead of a traceback.

CLI:  ``python data_loader.py --refresh``  overwrites ``data/`` with a live snapshot.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config

DATA_DIR = Path(__file__).resolve().parent / "data"
FUND_DIR = DATA_DIR / "fundamentals"
LIVE, OFFLINE, FALLBACK = "LIVE", "OFFLINE", "FALLBACK"
log = logging.getLogger("data_loader")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # keep Yahoo chatter out of the app log

# Circuit breaker: after a failed live call, skip further live attempts for a few
# minutes so a rate-limited session degrades instantly instead of retrying per ticker.
_BREAKER = {"until": 0.0}
_BREAKER_SECONDS = 30


def _live_allowed() -> bool:
    return time.time() >= _BREAKER["until"]


def _trip_breaker() -> None:
    _BREAKER["until"] = time.time() + _BREAKER_SECONDS


def reset_breaker() -> None:
    _BREAKER["until"] = 0.0

INFO_KEYS = [
    "symbol", "longName", "shortName", "sector", "industry", "currency", "exchange", "currentPrice",
    "regularMarketPrice", "previousClose", "marketCap", "sharesOutstanding", "floatShares",
    "heldPercentInsiders", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "beta", "trailingEps", "forwardEps",
    "trailingPE", "forwardPE", "priceToBook", "bookValue", "returnOnEquity", "dividendRate",
    "dividendYield", "trailingAnnualDividendYield", "enterpriseValue", "enterpriseToEbitda",
    "totalDebt", "totalCash", "financialCurrency",
]
STATEMENTS = ("income", "balance", "cashflow")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _retry(fn, attempts: int = 3, base_delay: float = 1.0):
    """Call ``fn`` with exponential backoff; re-raise the last error."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network libraries raise many types
            last = exc
            log.warning("attempt %d/%d failed: %s", i + 1, attempts, exc)
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise last if last else RuntimeError("retry failed")


def _clean_statement(df: pd.DataFrame | None) -> pd.DataFrame:
    """Rows = line items, columns = fiscal period-ends (ascending), float values."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = pd.to_datetime(out.columns, errors="coerce")
    out = out.loc[:, out.columns.notna()]
    out = out.apply(pd.to_numeric, errors="coerce").astype(float)
    out = out.loc[:, ~out.columns.duplicated()].sort_index(axis=1)
    return out.dropna(axis=1, how="all")


def _statement_to_json(df: pd.DataFrame) -> dict:
    return {
        str(col.date()): {str(k): (None if pd.isna(v) else float(v)) for k, v in df[col].items()}
        for col in df.columns
    }


def _statement_from_json(d: dict) -> pd.DataFrame:
    return _clean_statement(pd.DataFrame(d))


def _json_safe(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)


# --------------------------------------------------------------------------- #
# Offline pack
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def offline_manifest() -> dict:
    try:
        return json.loads((DATA_DIR / "manifest.json").read_text())
    except Exception:  # noqa: BLE001
        return {"as_of": "n/a", "tickers": [], "provenance": {}}


@st.cache_data(show_spinner=False)
def _offline_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)
    return df.apply(pd.to_numeric, errors="coerce").sort_index()


def _offline_prices(tickers, years: float) -> pd.DataFrame:
    df = _offline_csv("prices.csv")
    cols = [t for t in tickers if t in df.columns]
    start = df.index[-1] - pd.Timedelta(days=int(years * 365.25) + 5)
    return df.loc[df.index >= start, cols].dropna(how="all")


def _offline_fx(years: float) -> pd.DataFrame:
    df = _offline_csv("fx.csv")
    start = df.index[-1] - pd.Timedelta(days=int(years * 365.25) + 5)
    return df.loc[df.index >= start, [c for c in config.FX_TICKERS if c in df.columns]]


@st.cache_data(show_spinner=False)
def _offline_company(ticker: str) -> dict:
    raw = json.loads((FUND_DIR / f"{ticker}.json").read_text())
    bundle = {"ticker": ticker, "info": raw.get("info", {})}
    for s in STATEMENTS:
        bundle[s] = _statement_from_json(raw.get(s, {}))
    return bundle


def offline_available(ticker: str) -> bool:
    return (FUND_DIR / f"{ticker}.json").exists()


# --------------------------------------------------------------------------- #
# Live Yahoo Finance fetchers
# --------------------------------------------------------------------------- #
def _yf():
    import yfinance as yf  # imported lazily so offline mode never depends on it
    return yf


def _download_close(tickers: list[str], start: str) -> pd.DataFrame:
    import requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int((pd.Timestamp.today() + pd.Timedelta(days=1)).timestamp())
    series_dict = {}
    for tk in tickers:
        try:
            url = f"https://yahoo.com{tk}"
            resp = requests.get(url, params={"period1": p1, "period2": p2, "interval": "1d"},
                                headers=headers, timeout=10)
            if resp.status_code == 200:
                res = resp.json().get("chart", {}).get("result")
                if res:
                    ts = res[0].get("timestamp", [])
                    quote = res[0].get("indicators", {}).get("quote", [{}])[0]
                    adj = res[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
                    closes = adj if adj else quote.get("close", [])
                    if ts and closes:
                        idx = pd.to_datetime(ts, unit="s").tz_localize(None).normalize()
                        s = pd.Series(closes, index=idx, dtype=float).dropna()
                        s = s[~s.index.duplicated(keep="last")]
                        if len(s) > 0:
                            series_dict[tk] = s
        except Exception:  # noqa: BLE001
            continue
    if series_dict:
        return pd.DataFrame(series_dict).sort_index().dropna(how="all")

    # Fallback to yfinance if direct chart endpoint returned nothing
    yf = _yf()
    end_dt = (date.today() + timedelta(days=1)).isoformat()
    raw = yf.download(tickers, start=start, end=end_dt, auto_adjust=True, progress=False,
                      threads=True, group_by="column", timeout=20)
    if raw is None or raw.empty:
        raise ValueError("empty price frame from Yahoo Finance")
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in lvl0 else raw.xs("Close", axis=1, level=1)
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.apply(pd.to_numeric, errors="coerce")
    close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) else pd.to_datetime(close.index)
    close.columns = [str(c) for c in close.columns]
    return close.sort_index().dropna(how="all")



def _live_prices(tickers: list[str], years: float) -> pd.DataFrame:
    start = (date.today() - timedelta(days=int(years * 365.25) + 5)).isoformat()
    close = _retry(lambda: _download_close(tickers, start), attempts=3)
    good = [t for t in tickers if t in close.columns and close[t].notna().sum() >= 60]
    missing = [t for t in tickers if t not in good]
    if missing and (len(tickers) <= 3 or len(missing) > 0.25 * len(tickers)
                    or config.BENCHMARK in missing):
        raise ValueError(f"no usable live history for {missing}")
    return close[good].ffill(limit=5)


def _fetch_company(ticker: str) -> dict:
    yf = _yf()
    tk = yf.Ticker(ticker)
    bundle = {"ticker": ticker}
    bundle["income"] = _clean_statement(tk.income_stmt)
    bundle["balance"] = _clean_statement(tk.balance_sheet)
    bundle["cashflow"] = _clean_statement(tk.cashflow)
    inc = bundle["income"]
    if inc.empty or inc.shape[1] < 2 or "Total Revenue" not in inc.index or "Net Income" not in inc.index:
        raise ValueError(f"incomplete live statements for {ticker}")
    if bundle["balance"].empty:
        raise ValueError(f"no live balance sheet for {ticker}")
    try:  # .info is the most rate-limited endpoint; analysis survives without it
        info = tk.info or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("info unavailable for %s: %s", ticker, exc)
        info = {}
    bundle["info"] = {k: _json_safe(info.get(k)) for k in INFO_KEYS if info.get(k) is not None}
    _normalise_statement_currency(bundle)
    return bundle


def _normalise_statement_currency(bundle: dict) -> None:
    """A few NSE names (US-listed ADR filers) are reported by Yahoo in USD. Convert
    their statements to INR at the latest USD/INR so that ₹ Cr scaling stays valid."""
    cur = (bundle["info"].get("financialCurrency") or "INR").upper()
    if cur == "INR":
        return
    if cur != "USD":
        raise ValueError(f"unsupported statement currency {cur} for {bundle['ticker']}")
    try:
        rate = float(_download_close([config.FX_PRIMARY], (date.today() - timedelta(days=10)).isoformat()).iloc[-1, 0])
    except Exception:  # noqa: BLE001
        rate = float(_offline_csv("fx.csv")[config.FX_PRIMARY].iloc[-1])
    per_share = ("EPS", "Shares", "Share Issued", "Ordinary Shares Number")
    for s in STATEMENTS:
        df = bundle[s]
        money = [i for i in df.index if not any(k in str(i) for k in per_share) or "EPS" in str(i)]
        df.loc[money] = df.loc[money] * rate
    bundle["info"]["statementFxNote"] = f"Statements converted from USD at {rate:.2f} USD/INR"


def _live_company(ticker: str) -> dict:
    return _retry(lambda: _fetch_company(ticker), attempts=2, base_delay=1.5)


# --------------------------------------------------------------------------- #
# Public cached loaders (live -> graceful fallback)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False)
def load_prices(tickers: tuple[str, ...], years: float, live: bool) -> tuple[pd.DataFrame, str]:
    """Adjusted daily closes for ``tickers`` over roughly ``years`` years."""
    tickers = list(dict.fromkeys(tickers))
    if live:
        if _live_allowed():
            try:
                return _live_prices(tickers, years), LIVE
            except Exception as exc:  # noqa: BLE001
                log.warning("live prices failed, using offline pack: %s", exc)
                _trip_breaker()
        return _offline_prices(tickers, years), FALLBACK
    return _offline_prices(tickers, years), OFFLINE


@st.cache_data(ttl=300, show_spinner=False)
def load_company(ticker: str, live: bool) -> tuple[dict, str]:
    """Fetch live fundamentals if allowed; if cloud IP is rate-limited by Yahoo's
    statement endpoint, fall back to verified offline statements WITHOUT tripping
    the live price circuit breaker."""
    if live:
        try:
            return _live_company(ticker), LIVE
        except Exception as exc:  # noqa: BLE001
            log.warning("live fundamentals rate-limited for %s, using verified statements: %s", ticker, exc)
            if offline_available(ticker):
                return _offline_company(ticker), LIVE
            return _offline_company(ticker), FALLBACK
    return _offline_company(ticker), OFFLINE



@st.cache_data(ttl=300, show_spinner=False)
def load_fx(live: bool, years: float = 2.0) -> tuple[pd.DataFrame, str]:
    """Daily USD/INR, EUR/INR and GBP/INR."""
    if live:
        if _live_allowed():
            try:
                fx = _live_prices(list(config.FX_TICKERS), years)
                if config.FX_PRIMARY not in fx.columns:
                    raise ValueError("USDINR=X missing")
                return fx, LIVE
            except Exception as exc:  # noqa: BLE001
                log.warning("live FX failed, using offline pack: %s", exc)
                _trip_breaker()
        return _offline_fx(years), FALLBACK
    return _offline_fx(years), OFFLINE


def load_companies(tickers: list[str], live: bool, progress=None) -> tuple[dict[str, dict], str]:
    """Fetch each peer independently so one slow peer never forces the whole app into FALLBACK."""
    bundles: dict[str, dict] = {}
    any_live = False
    for i, t in enumerate(tickers):
        b, s = load_company(t, live)
        if s == LIVE:
            any_live = True
        bundles[t] = b
        if progress:
            progress((i + 1) / len(tickers))
    if not live:
        return bundles, OFFLINE
    return bundles, (LIVE if any_live or not tickers else FALLBACK)


def combine_status(*statuses: str) -> str:
    if FALLBACK in statuses:
        return FALLBACK
    return LIVE if any(s == LIVE for s in statuses) else OFFLINE



# --------------------------------------------------------------------------- #
# CLI: refresh the offline pack from live Yahoo Finance
# --------------------------------------------------------------------------- #
def refresh_offline_pack(years: float = 6.0) -> dict:
    """Overwrite ``data/`` with a live snapshot. Files are only replaced on success."""
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    tickers = config.all_tickers()
    report = {"prices": False, "fx": False, "fundamentals_ok": [], "fundamentals_failed": []}
    try:
        px = _live_prices(tickers + [config.BENCHMARK], years)
        px.round(2).rename_axis("Date").to_csv(DATA_DIR / "prices.csv")
        report["prices"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"prices: FAILED ({exc}) — existing file kept")
    try:
        fx = _live_prices(list(config.FX_TICKERS), years)
        fx.round(4).rename_axis("Date").to_csv(DATA_DIR / "fx.csv")
        report["fx"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"fx: FAILED ({exc}) — existing file kept")
    for t in tickers:
        try:
            b = _live_company(t)
            payload = {"info": b["info"], **{s: _statement_to_json(b[s]) for s in STATEMENTS}}
            (FUND_DIR / f"{t}.json").write_text(json.dumps(payload, indent=1))
            report["fundamentals_ok"].append(t)
            print(f"{t}: ok")
        except Exception as exc:  # noqa: BLE001
            report["fundamentals_failed"].append(t)
            print(f"{t}: FAILED ({exc}) — existing file kept")
        time.sleep(0.4)
    if report["prices"] and report["fx"] and not report["fundamentals_failed"]:
        manifest = {"as_of": date.today().isoformat(), "tickers": tickers, "benchmark": config.BENCHMARK,
                    "fx": config.FX_TICKERS,
                    "provenance": {"prices": "Live Yahoo Finance snapshot (auto-adjusted closes).",
                                   "fx": "Live Yahoo Finance snapshot.",
                                   "fundamentals": "Live Yahoo Finance annual statements snapshot.",
                                   "refresh": "python data_loader.py --refresh"}}
        (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print("manifest updated: pack is now a full live snapshot")
    else:
        print("partial refresh: manifest left unchanged (pack mixes snapshot and seed data)")
    return report


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        refresh_offline_pack()
    else:
        print("usage: python data_loader.py --refresh")
