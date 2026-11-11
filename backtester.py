"""
backtester.py — NIFTY multi-factor screener (Value / Quality / Momentum) and a
quarterly-rebalanced long-only backtest against the NIFTY 50 (^NSEI).

Point-in-time discipline
------------------------
* Momentum uses only prices up to each rebalance date.
* Value and Quality use the latest fiscal year whose statements were public on the
  rebalance date (fiscal year-end + 90-day reporting lag), priced at that date.
* Free data sources carry only 4–5 fiscal years. For rebalances that pre-date the first
  available statement, the earliest statement is used and the rebalance is counted in
  ``lookahead_rebalances`` so the limitation is visible rather than hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from fundamentals import CR

TRADING_DAYS = 252
WARMUP_DAYS = 253          # 12-1 momentum needs 252 trading days of history
FACTOR_COLS = ["Earnings Yield %", "FCF Yield %", "Book/Price", "ROE %", "F-Score", "Debt/Equity",
               "Mom 12-1 %", "6M Risk-Adj"]


# --------------------------------------------------------------------------- #
# Factor construction
# --------------------------------------------------------------------------- #
def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    s = s.clip(lo, hi)                                   # winsorise cross-sectional outliers
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index).where(s.notna())
    return ((s - s.mean()) / sd).clip(-3, 3)


def _fiscal_index(fund: dict, asof: pd.Timestamp) -> tuple[int, bool]:
    """Index of the latest fiscal year public on ``asof``; (0, False) if none was public yet."""
    lag = pd.Timedelta(days=config.REPORTING_LAG_DAYS)
    known = [i for i, p in enumerate(fund["periods"]) if p + lag <= asof]
    return (known[-1], True) if known else (0, False)


def raw_factors(funds: dict[str, dict], px: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Raw factor inputs for every ticker with enough price history on ``asof``."""
    hist = px.loc[:asof]
    rows = {}
    for tk, f in funds.items():
        if tk not in hist.columns:
            continue
        s = hist[tk].dropna()
        if len(s) < WARMUP_DAYS or s.index[-1] < asof - pd.Timedelta(days=10):
            continue
        price = float(s.iloc[-1])
        i, pit = _fiscal_index(f, asof)
        r = f["table"].iloc[i]
        fs = f["piotroski_history"][i]
        eps, bvps = r["EPS"], r["BVPS"]
        fcf_ps = np.nan if f["financial"] else r["FCF"] * CR / f["shares"] if f["shares"] else np.nan
        win = s.iloc[-127:]
        vol6 = win.pct_change().dropna().std() * np.sqrt(126)
        rows[tk] = {
            "Sector": f["sector"],
            "Earnings Yield %": eps / price * 100 if pd.notna(eps) else np.nan,
            "FCF Yield %": fcf_ps / price * 100 if pd.notna(fcf_ps) else np.nan,
            "Book/Price": bvps / price if pd.notna(bvps) else np.nan,
            "ROE %": r["ROE %"],
            "F-Score": (fs["score"] / fs["evaluated"] * 9) if fs and fs["evaluated"] else np.nan,
            "Debt/Equity": np.nan if f["financial"] else r["Debt/Equity"],
            "Mom 12-1 %": (s.iloc[-22] / s.iloc[-253] - 1) * 100,
            "6M Risk-Adj": (win.iloc[-1] / win.iloc[0] - 1) / vol6 if vol6 > 0 else np.nan,
            "Price": price, "Fiscal Year": f["years"][i], "Point-in-Time": pit,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def score_factors(raw: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Pillar z-scores (Value, Quality, Momentum) and the weighted composite rank."""
    if raw.empty:
        return raw
    out = raw.copy()
    z = {c: _zscore(raw[c]) for c in FACTOR_COLS}
    value = pd.concat([z["Earnings Yield %"], z["FCF Yield %"], z["Book/Price"]], axis=1).mean(axis=1, skipna=True)
    quality = pd.concat([z["ROE %"], z["F-Score"], -z["Debt/Equity"]], axis=1).mean(axis=1, skipna=True)
    momentum = pd.concat([z["Mom 12-1 %"], z["6M Risk-Adj"]], axis=1).mean(axis=1, skipna=True)
    out["Value"], out["Quality"], out["Momentum"] = _zscore(value), _zscore(quality), _zscore(momentum)
    tot = sum(weights.values()) or 1.0
    w = {k: v / tot for k, v in weights.items()}
    out["Composite"] = (w.get("value", 0) * out["Value"].fillna(0) + w.get("quality", 0) * out["Quality"].fillna(0)
                        + w.get("momentum", 0) * out["Momentum"].fillna(0))
    out = out.sort_values("Composite", ascending=False)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out


def screen(funds: dict[str, dict], px: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Current factor screen on the latest available date."""
    return score_factors(raw_factors(funds, px, px.index[-1]), weights)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def rebalance_dates(index: pd.DatetimeIndex, start: pd.Timestamp) -> list[pd.Timestamp]:
    """Start date plus the last trading day of every calendar quarter after it."""
    idx = index[index >= start]
    q_end = pd.Series(idx, index=idx).groupby([idx.year, idx.quarter]).last()
    dates = [idx[0]] + [d for d in q_end.tolist() if idx[0] < d < idx[-1]]
    return sorted(set(dates))


def _target_weights(picks: list[str], px: pd.DataFrame, asof: pd.Timestamp, scheme: str) -> pd.Series:
    if scheme == "inverse_vol":
        vol = px.loc[:asof, picks].iloc[-127:].pct_change().std()
        inv = 1 / vol.replace(0, np.nan)
        if inv.notna().all():
            return inv / inv.sum()
    return pd.Series(1 / len(picks), index=picks)


def performance_metrics(port: pd.Series, bench: pd.Series, rf: float) -> dict:
    """Risk/return statistics from two aligned NAV series."""
    rp, rb = port.pct_change().dropna(), bench.pct_change().dropna()
    rp, rb = rp.align(rb, join="inner")
    yrs = max((port.index[-1] - port.index[0]).days / 365.25, 1e-9)
    cagr_p = (port.iloc[-1] / port.iloc[0]) ** (1 / yrs) - 1
    cagr_b = (bench.iloc[-1] / bench.iloc[0]) ** (1 / yrs) - 1
    rf_d = rf / TRADING_DAYS
    vol_p, vol_b = rp.std() * np.sqrt(TRADING_DAYS), rb.std() * np.sqrt(TRADING_DAYS)
    ex_p, ex_b = rp.mean() * TRADING_DAYS - rf, rb.mean() * TRADING_DAYS - rf
    downside = np.sqrt(np.mean(np.minimum(rp - rf_d, 0) ** 2)) * np.sqrt(TRADING_DAYS)
    beta = rp.cov(rb) / rb.var() if rb.var() > 0 else np.nan
    active = rp - rb
    te = active.std() * np.sqrt(TRADING_DAYS)
    mp = port.resample("ME").last().pct_change().dropna()
    mb = bench.resample("ME").last().pct_change().dropna()
    mp, mb = mp.align(mb, join="inner")
    dd = lambda s: float((s / s.cummax() - 1).min())  # noqa: E731
    return {
        "Portfolio CAGR %": cagr_p * 100, "Benchmark CAGR %": cagr_b * 100,
        "Annualized Volatility %": vol_p * 100, "Benchmark Volatility %": vol_b * 100,
        "Sharpe Ratio": ex_p / vol_p if vol_p > 0 else np.nan,
        "Benchmark Sharpe": ex_b / vol_b if vol_b > 0 else np.nan,
        "Sortino Ratio": ex_p / downside if downside > 0 else np.nan,
        "Jensen's Alpha %": (ex_p - beta * ex_b) * 100, "Beta": beta,
        "Max Drawdown %": dd(port) * 100, "Benchmark Max Drawdown %": dd(bench) * 100,
        "Win Rate %": float((mp > mb).mean() * 100) if len(mp) else np.nan,
        "Positive Months %": float((mp > 0).mean() * 100) if len(mp) else np.nan,
        "Tracking Error %": te * 100, "Information Ratio": active.mean() * TRADING_DAYS / te if te > 0 else np.nan,
        "Years": yrs, "Months": int(len(mp)),
    }


def run_backtest(px: pd.DataFrame, funds: dict[str, dict], weights: dict[str, float], top_n: int = 5,
                 years: int = 3, scheme: str = "equal", cost: float = config.TXN_COST,
                 rf: float = config.RISK_FREE_RATE) -> dict:
    """Quarterly-rebalanced top-N factor portfolio vs NIFTY 50.

    ``cost`` is the round-trip execution cost. At each rebalance the fraction of the
    portfolio that is replaced (one-way turnover = ½ Σ|Δw|) is charged ``cost``; the
    initial purchase is half a round trip.
    """
    bench_col = config.BENCHMARK
    if bench_col not in px.columns:
        raise ValueError("benchmark history missing")
    stocks = [c for c in px.columns if c != bench_col and c in funds]
    px = px.sort_index()
    first_ok = px.index[min(WARMUP_DAYS, len(px.index) - 1)]
    start = max(px.index[-1] - pd.Timedelta(days=int(years * 365.25)), first_ok)
    start = px.index[px.index.searchsorted(start)]
    rb = rebalance_dates(px.index, start)
    rets = px[stocks].pct_change()

    nav, nav_val = {start: 1.0}, 1.0
    w_drift = pd.Series(dtype=float)
    log, lookahead, total_cost = [], 0, 0.0
    for k, d in enumerate(rb):
        scored = score_factors(raw_factors({t: funds[t] for t in stocks}, px[stocks], d), weights)
        if scored.empty:
            continue
        picks = scored.index[:min(top_n, len(scored))].tolist()
        target = _target_weights(picks, px, d, scheme)
        allk = target.index.union(w_drift.index)
        turnover = 0.5 * float((target.reindex(allk, fill_value=0) - w_drift.reindex(allk, fill_value=0)).abs().sum())
        c = turnover * cost
        nav_val *= 1 - c
        total_cost += c
        if not scored.loc[picks, "Point-in-Time"].all():
            lookahead += 1
        log.append({"Rebalance Date": d.date(), "Holdings": ", ".join(p.replace(".NS", "") for p in picks),
                    "One-way Turnover %": turnover * 100, "Cost (bps)": c * 1e4,
                    "Top Score": float(scored["Composite"].iloc[0])})
        end = rb[k + 1] if k + 1 < len(rb) else px.index[-1]
        window = rets.loc[(rets.index > d) & (rets.index <= end), picks].fillna(0.0)
        if window.empty:
            w_drift = target
            continue
        growth = (1 + window).cumprod() * target
        total = growth.sum(axis=1)
        for dt, v in total.items():
            nav[dt] = nav_val * v
        nav_val *= float(total.iloc[-1])
        w_drift = growth.iloc[-1] / total.iloc[-1]

    port = pd.Series(nav).sort_index()
    bench = px[bench_col].reindex(port.index).ffill()
    bench = bench / bench.iloc[0]
    metrics = performance_metrics(port, bench, rf)
    log_df = pd.DataFrame(log)
    return {
        "nav": port, "bench": bench, "metrics": metrics, "rebalances": log_df,
        "start": port.index[0], "end": port.index[-1], "n_rebalances": len(log),
        "lookahead_rebalances": lookahead, "total_cost_pct": total_cost * 100,
        "avg_turnover_pct": float(log_df["One-way Turnover %"].iloc[1:].mean()) if len(log_df) > 1 else np.nan,
        "requested_years": years, "universe_size": len(stocks), "scheme": scheme, "top_n": top_n,
        "drawdown": port / port.cummax() - 1, "bench_drawdown": bench / bench.cummax() - 1,
    }
