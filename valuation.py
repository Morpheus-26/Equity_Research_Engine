"""
valuation.py — India-calibrated DCF, 5x5 sensitivity matrix, NSE peer multiples,
two-year financial projections and INR currency (FX beta) analysis.

Non-financial companies are valued with a 5-year FCFF DCF. Banks are valued with a
5-year Excess-Return (residual income) model on book equity, because free cash flow
and enterprise value are not defined for a deposit-taking balance sheet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config
from fundamentals import CR, sdiv

STEP_BPS = [-0.010, -0.005, 0.0, 0.005, 0.010]


def default_params() -> dict:
    return {"rf": config.RISK_FREE_RATE, "erp": config.EQUITY_RISK_PREMIUM, "tax": config.TAX_RATE,
            "g": config.TERMINAL_GROWTH, "dcf_weight": config.DEFAULT_DCF_WEIGHT}


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def projected_labels(last_period: pd.Timestamp, n: int) -> list[str]:
    yy = last_period.year % 100
    if last_period.month == 3:
        return [f"FY{(yy + i) % 100:02d}E" for i in range(1, n + 1)]
    if last_period.month == 12:
        return [f"CY{(yy + i) % 100:02d}E" for i in range(1, n + 1)]
    return [f"FY{(yy + i) % 100:02d}E({last_period:%b})" for i in range(1, n + 1)]


# --------------------------------------------------------------------------- #
# Beta and cost of capital
# --------------------------------------------------------------------------- #
def estimate_beta(stock: pd.Series, bench: pd.Series) -> dict:
    """Weekly-return regression beta vs NIFTY 50 (up to 3 years), Blume-adjusted."""
    df = pd.concat([stock, bench], axis=1, keys=["s", "b"]).dropna()
    wk = df.resample("W-FRI").last().pct_change().dropna().iloc[-156:]
    if len(wk) < 30 or wk["b"].var() == 0:
        return {"raw": np.nan, "adjusted": 1.0, "n": len(wk), "note": "insufficient history — beta set to 1.0"}
    raw = float(wk["s"].cov(wk["b"]) / wk["b"].var())
    adj = float(np.clip(0.67 * raw + 0.33, 0.4, 2.0))
    return {"raw": raw, "adjusted": adj, "n": len(wk), "note": f"{len(wk)} weekly observations vs {config.BENCHMARK_NAME}"}


def cost_of_capital(fund: dict, beta: float, mcap_cr: float, p: dict) -> dict:
    """CAPM cost of equity and market-value-weighted WACC."""
    t = fund["table"]
    ke = p["rf"] + beta * p["erp"]
    debt = _f(t["Total Debt"].iloc[-1])
    debt = 0.0 if np.isnan(debt) else max(debt, 0.0)
    if fund["financial"]:
        return {"beta": beta, "ke": ke, "kd_pre": np.nan, "kd_post": np.nan, "we": 1.0, "wd": 0.0,
                "wacc": ke, "debt": debt, "equity": mcap_cr}
    avg_debt = np.nanmean(t["Total Debt"].iloc[-2:].astype(float)) if t["Total Debt"].iloc[-2:].notna().any() else np.nan
    interest = _f(t["Interest"].iloc[-1])
    kd = sdiv(interest, avg_debt)
    if np.isnan(kd) or kd <= 0:
        kd = config.DEFAULT_PRETAX_COST_OF_DEBT
    kd = float(np.clip(kd, p["rf"] + 0.005, 0.14))
    total = mcap_cr + debt
    we = mcap_cr / total if total > 0 else 1.0
    wd = 1 - we
    kd_post = kd * (1 - p["tax"])
    return {"beta": beta, "ke": ke, "kd_pre": kd, "kd_post": kd_post, "we": we, "wd": wd,
            "wacc": we * ke + wd * kd_post, "debt": debt, "equity": mcap_cr}


def growth_path(g0: float, g_terminal: float, n: int = config.PROJECTION_YEARS) -> list[float]:
    """Linear fade: year 1 grows at g0, each later year steps 1/n of the way to terminal growth."""
    return [g0 + (g_terminal - g0) * i / n for i in range(n)]


def starting_growth(fund: dict) -> tuple[float, float]:
    """Historical growth used to seed the projection, capped to 6–22 %.

    Mean of the 3Y revenue CAGR and the 3Y PAT CAGR (book-value CAGR and PAT CAGR for
    banks). The profit CAGR is first bounded to within ±10 pp of the top-line CAGR so
    that a low-base earnings recovery cannot masquerade as sustainable growth."""
    c = fund["cagr"]
    top = c["Equity CAGR"] if fund["financial"] else c["Revenue CAGR"]
    pat = c["PAT CAGR"]
    if pd.isna(top) and pd.isna(pat):
        raw = config.GROWTH_FLOOR
    elif pd.isna(top):
        raw = float(pat)
    elif pd.isna(pat):
        raw = float(top)
    else:
        raw = float((top + np.clip(pat, top - 0.10, top + 0.10)) / 2)
    return raw, float(np.clip(raw, config.GROWTH_FLOOR, config.GROWTH_CAP))


# --------------------------------------------------------------------------- #
# DCF (non-financial) and Excess-Return model (banks)
# --------------------------------------------------------------------------- #
def _base_fcf(fund: dict) -> tuple[float, str]:
    t = fund["table"]
    rev_last = _f(t["Revenue"].iloc[-1])
    margins = sdiv(t["FCF"], t["Revenue"]).iloc[-3:].dropna()
    if len(margins) and margins.mean() > 0 and rev_last > 0:
        return float(margins.mean() * rev_last), f"{len(margins)}Y average FCF margin ({margins.mean() * 100:.1f}%) × latest revenue"
    ebitda = _f(t["EBITDA"].iloc[-1])
    conv = sdiv(t["FCF"], t["EBITDA"])
    conv = conv[conv > 0]
    if len(conv):
        c, src = float(conv.median()), "median positive historical"
    else:
        c, src = config.SECTOR_FCF_CONVERSION.get(fund["sector"], 0.35), "sector default"
    if np.isnan(ebitda) or ebitda <= 0:
        return np.nan, "no positive FCF or EBITDA base"
    return ebitda * c, f"normalised EBITDA × {c * 100:.0f}% FCF conversion ({src}) — historical FCF negative"


def _dcf_equity_per_share(fcfs: list[float], wacc: float, g: float, net_debt: float, shares: float):
    if wacc - g < 0.005:
        return np.nan, np.nan, np.nan, np.nan
    pv = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(fcfs))
    tv = fcfs[-1] * (1 + g) / (wacc - g)
    pv_tv = tv / (1 + wacc) ** len(fcfs)
    ev = pv + pv_tv
    return (ev - net_debt) * CR / shares, ev, tv, pv_tv


def _ri_equity_per_share(bv0: float, roe0: float, g0: float, ke: float, g: float, shares: float, n: int = 5):
    """Excess-return model: value = BV0 + PV(residual income) + PV(terminal residual income)."""
    if ke - g < 0.005:
        return np.nan, None
    roe_t = ke + 0.5 * (roe0 - ke)                    # half of the excess return persists
    rows, bv, pv_sum = [], bv0, 0.0
    gp = growth_path(g0, g, n)
    for i in range(n):
        roe = roe0 + (roe_t - roe0) * (i + 1) / n
        gi = min(gp[i], roe * 0.95)                    # growth is funded by retained earnings
        ni, ri = roe * bv, (roe - ke) * bv
        pv = ri / (1 + ke) ** (i + 1)
        pv_sum += pv
        rows.append({"Opening BV": bv, "ROE %": roe * 100, "PAT": ni, "Residual Income": ri,
                     "Growth %": gi * 100, "Discount Factor": 1 / (1 + ke) ** (i + 1), "PV": pv})
        bv *= 1 + gi
    tv = (roe_t - ke) * bv / (ke - g)
    pv_tv = tv / (1 + ke) ** n
    eq = bv0 + pv_sum + pv_tv
    return eq * CR / shares, {"rows": rows, "pv_sum": pv_sum, "tv": tv, "pv_tv": pv_tv, "equity": eq, "roe_t": roe_t}


def run_dcf(fund: dict, price: float, beta: float, p: dict, g0_override: float | None = None) -> dict:
    """Intrinsic value per share with a 5x5 discount-rate vs terminal-growth sensitivity."""
    t = fund["table"]
    shares = fund["shares"]
    mcap = price * shares / CR
    coc = cost_of_capital(fund, beta, mcap, p)
    raw_g, g0 = starting_growth(fund)
    if g0_override is not None:
        g0 = float(g0_override)
    g = p["g"]
    labels = projected_labels(fund["periods"][-1], config.PROJECTION_YEARS)
    gp = growth_path(g0, g)
    out = {"coc": coc, "g0": g0, "g0_raw": raw_g, "g": g, "g_path": gp, "labels": labels, "price": price,
           "financial": fund["financial"], "rate_label": "Cost of Equity" if fund["financial"] else "WACC"}
    w = coc["wacc"]
    rates = [w + d for d in STEP_BPS]
    growths = [g + d for d in STEP_BPS]

    if fund["financial"]:
        bv0 = _f(t["Equity"].iloc[-1])
        roe0 = float(np.clip(np.nanmean(t["ROE %"].iloc[-3:]) / 100, 0.04, 0.26))
        val, d = _ri_equity_per_share(bv0, roe0, g0, w, g, shares)
        out.update({"method": "Excess-Return Model", "applicable": d is not None and not np.isnan(val)})
        if d is None:
            d = {"rows": [], "pv_sum": np.nan, "tv": np.nan, "pv_tv": np.nan, "equity": np.nan, "roe_t": np.nan}
        proj = pd.DataFrame(d["rows"], index=labels[:len(d["rows"])])
        sens = pd.DataFrame([[_ri_equity_per_share(bv0, roe0, g0, r, gg, shares)[0] for gg in growths] for r in rates],
                            index=rates, columns=growths)
        out.update({
            "intrinsic": val, "projection": proj, "sens": sens, "base_fcf": np.nan,
            "base_method": f"book equity {config.fmt_cr(bv0)}; starting ROE {roe0 * 100:.1f}% fading to {d['roe_t'] * 100:.1f}%",
            "ev": np.nan, "equity_value": d["equity"], "tv": d["tv"], "pv_tv": d["pv_tv"],
            "tv_share": sdiv(d["pv_tv"], d["equity"]), "implied_ev_ebitda": np.nan, "net_debt": np.nan,
            "bridge": [("Current book equity", bv0), ("(+) PV of 5Y residual income", d["pv_sum"]),
                       ("(+) PV of terminal residual income", d["pv_tv"]), ("Equity value", d["equity"]),
                       ("Diluted shares (Cr)", shares / CR)],
        })
    else:
        base, how = _base_fcf(fund)
        nd = _f(t["Net Debt"].iloc[-1])
        nd = 0.0 if np.isnan(nd) else nd
        ok = not np.isnan(base)
        fcfs, f = [], base if ok else np.nan
        for gi in gp:
            f = f * (1 + gi)
            fcfs.append(f)
        val, ev, tv, pv_tv = _dcf_equity_per_share(fcfs, w, g, nd, shares) if ok else (np.nan,) * 4
        dfs = [1 / (1 + w) ** (i + 1) for i in range(len(gp))]
        proj = pd.DataFrame({"Growth %": [x * 100 for x in gp], "FCF": fcfs, "Discount Factor": dfs,
                             "PV": [a * b for a, b in zip(fcfs, dfs)]}, index=labels)
        sens = pd.DataFrame([[(_dcf_equity_per_share(fcfs, r, gg, nd, shares)[0] if ok else np.nan) for gg in growths]
                             for r in rates], index=rates, columns=growths)
        pv_sum = float(proj["PV"].sum()) if ok else np.nan
        out.update({
            "method": "5-Year FCFF DCF", "applicable": ok and not np.isnan(val),
            "intrinsic": val, "projection": proj, "sens": sens, "base_fcf": base, "base_method": how,
            "ev": ev, "equity_value": ev - nd if ok else np.nan, "tv": tv, "pv_tv": pv_tv,
            "tv_share": sdiv(pv_tv, ev), "implied_ev_ebitda": sdiv(ev, _f(t["EBITDA"].iloc[-1])), "net_debt": nd,
            "bridge": [("PV of 5Y explicit FCF", pv_sum), ("(+) PV of terminal value", pv_tv),
                       ("Enterprise value", ev), ("(−) Net debt", nd),
                       ("Equity value", ev - nd if ok else np.nan), ("Diluted shares (Cr)", shares / CR)],
        })
    out["upside"] = sdiv(out["intrinsic"], price) - 1 if not pd.isna(out["intrinsic"]) else np.nan
    return out


# --------------------------------------------------------------------------- #
# Two-year financial projections (for the report's summary table)
# --------------------------------------------------------------------------- #
def project_financials(fund: dict, dcf: dict, n: int = 2) -> pd.DataFrame:
    """Deterministic FY+1/FY+2 projections consistent with the valuation growth path."""
    t = fund["table"]
    last = t.iloc[-1]
    labels = dcf["labels"][:n]
    shares_cr = fund["shares"] / CR
    payout = sdiv(last["Dividends"], last["PAT"])
    payout = 0.30 if np.isnan(payout) else float(np.clip(payout, 0.0, 1.0))
    rows, rev, eq, nd = [], _f(last["Revenue"]), _f(last["Equity"]), _f(last["Net Debt"])
    for i, lab in enumerate(labels):
        gi = dcf["g_path"][i]
        rev_n = rev * (1 + gi)
        r = {"Revenue": rev_n, "Revenue Growth %": gi * 100}
        if fund["financial"]:
            prow = dcf["projection"].iloc[i] if len(dcf["projection"]) > i else None
            pat = _f(prow["PAT"]) if prow is not None else _f(last["PAT"]) * (1 + gi)
            r.update({"EBITDA": np.nan, "EBITDA Margin %": np.nan, "FCF": np.nan, "Net Debt/EBITDA": np.nan})
            eq_n = eq + pat * (1 - payout)
        else:
            margin = np.nanmean(t["EBITDA Margin %"].iloc[-2:]) / 100
            ebitda = rev_n * margin
            da = rev_n * sdiv(last["D&A"], last["Revenue"]) if not np.isnan(sdiv(last["D&A"], last["Revenue"])) else 0.0
            below = _f(last["PBT"]) - _f(last["EBIT"])          # net other income less interest, held flat
            pbt = ebitda - da + (0.0 if np.isnan(below) else below)
            pat = pbt * (1 - _f(last["Effective Tax %"]) / 100)
            fcf = _f(dcf["projection"]["FCF"].iloc[i]) if dcf["applicable"] else np.nan
            eq_n = eq + pat * (1 - payout)
            nd = nd - (0.0 if np.isnan(fcf) else fcf) + max(pat, 0) * payout
            r.update({"EBITDA": ebitda, "EBITDA Margin %": margin * 100, "FCF": fcf, "Net Debt": nd,
                      "Net Debt/EBITDA": sdiv(nd, ebitda)})
        r.update({"PAT": pat, "EPS": sdiv(pat, shares_cr), "Equity": eq_n, "BVPS": sdiv(eq_n, shares_cr),
                  "ROE %": sdiv(pat, (eq + eq_n) / 2) * 100, "PAT Margin %": sdiv(pat, rev_n) * 100})
        rows.append(r)
        rev, eq = rev_n, eq_n
    return pd.DataFrame(rows, index=labels)


# --------------------------------------------------------------------------- #
# NSE peer relative valuation
# --------------------------------------------------------------------------- #
PEER_COLS = ["MCap (₹ Cr)", "P/E", "Fwd P/E", "EV/EBITDA", "P/B", "RoE %", "3Y PAT CAGR %"]


def multiples_row(fund: dict, price: float, info: dict | None = None) -> dict:
    """Multiples computed from the parsed statements and the latest price (consistent across peers)."""
    t = fund["table"].iloc[-1]
    info = info or {}
    mcap = price * fund["shares"] / CR
    eps, bvps = _f(t["EPS"]), _f(t["BVPS"])
    pe = sdiv(price, eps) if eps > 0 else np.nan
    fpe = _f(info.get("forwardPE"))
    if np.isnan(fpe) or fpe <= 0:
        g = fund["cagr"]["PAT CAGR"]
        g = 0.10 if pd.isna(g) else float(np.clip(g, 0.05, 0.18))
        fpe = sdiv(price, eps * (1 + g)) if eps > 0 else np.nan
    ev_ebitda = np.nan
    if not fund["financial"] and _f(t["EBITDA"]) > 0:
        ev_ebitda = sdiv(mcap + (0.0 if np.isnan(_f(t["Net Debt"])) else _f(t["Net Debt"])), _f(t["EBITDA"]))
    return {"MCap (₹ Cr)": mcap, "P/E": pe, "Fwd P/E": fpe, "EV/EBITDA": ev_ebitda,
            "P/B": sdiv(price, bvps) if bvps > 0 else np.nan, "RoE %": _f(t["ROE %"]),
            "3Y PAT CAGR %": fund["cagr"]["PAT CAGR"] * 100 if not pd.isna(fund["cagr"]["PAT CAGR"]) else np.nan}


def peer_valuation(target: str, funds: dict[str, dict], prices: dict[str, float], infos: dict[str, dict] | None = None) -> dict:
    """Peer table (target first, then peers, then the sector-median row) and implied relative price."""
    infos = infos or {}
    rows = {}
    for tk, f in funds.items():
        if tk in prices and not pd.isna(prices[tk]):
            rows[tk] = {"Company": config.name_of(tk), **multiples_row(f, prices[tk], infos.get(tk))}
    order = [target] + [k for k in rows if k != target]
    df = pd.DataFrame([rows[k] for k in order if k in rows], index=[k for k in order if k in rows])
    peers = df.drop(index=target, errors="ignore")
    med = peers[PEER_COLS].median(skipna=True) if len(peers) else pd.Series(np.nan, index=PEER_COLS)
    table = pd.concat([df, pd.DataFrame([{"Company": "Sector Median (peers)", **med.to_dict()}], index=["MEDIAN"])])

    f = funds[target]
    t = f["table"].iloc[-1]
    shares_cr = f["shares"] / CR
    implied = {}
    if _f(t["EPS"]) > 0 and not pd.isna(med["P/E"]):
        implied["P/E"] = med["P/E"] * _f(t["EPS"])
    if f["financial"]:
        if _f(t["BVPS"]) > 0 and not pd.isna(med["P/B"]):
            implied["P/B"] = med["P/B"] * _f(t["BVPS"])
    elif _f(t["EBITDA"]) > 0 and not pd.isna(med["EV/EBITDA"]):
        nd = 0.0 if np.isnan(_f(t["Net Debt"])) else _f(t["Net Debt"])
        v = (med["EV/EBITDA"] * _f(t["EBITDA"]) - nd) / shares_cr
        if v > 0:
            implied["EV/EBITDA"] = v
    price = prices.get(target, np.nan)
    ip = float(np.mean(list(implied.values()))) if implied else np.nan
    return {"table": table, "median": med, "implied": implied, "implied_price": ip,
            "upside": sdiv(ip, price) - 1 if not np.isnan(ip) else np.nan, "n_peers": len(peers)}


def target_and_rating(dcf_price: float, peer_price: float, price: float, dcf_weight: float) -> dict:
    """Blend intrinsic and relative value into a target price and map upside to a rating."""
    a, b = _f(dcf_price), _f(peer_price)
    if np.isnan(a) and np.isnan(b):
        return {"target": np.nan, "upside": np.nan, "rating": "NOT RATED", "basis": "no valuation available"}
    if np.isnan(b) or dcf_weight >= 1:
        tp, basis = a, "100% intrinsic value"
    elif np.isnan(a) or dcf_weight <= 0:
        tp, basis = b, "100% peer-relative value"
    else:
        tp = dcf_weight * a + (1 - dcf_weight) * b
        basis = f"{dcf_weight * 100:.0f}% intrinsic value + {(1 - dcf_weight) * 100:.0f}% peer-relative value"
    up = tp / price - 1
    rating = "BUY" if up > config.RATING_BUY else ("SELL" if up < config.RATING_SELL else "NEUTRAL")
    return {"target": tp, "upside": up, "rating": rating, "basis": basis}


# --------------------------------------------------------------------------- #
# INR currency macro
# --------------------------------------------------------------------------- #
def fx_analysis(stock: pd.Series, bench: pd.Series, fx: pd.DataFrame, sector: str) -> dict:
    """1Y INR moves, 90-day rolling correlation and FX beta of the stock vs USD/INR.
    A rising USD/INR is INR depreciation. The NIFTY-neutral beta controls for the
    market, because INR weakness usually coincides with broad risk-off selling."""
    usd = fx[config.FX_PRIMARY].dropna()
    out = {"pairs": {}, "sector_prior": config.FX_EXPOSURE.get(sector, ("mixed", ""))}
    for c in fx.columns:
        s = fx[c].dropna()
        if len(s) < 30:
            continue
        ref = s[s.index <= s.index[-1] - pd.Timedelta(days=365)]
        base = ref.iloc[-1] if len(ref) else s.iloc[0]
        out["pairs"][c] = {"last": float(s.iloc[-1]), "chg_1y": float(s.iloc[-1] / base - 1),
                           "high": float(s.iloc[-252:].max()), "low": float(s.iloc[-252:].min())}
    df = pd.concat([np.log(stock).diff(), np.log(bench).diff(), np.log(usd).diff()], axis=1,
                   keys=["s", "m", "fx"]).dropna()
    df = df[df.index >= df.index[-1] - pd.Timedelta(days=730)] if len(df) else df
    out["n"] = len(df)
    if len(df) < 60 or df["fx"].var() == 0:
        out.update({"fx_beta": np.nan, "fx_beta_neutral": np.nan, "t_stat": np.nan, "p_value": np.nan, "corr_90d": np.nan,
                    "corr_series": pd.Series(dtype=float), "corr_full": np.nan, "usd_inr": usd})
        return out
    out["fx_beta"] = float(stats.linregress(df["fx"].values, df["s"].values).slope)
    X = np.column_stack([np.ones(len(df)), df["m"].values, df["fx"].values])
    y = df["s"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - 3, 1)
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid / dof)
    se = float(np.sqrt(max(cov[2, 2], 1e-18)))
    corr = df["s"].rolling(90).corr(df["fx"]).dropna()
    out.update({"fx_beta_neutral": float(coef[2]), "t_stat": float(coef[2] / se), "market_beta_daily": float(coef[1]),
                "p_value": float(2 * stats.t.sf(abs(coef[2] / se), dof)),
                "corr_series": corr, "corr_90d": float(corr.iloc[-1]) if len(corr) else np.nan,
                "corr_full": float(df["s"].corr(df["fx"])), "usd_inr": usd})
    return out


# --------------------------------------------------------------------------- #
# Research pipeline orchestrator (shared by the dashboard and the PDF builder)
# --------------------------------------------------------------------------- #
def build_research_context(ticker: str, live: bool, params: dict | None = None, peers: list[str] | None = None,
                           g0_override: float | None = None) -> dict:
    """Load data, run fundamentals, valuation, peers and FX for one stock.

    If live mode was requested and any component had to fall back, the whole context is
    rebuilt from the offline pack so that a single analysis never mixes data sources."""
    import data_loader as dl
    import fundamentals as fa

    p = {**default_params(), **(params or {})}
    ticker = config.resolve_ticker(ticker)
    peers = [q for q in (peers if peers is not None else config.peers_of(ticker)) if q != ticker]

    def load(use_live: bool):
        b, s1 = dl.load_company(ticker, use_live)
        px, s2 = dl.load_prices(tuple([ticker, config.BENCHMARK] + peers), 6.2, use_live)
        fx, s3 = dl.load_fx(use_live, 2.0)
        pb, s4 = dl.load_companies(peers, use_live) if peers else ({}, s1)
        return b, px, fx, pb, dl.combine_status(s1, s2, s3, s4)

    bundle, px, fx, peer_bundles, status = load(live)
    if live and status == dl.FALLBACK:
        bundle, px, fx, peer_bundles, _ = load(False)
    if ticker not in px.columns or config.BENCHMARK not in px.columns:
        raise ValueError(f"no price history for {ticker}")

    stock, bench = px[ticker].dropna(), px[config.BENCHMARK].dropna()
    price = float(stock.iloc[-1])
    sector = config.sector_of(ticker)
    fund = fa.analyze(bundle, sector, price)
    beta = estimate_beta(stock, bench)
    if p.get("beta_override"):
        beta = {**beta, "adjusted": float(p["beta_override"]), "note": "user override"}
    dcf = run_dcf(fund, price, beta["adjusted"], p, g0_override)

    funds, prices, infos = {ticker: fund}, {ticker: price}, {ticker: bundle.get("info", {})}
    for q, qb in peer_bundles.items():
        if q in px.columns and px[q].notna().any():
            try:
                qp = float(px[q].dropna().iloc[-1])
                funds[q], prices[q], infos[q] = fa.analyze(qb, config.sector_of(q), qp), qp, qb.get("info", {})
            except Exception:  # noqa: BLE001 - a broken peer must not break the target analysis
                continue
    peer_val = peer_valuation(ticker, funds, prices, infos)
    rating = target_and_rating(dcf["intrinsic"], peer_val["implied_price"], price, p["dcf_weight"])
    fxa = fx_analysis(stock, bench, fx, sector)
    proj = project_financials(fund, dcf)

    info = bundle.get("info", {}) or {}
    yr = stock[stock.index >= stock.index[-1] - pd.Timedelta(days=365)]
    mcap = price * fund["shares"] / CR
    last = fund["table"].iloc[-1]
    div_ps = _f(info.get("dividendRate"))
    if np.isnan(div_ps) or div_ps <= 0:
        div_ps = sdiv(_f(last["Dividends"]) * CR, fund["shares"])
    ff = sdiv(_f(info.get("floatShares")), _f(info.get("sharesOutstanding")))
    if np.isnan(ff) and not np.isnan(_f(info.get("heldPercentInsiders"))):
        ff = 1 - _f(info.get("heldPercentInsiders"))
    market = {
        "mcap": mcap, "high_52w": float(yr.max()), "low_52w": float(yr.min()),
        "shares_cr": fund["shares"] / CR, "free_float": float(np.clip(ff, 0, 1)) if not np.isnan(ff) else np.nan,
        "div_yield": sdiv(div_ps, price), "pe": sdiv(price, _f(last["EPS"])) if _f(last["EPS"]) > 0 else np.nan,
        "pb": sdiv(price, _f(last["BVPS"])), "roe": _f(last["ROE %"]),
        "ret_1y": float(stock.iloc[-1] / yr.iloc[0] - 1),
        "bench_ret_1y": float(bench.iloc[-1] / bench[bench.index >= yr.index[0]].iloc[0] - 1),
    }
    return {
        "ticker": ticker, "name": info.get("longName") or config.name_of(ticker), "sector": sector,
        "status": status, "as_of": stock.index[-1], "price": price, "params": p, "peers_list": peers,
        "bundle": bundle, "info": info, "prices": px, "fx_prices": fx, "fund": fund, "beta": beta, "dcf": dcf,
        "peers": peer_val, "rating": rating, "fx": fxa, "projection": proj, "market": market,
    }
