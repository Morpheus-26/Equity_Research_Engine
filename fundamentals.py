"""
fundamentals.py — statement parser (₹ Cr), growth/margins, ROIC, 5-step DuPont,
Piotroski F-Score and Altman Z-Score.

Input is a company bundle from ``data_loader`` (yfinance layout, raw INR). Every
line item is looked up defensively and every ratio is zero-division guarded, so a
missing row degrades to NaN ("n.a.") instead of raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

CR = config.CRORE


# --------------------------------------------------------------------------- #
# Safe helpers
# --------------------------------------------------------------------------- #
def sdiv(a, b):
    """Zero/NaN-guarded division for scalars or aligned Series."""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        a = pd.Series(a, index=b.index) if not isinstance(a, pd.Series) else a
        b = pd.Series(b, index=a.index) if not isinstance(b, pd.Series) else b
        return (a / b.where(b.abs() > 1e-12)).astype(float)
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or abs(b) < 1e-12:
            return np.nan
        return float(a) / float(b)
    except (TypeError, ValueError):
        return np.nan


def cagr(first, last, years):
    if years is None or years <= 0 or pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0:
        return np.nan
    return (last / first) ** (1 / years) - 1


def fy_label(ts: pd.Timestamp) -> str:
    if ts.month == 3:
        return f"FY{ts:%y}"
    if ts.month == 12:
        return f"CY{ts:%y}"
    return f"FY{ts:%y}({ts:%b})"


def _row(df: pd.DataFrame, names: list[str], cols) -> pd.Series:
    """First available line item among ``names`` re-indexed to ``cols`` (NaN if absent)."""
    if df is not None and not df.empty:
        for n in names:
            if n in df.index:
                s = pd.to_numeric(df.loc[n], errors="coerce")
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[0]
                if s.notna().any():
                    return s.reindex(cols).astype(float)
    return pd.Series(np.nan, index=cols, dtype=float)


def _align(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Align a statement to the income-statement period-ends (same fiscal year match)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = df[c]
        else:
            near = [d for d in df.columns if abs((d - c).days) <= 45]
            out[c] = df[near[0]] if near else pd.Series(np.nan, index=df.index)
    return pd.DataFrame(out)


def _avg(s: pd.Series) -> pd.Series:
    """Average of opening and closing balance; the first year uses its closing balance."""
    return ((s + s.shift(1)) / 2).fillna(s)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def analyze(bundle: dict, sector: str, price: float | None = None) -> dict:
    """Parse a bundle into an institutional fundamentals pack (all money in ₹ Cr)."""
    financial = sector in config.FINANCIAL_SECTORS
    info = bundle.get("info", {}) or {}
    inc_raw = bundle.get("income", pd.DataFrame())
    if inc_raw is None or inc_raw.empty:
        raise ValueError("no income statement available")
    rev_all = _row(inc_raw, ["Total Revenue", "Operating Revenue"], inc_raw.columns)
    cols = list(rev_all.dropna().index[-5:])
    if len(cols) < 2:
        raise ValueError("need at least two fiscal years of statements")
    inc = inc_raw[cols]
    bal = _align(bundle.get("balance"), cols)
    cfs = _align(bundle.get("cashflow"), cols)
    g = lambda df, names: _row(df, names, cols)  # noqa: E731

    revenue = g(inc, ["Total Revenue", "Operating Revenue"])
    pbt = g(inc, ["Pretax Income"])
    tax = g(inc, ["Tax Provision"])
    pat = g(inc, ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"])
    interest = g(inc, ["Interest Expense", "Interest Expense Non Operating"]).abs()
    da = g(inc, ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"])
    if da.isna().all():
        da = g(cfs, ["Depreciation And Amortization", "Depreciation Amortization Depletion"])
    gross = g(inc, ["Gross Profit"])
    if financial:
        # Banks have no meaningful EBITDA; pre-tax profit is the operating-profit proxy.
        ebit = pbt.copy()
        ebitda = pd.Series(np.nan, index=cols, dtype=float)
    else:
        ebit = g(inc, ["EBIT", "Operating Income"])
        ebitda = g(inc, ["EBITDA", "Normalized EBITDA"])
        ebitda = ebitda.fillna(ebit + da)
        ebit = ebit.fillna(ebitda - da).fillna(pbt + interest.fillna(0))
    cfo = g(cfs, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex = g(cfs, ["Capital Expenditure", "Purchase Of PPE", "Net PPE Purchase And Sale"]).abs()
    fcf = (cfo - capex.fillna(0)).fillna(g(cfs, ["Free Cash Flow"]))
    dividends = g(cfs, ["Cash Dividends Paid", "Common Stock Dividend Paid"]).abs()

    ta = g(bal, ["Total Assets"])
    equity = g(bal, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    debt = g(bal, ["Total Debt"])
    if debt.isna().all():
        debt = g(bal, ["Long Term Debt"]).fillna(0) + g(bal, ["Current Debt"]).fillna(0)
    ltd = g(bal, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]).fillna(debt)
    cash = g(bal, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
    ca = g(bal, ["Current Assets"])
    cl = g(bal, ["Current Liabilities"])
    re = g(bal, ["Retained Earnings"])
    tl = g(bal, ["Total Liabilities Net Minority Interest"]).fillna(ta - equity)
    shares = g(inc, ["Diluted Average Shares", "Basic Average Shares"]).fillna(g(bal, ["Ordinary Shares Number", "Share Issued"]))
    if shares.isna().all() and info.get("sharesOutstanding"):
        shares = pd.Series(float(info["sharesOutstanding"]), index=cols)
    shares = shares.ffill().bfill()

    net_debt = pd.Series(np.nan, index=cols, dtype=float) if financial else debt.fillna(0) - cash.fillna(0)
    ic = equity.copy() if financial else equity + debt.fillna(0) - cash.fillna(0)
    eff_tax = sdiv(tax, pbt).clip(0, 0.40).where(pbt > 0).fillna(config.TAX_RATE)
    nopat = ebit * (1 - eff_tax)
    avg_ic = _avg(ic)
    roic = sdiv(nopat, avg_ic.where(avg_ic > 0))
    if financial:
        roic[:] = np.nan
    avg_eq, avg_ta = _avg(equity), _avg(ta)

    labels = [fy_label(c) for c in cols]
    t = pd.DataFrame(index=cols)
    for name, s in [("Revenue", revenue), ("EBITDA", ebitda), ("EBIT", ebit), ("PBT", pbt), ("PAT", pat),
                    ("CFO", cfo), ("CapEx", capex), ("FCF", fcf), ("Total Debt", debt), ("Cash", cash),
                    ("Net Debt", net_debt), ("Equity", equity), ("Total Assets", ta), ("Invested Capital", ic),
                    ("Dividends", dividends), ("Interest", interest), ("Tax", tax), ("D&A", da),
                    ("Gross Profit", gross), ("Current Assets", ca), ("Current Liabilities", cl),
                    ("Retained Earnings", re), ("Total Liabilities", tl), ("LT Debt", ltd)]:
        t[name] = s / CR
    t["Shares (Cr)"] = shares / CR
    t["EPS"] = sdiv(pat, shares)
    t["BVPS"] = sdiv(equity, shares)
    t["Revenue Growth %"] = revenue.pct_change() * 100
    t["EBITDA Margin %"] = sdiv(ebitda, revenue) * 100
    t["EBIT Margin %"] = sdiv(ebit, revenue) * 100
    t["PAT Margin %"] = sdiv(pat, revenue) * 100
    t["Gross Margin %"] = sdiv(gross, revenue) * 100
    t["FCF Conversion %"] = sdiv(fcf, pat.where(pat > 0)) * 100
    t["ROE %"] = sdiv(pat, avg_eq.where(avg_eq > 0)) * 100
    t["ROA %"] = sdiv(pat, avg_ta) * 100
    t["ROIC %"] = roic * 100
    t["Net Debt/EBITDA"] = sdiv(net_debt, ebitda.where(ebitda > 0))
    t["Debt/Equity"] = sdiv(debt, equity.where(equity > 0))
    t["Current Ratio"] = sdiv(ca, cl)
    t["Asset Turnover"] = sdiv(revenue, avg_ta)
    t["Effective Tax %"] = eff_tax * 100
    t.index = labels

    n = len(cols)
    span = min(3, n - 1)
    cg = {f"{k} CAGR": cagr(t[k].iloc[-1 - span], t[k].iloc[-1], span) for k in ("Revenue", "EBITDA", "PAT")}
    cg["Equity CAGR"] = cagr(t["Equity"].iloc[-1 - span], t["Equity"].iloc[-1], span)
    cg["span"] = span

    # 5-step DuPont
    dp = pd.DataFrame(index=labels)
    dp["Tax Burden (PAT/PBT)"] = sdiv(pat, pbt).values
    dp["Interest Burden (PBT/EBIT)"] = sdiv(pbt, ebit).values
    dp["EBIT Margin (EBIT/Revenue)"] = sdiv(ebit, revenue).values
    dp["Asset Turnover (Revenue/Avg Assets)"] = sdiv(revenue, avg_ta).values
    dp["Financial Leverage (Avg Assets/Avg Equity)"] = sdiv(avg_ta, avg_eq.where(avg_eq > 0)).values
    dp["ROE (product)"] = dp.iloc[:, :5].prod(axis=1, skipna=False)

    fscores = [None] + [piotroski(t, i, financial) for i in range(1, n)]
    last_price = price if price else info.get("currentPrice") or info.get("regularMarketPrice")
    sh_last = float(shares.iloc[-1]) if pd.notna(shares.iloc[-1]) else np.nan
    mcap_cr = (last_price * sh_last / CR) if last_price and pd.notna(sh_last) else sdiv(info.get("marketCap"), CR)

    return {
        "ticker": bundle.get("ticker"), "sector": sector, "financial": financial,
        "periods": cols, "years": labels, "table": t, "cagr": cg, "dupont": dp,
        "piotroski": fscores[-1], "piotroski_history": fscores,
        "altman": altman_z(t, mcap_cr, financial),
        "shares": sh_last, "market_cap_cr": mcap_cr,
    }


# --------------------------------------------------------------------------- #
# Piotroski F-Score
# --------------------------------------------------------------------------- #
def piotroski(t: pd.DataFrame, i: int, financial: bool = False) -> dict:
    """Nine-point F-Score for fiscal-year row ``i`` versus row ``i-1``.
    A criterion whose inputs are missing scores 0 and is reported as "n.a."."""
    cur, prv = t.iloc[i], t.iloc[i - 1]

    def test(name, group, value, cond, fmt):
        ok = bool(cond) if not (isinstance(value, float) and np.isnan(value)) else False
        na = isinstance(value, float) and np.isnan(value)
        return {"criterion": name, "group": group, "passed": ok, "available": not na,
                "detail": "n.a. (line item not reported)" if na else fmt}

    roa, roa_p = sdiv(cur["PAT"], cur["Total Assets"]), sdiv(prv["PAT"], prv["Total Assets"])
    cfo_ta = sdiv(cur["CFO"], cur["Total Assets"])
    lev, lev_p = sdiv(cur["LT Debt"], cur["Total Assets"]), sdiv(prv["LT Debt"], prv["Total Assets"])
    cr_, cr_p = cur["Current Ratio"], prv["Current Ratio"]
    gm, gm_p = cur["Gross Margin %"], prv["Gross Margin %"]
    at, at_p = sdiv(cur["Revenue"], cur["Total Assets"]), sdiv(prv["Revenue"], prv["Total Assets"])
    sh, sh_p = cur["Shares (Cr)"], prv["Shares (Cr)"]
    d = lambda a, b: a - b if not (pd.isna(a) or pd.isna(b)) else np.nan  # noqa: E731
    c = [
        test("ROA > 0", "Profitability", roa, roa > 0, f"ROA {roa * 100:.2f}%"),
        test("Operating cash flow > 0", "Profitability", cur["CFO"], cur["CFO"] > 0, f"CFO ₹{cur['CFO']:,.0f} Cr"),
        test("ROA improved YoY", "Profitability", d(roa, roa_p), d(roa, roa_p) > 0,
             f"{roa_p * 100:.2f}% → {roa * 100:.2f}%"),
        test("Accruals: CFO/Assets > ROA", "Profitability", d(cfo_ta, roa), d(cfo_ta, roa) > 0,
             f"CFO/TA {cfo_ta * 100:.2f}% vs ROA {roa * 100:.2f}%"),
        test("Long-term leverage lower YoY", "Leverage & Liquidity", d(lev, lev_p), d(lev, lev_p) <= 0,
             f"LTD/TA {lev_p * 100:.2f}% → {lev * 100:.2f}%"),
        test("Current ratio improved YoY", "Leverage & Liquidity", d(cr_, cr_p), d(cr_, cr_p) > 0,
             f"{cr_p:.2f}x → {cr_:.2f}x"),
        test("No equity dilution", "Leverage & Liquidity", d(sh, sh_p), sh <= sh_p * 1.005,
             f"{sh_p:,.1f} Cr → {sh:,.1f} Cr shares"),
        test("Gross margin improved YoY", "Operating Efficiency", d(gm, gm_p), d(gm, gm_p) > 0,
             f"{gm_p:.2f}% → {gm:.2f}%"),
        test("Asset turnover improved YoY", "Operating Efficiency", d(at, at_p), d(at, at_p) > 0,
             f"{at_p:.3f}x → {at:.3f}x"),
    ]
    score = sum(x["passed"] for x in c)
    evaluated = sum(x["available"] for x in c)
    note = ("Current-ratio and gross-margin tests are not defined for banks; the score is out of "
            f"{evaluated} evaluable criteria.") if financial and evaluated < 9 else ""
    return {"score": score, "evaluated": evaluated, "criteria": c, "year": t.index[i], "note": note}


# --------------------------------------------------------------------------- #
# Altman Z-Score
# --------------------------------------------------------------------------- #
def altman_z(t: pd.DataFrame, market_cap_cr: float, financial: bool) -> dict:
    """Classic Altman (1968) Z for listed non-financial companies plus the Z''
    emerging-market variant. Flagged not applicable for banks/NBFCs."""
    r = t.iloc[-1]
    ta, tl = r["Total Assets"], r["Total Liabilities"]
    x1 = sdiv(r["Current Assets"] - r["Current Liabilities"], ta)
    x2 = sdiv(r["Retained Earnings"], ta)
    x3 = sdiv(r["EBIT"], ta)
    x4 = sdiv(market_cap_cr, tl)
    x4b = sdiv(r["Equity"], tl)
    x5 = sdiv(r["Revenue"], ta)
    comps = {"X1 Working Capital / Total Assets": x1, "X2 Retained Earnings / Total Assets": x2,
             "X3 EBIT / Total Assets": x3, "X4 Market Cap / Total Liabilities": x4,
             "X5 Revenue / Total Assets": x5}
    if financial:
        return {"applicable": False, "z": np.nan, "z_em": np.nan, "zone": "Not applicable", "components": comps,
                "note": ("Altman Z is calibrated on industrial companies. For banks and NBFCs, deposits and "
                         "borrowings are operating liabilities and working capital is undefined, so the score "
                         "is not meaningful — use capital adequacy (CET-1), GNPA/NNPA and RoA instead.")}
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    z_em = 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4b
    if pd.isna(z):
        zone = "n.a."
    else:
        zone = "Safe" if z > 2.99 else ("Grey" if z >= 1.81 else "Distress")
    return {"applicable": True, "z": z, "z_em": z_em, "zone": zone, "components": comps,
            "note": ("Applicable: non-financial listed company. Z > 2.99 safe, 1.81–2.99 grey zone, "
                     "< 1.81 distress. Z''-EM (book-value, sector-neutral) > 5.85 safe, < 4.15 distress.")}
