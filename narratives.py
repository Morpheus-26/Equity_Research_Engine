"""
narratives.py — institutional analyst prose.

Default engine: a deterministic Python financial NLG engine (zero API keys). Every
sentence is assembled from computed metrics, so the text can never contradict the
tables. Optional: a user-supplied free Google Gemini or Groq key rewrites the thesis
block for fluency; numbers are passed as a fact sheet and any failure silently falls
back to the deterministic text.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests

import config
from config import fmt_cr, fmt_inr, fmt_num, fmt_pct, fmt_x

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
ENGINE_DETERMINISTIC = "Deterministic NLG engine"

SECTOR_TEXT = {
    "IT Services": {
        "desc": "a tier-1 Indian IT services exporter with a diversified BFSI, retail and manufacturing client base",
        "bull": "deal-win conversion accelerates as GenAI-led modernisation budgets scale, lifting utilisation and pricing",
        "bear": "discretionary tech spend stays muted and AI-driven productivity is passed through to clients as price deflation",
        "catalysts": ["Large-deal TCV announcements and book-to-bill above 1.2x", "Recovery in BFSI and hi-tech discretionary spend",
                      "Margin levers: pyramid rationalisation, utilisation and lower sub-contracting"],
        "risks": ["Client budget cuts in the US and Europe", "Pricing deflation from GenAI productivity gains",
                  "Wage inflation and attrition", "US visa / outsourcing regulation"],
    },
    "Private Banks": {
        "desc": "a leading Indian private-sector bank with a granular retail deposit franchise",
        "bull": "deposit repricing lags loan yields in an easing cycle, expanding NIM while credit costs stay benign",
        "bear": "deposit competition compresses NIM and unsecured retail slippages push credit costs above guidance",
        "catalysts": ["CASA accretion and loan growth ahead of system", "NIM stabilisation as term-deposit repricing completes",
                      "Credit cost normalisation in unsecured retail and MFI"],
        "risks": ["Net interest margin compression", "Asset-quality slippage in unsecured lending",
                  "Regulatory changes on liquidity, provisioning or priority-sector norms", "Slower deposit mobilisation"],
    },
    "Oil & Gas / Conglomerate": {
        "desc": "a large-cap Indian energy major with integrated hydrocarbon operations",
        "bull": "refining and marketing margins hold above mid-cycle while new-energy and consumer capex begins to earn returns",
        "bear": "crude volatility, under-recoveries on regulated fuels and elevated capex suppress free cash flow",
        "catalysts": ["Gross refining margin and marketing margin trajectory", "Government pricing and subsidy policy on regulated fuels",
                      "Commissioning of growth capex and deleveraging"],
        "risks": ["Crude oil and LNG price volatility", "Policy intervention in fuel pricing",
                  "Capex intensity and project execution delays", "INR depreciation raising the import bill"],
    },
    "Pharmaceuticals": {
        "desc": "a leading Indian pharmaceutical company with branded domestic formulations and a US generics pipeline",
        "bull": "complex generics and specialty launches outrun US price erosion while the India branded business compounds",
        "bear": "US FDA compliance actions delay launches and base-business price erosion accelerates",
        "catalysts": ["US FDA approvals and limited-competition launches", "India branded formulation growth above the IPM",
                      "Specialty / biosimilar portfolio scale-up"],
        "risks": ["US FDA inspection outcomes", "US generics price erosion and tariff policy",
                  "NLEM / DPCO price controls in India", "R&D pipeline execution"],
    },
    "Automobiles": {
        "desc": "a leading Indian automobile manufacturer",
        "bull": "premiumisation and new-model cycles lift realisations while operating leverage expands margins",
        "bear": "demand moderates, discounting rises and the EV transition forces capex ahead of returns",
        "catalysts": ["New model launches and order-book conversion", "Rural demand recovery and festive-season volumes",
                      "Commodity cost tailwinds and operating leverage"],
        "risks": ["Cyclical demand slowdown", "Commodity and component cost inflation",
                  "EV transition and regulatory (CAFE / emission) costs", "Competitive intensity and discounting"],
    },
    "FMCG": {
        "desc": "a leading Indian consumer staples franchise with deep distribution reach",
        "bull": "rural volume recovery and premiumisation restore operating leverage as input costs ease",
        "bear": "urban demand stays soft, regional and D2C brands take share and input inflation squeezes gross margin",
        "catalysts": ["Volume growth recovery in rural and urban channels", "Gross margin expansion from commodity deflation",
                      "Premiumisation and quick-commerce channel mix"],
        "risks": ["Weak volume growth and down-trading", "Agri-commodity and palm oil inflation",
                  "Competition from regional and D2C brands", "Taxation or regulatory changes on key categories"],
    },
    "Capital Goods": {
        "desc": "a leading Indian capital goods and engineering company leveraged to the domestic capex cycle",
        "bull": "order inflows from power, defence and infrastructure sustain a multi-year execution runway with operating leverage",
        "bear": "ordering slows, execution slips and working capital stretches as public capex normalises",
        "catalysts": ["Order inflow momentum and book-to-bill", "Government capex in power T&D, defence and railways",
                      "Margin expansion from execution mix and localisation"],
        "risks": ["Slowdown in public and private capex", "Working-capital intensity and receivable delays",
                  "Commodity cost inflation on fixed-price contracts", "Execution and project-timing risk"],
    },
    "Telecom & Digital Infra": {
        "desc": "a leading Indian telecom and digital-infrastructure operator",
        "bull": "tariff repair lifts ARPU, capex intensity falls post-5G rollout and free cash flow deleverages the balance sheet",
        "bear": "tariff hikes stall, competitive intensity returns and spectrum / AGR liabilities absorb cash flow",
        "catalysts": ["Industry tariff hikes and ARPU expansion", "Declining capex intensity after 5G rollout",
                      "Deleveraging and rising shareholder payouts"],
        "risks": ["Regulatory and AGR / spectrum liabilities", "Tariff discipline breaking down",
                  "High leverage and refinancing costs", "Capex intensity for new technology cycles"],
    },
}


def _v(x) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _ok(x) -> bool:
    return not np.isnan(_v(x))


# --------------------------------------------------------------------------- #
# Deterministic engine
# --------------------------------------------------------------------------- #
def build_narrative(ctx: dict) -> dict:
    """Full analyst narrative assembled from the research context."""
    f, d, pv, tr, fxa = ctx["fund"], ctx["dcf"], ctx["peers"], ctx["rating"], ctx["fx"]
    t = f["table"]
    last, first = t.iloc[-1], t.iloc[0]
    y0, y1 = f["years"][max(len(t) - 1 - f["cagr"]["span"], 0)], f["years"][-1]
    name, tk, price = ctx["name"], ctx["ticker"], ctx["price"]
    sec = SECTOR_TEXT.get(ctx["sector"], SECTOR_TEXT["Capital Goods"])
    fin = f["financial"]
    rev_c, pat_c = _v(f["cagr"]["Revenue CAGR"]) * 100, _v(f["cagr"]["PAT CAGR"]) * 100
    coc = d["coc"]
    rate_name = d["rate_label"]

    # ---- Paragraph 1: franchise and growth
    growth_tag = ("top-quartile growth" if rev_c > 15 else "steady compounding" if rev_c >= 8
                  else "a mature, low-growth profile") if _ok(rev_c) else "a volatile growth record"
    p1 = (f"{name} ({tk}) is {sec['desc']}. Between {y0} and {y1} revenue compounded at "
          f"{fmt_pct(rev_c)} to {fmt_cr(last['Revenue'])}")
    p1 += f" and PAT at {fmt_pct(pat_c)} to {fmt_cr(last['PAT'])}" if _ok(pat_c) else \
          f", while PAT moved from {fmt_cr(t['PAT'].iloc[-1 - f['cagr']['span']])} to {fmt_cr(last['PAT'])}"
    if fin:
        p1 += (f"; RoA stands at {fmt_pct(last['ROA %'], 2)} on a {fmt_x(_v(last['Total Assets']) / _v(last['Equity']))} "
               f"assets-to-equity balance sheet. ")
    else:
        m0, m1 = _v(t["EBITDA Margin %"].iloc[-1 - f["cagr"]["span"]]), _v(last["EBITDA Margin %"])
        bps = (m1 - m0) * 100
        move = "expanding" if bps > 25 else "contracting" if bps < -25 else "holding broadly flat, moving"
        p1 += f", with EBITDA margin {move} {abs(bps):.0f} bps to {fmt_pct(m1)}. "
    p1 += f"The record points to {growth_tag}; our forecasts fade growth from {fmt_pct(d['g0'] * 100)} toward {fmt_pct(d['g'] * 100)} over five years."

    # ---- Paragraph 2: returns and balance-sheet quality
    roe, roic = _v(last["ROE %"]), _v(last["ROIC %"])
    fs, z = f["piotroski"], f["altman"]
    if fin:
        spread = roe - coc["ke"] * 100
        p2 = (f"RoE of {fmt_pct(roe)} sits {abs(spread) * 100:.0f} bps {'above' if spread >= 0 else 'below'} our "
              f"{fmt_pct(coc['ke'] * 100)} cost of equity, so book value is being compounded "
              f"{'at a value-accretive rate' if spread >= 0 else 'below its cost'}. ")
    else:
        spread = roic - coc["wacc"] * 100
        nd, nde = _v(last["Net Debt"]), _v(last["Net Debt/EBITDA"])
        lev = (f"net cash of {fmt_cr(-nd)}" if nd < 0 else f"net debt of {fmt_x(nde)} EBITDA") if _ok(nd) else "leverage not reported"
        p2 = (f"RoIC of {fmt_pct(roic)} against a {fmt_pct(coc['wacc'] * 100)} WACC implies a "
              f"{'value-creating' if spread >= 0 else 'value-dilutive'} spread of {abs(spread):.1f} pp, RoE is {fmt_pct(roe)}, "
              f"FCF conversion is {fmt_pct(last['FCF Conversion %'], 0)} of PAT and the balance sheet carries {lev}. ")
    if fs:
        q = "strong" if fs["score"] >= 7 else "adequate" if fs["score"] >= 5 else "weak"
        p2 += f"A Piotroski F-Score of {fs['score']}/{fs['evaluated']} signals {q} fundamental momentum"
    p2 += (f" and an Altman Z of {fmt_num(z['z'])} places the company in the {z['zone'].lower()} zone."
           if z["applicable"] and _ok(z["z"]) else "; Altman Z is not applicable to a banking balance sheet.")

    # ---- Paragraph 3: valuation and stance
    med_pe, own = _v(pv["median"].get("P/E")), pv["table"].iloc[0]
    own_pe = _v(own["P/E"])
    stance = {"BUY": "see the risk-reward as favourable", "NEUTRAL": "see the risk-reward as balanced",
              "SELL": "see the risk-reward as unfavourable"}.get(tr["rating"], "do not assign a rating")
    p3 = (f"We rate {name} {tr['rating']} with a target price of {fmt_inr(tr['target'], 0)}, {fmt_pct(tr['upside'] * 100, 1, True)} "
          f"versus the current {fmt_inr(price, 0)}, and {stance}. Our {'excess-return model' if fin else 'five-year DCF'} ({rate_name} {fmt_pct(coc['wacc'] * 100)}, "
          f"terminal growth {fmt_pct(d['g'] * 100)}) values the equity at {fmt_inr(d['intrinsic'], 0)} per share")
    p3 += f", while NSE peer medians imply {fmt_inr(pv['implied_price'], 0)}. " if _ok(pv["implied_price"]) else ". "
    if _ok(own_pe) and _ok(med_pe) and med_pe > 0:
        prem = (own_pe / med_pe - 1) * 100
        p3 += (f"At {fmt_x(own_pe)} earnings the stock trades at a {'premium' if prem >= 0 else 'discount'} of "
               f"{abs(prem):.0f}% to the peer median of {fmt_x(med_pe)}. ")
    if _ok(d["tv_share"]) and not fin:
        p3 += f"Terminal value is {d['tv_share'] * 100:.0f}% of enterprise value, so the call is sensitive to long-run assumptions."

    # ---- Bull / bear from the sensitivity corners
    sens = d["sens"]
    bull_v, bear_v = _v(sens.iloc[0, -1]), _v(sens.iloc[-1, 0])
    def case(v, label, driver):
        if not _ok(v):
            return f"{label}: {driver[0].upper()}{driver[1:]}."
        return (f"{fmt_inr(v, 0)} per share ({fmt_pct((v / price - 1) * 100, 0, True)} vs CMP). {label} "
                f"and {driver}.")
    bull = case(bull_v, f"{rate_name} 100 bps lower, terminal growth 100 bps higher", sec["bull"])
    bear = case(bear_v, f"{rate_name} 100 bps higher, terminal growth 100 bps lower", sec["bear"])

    # ---- Catalysts (sector + data-driven)
    cats = list(sec["catalysts"])
    usd = fxa["pairs"].get(config.FX_PRIMARY, {})
    prior = fxa["sector_prior"][0]
    if usd and prior == "tailwind" and usd.get("chg_1y", 0) > 0.02:
        cats.append(f"INR depreciation of {usd['chg_1y'] * 100:.1f}% over 12 months flowing through to export realisations")
    elif not fin and _ok(last["Net Debt"]) and last["Net Debt"] < 0 and -last["Net Debt"] > 0.05 * ctx["market"]["mcap"]:
        cats.append(f"Capital-return optionality from {fmt_cr(-last['Net Debt'])} of net cash (buybacks / special dividends)")
    elif _ok(d["g0_raw"]) and d["g0_raw"] > config.GROWTH_CAP:
        cats.append("Sustained earnings momentum above our capped 22% starting growth assumption")
    else:
        cats.append("Earnings delivery versus consensus over the next two quarterly results")

    return {
        "thesis": [p1, p2, p3], "bull": bull, "bear": bear, "catalysts": cats[:4],
        "dupont": dupont_commentary(f), "forensic": forensic_commentary(f),
        "fx": fx_commentary(ctx), "valuation": valuation_commentary(ctx),
        "risks": risk_list(ctx), "engine": ENGINE_DETERMINISTIC,
    }


def dupont_commentary(f: dict) -> str:
    dp = f["dupont"].dropna(subset=["ROE (product)"])
    if len(dp) < 2:
        return "Insufficient history for a DuPont trend."
    a, b = dp.iloc[0], dp.iloc[-1]
    comps = list(dp.columns[:5])
    contrib = {}
    for c in comps:
        if a[c] > 0 and b[c] > 0:
            contrib[c] = float(np.log(b[c] / a[c]))
    txt = (f"RoE moved from {a['ROE (product)'] * 100:.1f}% in {dp.index[0]} to {b['ROE (product)'] * 100:.1f}% in "
           f"{dp.index[-1]}. ")
    if contrib:
        top = sorted(contrib, key=lambda k: abs(contrib[k]), reverse=True)
        k1 = top[0]
        short = lambda k: k.split(" (")[0].lower().replace("ebit", "EBIT")  # noqa: E731
        fmt = lambda k, v: f"{v * 100:.1f}%" if "Margin" in k else f"{v:.2f}x"  # noqa: E731
        txt += (f"The dominant driver was {short(k1)} ({fmt(k1, a[k1])} to {fmt(k1, b[k1])}, "
                f"{'adding to' if contrib[k1] > 0 else 'subtracting from'} RoE)")
        if len(top) > 1:
            k2 = top[1]
            txt += f", followed by {short(k2)} ({fmt(k2, a[k2])} to {fmt(k2, b[k2])})"
        txt += ". "
    lev = b["Financial Leverage (Avg Assets/Avg Equity)"]
    if f["financial"]:
        txt += (f"Leverage of {lev:.1f}x is structural for a bank; interest burden is shown as 1.00x because "
                f"interest expense is an operating cost and pre-tax profit is used as the operating-profit line.")
    else:
        txt += (f"Financial leverage of {lev:.2f}x indicates returns are "
                f"{'predominantly operationally earned' if lev < 2.5 else 'materially assisted by balance-sheet leverage'}, "
                f"and an interest burden of {b['Interest Burden (PBT/EBIT)']:.2f}x shows "
                f"{'negligible drag from financing costs' if b['Interest Burden (PBT/EBIT)'] >= 0.95 else 'a visible drag from financing costs'}.")
    return txt


def forensic_commentary(f: dict) -> str:
    fs, z = f["piotroski"], f["altman"]
    out = ""
    if fs:
        fails = [c["criterion"] for c in fs["criteria"] if c["available"] and not c["passed"]]
        out += f"Piotroski F-Score {fs['score']}/{fs['evaluated']} for {fs['year']}. "
        out += (f"Criteria not met: {'; '.join(fails)}. " if fails else "All evaluable criteria are met. ")
        if fs["note"]:
            out += fs["note"] + " "
    if z["applicable"] and _ok(z["z"]):
        out += f"Altman Z-Score {z['z']:.2f} ({z['zone']} zone); Z''-EM {fmt_num(z['z_em'])}."
    else:
        out += "Altman Z-Score: not applicable to banks / NBFCs."
    return out


def fx_commentary(ctx: dict) -> str:
    fxa, name = ctx["fx"], ctx["name"]
    usd = fxa["pairs"].get(config.FX_PRIMARY)
    if not usd:
        return "USD/INR history unavailable."
    chg = usd["chg_1y"] * 100
    direction = "depreciated" if chg > 0 else "appreciated"
    prior, why = fxa["sector_prior"]
    txt = (f"USD/INR is at {usd['last']:.2f}; the rupee has {direction} {abs(chg):.1f}% against the dollar over 12 months. ")
    b, bn, ts = _v(fxa.get("fx_beta")), _v(fxa.get("fx_beta_neutral")), _v(fxa.get("t_stat"))
    if _ok(bn):
        sig = "statistically significant" if abs(ts) >= 2 else "not statistically significant"
        txt += (f"{name}'s two-year FX beta to USD/INR is {b:.2f} (NIFTY-neutral {bn:.2f}, t-stat {ts:.1f}, {sig}) "
                f"and the latest 90-day rolling correlation is {fmt_num(fxa.get('corr_90d'))}. ")
    label = {"tailwind": "an earnings tailwind", "headwind": "an earnings headwind",
             "mixed": "a mixed earnings effect", "neutral": "a broadly neutral direct earnings effect"}[prior]
    txt += f"Structurally, INR depreciation is {label} for the sector given {why}. "
    if _ok(bn) and abs(ts) >= 2:
        emp = "outperform" if bn > 0 else "underperform"
        agree = (bn > 0 and prior == "tailwind") or (bn < 0 and prior == "headwind")
        txt += (f"Price behaviour {'corroborates' if agree else 'does not mirror'} this: the stock tends to {emp} the "
                f"NIFTY 50 on days the rupee weakens.")
    elif _ok(bn):
        txt += "The market-neutral price sensitivity is too weak to treat the rupee as a standalone trading driver."
    return txt


def valuation_commentary(ctx: dict) -> str:
    d, pv, tr = ctx["dcf"], ctx["peers"], ctx["rating"]
    coc = d["coc"]
    if d["financial"]:
        txt = (f"Banks are valued on an excess-return model: book equity plus the present value of returns earned above the "
               f"{fmt_pct(coc['ke'] * 100)} cost of equity (beta {coc['beta']:.2f}). Inputs: {d['base_method']}. ")
    else:
        txt = (f"The DCF starts from {d['base_method']} ({fmt_cr(d['base_fcf'])}) and fades growth linearly from "
               f"{fmt_pct(d['g0'] * 100)} to the {fmt_pct(d['g'] * 100)} terminal rate. WACC of {fmt_pct(coc['wacc'] * 100)} "
               f"blends a {fmt_pct(coc['ke'] * 100)} cost of equity (beta {coc['beta']:.2f}) at {coc['we'] * 100:.0f}% weight "
               f"with a {fmt_pct(coc['kd_post'] * 100)} post-tax cost of debt. ")
    if _ok(pv["implied_price"]):
        used = ", ".join(pv["implied"].keys())
        txt += (f"Applying peer-median {used} to the company's own earnings base gives {fmt_inr(pv['implied_price'], 0)} per share. ")
    txt += f"The target price of {fmt_inr(tr['target'], 0)} reflects {tr['basis']}."
    return txt


def risk_list(ctx: dict) -> list[str]:
    sec = SECTOR_TEXT.get(ctx["sector"], SECTOR_TEXT["Capital Goods"])
    d = ctx["dcf"]
    risks = list(sec["risks"])
    if _ok(d["tv_share"]) and not d["financial"] and d["tv_share"] > 0.7:
        risks.append(f"Valuation: {d['tv_share'] * 100:.0f}% of enterprise value sits in the terminal value; a 50 bps "
                     f"change in WACC or terminal growth moves fair value materially (see sensitivity matrix)")
    else:
        risks.append("Valuation: fair value is sensitive to the discount rate and long-run growth assumptions (see sensitivity matrix)")
    return risks[:5]


# --------------------------------------------------------------------------- #
# Optional LLM polish (free-tier Gemini or Groq, user-supplied key)
# --------------------------------------------------------------------------- #
def _fact_sheet(ctx: dict, base: dict) -> dict:
    return {"company": ctx["name"], "ticker": ctx["ticker"], "sector": ctx["sector"],
            "rating": ctx["rating"]["rating"], "deterministic_draft": {k: base[k] for k in ("thesis", "bull", "bear", "catalysts")}}


def _prompt(ctx: dict, base: dict) -> str:
    return (
        "You are a sell-side India equity research editor. Rewrite the draft below in crisp institutional prose. "
        "STRICT RULES: use only the numbers already present in the draft; do not add, change or round any figure; "
        "do not change the rating; keep ₹ and 'Cr' units; no markdown. Return ONLY a JSON object with keys "
        '"thesis" (array of exactly 3 paragraphs, each at most 75 words), "bull" (at most 45 words), '
        '"bear" (at most 45 words) and "catalysts" (array of exactly 4 short bullet strings).\n\n'
        + json.dumps(_fact_sheet(ctx, base), ensure_ascii=False)
    )


def _call_gemini(prompt: str, key: str, timeout: int = 30) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, timeout=timeout,
                      json={"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}})
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(prompt: str, key: str, timeout: int = 30) -> str:
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=timeout,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"model": GROQ_MODEL, "temperature": 0.3, "response_format": {"type": "json_object"},
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _clip_words(s: str, n: int) -> str:
    words = str(s).split()
    return " ".join(words) if len(words) <= n else " ".join(words[:n]).rstrip(",;:") + "…"


def _parse_llm(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    thesis, cats = data["thesis"], data["catalysts"]
    if not (isinstance(thesis, list) and len(thesis) == 3 and isinstance(cats, list) and len(cats) >= 3):
        raise ValueError("unexpected LLM response shape")
    return {"thesis": [_clip_words(p, 90) for p in thesis], "bull": _clip_words(data["bull"], 55),
            "bear": _clip_words(data["bear"], 55), "catalysts": [_clip_words(c, 22) for c in cats[:4]]}


def enhance_with_llm(base: dict, ctx: dict, provider: str, api_key: str) -> tuple[dict, str | None]:
    """Return ``(narrative, error)``. On any failure the deterministic narrative is returned unchanged."""
    if not api_key or provider not in ("Google Gemini", "Groq"):
        return base, None
    try:
        prompt = _prompt(ctx, base)
        raw = _call_gemini(prompt, api_key.strip()) if provider == "Google Gemini" else _call_groq(prompt, api_key.strip())
        out = dict(base)
        out.update(_parse_llm(raw))
        out["engine"] = f"{provider} ({GEMINI_MODEL if provider == 'Google Gemini' else GROQ_MODEL}) over deterministic facts"
        return out, None
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if api_key and api_key.strip() in msg:
            msg = msg.replace(api_key.strip(), "***")
        return base, f"LLM enhancement unavailable ({type(exc).__name__}); showing the deterministic narrative."
