# Institutional Indian Equity Research & Factor Backtesting Engine

A zero-cost Streamlit application for NSE-listed Indian equities: statement forensics, an
India-calibrated DCF, NSE peer multiples, INR currency sensitivity, a NIFTY multi-factor
backtester and a three-page institutional research note in PDF.

**Scope is strict:** NSE tickers (`.NS`), benchmark NIFTY 50 (`^NSEI`), currency pairs
`USDINR=X`, `EURINR=X`, `GBPINR=X`. BSE (`.BO`) and non-Indian tickers are rejected.
All absolute figures are in **₹ Crores** (`value / 1e7`, Indian digit grouping); prices and
EPS are in **₹**.

> Educational tool. Not investment advice, and not a product of any SEBI-registered research
> analyst, brokerage or bank.

---

## 1. Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in **Offline Demo Mode**, which needs no network and no keys. Switch the sidebar
**Data mode** to *Live Data (Yahoo Finance)* for live prices and statements.

| Sidebar control | What it does |
|---|---|
| Data mode | Offline Demo Mode / Live Data. Live falls back to `data/` automatically. |
| Sector → NSE stock | 39 liquid large caps in 8 sectors; peers are pre-populated. |
| Macro & valuation assumptions | Risk-free rate, ERP, tax rate, terminal growth, target-price blend, growth override. |
| Narrative engine | Deterministic NLG by default; optional free Gemini or Groq key. |

### Tabs
1. **Overview** — price vs NIFTY 50, key market and valuation data.
2. **Fundamentals & Forensics** — ₹ Cr statements, growth and margins, RoIC, 5-step DuPont, Piotroski F-Score, Altman Z.
3. **Valuation** — DCF projection and bridge, 5×5 sensitivity heatmap, NSE peer table and implied price.
4. **INR FX Macro** — USD/EUR/GBP vs INR, FX beta, 90-day rolling correlation, interpretation.
5. **Factor Screener & Backtest** — Value/Quality/Momentum screen and quarterly rebalanced backtest vs `^NSEI`.
6. **Research Report (PDF)** — narrative and the downloadable three-page note.

---

## 2. Repository structure

```text
india-equity-research-engine/
├── app.py                  # Streamlit multi-tab dashboard and UI controller
├── config.py               # NSE universe, sector peer maps, Indian macro defaults, ₹ formatting
├── data_loader.py          # yfinance fetcher with st.cache_data, retries, circuit breaker, offline fallback
├── fundamentals.py         # Statement parser, ₹ Cr scaling, DuPont, Piotroski, Altman Z
├── valuation.py            # DCF / bank excess-return model, sensitivity, peers, FX beta, research pipeline
├── backtester.py           # Multi-factor screener and quarterly rebalanced backtest engine
├── narratives.py           # Deterministic financial NLG + optional Gemini / Groq polish
├── report_generator.py     # Three-page institutional PDF (ReportLab + Matplotlib)
├── requirements.txt        # Pinned dependencies (tested together on Python 3.13)
├── README.md
└── data/                   # Offline pack
    ├── prices.csv          # Daily closes: 39 stocks + ^NSEI, Nov-2020 to Nov-2026
    ├── fx.csv              # Daily USDINR=X, EURINR=X, GBPINR=X
    ├── manifest.json       # As-of date and provenance of the pack
    └── fundamentals/       # One JSON per ticker: info + income / balance / cash-flow statements
```

---

## 3. Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. On <https://share.streamlit.io> choose **New app**, select the repository and `app.py`.
3. In **Advanced settings** choose **Python 3.13** (the pinned versions were tested on 3.13).
4. Deploy. No secrets, no `packages.txt` and no apt packages are required: the PDF uses
   ReportLab and Matplotlib only, and the ₹ glyph comes from the DejaVu font bundled with Matplotlib.

Yahoo Finance frequently rate-limits shared cloud IPs. When that happens the app shows an amber
badge and serves the offline pack; **Retry live connection** clears the cache and tries again.

---

## 4. Data and reliability

### Live mode
`yfinance` supplies adjusted daily closes, `.info`, and annual income statement, balance sheet
and cash-flow statement. Each call is retried with exponential backoff. A failed call trips a
five-minute circuit breaker so a rate-limited session degrades immediately instead of retrying
per ticker. If any component of an analysis falls back, the **whole** analysis is rebuilt from
the offline pack so one page never mixes sources. Statements that Yahoo reports in USD are
converted to INR at the latest USD/INR and flagged on screen.

### Offline pack — read this before quoting numbers from it
Yahoo Finance was not reachable from the environment in which this pack was built, so the
shipped pack is a **calibrated demo dataset, not a Yahoo snapshot**:

* **Prices** are simulated daily paths (market + sector + currency factor model) pinned to
  anchor levels. Anchors for Nov-2024, Nov-2025 and 09-Nov-2026 are actual NSE closes
  (adjusted for splits and bonuses); NIFTY 50 anchors from Oct-2024 onward are actual closes;
  earlier anchors are approximate. Day-to-day moves, betas and correlations in between are
  model output.
* **FX**: simulated paths pinned to approximate historical levels; 02–09 Nov 2026 are RBI
  reference rates.
* **Fundamentals** are approximate, rounded figures in the yfinance layout. Headline lines
  (revenue, EBITDA margin, PAT, net worth) approximate reported numbers; secondary lines
  (working capital, D&A, capex, cash flow) are derived from sector ratios. They are not audited
  and FY26 figures in particular should be treated as placeholders for the real filings.

To replace the pack with a real snapshot, run once from a machine with internet access:

```bash
python data_loader.py --refresh
```

This rewrites `prices.csv`, `fx.csv` and every fundamentals JSON from Yahoo Finance and updates
`manifest.json`. Files are only replaced when their download succeeds.

### NSE symbol changes handled
| Requested | Resolved to | Reason |
|---|---|---|
| `TATAMOTORS.NS` | `TMPV.NS` | Oct-2025 demerger; the listed entity became Tata Motors Passenger Vehicles. `TATAMOTORS` last traded on 23-Oct-2025. |
| `LTIM.NS` | `LTM.NS` | Symbol change in Feb-2026. |

---

## 5. Methodology

### Indian institutional defaults (`config.py`)
| Parameter | Value |
|---|---|
| Risk-free rate (10Y G-Sec) | 6.85% |
| India equity risk premium | 6.25% |
| Marginal corporate tax (Sec 115BAA) | 25.17% |
| Terminal growth | 5.00% |
| Round-trip execution cost | 0.25% |

### Fundamentals (`fundamentals.py`)
* FCF = CFO − CapEx. Net debt = total debt − cash and short-term investments.
  Invested capital = equity + total debt − cash.
* RoIC = EBIT × (1 − effective tax) / average invested capital. RoE and RoA use average balances.
* CAGRs span three years (four data points) or the longest available span.
* **5-step DuPont:** RoE = (PAT/PBT) × (PBT/EBIT) × (EBIT/Revenue) × (Revenue/Avg assets) × (Avg assets/Avg equity).
* **Piotroski F-Score:** nine pass/fail tests with evidence. A test whose inputs are not
  reported scores 0 and is shown as n.a.; the denominator is the number of evaluable tests.
* **Altman Z** = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5, with the Z''-EM book-value
  variant alongside. Flagged **not applicable** for banks and NBFCs.
* **Banks:** EBITDA, FCF ratios, net debt and RoIC are shown as n.m.; pre-tax profit is the
  operating-profit line and revenue is net total income.

### Valuation (`valuation.py`)
* **Beta:** three years of weekly returns vs NIFTY 50, Blume-adjusted (0.67·raw + 0.33).
* **WACC:** Ke = Rf + β·ERP; Kd = interest / average debt (bounded), post-tax at 25.17%;
  weights by market capitalisation and total debt.
* **Base FCF:** three-year average FCF margin × latest revenue. If that is not positive:
  latest EBITDA × median positive historical FCF/EBITDA conversion, else a sector default.
* **Growth:** mean of 3Y revenue and PAT CAGR (PAT CAGR bounded to ±10 pp of revenue CAGR so a
  low-base recovery is not extrapolated), capped to 6–22%, then faded linearly to terminal
  growth over five years.
* **Value:** EV = Σ PV(FCF) + PV(Gordon terminal value); equity = EV − net debt; per share on
  diluted shares. **5×5 sensitivity:** WACC ±100 bps × terminal growth ±100 bps in 50 bps steps.
* **Banks** use a five-year excess-return model: book equity + PV of (RoE − Ke) × book value,
  with RoE fading halfway to the cost of equity, and the same 5×5 grid on Ke and growth.
* **Peers:** P/E, forward P/E, EV/EBITDA, P/B, RoE and 3Y PAT CAGR computed from statements
  and current prices; medians across peers only. Implied price = average of the P/E route and
  the EV/EBITDA route (P/E and P/B for banks).
* **Target price** = 50% intrinsic value + 50% peer-implied value (adjustable).
  **BUY** above +15%, **NEUTRAL** −10% to +15%, **SELL** below −10%.
* **FX:** 1Y change in each pair; FX beta = slope of daily stock returns on USD/INR returns over
  two years; a NIFTY-neutral beta (stock on NIFTY and USD/INR) with t-statistic; 90-day rolling
  correlation; a sector prior on whether INR depreciation helps or hurts earnings.

A five-year explicit window with a 5% terminal rate is conservative for long-duration Indian
compounders trading at 40–60× earnings, so mechanical intrinsic values often sit well below
market prices. That is a property of the specified model, not a data error; the sidebar exposes
every assumption.

### Factor engine (`backtester.py`)
* **Value** = z(earnings yield) + z(FCF yield) + z(book/price).
  **Quality** = z(RoE) + z(F-Score) − z(debt/equity).
  **Momentum** = z(12-1 month return) + z(6-month return / 6-month volatility).
  Inputs are winsorised at 5/95% and pillars re-standardised; missing inputs are skipped.
* Rebalance on the last trading day of each calendar quarter; equal or inverse-volatility
  weights; weights drift between rebalances.
* **Costs:** one-way turnover (½ Σ|Δw|) × 0.25% at each rebalance; the initial purchase is
  half a round trip.
* **Point-in-time:** prices up to the rebalance date only; fundamentals from the latest fiscal
  year available on that date (year-end + 90 days), priced at that date.
* **Metrics:** CAGR, annualised volatility, Sharpe and Sortino (Rf 6.85%), Jensen's alpha,
  beta, max drawdown, monthly win rate vs NIFTY 50, tracking error and information ratio.

Known limitations, shown in the UI where relevant: free sources carry only 4–5 fiscal years,
so early rebalances in a 5Y test use the earliest statement and are counted as look-ahead
rebalances; the universe is today's large-cap list (survivorship bias); the index is a price
index while live stock prices are dividend-adjusted; results are before taxes.

### Narrative (`narratives.py`)
The default engine composes every sentence from computed metrics, so text and tables cannot
disagree. With a user-supplied key, Gemini (`gemini-2.5-flash`) or Groq
(`llama-3.3-70b-versatile`) rewrites the thesis, bull/bear cases and catalysts from that draft
under an instruction not to alter any figure. Any API or parsing failure returns the
deterministic text. Keys are held only in the session.

### PDF (`report_generator.py`)
A4 portrait, 0.5-inch margins, Helvetica, navy `#002D72` / slate `#005596` / gray `#4A5568` /
tint `#F4F7FA`.
* **Page 1** — banner, title block with price, target, upside and rating badge; thesis,
  bull/bear boxes and catalysts (65%) beside the market-data box (35%); financial summary with
  four historical and two projected years.
* **Page 2** — revenue/margin and 1-year relative-performance charts at 300 DPI; valuation
  bridge and assumptions beside the 5×5 sensitivity matrix; peer comparison with the median row.
* **Page 3** — DuPont table and interpretation, Piotroski and Altman tables, INR macro table
  and chart, key risks, disclosures.

Each page body is laid out in a shrink-to-fit frame, so long names or LLM text cannot overlap,
clip or create a fourth page.

---

## 6. Verification performed

* Hand recomputation from the raw JSON of ₹ Cr scaling, FCF, net debt, RoIC, CAGR, the DuPont
  product (equal to PAT / average equity), all nine Piotroski tests and Altman Z.
* Independent rebuild of WACC, the FCF path, terminal value and intrinsic value; the bank
  excess-return value; sensitivity axes, centre cell and monotonicity; rating thresholds.
* Backtest NAV replayed independently from the holdings log; Sharpe, Sortino, max drawdown and
  win rate recomputed; beta and Jensen's alpha matched to a `scipy.stats.linregress` CAPM
  regression; a price-tampering test confirming no look-ahead in momentum.
* The live code path exercised against a stand-in for `yfinance` (newest-first statements,
  multi-index downloads, USD-reported statements, empty frames, rate-limit errors), and the
  real fallback exercised against a blocked Yahoo endpoint.
* The app run headlessly for all 39 tickers, extreme inputs and an invalid LLM key with no
  exceptions; PDFs for all 39 tickers checked for exactly three pages and expected content.

Not verified: a successful live fetch from Yahoo Finance itself, and real calls to the Gemini
and Groq APIs, because neither was reachable from the build environment.
