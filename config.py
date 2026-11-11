"""
config.py — NSE universe, sector peer maps and Indian institutional macro defaults.

Scope is deliberately strict: NSE-listed equities (``.NS``), the NIFTY 50 benchmark
(``^NSEI``) and INR currency pairs. BSE (``.BO``) and non-NSE suffixes are rejected.

v2 (additive): 52 covered stocks in 11 analytical sectors, universal NSE ticker
resolution with safe sector/peer fallbacks, the interface design tokens, and a small
extension hook that serves offline demo data for the newly covered tickers from
``data_ext/`` without modifying ``data/`` or any engine module.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

APP_TITLE = "Institutional Indian Equity Research & Factor Backtesting Engine"

# --------------------------------------------------------------------------- #
# Indian institutional macro defaults
# --------------------------------------------------------------------------- #
RISK_FREE_RATE = 0.0685        # 10-Year Indian G-Sec yield
EQUITY_RISK_PREMIUM = 0.0625   # India equity risk premium
TAX_RATE = 0.2517              # Section 115BAA marginal corporate tax rate
TERMINAL_GROWTH = 0.05         # Nominal India GDP long-run proxy
TXN_COST = 0.0025              # Round-trip cost: STT + brokerage + stamp duty + slippage
DEFAULT_PRETAX_COST_OF_DEBT = 0.0885   # G-Sec + 200 bps when no interest history exists
GROWTH_FLOOR, GROWTH_CAP = 0.06, 0.22  # Bounds on the starting DCF growth rate
PROJECTION_YEARS = 5
DEFAULT_DCF_WEIGHT = 0.5       # Target price = w * DCF value + (1 - w) * peer-implied value
REPORTING_LAG_DAYS = 90        # Statements become "known" 90 days after fiscal year-end
CRORE = 1e7

BENCHMARK = "^NSEI"
BENCHMARK_NAME = "NIFTY 50"
FX_TICKERS = ["USDINR=X", "EURINR=X", "GBPINR=X"]
FX_PRIMARY = "USDINR=X"

RATING_BUY, RATING_SELL = 0.15, -0.10   # BUY > +15 %, SELL < -10 %, NEUTRAL in between

# --------------------------------------------------------------------------- #
# Curated NSE universe (52 liquid large caps) grouped by analytical sector.
# The sector key drives the financial/non-financial model, the INR exposure
# prior and the narrative phrase bank, so existing keys are never renamed.
# --------------------------------------------------------------------------- #
UNIVERSE: dict[str, dict[str, str]] = {
    "IT Services": {
        "TCS.NS": "Tata Consultancy Services", "INFY.NS": "Infosys", "HCLTECH.NS": "HCL Technologies",
        "WIPRO.NS": "Wipro", "TECHM.NS": "Tech Mahindra", "LTM.NS": "LTIMindtree",
    },
    "Private Banks": {
        "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "AXISBANK.NS": "Axis Bank", "INDUSINDBK.NS": "IndusInd Bank",
    },
    "PSU Banks": {
        "SBIN.NS": "State Bank of India", "BANKBARODA.NS": "Bank of Baroda", "PNB.NS": "Punjab National Bank",
        "CANBK.NS": "Canara Bank", "UNIONBANK.NS": "Union Bank of India",
    },
    "Oil & Gas / Conglomerate": {
        "RELIANCE.NS": "Reliance Industries", "ONGC.NS": "ONGC", "IOC.NS": "Indian Oil Corporation",
        "BPCL.NS": "Bharat Petroleum", "GAIL.NS": "GAIL (India)",
    },
    "Power Utilities": {
        "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid Corporation of India",
    },
    "Pharmaceuticals": {
        "SUNPHARMA.NS": "Sun Pharmaceutical", "DRREDDY.NS": "Dr. Reddy's Laboratories", "CIPLA.NS": "Cipla",
        "DIVISLAB.NS": "Divi's Laboratories", "LUPIN.NS": "Lupin",
    },
    "Automobiles": {
        "TMPV.NS": "Tata Motors Passenger Vehicles", "MARUTI.NS": "Maruti Suzuki", "M&M.NS": "Mahindra & Mahindra",
        "BAJAJ-AUTO.NS": "Bajaj Auto", "EICHERMOT.NS": "Eicher Motors",
    },
    "FMCG": {
        "ITC.NS": "ITC", "HINDUNILVR.NS": "Hindustan Unilever", "NESTLEIND.NS": "Nestle India",
        "BRITANNIA.NS": "Britannia Industries", "DABUR.NS": "Dabur India",
    },
    "Consumer Tech & Retail": {
        "ETERNAL.NS": "Eternal (Zomato)", "TRENT.NS": "Trent", "DMART.NS": "Avenue Supermarts (DMart)",
        "NAUKRI.NS": "Info Edge (Naukri)", "NYKAA.NS": "FSN E-Commerce (Nykaa)",
    },
    "Capital Goods": {
        "LT.NS": "Larsen & Toubro", "BEL.NS": "Bharat Electronics", "ABB.NS": "ABB India",
        "BHEL.NS": "Bharat Heavy Electricals", "CUMMINSIND.NS": "Cummins India", "HAL.NS": "Hindustan Aeronautics",
    },
    "Telecom & Digital Infra": {
        "BHARTIARTL.NS": "Bharti Airtel", "INDUSTOWER.NS": "Indus Towers", "TATACOMM.NS": "Tata Communications",
    },
}

# Coverage groups as shown in the interface (group label -> analytical sectors).
# Power PSUs sit under Energy; defence PSUs (HAL, BEL) sit under Capital Goods.
SECTOR_GROUPS: dict[str, list[str]] = {
    "Private Banks": ["Private Banks"],
    "PSU Banks": ["PSU Banks"],
    "IT Services": ["IT Services"],
    "Energy & Conglomerate": ["Oil & Gas / Conglomerate", "Power Utilities"],
    "Pharmaceuticals": ["Pharmaceuticals"],
    "Automobiles": ["Automobiles"],
    "FMCG": ["FMCG"],
    "Consumer Tech & Retail": ["Consumer Tech & Retail"],
    "Telecom": ["Telecom & Digital Infra"],
    "Capital Goods": ["Capital Goods"],
}

# NSE symbol changes. The old symbols no longer trade, so they are mapped to the
# continuing listed entity (verified against NSE bhavcopy, Nov-2026).
TICKER_ALIASES = {
    "TATAMOTORS.NS": "TMPV.NS",   # Oct-2025 demerger: listed entity renamed Tata Motors Passenger Vehicles
    "LTIM.NS": "LTM.NS",          # Feb-2026 symbol change after the LTIMindtree rebrand
    "ZOMATO.NS": "ETERNAL.NS",    # Apr-2025 symbol change after Zomato Ltd became Eternal Ltd
}

# Extra cross-sector peers appended to the sector list.
PEER_OVERRIDES = {
    "BHARTIARTL.NS": ["RELIANCE.NS"],          # Jio is the direct competitor
    "INDUSTOWER.NS": ["RELIANCE.NS"],
    "TATACOMM.NS": ["RELIANCE.NS"],
    "NTPC.NS": ["GAIL.NS", "ONGC.NS", "IOC.NS", "BPCL.NS"],        # energy PSUs complete the peer set
    "POWERGRID.NS": ["GAIL.NS", "ONGC.NS", "IOC.NS", "BPCL.NS"],
}

FINANCIAL_SECTORS = {"Private Banks", "PSU Banks"}

# Structural INR exposure prior per sector: sign of the earnings response to INR depreciation.
FX_EXPOSURE = {
    "IT Services": ("tailwind", "USD/EUR/GBP-billed export revenue against a largely INR cost base"),
    "Pharmaceuticals": ("tailwind", "US generics and export formulations/API revenue billed in USD"),
    "Oil & Gas / Conglomerate": ("headwind", "USD-denominated crude and LNG import bill, partly offset by dollar-linked refining and upstream realisations"),
    "Automobiles": ("mixed", "imported components and commodities against export and overseas-subsidiary earnings"),
    "FMCG": ("headwind", "imported palm oil, crude derivatives and packaging inputs with largely domestic INR revenue"),
    "Capital Goods": ("mixed", "imported components against export orders and overseas project revenue"),
    "Private Banks": ("neutral", "limited direct translation exposure; the channel is indirect via FPI flows, liquidity and rates"),
    "Telecom & Digital Infra": ("headwind", "USD-linked network capex, spectrum-era foreign currency debt and equipment imports"),
    "PSU Banks": ("neutral", "limited direct translation exposure; the channel is indirect via bond yields, treasury gains and FPI flows"),
    "Consumer Tech & Retail": ("neutral", "domestic INR revenue and cost base; the channel is indirect via imported inflation and discretionary demand"),
    "Power Utilities": ("mixed", "imported coal and foreign-currency borrowings, largely passed through under regulated tariffs"),
}

# Default EBITDA-to-FCF conversion used only when historical FCF is negative and no
# positive conversion history exists.
SECTOR_FCF_CONVERSION = {
    "IT Services": 0.60, "Pharmaceuticals": 0.40, "Oil & Gas / Conglomerate": 0.30, "Automobiles": 0.35,
    "FMCG": 0.55, "Capital Goods": 0.40, "Telecom & Digital Infra": 0.25,
    "Consumer Tech & Retail": 0.35, "Power Utilities": 0.25,
}

# Narrative phrase banks for sectors added in v2. Registered into the narrative
# engine at runtime by ``install_extensions`` (same shape as narratives.SECTOR_TEXT).
SECTOR_NARRATIVES = {
    "PSU Banks": {
        "desc": "a large Indian public-sector bank with a nationwide deposit franchise and sovereign ownership",
        "bull": "credit costs stay near cycle lows, treasury gains accrue as yields fall and operating leverage lifts RoA toward private-bank levels",
        "bear": "margin compression in an easing cycle, wage revisions and a turn in the MSME and agri credit cycle pull RoA back below 1%",
        "catalysts": ["Sustained RoA above 1% with credit costs below guidance", "Treasury gains as the G-Sec curve shifts lower",
                      "Capital raising and stake-sale clarity from the Government of India"],
        "risks": ["Net interest margin compression as loans reprice faster than deposits", "Asset-quality slippage in MSME, agri and unsecured books",
                  "Wage revisions and pension provisioning", "Policy-directed lending and government ownership overhang"],
    },
    "Consumer Tech & Retail": {
        "desc": "a leading Indian consumer internet and organised retail franchise",
        "bull": "store and dark-store expansion compounds on improving unit economics, with operating leverage turning growth into free cash flow",
        "bear": "competitive intensity in quick commerce and value retail forces heavier discounting and capex, pushing out the path to cash generation",
        "catalysts": ["Contribution margin and adjusted EBITDA progression", "Store, dark-store and city expansion ahead of plan",
                      "Private-label mix and advertising income lifting take rates"],
        "risks": ["Competitive intensity and discounting in quick commerce and value retail", "Elevated valuation multiples sensitive to growth disappointment",
                  "Regulatory changes on e-commerce, gig work and FDI in retail", "Slower urban discretionary demand"],
    },
    "Power Utilities": {
        "desc": "a regulated Indian power utility with government ownership and long-dated contracted cash flows",
        "bull": "capacity and transmission additions expand the regulated equity base on schedule while renewables and storage add a second growth leg",
        "bear": "project delays, fuel-supply constraints and adverse tariff orders compress regulated returns while capex keeps free cash flow thin",
        "catalysts": ["Commissioning of thermal, renewable and transmission capacity", "CERC tariff-regulation outcomes on regulated RoE",
                      "Order wins in transmission bids, storage and green-energy subsidiaries"],
        "risks": ["Adverse changes to regulated return on equity", "Project execution delays and cost overruns",
                  "State distribution-company receivables", "Fuel availability and the pace of energy transition"],
    },
}

# Engine palette (also used by the PDF builder; no third-party branding is reproduced).
NAVY, SLATE, GRAY, TINT = "#002D72", "#005596", "#4A5568", "#F4F7FA"
GREEN, RED, AMBER = "#1B7F4B", "#B42318", "#B7791F"

# Interface design tokens (dashboard only; the PDF keeps its own palette above).
UI = {
    "obsidian": "#0B0D10", "card": "#12161C", "raised": "#171C24", "hairline": "rgba(255, 255, 255, 0.08)",
    "text": "#F3F4F6", "muted": "#8E96A4", "gold": "#D4AF37",
    "buy": "#7FB89A", "neutral": "#D9A441", "sell": "#C8644F",      # sage emerald / warm amber / terracotta crimson
}
# Chart series colours, validated for the card surface (lightness band, chroma,
# colour-vision separation, 3:1 contrast). Fixed order; never cycled.
CHART = {"series": ["#B08A26", "#3987E5", "#D95926", "#199E70"], "reference": "#8E96A4",
         "positive": "#4F9B79", "negative": "#C8644F", "midpoint": "#2A2F37"}

# --------------------------------------------------------------------------- #
# Universal NSE ticker handling
# --------------------------------------------------------------------------- #
DEFAULT_CUSTOM_SECTOR = "Capital Goods"     # benchmark peer sector when none is chosen
_CUSTOM_SECTORS: dict[str, str] = {}        # ticker -> benchmark peer sector chosen by the user
_NSE_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9&\-]{0,24}$")


def all_tickers() -> list[str]:
    return [t for sec in UNIVERSE.values() for t in sec]


def in_universe(ticker: str) -> bool:
    return any(ticker in members for members in UNIVERSE.values())


def resolve_ticker(ticker: str) -> str:
    """Normalise user input to an NSE ticker (``SYMBOL.NS``).

    Accepts any syntactically valid NSE symbol with or without ``.NS`` and applies
    symbol-change aliases (ZOMATO -> ETERNAL.NS, TATAMOTORS -> TMPV.NS, LTIM -> LTM.NS).
    Raises ValueError for BSE tickers, other exchange suffixes, indices and FX pairs.
    A ticker outside the curated universe is still returned; ``sector_of`` and
    ``peers_of`` fall back to a benchmark peer sector for it.
    """
    t = (ticker or "").strip().upper().replace(" ", "")
    if t.startswith("NSE:"):
        t = t[4:]
    if not t:
        raise ValueError("Enter an NSE ticker, for example TCS.NS.")
    if t.endswith(".BO") or t.endswith(".BSE"):
        raise ValueError("BSE (.BO) tickers are out of scope — this engine covers NSE (.NS) only.")
    if t.startswith("^") or "=" in t:
        raise ValueError("Indices and currency pairs are not equities; enter an NSE stock ticker.")
    base, dot, suffix = t.rpartition(".")
    if dot:
        if suffix != "NS":
            raise ValueError(f"{ticker!r} is not an NSE ticker — only the .NS suffix is supported.")
    else:
        base = t
    if not _NSE_SYMBOL.match(base):
        raise ValueError(f"{ticker!r} is not a valid NSE symbol.")
    t = TICKER_ALIASES.get(f"{base}.NS", f"{base}.NS")
    install_extensions()
    return t


def register_custom_ticker(ticker: str, sector: str | None = None) -> str:
    """Attach a benchmark peer sector to a ticker outside the curated universe."""
    t = resolve_ticker(ticker)
    if not in_universe(t):
        _CUSTOM_SECTORS[t] = sector if sector in UNIVERSE else DEFAULT_CUSTOM_SECTOR
    return t


def sector_of(ticker: str) -> str:
    """Analytical sector. Never raises: an uncovered ticker gets its registered
    benchmark peer sector, or the default one."""
    for sec, members in UNIVERSE.items():
        if ticker in members:
            return sec
    return _CUSTOM_SECTORS.get(ticker, DEFAULT_CUSTOM_SECTOR)


def name_of(ticker: str) -> str:
    if ticker == BENCHMARK:
        return BENCHMARK_NAME
    for members in UNIVERSE.values():
        if ticker in members:
            return members[ticker]
    return str(ticker).replace(".NS", "")


def is_financial(ticker: str) -> bool:
    return sector_of(ticker) in FINANCIAL_SECTORS


def peers_of(ticker: str, n: int = 5) -> list[str]:
    """Closest NSE sector peers (excluding the stock itself), at most ``n``."""
    sec = sector_of(ticker)
    peers = [t for t in UNIVERSE[sec] if t != ticker] + PEER_OVERRIDES.get(ticker, [])
    if not in_universe(ticker) and len(peers) < 4:      # thin benchmark sector: top up from its coverage group
        peers += [t for t in group_tickers(group_of(sec)) if t not in peers and t != ticker]
    return peers[:n]


def display_label(ticker: str) -> str:
    old = [k for k, v in TICKER_ALIASES.items() if v == ticker]
    suffix = f" (formerly {old[0]})" if old else ""
    return f"{ticker} — {name_of(ticker)}{suffix}"


def group_of(sector: str) -> str:
    """Interface coverage group for an analytical sector."""
    for group, sectors in SECTOR_GROUPS.items():
        if sector in sectors:
            return group
    return sector


def group_tickers(group: str) -> list[str]:
    return [t for sec in SECTOR_GROUPS.get(group, []) for t in UNIVERSE.get(sec, {})]


# --------------------------------------------------------------------------- #
# Runtime extensions (no engine file or data/ file is modified)
# --------------------------------------------------------------------------- #
EXT_DIR = Path(__file__).resolve().parent / "data_ext"
_EXT_STATE = {"installed": False, "prices": None}


def offline_available(ticker: str) -> bool:
    """True when an offline dataset exists for ``ticker`` in data/ or data_ext/."""
    root = Path(__file__).resolve().parent
    return (root / "data" / "fundamentals" / f"{ticker}.json").exists() or \
           (EXT_DIR / "fundamentals" / f"{ticker}.json").exists()


def install_extensions() -> bool:
    """Register v2 additions with the engines. Idempotent and failure-safe.

    1. Adds the new sectors' phrase banks to ``narratives.SECTOR_TEXT``.
    2. Lets the offline loader fall through to ``data_ext/`` for tickers that are
       absent from ``data/``. Tickers present in ``data/`` take the original path,
       so behaviour for the original pack (or a refreshed one) is unchanged.
    """
    if _EXT_STATE["installed"]:
        return True
    try:
        import data_loader as dl
        import narratives as na
        import pandas as pd
    except Exception:  # noqa: BLE001 - engines not importable yet (e.g. during their own import)
        return False
    if not all(hasattr(dl, a) for a in ("_offline_company", "_offline_prices", "_statement_from_json")) \
            or not hasattr(na, "SECTOR_TEXT"):
        return False

    for sector, text in SECTOR_NARRATIVES.items():
        na.SECTOR_TEXT.setdefault(sector, text)

    base_company, base_prices = dl._offline_company, dl._offline_prices

    def _ext_prices_frame():
        if _EXT_STATE["prices"] is None:
            path = EXT_DIR / "prices.csv"
            _EXT_STATE["prices"] = (pd.read_csv(path, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce").sort_index()
                                    if path.exists() else pd.DataFrame())
        return _EXT_STATE["prices"]

    def offline_company(ticker: str) -> dict:
        if (dl.FUND_DIR / f"{ticker}.json").exists():
            return base_company(ticker)
        path = EXT_DIR / "fundamentals" / f"{ticker}.json"
        if not path.exists():
            raise FileNotFoundError(f"No offline dataset for {ticker}. Switch to Live Data to analyse it.")
        raw = json.loads(path.read_text())
        bundle = {"ticker": ticker, "info": raw.get("info", {})}
        for s in ("income", "balance", "cashflow"):
            bundle[s] = dl._statement_from_json(raw.get(s, {}))
        return bundle

    def offline_prices(tickers, years: float):
        df = base_prices(tickers, years)
        missing = [t for t in tickers if t not in df.columns]
        ext = _ext_prices_frame()
        cols = [t for t in missing if t in ext.columns]
        if not cols:
            return df
        if df.empty:
            start = ext.index[-1] - pd.Timedelta(days=int(years * 365.25) + 5)
            return ext.loc[ext.index >= start, cols].dropna(how="all")
        add = ext[cols].reindex(df.index.union(ext.index)).ffill(limit=5).reindex(df.index)
        merged = df.join(add)
        return merged[[t for t in tickers if t in merged.columns]]

    dl._offline_company, dl._offline_prices = offline_company, offline_prices
    _EXT_STATE["installed"] = True
    return True


# --------------------------------------------------------------------------- #
# Indian number formatting (lakh / crore digit grouping)
# --------------------------------------------------------------------------- #
def _isnan(x) -> bool:
    try:
        return x is None or x != x or x in (float("inf"), float("-inf"))
    except TypeError:
        return True


def indian_number(x, decimals: int = 0) -> str:
    """1234567.8 -> '12,34,568' (Indian grouping: last three digits, then pairs)."""
    if _isnan(x):
        return "n.a."
    neg = x < 0
    s = f"{abs(float(x)):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + (f".{frac}" if frac else "")


def fmt_cr(x, decimals: int = 0) -> str:
    """Absolute financial figure in ₹ Crores."""
    return "n.a." if _isnan(x) else f"{'-' if x < 0 else ''}₹{indian_number(abs(x), decimals)} Cr"


def fmt_inr(x, decimals: int = 2) -> str:
    """Per-share price / EPS in ₹."""
    return "n.a." if _isnan(x) else f"{'-' if x < 0 else ''}₹{indian_number(abs(x), decimals)}"


def fmt_pct(x, decimals: int = 1, signed: bool = False) -> str:
    """``x`` is already in percent units."""
    if _isnan(x):
        return "n.a."
    return f"{x:+.{decimals}f}%" if signed else f"{x:.{decimals}f}%"


def fmt_x(x, decimals: int = 1) -> str:
    return "n.m." if _isnan(x) else f"{x:.{decimals}f}x"


def fmt_num(x, decimals: int = 2) -> str:
    return "n.a." if _isnan(x) else f"{x:.{decimals}f}"


def fmt_cr_compact(x) -> str:
    """Large ₹ Cr figures in lakh crore (1 L Cr = ₹1 trillion) for headline tiles."""
    if _isnan(x):
        return "n.a."
    return f"₹{x / 1e5:.2f} L Cr" if abs(x) >= 1e5 else fmt_cr(x)
