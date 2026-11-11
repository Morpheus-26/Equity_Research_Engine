"""
app.py — editorial research interface for the Institutional Indian Equity Research
& Factor Backtesting Engine (NSE equities, NIFTY 50 benchmark, INR FX macro).

This file is presentation only. Every number on screen comes from the unchanged
engines (fundamentals, valuation, backtester, narratives, report_generator).

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtester as bt
import config
import data_loader as dl
import fundamentals as fa
import narratives as na
import report_generator as rg
import valuation as va
from config import fmt_cr, fmt_cr_compact, fmt_inr, fmt_num, fmt_pct, fmt_x, indian_number

st.set_page_config(page_title="India Equity Research", page_icon="📈", layout="wide",
                   initial_sidebar_state="auto")
config.install_extensions()

UI, CH = config.UI, config.CHART
S1, S2, S3, S4 = CH["series"]
IST = timezone(timedelta(hours=5, minutes=30))
PEER_PE_CAP = 200.0          # peers above this P/E are not meaningful inputs to a median
TIMEFRAMES = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365, "3Y": 1095, "MAX": None}
SANDBOX_DEFAULTS = {"sb_rf": round(config.RISK_FREE_RATE * 100, 2), "sb_erp": round(config.EQUITY_RISK_PREMIUM * 100, 2),
                    "sb_tax": round(config.TAX_RATE * 100, 2), "sb_g": round(config.TERMINAL_GROWTH * 100, 2),
                    "sb_w": int(config.DEFAULT_DCF_WEIGHT * 100), "sb_use_g0": False, "sb_g0": 12.0, "sb_guard": True}


# --------------------------------------------------------------------------- #
# Theme: Streamlit base theme (so native widgets, menus and popovers are dark)
# plus the editorial layer in CSS. Both are applied from this file alone.
# --------------------------------------------------------------------------- #
def _apply_base_theme() -> None:
    try:
        from streamlit import config as _stc
        first = _stc.get_option("theme.primaryColor") != UI["gold"]
        opts = {"theme.base": "dark", "theme.primaryColor": UI["gold"], "theme.backgroundColor": UI["obsidian"],
                "theme.secondaryBackgroundColor": UI["card"], "theme.textColor": UI["text"]}
        for k, v in opts.items():
            _stc.set_option(k, v)
        for k, v in {"theme.font": "Inter, sans-serif", "theme.codeFont": "'JetBrains Mono', monospace"}.items():
            try:
                _stc.set_option(k, v)
            except Exception:  # noqa: BLE001 - optional on older Streamlit builds
                pass
    except Exception:  # noqa: BLE001 - the CSS layer below still themes the page
        return
    if first:
        st.rerun()      # the first visitor after a cold start gets the dark theme immediately


_apply_base_theme()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--obsidian:__OBSIDIAN__;--card:__CARD__;--raised:__RAISED__;--hair:__HAIR__;--text:__TEXT__;--muted:__MUTED__;--gold:__GOLD__;
--buy:__BUY__;--neutral:__NEUTRAL__;--sell:__SELL__;--pos:#7FB89A;--neg:#D98A76;
--serif:'Instrument Serif','Playfair Display',Georgia,'Times New Roman',serif;
--sans:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
--mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace;}
.stApp{background:var(--obsidian);color:var(--text);font-family:var(--sans);}
#MainMenu,footer,[data-testid="stDecoration"],[data-testid="stAppDeployButton"],.stDeployButton,[data-testid="stStatusWidget"]{display:none !important;}
[data-testid="stHeader"]{background:transparent !important;}
.block-container{padding-top:1.1rem;padding-bottom:3.5rem;max-width:1400px;}
[data-testid="stSidebar"]{background:#0E1116;border-right:1px solid var(--hair);}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarUserContent"]{padding-top:.6rem;}
.stApp p,.stApp li,.stApp label,[data-testid="stWidgetLabel"] p{font-family:var(--sans);}
[data-testid="stWidgetLabel"] p{color:var(--muted);font-size:.8rem;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--muted) !important;font-size:.8rem;line-height:1.55;}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:var(--serif);font-weight:400;letter-spacing:0;color:var(--text);}
/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:30px;border-bottom:1px solid var(--hair);}
.stTabs [data-baseweb="tab"]{height:46px;padding:0;background:transparent !important;color:var(--muted);}
.stTabs [data-baseweb="tab"] p{font-family:var(--sans);font-size:.9rem;font-weight:500;}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--text);}
.stTabs [data-baseweb="tab"]:hover{color:var(--text);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--gold);height:1px;}
.stTabs [data-baseweb="tab-border"]{display:none;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.5rem;}
/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:3px;border:1px solid var(--hair);background:transparent;color:var(--text);
font-family:var(--sans);font-weight:500;font-size:.86rem;transition:border-color .15s ease,color .15s ease,background .15s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--gold);color:var(--gold);background:transparent;}
.stDownloadButton>button[kind="primary"],.stDownloadButton>button[data-testid="stBaseButton-primary"]{background:var(--gold);border-color:var(--gold);color:#0B0D10;font-weight:600;}
.stDownloadButton>button[kind="primary"]:hover,.stDownloadButton>button[data-testid="stBaseButton-primary"]:hover{background:#E3C55C;border-color:#E3C55C;color:#0B0D10;}
.stApp button:focus-visible,.stApp input:focus-visible,.stApp [role="tab"]:focus-visible{outline:2px solid var(--gold);outline-offset:2px;}
/* inputs */
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="select"]>div,[data-testid="stNumberInputContainer"]{background:var(--card) !important;border-color:var(--hair) !important;border-radius:3px !important;}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{font-family:var(--mono);font-size:.86rem;color:var(--text);}
[data-testid="stNumberInput"] button{background:transparent;color:var(--muted);}
.stMultiSelect [data-baseweb="tag"]{background-color:rgba(212,175,55,.10) !important;border:1px solid rgba(212,175,55,.38);border-radius:2px !important;color:var(--text) !important;}
.stMultiSelect [data-baseweb="tag"] span,.stMultiSelect [data-baseweb="tag"] svg{color:var(--text) !important;fill:var(--text) !important;font-family:var(--mono);font-size:.74rem;}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:var(--raised) !important;border:1px solid var(--hair);}
[data-testid="stExpander"] details{border:1px solid var(--hair) !important;border-radius:3px;background:transparent;}
[data-testid="stExpander"] summary p{font-size:.86rem;color:var(--text);}
[data-testid="stVerticalBlockBorderWrapper"],[data-testid="stLayoutWrapper"]>[data-testid="stVerticalBlock"][class*="border"]{border-color:var(--hair) !important;border-radius:4px !important;}
div[data-testid="stVerticalBlock"]:has(>div>.cardmark){background:var(--card);}
[data-testid="stAlert"]{background:var(--card);border:1px solid var(--hair);border-radius:3px;}
hr{border-color:var(--hair) !important;}
/* figures */
.num,.stat-v,.tbl td,.kpi-v,.pulse-v,.range .num{font-family:var(--mono);font-variant-numeric:tabular-nums;}
/* masthead */
.mast{display:flex;justify-content:space-between;align-items:baseline;gap:24px;flex-wrap:wrap;padding:2px 0 14px;}
.mast-brand{font-family:var(--serif);font-size:1.35rem;color:var(--text);}
.mast-line{font-family:var(--serif);font-size:1.12rem;color:var(--muted);}
.mast-line em{color:var(--gold);font-style:italic;}
/* pulse strip */
.pulse{display:flex;flex-wrap:wrap;align-items:stretch;border:1px solid var(--hair);border-radius:4px;background:var(--card);}
.pulse-item{padding:11px 20px;border-right:1px solid var(--hair);min-width:150px;}
.pulse-l{font-size:.7rem;color:var(--muted);letter-spacing:.02em;}
.pulse-v{font-size:1rem;color:var(--text);margin-top:2px;}
.pulse-d{font-family:var(--mono);font-size:.74rem;margin-top:1px;}
.pos{color:var(--pos);} .neg{color:var(--neg);} .flat{color:var(--muted);}
.pulse-meta{margin-left:auto;padding:11px 18px;display:flex;flex-direction:column;justify-content:center;align-items:flex-end;gap:5px;}
.feed{display:inline-flex;align-items:center;gap:8px;font-size:.68rem;font-weight:600;letter-spacing:.09em;color:var(--text);}
.dot{width:7px;height:7px;border-radius:50%;background:var(--muted);display:inline-block;}
.dot.live{background:var(--buy);box-shadow:0 0 0 0 rgba(127,184,154,.6);animation:pulse 2s infinite;}
.dot.fall{background:var(--neutral);}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(127,184,154,.55);}70%{box-shadow:0 0 0 7px rgba(127,184,154,0);}100%{box-shadow:0 0 0 0 rgba(127,184,154,0);}}
.stamp{font-family:var(--mono);font-size:.7rem;color:var(--muted);}
/* marquee */
.marquee{overflow:hidden;border-bottom:1px solid var(--hair);padding:9px 0;margin-top:2px;-webkit-mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent);mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent);}
.marquee-track{display:inline-flex;white-space:nowrap;animation:scroll 60s linear infinite;}
.marquee:hover .marquee-track{animation-play-state:paused;}
.marquee-item{font-size:.78rem;color:var(--muted);padding:0 18px;}
.marquee-item b{font-weight:500;color:var(--text);}
.marquee-item i{font-style:normal;font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-left:6px;}
.marquee-sep{color:var(--gold);opacity:.7;}
@keyframes scroll{from{transform:translateX(0);}to{transform:translateX(-50%);}}
/* hero */
.hero-tags{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:26px 0 10px;}
.badge-t{font-family:var(--mono);font-size:.74rem;color:var(--gold);border:1px solid rgba(212,175,55,.4);border-radius:2px;padding:3px 9px;}
.tag{font-size:.74rem;color:var(--muted);border:1px solid var(--hair);border-radius:2px;padding:3px 9px;}
.hero-name{font-family:var(--serif);font-size:clamp(2.2rem,4.4vw,3.7rem);line-height:1.04;color:var(--text);margin:0;font-weight:400;}
.hero-sub{color:var(--muted);font-size:.84rem;margin:12px 0 14px;max-width:70ch;line-height:1.55;}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border:1px solid var(--hair);border-radius:4px;background:var(--card);margin-top:6px;}
.kpi{padding:14px 20px;border-right:1px solid var(--hair);min-width:0;}
.kpi:last-child{border-right:0;}
.kpi-l{font-size:.7rem;color:var(--muted);}
.kpi-v{font-size:1.4rem;color:var(--text);margin-top:5px;white-space:nowrap;}
.pill{display:inline-block;font-family:var(--sans);font-size:.74rem;font-weight:600;letter-spacing:.08em;padding:5px 13px;border-radius:999px;border:1px solid;margin-top:5px;}
.pill.buy{color:var(--buy);border-color:rgba(127,184,154,.5);background:rgba(127,184,154,.10);}
.pill.neutral{color:var(--neutral);border-color:rgba(217,164,65,.5);background:rgba(217,164,65,.10);}
.pill.sell{color:var(--sell);border-color:rgba(200,100,79,.55);background:rgba(200,100,79,.12);}
.pill.nr{color:var(--muted);border-color:var(--hair);}
.pill.sm{font-size:.66rem;padding:2px 9px;margin:0;letter-spacing:.05em;}
/* 52-week range */
.range{margin:26px 0 6px;}
.range-head{display:flex;justify-content:space-between;font-size:.76rem;color:var(--muted);margin-bottom:30px;}
.range-track{position:relative;height:2px;background:rgba(255,255,255,.14);border-radius:2px;}
.range-fill{position:absolute;left:0;top:0;height:2px;background:linear-gradient(90deg,rgba(212,175,55,.25),var(--gold));border-radius:2px;}
.range-pin{position:absolute;top:-5px;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:var(--gold);border:2px solid var(--obsidian);box-shadow:0 0 0 1px var(--gold);}
.range-pin span{position:absolute;bottom:17px;left:50%;transform:translateX(-50%);font-family:var(--mono);font-size:.76rem;color:var(--text);white-space:nowrap;}
.range-tp{position:absolute;top:-6px;width:1px;height:14px;background:var(--text);opacity:.7;}
.range-tp span{position:absolute;top:17px;left:50%;transform:translateX(-50%);font-size:.66rem;color:var(--muted);white-space:nowrap;}
.range-ends{display:flex;justify-content:space-between;font-size:.76rem;color:var(--muted);margin-top:12px;}
.range-ends .num{color:var(--text);}
/* stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border:1px solid var(--hair);border-radius:4px;background:var(--card);margin:4px 0 18px;}
.stat{padding:14px 18px;border-right:1px solid var(--hair);}
.stat-l{font-size:.72rem;color:var(--muted);}
.stat-v{font-size:1.22rem;color:var(--text);margin-top:5px;white-space:nowrap;}
.stat-s{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-top:3px;}
/* tables */
.tblwrap{border:1px solid var(--hair);border-radius:4px;overflow:auto;background:var(--card);margin:4px 0 14px;}
.tbl{width:100%;border-collapse:collapse;font-size:.8rem;}
.tbl th{position:sticky;top:0;background:var(--raised);color:var(--muted);font-family:var(--sans);font-weight:500;font-size:.72rem;text-align:right;padding:9px 14px;border-bottom:1px solid var(--hair);white-space:nowrap;}
.tbl td{padding:8px 14px;text-align:right;color:var(--text);border-bottom:1px solid rgba(255,255,255,.045);white-space:nowrap;font-size:.78rem;}
.tbl th:first-child,.tbl td:first-child,.tbl .l{text-align:left;}
.tbl td:first-child{font-family:var(--sans);color:var(--muted);font-size:.8rem;}
.tbl tr:last-child td{border-bottom:0;}
.tbl tr.b td{color:var(--text);font-weight:500;border-top:1px solid var(--hair);}
.tbl tr.b td:first-child{color:var(--text);}
.tbl tr.h td{background:rgba(212,175,55,.06);}
.tbl td.wrap{white-space:normal;font-family:var(--sans);color:var(--muted);text-align:left;}
.tbl td.t{font-family:var(--sans);color:var(--muted);text-align:left;font-size:.8rem;}
.bar{display:inline-block;height:6px;border-radius:0 3px 3px 0;background:__S1__;vertical-align:middle;margin-right:8px;}
/* prose */
.sec{font-family:var(--serif);font-size:1.5rem;color:var(--text);margin:18px 0 10px;font-weight:400;}
.sec em{color:var(--gold);}
.lead{font-size:.95rem;line-height:1.7;color:var(--text);max-width:74ch;}
.prose{font-size:.9rem;line-height:1.7;color:#D5D9E0;max-width:74ch;margin:0 0 12px;}
.note{font-size:.78rem;line-height:1.6;color:var(--muted);max-width:92ch;margin:6px 0 12px;}
.case{border:1px solid var(--hair);border-left:2px solid;border-radius:3px;background:var(--card);padding:14px 16px;font-size:.86rem;line-height:1.6;color:#D5D9E0;height:100%;}
.case b{display:block;font-size:.72rem;letter-spacing:.06em;margin-bottom:6px;}
.case.bull{border-left-color:var(--buy);} .case.bull b{color:var(--buy);}
.case.bear{border-left-color:var(--sell);} .case.bear b{color:var(--sell);}
.list{margin:0;padding-left:18px;color:#D5D9E0;font-size:.88rem;line-height:1.75;}
.empty{border:1px solid var(--hair);border-radius:4px;background:var(--card);padding:34px 30px;margin-top:28px;max-width:760px;}
.empty h3{font-family:var(--serif);font-size:1.7rem;margin:0 0 10px;font-weight:400;}
.empty p{color:var(--muted);font-size:.9rem;line-height:1.65;margin:0 0 8px;}
.side-brand{font-family:var(--serif);font-size:1.45rem;line-height:1.15;color:var(--text);margin:0 0 2px;}
.side-sub{font-size:.76rem;color:var(--muted);margin:0 0 6px;}
@media (max-width:900px){.kpis{grid-template-columns:repeat(2,1fr);}.kpi:nth-child(2){border-right:0;}.kpi:nth-child(-n+2){border-bottom:1px solid var(--hair);}.pulse-meta{margin-left:0;align-items:flex-start;}}
@media (prefers-reduced-motion:reduce){.marquee-track{animation:none;}.dot.live{animation:none;}}
</style>
"""
for _k, _v in {"__OBSIDIAN__": UI["obsidian"], "__CARD__": UI["card"], "__RAISED__": UI["raised"], "__HAIR__": UI["hairline"],
               "__TEXT__": UI["text"], "__MUTED__": UI["muted"], "__GOLD__": UI["gold"], "__BUY__": UI["buy"],
               "__NEUTRAL__": UI["neutral"], "__SELL__": UI["sell"], "__S1__": S1}.items():
    CSS = CSS.replace(_k, _v)
st.html(CSS)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
esc = html.escape


_RAW = "\x00"


def Raw(markup: str) -> str:
    """Marks a table cell as trusted HTML (built here, never from user input).
    A sentinel prefix is used because DataFrames do not preserve str subclasses."""
    return _RAW + markup


def H(markup: str) -> None:
    st.html(markup)


@contextmanager
def safe(section: str):
    """Render a section; on any error show a calm notice instead of a traceback."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        st.warning(f"{section} could not be rendered for this selection ({type(exc).__name__}: {str(exc)[:160]}). "
                   "Other sections are unaffected.")


def tone(x) -> str:
    if x is None or pd.isna(x) or abs(x) < 1e-12:
        return "flat"
    return "pos" if x > 0 else "neg"


def arrow(x) -> str:
    return "" if x is None or pd.isna(x) or abs(x) < 1e-12 else ("▲ " if x > 0 else "▼ ")


def stats(items: list[tuple]) -> None:
    """Row of figures: (label, value, sub-line, tone)."""
    cells = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else ""
        cls = it[3] if len(it) > 3 else ""
        cells.append(f'<div class="stat"><div class="stat-l">{esc(str(label))}</div><div class="stat-v">{esc(str(value))}</div>'
                     + (f'<div class="stat-s {cls}">{esc(str(sub))}</div>' if sub else "") + "</div>")
    H(f'<div class="stats">{"".join(cells)}</div>')


def tbl(df: pd.DataFrame, corner: str = "", bold: tuple = (), hl: tuple = (), max_h: int | None = None,
        wrap_cols: tuple = (), show_index: bool = True, header: bool = True, text_cols: tuple = ()) -> None:
    """Hairline table. Cells are pre-formatted strings; figures are set in the mono face."""
    head = (f"<th>{esc(corner)}</th>" if show_index else "") + "".join(
        f'<th class="l">{esc(str(c))}</th>' if c in text_cols or c in wrap_cols else f"<th>{esc(str(c))}</th>" for c in df.columns)
    rows = []
    for i, (idx, r) in enumerate(df.iterrows()):
        cls = " ".join(c for c, on in (("b", idx in bold or i in bold), ("h", idx in hl or i in hl)) if on)
        tds = f"<td>{esc(str(idx))}</td>" if show_index else ""
        for c, v in r.items():
            cell = v[1:] if isinstance(v, str) and v.startswith(_RAW) else esc(str(v))
            tds += f'<td class="wrap">{cell}</td>' if c in wrap_cols else (f'<td class="t">{cell}</td>' if c in text_cols else f"<td>{cell}</td>")
        rows.append(f'<tr class="{cls}">{tds}</tr>')
    style = f' style="max-height:{max_h}px"' if max_h else ""
    thead = f"<thead><tr>{head}</tr></thead>" if header else ""
    H(f'<div class="tblwrap"{style}><table class="tbl">{thead}<tbody>{"".join(rows)}</tbody></table></div>')


def money_table(t: pd.DataFrame, rows: dict) -> pd.DataFrame:
    """Transpose a fundamentals table into a display frame (years as columns)."""
    out = {}
    for label, (col, kind) in rows.items():
        s = t[col]
        if kind == "cr":
            out[label] = [indian_number(v, 0) if pd.notna(v) else "n.m." for v in s]
        elif kind == "pct":
            out[label] = [f"{v:.1f}%" if pd.notna(v) else "n.m." for v in s]
        elif kind == "x":
            out[label] = [f"{v:.2f}x" if pd.notna(v) else "n.m." for v in s]
        else:
            out[label] = [indian_number(v, 2) if pd.notna(v) else "n.m." for v in s]
    return pd.DataFrame(out, index=t.index).T


def section(title: str) -> None:
    H(f'<div class="sec">{title}</div>')


def note(text: str) -> None:
    H(f'<p class="note">{esc(text)}</p>')


def card():
    """Bordered surface for a chart or a control group."""
    c = st.container(border=True)
    c.html('<span class="cardmark"></span>')
    return c


def fig_style(fig: go.Figure, height: int = 360, title: str | None = None, unified: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=8, r=12, t=46 if title else 14, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=UI["muted"]),
        title=dict(text=title, font=dict(family="Inter, system-ui, sans-serif", size=14, color=UI["text"]), x=0, xanchor="left", y=0.97) if title else None,
        legend=dict(orientation="h", yanchor="top", y=-0.13, x=0, font=dict(color=UI["text"], size=12), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified" if unified else "closest",
        hoverlabel=dict(bgcolor=UI["raised"], bordercolor="rgba(255,255,255,0.18)",
                        font=dict(family="JetBrains Mono, monospace", size=12, color=UI["text"])),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.12)", tickcolor="rgba(255,255,255,0.12)", zeroline=False,
                     showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="dot",
                     spikecolor="rgba(212,175,55,0.55)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.055)", zerolinecolor="rgba(255,255,255,0.14)", linecolor="rgba(0,0,0,0)",
                     showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="dot",
                     spikecolor="rgba(212,175,55,0.35)", title_font=dict(size=11), automargin=True)
    fig.update_xaxes(automargin=True)
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def status_badge(status: str) -> str:
    as_of = dl.offline_manifest().get("as_of", "n/a")
    if status == dl.LIVE:
        return '<span class="feed"><span class="dot live"></span>LIVE NSE FEED</span>'
    if status == dl.FALLBACK:
        return f'<span class="feed" title="Live source unavailable; showing the offline cache as of {esc(as_of)}"><span class="dot fall"></span>LIVE UNAVAILABLE · OFFLINE DEMO CACHE</span>'
    return f'<span class="feed" title="Offline cache as of {esc(as_of)}"><span class="dot"></span>OFFLINE DEMO CACHE</span>'


def empty_state(title: str, lines: list[str]) -> None:
    H(f'<div class="empty"><h3>{esc(title)}</h3>{"".join(f"<p>{esc(x)}</p>" for x in lines)}</div>')


# --------------------------------------------------------------------------- #
# Data access (cached wrappers around the unchanged engines)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=1800, show_spinner=False)
def get_context(ticker: str, live: bool, params: tuple, peers: tuple, g0: float | None,
                custom_sector: str | None, guard: bool) -> dict:
    """Research context with two presentation-level safeguards applied through the
    engine's own parameters: outlier peers are dropped before medians are taken, and
    a non-positive intrinsic value is not blended into the target price."""
    if custom_sector:
        config.register_custom_ticker(ticker, custom_sector)
    p, pl, notes = dict(params), list(peers), {}

    def build(peer_list, prm):
        return va.build_research_context(ticker, live, prm, list(peer_list), g0)

    try:
        ctx = build(pl, p)
    except Exception:
        avail = [q for q in pl if config.offline_available(q)]
        if avail == pl:
            raise
        notes["peers_dropped"] = [q for q in pl if q not in avail]
        pl = avail
        ctx = build(pl, p)
    if guard:
        tab = ctx["peers"]["table"]
        out = [i for i in tab.index if i not in (ctx["ticker"], "MEDIAN") and pd.notna(tab.loc[i, "P/E"]) and tab.loc[i, "P/E"] > PEER_PE_CAP]
        keep = [q for q in ctx["peers_list"] if q not in out]
        if out and len(keep) >= 2:
            ctx = build(keep, p)
            notes["peer_outliers"] = out
    iv, pv_ = ctx["dcf"]["intrinsic"], ctx["peers"]["implied_price"]
    if pd.notna(iv) and iv <= 0 and pd.notna(pv_) and p.get("dcf_weight", 0) > 0:
        ctx = build(ctx["peers_list"], {**p, "dcf_weight": 0.0})
        notes["dcf_non_positive"] = True
    ctx["ui_notes"] = notes
    return ctx


@st.cache_data(ttl=3600, show_spinner=False)
def get_universe(live: bool) -> dict:
    """Prices and parsed fundamentals for the whole NSE basket (for the factor engine)."""
    tickers = config.all_tickers()

    def load(use_live: bool, tks: list[str]):
        bundles, s1 = dl.load_companies(tks, use_live)
        px, s2 = dl.load_prices(tuple(tks + [config.BENCHMARK]), 6.3, use_live)
        return bundles, px, dl.combine_status(s1, s2)

    bundles = None
    if live:
        try:
            bundles, px, status = load(True, tickers)
        except Exception:  # noqa: BLE001
            bundles = None
    if bundles is None or (live and status == dl.FALLBACK):
        tks = [t for t in tickers if config.offline_available(t)]
        bundles, px, _ = load(False, tks)
        status = dl.FALLBACK if live else dl.OFFLINE
    funds = {}
    for t, b in bundles.items():
        if t in px.columns and px[t].notna().sum() > 60:
            try:
                funds[t] = fa.analyze(b, config.sector_of(t), float(px[t].dropna().iloc[-1]))
            except Exception:  # noqa: BLE001 - skip names with unusable statements
                continue
    return {"funds": funds, "px": px, "status": status}


@st.cache_data(ttl=240, show_spinner=False)
def pulse_quotes(tickers: tuple, live: bool) -> dict:
    """Last and previous close for the pulse strip. Live quotes are re-fetched at most
    every four minutes; anything that fails falls back to the offline cache."""
    out = {"status": dl.OFFLINE, "quotes": {}, "as_of": None}
    if live and dl._live_allowed():
        try:
            close = dl._download_close(list(tickers), (date.today() - timedelta(days=12)).isoformat())
            for t in tickers:
                if t in close.columns:
                    s = close[t].dropna()
                    if len(s) >= 2:
                        out["quotes"][t] = (float(s.iloc[-1]), float(s.iloc[-2]))
            if config.BENCHMARK in out["quotes"]:
                out.update(status=dl.LIVE, as_of=close.index[-1])
                return out
        except Exception:  # noqa: BLE001
            pass
        out["quotes"] = {}
    eq = [t for t in tickers if "=" not in t]
    try:
        px, _ = dl.load_prices(tuple(eq), 0.25, False)
        fx, _ = dl.load_fx(False, 0.25)
        for frame in (px, fx):
            for t in tickers:
                if t in frame.columns:
                    s = frame[t].dropna()
                    if len(s) >= 2:
                        out["quotes"][t] = (float(s.iloc[-1]), float(s.iloc[-2]))
        out["as_of"] = px.index[-1] if len(px) else None
    except Exception:  # noqa: BLE001
        pass
    out["status"] = dl.FALLBACK if live else dl.OFFLINE
    return out


@st.cache_resource
def _pdf_store() -> dict:
    return {}


def pdf_provider(ctx: dict, nar: dict, key: str):
    """Deferred PDF builder: the note is rendered when a download button is clicked,
    then kept so the other buttons serve the same bytes."""
    store = _pdf_store()

    def make() -> bytes:
        if key not in store:
            if len(store) > 24:
                store.clear()
            store[key] = rg.build_report(ctx, nar)
        return store[key]
    return make


# --------------------------------------------------------------------------- #
# Sidebar: coverage, universal search, narrative engine, export
# --------------------------------------------------------------------------- #
ss = st.session_state
with st.sidebar:
    H('<div class="side-brand">India Equity Research</div><div class="side-sub">NSE equities, NIFTY 50 benchmark, INR macro</div>')
    mode = st.radio("Data mode", ["Offline Demo Mode", "Live Data (Yahoo Finance)"], index=0,
                    help="Live mode pulls from Yahoo Finance and falls back to the offline cache automatically "
                         "if the source is empty, failing or rate-limited.")
    live = mode.startswith("Live")
    if st.button("Retry live connection", disabled=not live, help="Clear cached results and try Yahoo Finance again.",
                 width="stretch"):
        dl.reset_breaker()
        st.cache_data.clear()

    groups = list(config.SECTOR_GROUPS)
    group = st.selectbox("Sector", groups, index=groups.index("IT Services"))
    picked = st.selectbox("NSE stock", config.group_tickers(group), format_func=config.display_label)
    query = st.text_input("Search any NSE ticker", placeholder="ETERNAL.NS, BEL.NS, TATAPOWER.NS",
                          help="With or without .NS. Old symbols resolve automatically "
                               "(ZOMATO → ETERNAL.NS, TATAMOTORS → TMPV.NS, LTIM → LTM.NS). Clear the box to return to the lists above.")
    ticker, custom_sector = picked, None
    if query.strip():
        try:
            found = config.resolve_ticker(query)
            ticker = found
            if config.in_universe(found):
                st.caption(f"Covered: {config.name_of(found)} · {config.group_of(config.sector_of(found))}")
            else:
                sectors = list(config.UNIVERSE)
                default_sector = config.SECTOR_GROUPS[group][0]
                custom_sector = st.selectbox("Benchmark peer sector", sectors, index=sectors.index(default_sector),
                                             key=f"bench_{found}",
                                             help="Sets the peer group, valuation model (bank or non-bank) and "
                                                  "sector commentary for a ticker outside the covered list.")
                config.register_custom_ticker(found, custom_sector)
                st.caption(f"{found} is outside the covered list; benchmarked against {custom_sector}.")
        except ValueError as exc:
            st.warning(str(exc))

    with st.expander("Narrative engine (optional LLM)"):
        provider = st.selectbox("Engine", ["Deterministic NLG (no key)", "Google Gemini", "Groq"])
        api_key = st.text_input("Your free API key", type="password", disabled=provider.startswith("Deterministic"),
                                help=f"Gemini uses {na.GEMINI_MODEL}; Groq uses {na.GROQ_MODEL}. The key is used only "
                                     "for this session's request and is never stored.")
    side_export = st.container()
    st.caption("Educational tool. Not investment advice.")

sector_now = config.sector_of(ticker)
is_custom = not config.in_universe(ticker)

# Valuation sandbox state (the controls themselves live in the Valuation tab).
rf, erp = ss.get("sb_rf", SANDBOX_DEFAULTS["sb_rf"]) / 100, ss.get("sb_erp", SANDBOX_DEFAULTS["sb_erp"]) / 100
tax, g = ss.get("sb_tax", SANDBOX_DEFAULTS["sb_tax"]) / 100, ss.get("sb_g", SANDBOX_DEFAULTS["sb_g"]) / 100
w_dcf = ss.get("sb_w", SANDBOX_DEFAULTS["sb_w"]) / 100
g0 = ss.get("sb_g0", SANDBOX_DEFAULTS["sb_g0"]) / 100 if ss.get("sb_use_g0", False) else None
guard = bool(ss.get("sb_guard", True))
peer_key = f"peers_{ticker}_{sector_now}"
peer_pool = [t for t in config.all_tickers() if t != ticker and (live or config.offline_available(t))]
default_peers = [p for p in config.peers_of(ticker) if p in peer_pool]
peers = [p for p in ss.get(peer_key, default_peers) if p in peer_pool]
params = (("rf", rf), ("erp", erp), ("tax", tax), ("g", g), ("dcf_weight", w_dcf))


# --------------------------------------------------------------------------- #
# Masthead, market pulse (auto-refreshing fragment) and coverage marquee
# --------------------------------------------------------------------------- #
H('<div class="mast"><span class="mast-brand">India Equity Research</span>'
  '<span class="mast-line">Institutional research built for <em>conviction</em>.</span></div>')


@st.fragment(run_every="5m")
def market_pulse(stock: str, use_live: bool) -> None:
    """Refreshes on its own every five minutes; the rest of the page (tabs, sliders,
    inputs) is not re-run, so nothing the user has set is disturbed."""
    q = pulse_quotes((config.BENCHMARK, "USDINR=X", "EURINR=X", stock), use_live)
    labels = [(config.BENCHMARK, "NIFTY 50", 2, ""), ("USDINR=X", "USD / INR", 4, ""), ("EURINR=X", "EUR / INR", 4, ""),
              (stock, stock.replace(".NS", ""), 2, "₹")]
    cells = []
    for tk, label, dec, unit in labels:
        if tk in q["quotes"]:
            last, prev = q["quotes"][tk]
            chg, pct = last - prev, (last / prev - 1) * 100 if prev else np.nan
            delta = f"{arrow(chg)}{'+' if chg >= 0 else '−'}{unit}{indian_number(abs(chg), 2)} ({pct:+.2f}%)"
            cells.append(f'<div class="pulse-item"><div class="pulse-l">{esc(label)}</div>'
                         f'<div class="pulse-v">{unit}{indian_number(last, dec)}</div><div class="pulse-d {tone(chg)}">{delta}</div></div>')
        else:
            cells.append(f'<div class="pulse-item"><div class="pulse-l">{esc(label)}</div><div class="pulse-v">—</div>'
                         f'<div class="pulse-d flat">no quote</div></div>')
    now = datetime.now(IST)
    stamp = f"Updated {now:%d %b %Y, %H:%M} IST"
    if q["as_of"] is not None:
        stamp += f" · last session {pd.Timestamp(q['as_of']):%d %b %Y}"
    H(f'<div class="pulse">{"".join(cells)}<div class="pulse-meta">{status_badge(q["status"])}<span class="stamp">{stamp}</span></div></div>')


market_pulse(ticker, live)

_items = "".join(f'<span class="marquee-item"><b>{esc(grp)}</b><i>{len(config.group_tickers(grp))}</i></span><span class="marquee-sep">·</span>'
                 for grp in config.SECTOR_GROUPS)
H(f'<div class="marquee" aria-label="Covered NSE sectors"><div class="marquee-track">{_items}{_items}</div></div>')


# --------------------------------------------------------------------------- #
# Load the research context
# --------------------------------------------------------------------------- #
if not live and not config.offline_available(ticker):
    empty_state(f"{ticker} is not in the offline cache",
                ["The offline demo cache holds the 52 covered stocks. Any other NSE ticker needs live data.",
                 "Switch Data mode to Live Data (Yahoo Finance) in the sidebar, or clear the search box to return to a covered stock."])
    st.stop()
try:
    with st.spinner("Loading market data and running models…"):
        ctx = get_context(ticker, live, params, tuple(peers), g0, custom_sector, guard)
except Exception as exc:  # noqa: BLE001
    if is_custom:
        empty_state(f"No usable data for {ticker}",
                    ["Yahoo Finance returned no price history or annual statements for this symbol, and it is not in the offline cache.",
                     "Check the NSE symbol (for example TATAPOWER.NS), or clear the search box to return to a covered stock."])
    else:
        empty_state(f"Analysis for {ticker} is unavailable right now",
                    [f"{type(exc).__name__}: {str(exc)[:200]}", "Switch to Offline Demo Mode or pick another stock."])
    st.stop()

f, d, pv, tr, fxa, mk = ctx["fund"], ctx["dcf"], ctx["peers"], ctx["rating"], ctx["fx"], ctx["market"]
t = f["table"]
last = t.iloc[-1]
ui_notes = ctx.get("ui_notes", {})

# Narrative (deterministic by default; optional LLM polish) — needed before the export buttons.
nar_det = na.build_narrative(ctx)
nar, nar_err = nar_det, None
if not provider.startswith("Deterministic") and api_key:
    ck = ("nar", ctx["ticker"], ctx["status"], params, tuple(ctx["peers_list"]), g0, provider)
    if ss.get("nar_key") != ck:
        with st.spinner(f"Polishing the narrative with {provider}…"):
            ss["nar_val"] = na.enhance_with_llm(nar_det, ctx, provider, api_key)
            ss["nar_key"] = ck
    nar, nar_err = ss["nar_val"]
pdf_key = hashlib.sha1(repr((ctx["ticker"], ctx["status"], str(ctx["as_of"]), params, tuple(ctx["peers_list"]), g0,
                             ctx["rating"]["target"], nar["engine"], nar["thesis"])).encode()).hexdigest()
make_pdf = pdf_provider(ctx, nar, pdf_key)
pdf_name = rg.report_filename(ctx)

with side_export:
    st.download_button("Download research note (PDF)", data=make_pdf, file_name=pdf_name, mime="application/pdf",
                       type="primary", key="dl_side", width="stretch",
                       help="Three-page institutional note for the selected stock and current assumptions.")

# --------------------------------------------------------------------------- #
# Hero: name, rating, target, price, upside, export, 52-week range
# --------------------------------------------------------------------------- #
grp_now = config.group_of(ctx["sector"])
tags = f'<span class="badge-t">NSE: {esc(ctx["ticker"])}</span><span class="tag">{esc(grp_now)}</span>'
if ctx["sector"] != grp_now:
    tags += f'<span class="tag">{esc(ctx["sector"])}</span>'
if is_custom:
    tags += '<span class="tag">Outside covered list</span>'
rating = tr["rating"]
pill_cls = {"BUY": "buy", "NEUTRAL": "neutral", "SELL": "sell"}.get(rating, "nr")
up = tr["upside"] * 100 if pd.notna(tr["upside"]) else np.nan

H(f'<div class="hero-tags">{tags}</div><h1 class="hero-name">{esc(ctx["name"])}</h1>'
  f'<p class="hero-sub">Price as of {ctx["as_of"]:%d %b %Y}. Target price is {esc(tr["basis"])}.</p>')
hc1, hc2 = st.columns([3.4, 1], gap="medium", vertical_alignment="center")
with hc1:
    H(f'<div class="kpis"><div class="kpi"><div class="kpi-l">Rating</div><span class="pill {pill_cls}">{esc(rating)}</span></div>'
      f'<div class="kpi"><div class="kpi-l">Target price</div><div class="kpi-v">{fmt_inr(tr["target"], 0)}</div></div>'
      f'<div class="kpi"><div class="kpi-l">Current price</div><div class="kpi-v">{fmt_inr(ctx["price"], 2)}</div></div>'
      f'<div class="kpi"><div class="kpi-l">Implied {"upside" if not pd.notna(up) or up >= 0 else "downside"}</div>'
      f'<div class="kpi-v {tone(up)}">{arrow(up)}{fmt_pct(up, 1, True)}</div></div></div>')
with hc2:
    st.download_button("Download research note (PDF)", data=make_pdf, file_name=pdf_name, mime="application/pdf",
                       type="primary", key="dl_hero", width="stretch")

lo, hi, cmp_ = mk["low_52w"], mk["high_52w"], ctx["price"]
pos = float(np.clip((cmp_ - lo) / (hi - lo), 0, 1)) * 100 if hi > lo else 50.0
tp_mark = ""
if pd.notna(tr["target"]) and lo <= tr["target"] <= hi:
    tp_mark = f'<div class="range-tp" style="left:{(tr["target"] - lo) / (hi - lo) * 100:.2f}%"><span>target {fmt_inr(tr["target"], 0)}</span></div>'
H(f'<div class="range"><div class="range-head"><span>52-week price range</span>'
  f'<span>{(cmp_ / lo - 1) * 100:.1f}% above low, {(1 - cmp_ / hi) * 100:.1f}% below high</span></div>'
  f'<div class="range-track"><div class="range-fill" style="width:{pos:.2f}%"></div>{tp_mark}'
  f'<div class="range-pin" style="left:{pos:.2f}%"><span>{fmt_inr(cmp_, 2)}</span></div></div>'
  f'<div class="range-ends"><span>52W low <span class="num">{fmt_inr(lo, 2)}</span></span>'
  f'<span>52W high <span class="num">{fmt_inr(hi, 2)}</span></span></div></div>')

stats([("Intrinsic value", fmt_inr(d["intrinsic"], 0) if pd.notna(d["intrinsic"]) and d["intrinsic"] > 0 else "n.m.",
        fmt_pct(d["upside"] * 100, 1, True) + " vs price" if pd.notna(d["upside"]) and d["intrinsic"] > 0 else d["method"], tone(d["upside"])),
       ("Peer-implied value", fmt_inr(pv["implied_price"], 0),
        fmt_pct(pv["upside"] * 100, 1, True) + " vs price" if pd.notna(pv["upside"]) else "no peer basis", tone(pv["upside"])),
       ("Market cap", fmt_cr_compact(mk["mcap"]), fmt_cr(mk["mcap"])),
       ("P/E, latest FY", fmt_x(mk["pe"]), f"P/B {fmt_x(mk['pb'])}"),
       ("RoE", fmt_pct(mk["roe"]), f"{d['rate_label']} {d['coc']['wacc'] * 100:.1f}%"),
       ("1Y return", fmt_pct(mk["ret_1y"] * 100, 1, True), f"NIFTY 50 {mk['bench_ret_1y'] * 100:+.1f}%", tone(mk["ret_1y"]))])
if ctx["info"].get("statementFxNote"):
    note(ctx["info"]["statementFxNote"])
if ctx["status"] == dl.FALLBACK:
    note("Live data was unavailable, so this page is running on the offline demo cache. Use Retry live connection in the sidebar to try again.")
if ui_notes.get("dcf_non_positive"):
    note("The DCF equity value is not positive (net debt exceeds the discounted enterprise value), so the target price rests on peer-relative value only.")

tabs = st.tabs(["Overview", "Fundamentals & Forensics", "Valuation", "INR FX Macro",
                "Factor Screener & Backtest", "Research Report (PDF)"])

# --------------------------------------------------------------------------- #
# Tab 1 — Overview
# --------------------------------------------------------------------------- #
with tabs[0], safe("Overview"):
    px = ctx["prices"]
    stock, bench = px[ctx["ticker"]].dropna(), px[config.BENCHMARK].dropna()
    c1, c2 = st.columns([1.9, 1], gap="large")
    with c1, card():
        tf = st.segmented_control("Timeframe", list(TIMEFRAMES), default="1Y", key="tf", label_visibility="collapsed") or "1Y"
        k2, k3, k4, _ = st.columns([1, 1, 1.5, 1.2], vertical_alignment="center")
        show50 = k2.toggle("50-DMA", value=True, key="ma50")
        show200 = k3.toggle("200-DMA", value=True, key="ma200")
        overlay = k4.toggle("NIFTY 50 overlay", value=False, key="ovl", help="Rebases the stock and the NIFTY 50 to 100 at the start of the window.")
        start = stock.index[0] if TIMEFRAMES[tf] is None else stock.index[-1] - pd.Timedelta(days=TIMEFRAMES[tf])
        win = stock[stock.index >= start]
        ma50, ma200 = stock.rolling(50).mean(), stock.rolling(200).mean()
        base = float(win.iloc[0])
        scale = (lambda s: s / base * 100) if overlay else (lambda s: s)
        unit = "" if overlay else "₹"
        fig = go.Figure()
        fig.add_scatter(x=win.index, y=scale(win), name=ctx["ticker"], line=dict(color=S1, width=2),
                        hovertemplate=f"{unit}%{{y:,.2f}}<extra>{ctx['ticker']}</extra>")
        if overlay:
            b = bench[bench.index >= win.index[0]]
            fig.add_scatter(x=b.index, y=b / float(b.iloc[0]) * 100, name="NIFTY 50", line=dict(color=CH["reference"], width=1.6),
                            hovertemplate="%{y:,.2f}<extra>NIFTY 50</extra>")
        if show50 and ma50.notna().any():
            m = ma50[ma50.index >= start].dropna()
            fig.add_scatter(x=m.index, y=scale(m), name="50-DMA", line=dict(color=S2, width=1.3),
                            hovertemplate=f"{unit}%{{y:,.2f}}<extra>50-DMA</extra>")
        if show200 and ma200.notna().any():
            m = ma200[ma200.index >= start].dropna()
            fig.add_scatter(x=m.index, y=scale(m), name="200-DMA", line=dict(color=S3, width=1.3),
                            hovertemplate=f"{unit}%{{y:,.2f}}<extra>200-DMA</extra>")
        fig.update_yaxes(title="Rebased to 100" if overlay else "Price (₹)", tickprefix="" if overlay else "₹", tickformat=",.0f")
        chg = (win.iloc[-1] / win.iloc[0] - 1) * 100
        bwin = bench[bench.index >= win.index[0]]
        bchg = (bwin.iloc[-1] / bwin.iloc[0] - 1) * 100 if len(bwin) > 1 else np.nan
        H(f'<p class="note" style="margin:2px 0 0"><span class="num" style="color:{UI["text"]};font-size:1.05rem">{fmt_inr(win.iloc[-1], 2)}</span>'
          f'&nbsp;&nbsp;<span class="num {tone(chg)}">{arrow(chg)}{chg:+.1f}%</span> over {esc(tf.lower() if tf == "MAX" else tf)}'
          f'&nbsp;&nbsp;·&nbsp;&nbsp;NIFTY 50 <span class="num {tone(bchg)}">{bchg:+.1f}%</span></p>')
        plot(fig_style(fig, 400))
    with c2:
        section("Market and valuation data")
        snap = pd.DataFrame({"": {
            "52W high / low": f'{fmt_inr(mk["high_52w"], 0)} / {fmt_inr(mk["low_52w"], 0)}',
            "Beta vs NIFTY 50": f'{ctx["beta"]["adjusted"]:.2f} (raw {fmt_num(ctx["beta"]["raw"])})',
            "Shares outstanding": f'{indian_number(mk["shares_cr"], 1)} Cr',
            "Free float": fmt_pct(mk["free_float"] * 100, 0),
            d["rate_label"]: fmt_pct(d["coc"]["wacc"] * 100, 2),
            "P/E, latest FY": fmt_x(mk["pe"]), "P/B": fmt_x(mk["pb"]),
            "Implied EV/EBITDA (DCF)": fmt_x(d["implied_ev_ebitda"]),
            "RoE": fmt_pct(mk["roe"]), "Dividend yield": fmt_pct(mk["div_yield"] * 100),
            "USD/INR beta (2Y)": fmt_num(fxa.get("fx_beta")),
        }})
        tbl(snap, header=False)
    section("Investment view")
    H(f'<p class="lead">{esc(nar["thesis"][2])}</p>')

# --------------------------------------------------------------------------- #
# Tab 2 — Fundamentals & Forensics
# --------------------------------------------------------------------------- #
with tabs[1], safe("Fundamentals"):
    span = f["cagr"]["span"]
    stats([(f"{span}Y revenue CAGR", fmt_pct(f["cagr"]["Revenue CAGR"] * 100)),
           (f"{span}Y EBITDA CAGR", fmt_pct(f["cagr"]["EBITDA CAGR"] * 100) if not f["financial"] else "n.m."),
           (f"{span}Y PAT CAGR", fmt_pct(f["cagr"]["PAT CAGR"] * 100)),
           ("RoE", fmt_pct(last["ROE %"])), ("RoIC" if not f["financial"] else "RoA", fmt_pct(last["ROIC %"]) if not f["financial"] else fmt_pct(last["ROA %"], 2)),
           ("FCF conversion", fmt_pct(last["FCF Conversion %"], 0) if not f["financial"] else "n.m.")])
    if f["financial"]:
        note("Bank: EBITDA, FCF-based ratios, net debt and RoIC are not meaningful and show as n.m.; "
             "revenue is net total income and pre-tax profit is the operating-profit line.")
    section("Financial statements, ₹ Cr")
    tbl(money_table(t, {
        "Revenue": ("Revenue", "cr"), "EBITDA": ("EBITDA", "cr"), "EBIT": ("EBIT", "cr"), "PAT": ("PAT", "cr"),
        "Cash flow from operations": ("CFO", "cr"), "CapEx": ("CapEx", "cr"), "Free cash flow": ("FCF", "cr"),
        "Net debt (net cash if negative)": ("Net Debt", "cr"), "Invested capital": ("Invested Capital", "cr"),
        "Net worth": ("Equity", "cr"), "EPS (₹)": ("EPS", "num"), "Book value per share (₹)": ("BVPS", "num"),
    }), bold=("Revenue", "PAT"))
    section("Growth, margins and returns")
    tbl(money_table(t, {
        "Revenue growth": ("Revenue Growth %", "pct"), "EBITDA margin": ("EBITDA Margin %", "pct"),
        "PAT margin": ("PAT Margin %", "pct"), "FCF conversion (FCF/PAT)": ("FCF Conversion %", "pct"),
        "RoE": ("ROE %", "pct"), "RoIC (NOPAT / avg invested capital)": ("ROIC %", "pct"), "RoA": ("ROA %", "pct"),
        "Net debt / EBITDA": ("Net Debt/EBITDA", "x"), "Debt / equity": ("Debt/Equity", "x"),
    }))
    c1, c2 = st.columns(2, gap="large")
    with c1, card():
        line_col, line_name = ("ROE %", "RoE") if f["financial"] else ("EBITDA Margin %", "EBITDA margin")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.09)
        fig.add_bar(x=t.index, y=t["Revenue"], name="Revenue", marker=dict(color=S1, cornerradius=4), width=0.5,
                    hovertemplate="₹%{y:,.0f} Cr<extra>Revenue</extra>", row=1, col=1)
        fig.add_scatter(x=t.index, y=t[line_col], name=line_name, mode="lines+markers", line=dict(color=S2, width=2),
                        marker=dict(size=8, line=dict(color=UI["card"], width=2)),
                        hovertemplate="%{y:.1f}%<extra>" + line_name + "</extra>", row=2, col=1)
        fig.update_yaxes(title_text="Revenue, ₹ Cr", tickformat=",.0f", row=1, col=1)
        fig.update_yaxes(title_text=f"{line_name}, %", tickformat=".1f", row=2, col=1)
        fig.update_layout(showlegend=False, bargap=0.4)
        plot(fig_style(fig, 400, f"Revenue and {line_name}"))
    with c2, card():
        fig = go.Figure()
        series = [("PAT", "PAT", S1), ("CFO", "Operating cash flow", S2)] + ([] if f["financial"] else [("FCF", "Free cash flow", S3)])
        for col, name, colr in series:
            fig.add_bar(x=t.index, y=t[col], name=name, marker=dict(color=colr, cornerradius=3, line=dict(color=UI["card"], width=1)),
                        hovertemplate="₹%{y:,.0f} Cr<extra>" + name + "</extra>")
        fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
        fig.update_yaxes(title="₹ Cr", tickformat=",.0f")
        plot(fig_style(fig, 400, "Earnings against cash generation"))

    section("Five-step DuPont decomposition of RoE")
    dp = f["dupont"]
    show = pd.DataFrame({yr: ["n.m." if pd.isna(v) else (f"{v * 100:.1f}%" if i in (2, 5) else f"{v:.2f}x")
                              for i, v in enumerate(dp.loc[yr])] for yr in dp.index},
                        index=[c.replace("ROE (product)", "Return on equity") for c in dp.columns])
    tbl(show, bold=("Return on equity",))
    note("RoE = tax burden × interest burden × EBIT margin × asset turnover × financial leverage. " + nar["dupont"])

    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        fs = f["piotroski"]
        if fs:
            section(f"Piotroski F-Score <em>{fs['score']} / {fs['evaluated']}</em>")
            rows = pd.DataFrame([{"Criterion": c["criterion"],
                                  "Result": Raw(f'<span class="pill sm {"buy" if c["passed"] else ("nr" if not c["available"] else "sell")}">'
                                                f'{"Pass" if c["passed"] else ("n.a." if not c["available"] else "Fail")}</span>'),
                                  "Evidence": c["detail"]} for c in fs["criteria"]]).set_index("Criterion")
            tbl(rows, corner=f"{fs['year']}", wrap_cols=("Evidence",))
            if fs["note"]:
                note(fs["note"])
    with c2:
        z = f["altman"]
        if z["applicable"] and pd.notna(z["z"]):
            section(f"Altman Z-Score <em>{z['z']:.2f}</em>, {esc(z['zone'].lower())} zone")
        else:
            section("Altman Z-Score, not applicable")
        zrows = {k_: fmt_num(v, 3) for k_, v in z["components"].items()}
        if z["applicable"]:
            zrows["Z''-EM (book-value variant)"] = fmt_num(z["z_em"])
        tbl(pd.DataFrame({"Value": zrows}), header=False)
        note(z["note"])

# --------------------------------------------------------------------------- #
# Tab 3 — Valuation sandbox
# --------------------------------------------------------------------------- #
def _reset_sandbox(extra_key: str) -> None:
    for k_ in list(SANDBOX_DEFAULTS) + [extra_key]:
        st.session_state.pop(k_, None)


with tabs[2], safe("Valuation"):
    coc = d["coc"]
    with card():
        H('<div class="sec" style="margin-top:2px">Valuation sandbox</div>')
        s1, s2, s3, s4 = st.columns(4)
        s1.number_input("Risk-free rate, 10Y G-Sec (%)", 3.0, 12.0, SANDBOX_DEFAULTS["sb_rf"], 0.05, key="sb_rf")
        s2.number_input("India equity risk premium (%)", 3.0, 12.0, SANDBOX_DEFAULTS["sb_erp"], 0.25, key="sb_erp")
        s3.number_input("Corporate tax rate, Sec 115BAA (%)", 10.0, 40.0, SANDBOX_DEFAULTS["sb_tax"], 0.01, key="sb_tax")
        s4.number_input("Terminal growth (%)", 2.0, 7.0, SANDBOX_DEFAULTS["sb_g"], 0.25, key="sb_g")
        s1, s2, s3, s4 = st.columns([1.3, 1.3, 1, 0.8], vertical_alignment="bottom")
        s1.slider("Target price weight on intrinsic value (%)", 0, 100, SANDBOX_DEFAULTS["sb_w"], 5, key="sb_w",
                  help="The remainder is weighted on the NSE peer-relative value.")
        with s2:
            use_g0 = st.checkbox("Override year-1 growth", key="sb_use_g0")
            st.slider("Year-1 growth (%)", 6.0, 22.0, SANDBOX_DEFAULTS["sb_g0"], 0.5, key="sb_g0", disabled=not use_g0,
                      label_visibility="collapsed")
        s3.toggle("Drop outlier peers", value=True, key="sb_guard",
                  help=f"Leaves peers trading above {PEER_PE_CAP:.0f}× earnings out of the medians, as long as two peers remain.")
        s4.button("Reset", on_click=_reset_sandbox, args=(peer_key,), width="stretch", help="Return every input to the India defaults.")
        st.multiselect("Comparison peers", peer_pool, default=default_peers, key=peer_key,
                       format_func=lambda x: x.replace(".NS", ""),
                       help="Pre-populated with the closest NSE sector peers. Any covered stock can be added.")
    if ui_notes.get("peer_outliers"):
        note("Left out of the peer medians as outliers (P/E above "
             f"{PEER_PE_CAP:.0f}×): {', '.join(x.replace('.NS', '') for x in ui_notes['peer_outliers'])}.")
    if ui_notes.get("peers_dropped"):
        note("No offline data for: " + ", ".join(x.replace(".NS", "") for x in ui_notes["peers_dropped"]) + ".")

    stats([("Target price", fmt_inr(tr["target"], 0), tr["rating"]),
           ("Implied upside" if not pd.notna(up) or up >= 0 else "Implied downside", f"{arrow(up)}{fmt_pct(up, 1, True)}", f"vs {fmt_inr(ctx['price'], 0)}", tone(up)),
           ("Intrinsic value", fmt_inr(d["intrinsic"], 0), d["method"]),
           ("Peer-implied value", fmt_inr(pv["implied_price"], 0), ", ".join(pv["implied"].keys()) or "n.a."),
           (d["rate_label"], fmt_pct(coc["wacc"] * 100, 2), f"beta {coc['beta']:.2f}"),
           ("Year-1 growth", fmt_pct(d["g0"] * 100), f"historical {d['g0_raw'] * 100:.1f}%"),
           ("Terminal value share", fmt_pct(d["tv_share"] * 100, 0) if pd.notna(d["tv_share"]) else "n.a.",
            f"E/D {coc['we'] * 100:.0f}/{coc['wd'] * 100:.0f}")])
    if not d["applicable"]:
        st.warning("An intrinsic value could not be computed (no positive cash-flow base or discount rate too close to growth).")

    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        section("Five-year projection")
        pr = d["projection"]
        disp = {}
        for c in pr.columns:
            if "%" in c:
                disp[c] = [f"{v:.1f}%" for v in pr[c]]
            elif c == "Discount Factor":
                disp["Discount factor"] = [f"{v:.3f}" for v in pr[c]]
            else:
                disp[f"{c}, ₹ Cr"] = [indian_number(v, 0) for v in pr[c]]
        tbl(pd.DataFrame(disp, index=pr.index).T)
        note(f"Base: {d['base_method']}.")
    with c2:
        section("Valuation bridge")
        br = {lab: (indian_number(v, 1) if "shares" in lab.lower() else fmt_cr(v)) for lab, v in d["bridge"]}
        br["Intrinsic value per share"] = fmt_inr(d["intrinsic"], 0)
        br["Upside / downside vs price"] = fmt_pct(d["upside"] * 100, 1, True)
        tbl(pd.DataFrame({"": br}), bold=("Enterprise value", "Equity value", "Intrinsic value per share"), header=False)

    section(f"Sensitivity of value per share to {esc(d['rate_label'])} and terminal growth")
    with card():
        sm = d["sens"]
        zv = sm.values.astype(float) / ctx["price"] - 1
        lim = float(np.nanmax(np.abs(zv))) if np.isfinite(zv).any() else 1.0
        text = [[fmt_inr(v, 0) for v in row] for row in sm.values.astype(float)]
        fig = go.Figure(go.Heatmap(
            z=zv, x=[f"g {c * 100:.1f}%" for c in sm.columns], y=[f"{r * 100:.1f}%" for r in sm.index], text=text,
            texttemplate="%{text}", textfont=dict(family="JetBrains Mono, monospace", size=13, color=UI["text"]),
            colorscale=[[0, CH["negative"]], [0.5, CH["midpoint"]], [1, CH["positive"]]], zmin=-lim, zmax=lim, xgap=2, ygap=2,
            colorbar=dict(title=dict(text="vs price", font=dict(size=11)), tickformat="+.0%", thickness=10, outlinewidth=0, len=0.9),
            hovertemplate="Rate %{y}, %{x}<br>%{text} per share<br>%{z:+.1%} vs price<extra></extra>"))
        fig.add_shape(type="rect", x0=1.5, x1=2.5, y0=1.5, y1=2.5, line=dict(color=UI["gold"], width=2))
        fig.update_yaxes(autorange="reversed", title=d["rate_label"])
        fig = fig_style(fig, 340, unified=False)
        fig.update_xaxes(showspikes=False, side="top")
        fig.update_yaxes(showspikes=False, showgrid=False)
        plot(fig)
    note("Base case outlined in gold. Cell colour shows the gap to the current market price: green above, red below, grey near parity.")

    section("NSE sector peer comparison")
    pt = pv["table"]
    disp = pd.DataFrame({
        "MCap, ₹ Cr": [indian_number(v, 0) if i != "MEDIAN" else "—" for i, v in pt["MCap (₹ Cr)"].items()],
        "P/E": pt["P/E"].map(fmt_x), "Fwd P/E": pt["Fwd P/E"].map(fmt_x), "EV/EBITDA": pt["EV/EBITDA"].map(fmt_x),
        "P/B": pt["P/B"].map(fmt_x), "RoE": pt["RoE %"].map(fmt_pct), "3Y PAT CAGR": pt["3Y PAT CAGR %"].map(fmt_pct)})
    disp.index = [r["Company"] if i == "MEDIAN" else f'{r["Company"]} ({i.replace(".NS", "")})' for i, r in pt.iterrows()]
    tbl(disp, corner="Company", bold=(len(disp) - 1,), hl=(0,))
    imp = "; ".join(f"{k_} route {fmt_inr(v, 0)}" for k_, v in pv["implied"].items()) or "n.a."
    note(f"Implied price from peer medians: {imp}. Average {fmt_inr(pv['implied_price'], 0)}. "
         f"Target price {fmt_inr(tr['target'], 0)} is {tr['basis']}. The first row is the subject company; the median covers peers only.")

# --------------------------------------------------------------------------- #
# Tab 4 — INR FX Macro
# --------------------------------------------------------------------------- #
with tabs[3], safe("INR FX macro"):
    names = {"USDINR=X": "USD / INR", "EURINR=X": "EUR / INR", "GBPINR=X": "GBP / INR"}
    items = [(names.get(k_, k_), f"{v['last']:.2f}", f"{v['chg_1y'] * 100:+.1f}% over 1Y") for k_, v in fxa["pairs"].items()]
    items += [("FX beta vs USD/INR", fmt_num(fxa.get("fx_beta")), "2Y daily returns"),
              ("NIFTY-neutral FX beta", fmt_num(fxa.get("fx_beta_neutral")), f"t {fmt_num(fxa.get('t_stat'), 1)}, p {fmt_num(fxa.get('p_value'), 3)}"),
              ("90-day correlation", fmt_num(fxa.get("corr_90d")), "stock vs USD/INR")]
    stats(items)
    c1, c2 = st.columns(2, gap="large")
    with c1, card():
        fxp = ctx["fx_prices"]
        reb = fxp / fxp.iloc[0] * 100
        fig = go.Figure()
        for c, colr in zip(reb.columns, [S1, S2, S3]):
            fig.add_scatter(x=reb.index, y=reb[c], name=names.get(c, c), line=dict(color=colr, width=2),
                            hovertemplate="%{y:.1f}<extra>" + names.get(c, c) + "</extra>")
        fig.update_yaxes(title="Rebased to 100 (higher = weaker rupee)")
        plot(fig_style(fig, 400, "The rupee against the dollar, euro and pound, two years"))
    with c2, card():
        usd = fxa["usd_inr"]
        cs = fxa.get("corr_series", pd.Series(dtype=float))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42], vertical_spacing=0.09)
        fig.add_scatter(x=usd.index, y=usd.values, name="USD/INR", line=dict(color=S1, width=2),
                        hovertemplate="₹%{y:.2f}<extra>USD/INR</extra>", row=1, col=1)
        if len(cs):
            fig.add_scatter(x=cs.index, y=cs.values, name="90-day correlation", line=dict(color=S2, width=2),
                            hovertemplate="%{y:.2f}<extra>90-day correlation</extra>", row=2, col=1)
        fig.update_yaxes(title_text="USD/INR", row=1, col=1)
        fig.update_yaxes(title_text="Correlation", range=[-1, 1], row=2, col=1)
        fig.update_layout(showlegend=False)
        plot(fig_style(fig, 400, f"USD/INR, and its 90-day correlation with {ctx['ticker']}"))
    prior, why = fxa["sector_prior"]
    section(f"A weaker rupee is <em>{esc({'tailwind': 'a tailwind', 'headwind': 'a headwind', 'mixed': 'mixed', 'neutral': 'broadly neutral'}[prior])}</em> for {esc(ctx['sector'])}")
    H(f'<p class="prose">{esc(nar["fx"])}</p>')
    note("FX beta is the slope of daily stock returns on USD/INR returns over two years. The NIFTY-neutral beta adds "
         "the NIFTY 50 as a control, separating currency sensitivity from general risk-off selling.")

# --------------------------------------------------------------------------- #
# Tab 5 — Factor Screener & Backtest
# --------------------------------------------------------------------------- #
with tabs[4], safe("Factor engine"):
    section("NIFTY multi-factor screener")
    with card():
        c = st.columns([1, 1, 0.8, 0.9, 0.8, 1.1], vertical_alignment="bottom")
        w_val = c[0].slider("Value weight (%)", 0, 100, 40, 5)
        w_qual = c[1].slider("Quality weight (%)", 0, 100 - w_val, min(30, 100 - w_val), 5) if w_val < 100 else 0
        w_mom = 100 - w_val - w_qual
        c[2].markdown(f"<div style='font-size:.8rem;color:{UI['muted']}'>Momentum weight (%)</div>"
                      f"<div class='num' style='font-size:1.3rem;padding:2px 0 8px'>{w_mom}</div>", unsafe_allow_html=True)
        top_n = c[3].selectbox("Portfolio size", [5, 10], format_func=lambda n: f"Top {n}")
        yrs = c[4].selectbox("Horizon", [3, 5], format_func=lambda y: f"{y}Y")
        scheme = c[5].selectbox("Weighting", ["equal", "inverse_vol"],
                                format_func=lambda s_: {"equal": "Equal weight", "inverse_vol": "Inverse volatility"}[s_])
    weights = {"value": w_val, "quality": w_qual, "momentum": w_mom}

    with st.spinner("Loading the NSE basket…"):
        uni = get_universe(live)
    scr = bt.screen(uni["funds"], uni["px"], weights)
    cmax = float(scr["Composite"].abs().max()) or 1.0
    cols = ["Value", "Quality", "Momentum", "Earnings Yield %", "FCF Yield %", "Book/Price", "ROE %", "F-Score", "Debt/Equity",
            "Mom 12-1 %", "6M Risk-Adj"]
    rows = []
    for tk, r in scr.iterrows():
        row = {"Sector": config.group_of(r["Sector"]),
               "Composite": Raw(f'<span class="bar" style="width:{max(r["Composite"], 0) / cmax * 54:.0f}px"></span>{r["Composite"]:+.2f}')}
        row.update({c_: ("n.m." if pd.isna(r[c_]) else f"{r[c_]:.2f}") for c_ in cols})
        rows.append(pd.Series(row, name=f'{int(r["Rank"])}  {tk.replace(".NS", "")}'))
    tbl(pd.DataFrame(rows), corner="Rank and stock", max_h=372, hl=tuple(range(top_n)), text_cols=("Sector",))
    H(f'<p class="note">{status_badge(uni["status"])}&nbsp;&nbsp; {len(scr)} stocks screened. Value is earnings yield, FCF yield and book-to-price; '
      'Quality is RoE, Piotroski F-Score and low debt/equity; Momentum is the 12-1 month return and the 6-month risk-adjusted return. '
      f'Scores are cross-sectional z-scores; the top {top_n} are shaded.</p>')

    res = bt.run_backtest(uni["px"], uni["funds"], weights, top_n, yrs, scheme, config.TXN_COST, rf)
    mt = res["metrics"]
    section(f"Backtest: top {top_n}, rebalanced quarterly, {res['start']:%b %Y} to {res['end']:%b %Y}")
    ex = mt["Portfolio CAGR %"] - mt["Benchmark CAGR %"]
    stats([("Portfolio CAGR", fmt_pct(mt["Portfolio CAGR %"]), f"{arrow(ex)}{ex:+.1f} pp vs NIFTY 50", tone(ex)),
           ("NIFTY 50 CAGR", fmt_pct(mt["Benchmark CAGR %"]), "price index"),
           ("Annualised volatility", fmt_pct(mt["Annualized Volatility %"]), f"NIFTY 50 {mt['Benchmark Volatility %']:.1f}%"),
           ("Sharpe ratio", fmt_num(mt["Sharpe Ratio"]), f"Rf {rf * 100:.2f}%"),
           ("Sortino ratio", fmt_num(mt["Sortino Ratio"]), "downside deviation")])
    stats([("Jensen's alpha", fmt_pct(mt["Jensen's Alpha %"], 2), "annualised", tone(mt["Jensen's Alpha %"])),
           ("Beta", fmt_num(mt["Beta"]), "vs NIFTY 50"),
           ("Max drawdown", fmt_pct(mt["Max Drawdown %"]), f"NIFTY 50 {mt['Benchmark Max Drawdown %']:.1f}%"),
           ("Win rate", fmt_pct(mt["Win Rate %"], 0), "months ahead of NIFTY 50"),
           ("Information ratio", fmt_num(mt["Information Ratio"]), f"tracking error {mt['Tracking Error %']:.1f}%")])
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1, card():
        fig = go.Figure()
        fig.add_scatter(x=res["nav"].index, y=res["nav"] * 100, name="Factor portfolio, net of costs", line=dict(color=S1, width=2),
                        hovertemplate="₹%{y:.1f}<extra>Factor portfolio</extra>")
        fig.add_scatter(x=res["bench"].index, y=res["bench"] * 100, name="NIFTY 50", line=dict(color=CH["reference"], width=1.6),
                        hovertemplate="₹%{y:.1f}<extra>NIFTY 50</extra>")
        fig.update_yaxes(title="Growth of ₹100")
        plot(fig_style(fig, 380, "Cumulative performance"))
    with c2, card():
        fig = go.Figure()
        fig.add_scatter(x=res["drawdown"].index, y=res["drawdown"] * 100, name="Factor portfolio", line=dict(color=S1, width=2),
                        fill="tozeroy", fillcolor="rgba(176,138,38,0.14)", hovertemplate="%{y:.1f}%<extra>Factor portfolio</extra>")
        fig.add_scatter(x=res["bench_drawdown"].index, y=res["bench_drawdown"] * 100, name="NIFTY 50",
                        line=dict(color=CH["reference"], width=1.6), hovertemplate="%{y:.1f}%<extra>NIFTY 50</extra>")
        fig.update_yaxes(title="Drawdown, %")
        plot(fig_style(fig, 380, "Drawdown from peak"))
    with st.expander("Rebalance log, turnover and costs"):
        lg = res["rebalances"].copy()
        lg["One-way Turnover %"] = lg["One-way Turnover %"].map("{:.1f}".format)
        lg["Cost (bps)"] = lg["Cost (bps)"].map("{:.1f}".format)
        lg["Top Score"] = lg["Top Score"].map("{:.2f}".format)
        lg["Rebalance Date"] = lg["Rebalance Date"].astype(str)
        tbl(lg.set_index("Rebalance Date"), corner="Rebalance date", max_h=340, wrap_cols=("Holdings",))
        note(f"{res['n_rebalances']} rebalances at {config.TXN_COST * 100:.2f}% round-trip cost on turnover. Total cost drag "
             f"{res['total_cost_pct']:.2f}% over the period; average one-way turnover {fmt_num(res['avg_turnover_pct'], 1)}% per rebalance.")
    notes_ = []
    if res["lookahead_rebalances"]:
        notes_.append(f"{res['lookahead_rebalances']} of {res['n_rebalances']} rebalances pre-date the first available fiscal "
                      "year (free sources carry 4–5 years), so the earliest statement was used for Value and Quality there; "
                      "Momentum is fully point-in-time.")
    if uni["status"] != dl.LIVE:
        notes_.append("Offline demo prices are simulated daily paths anchored to periodic market levels, so this backtest "
                      "demonstrates the engine rather than evidencing a factor premium. Switch to Live Data for real history.")
    notes_.append("The basket is today's large-cap list applied to the past (survivorship bias); results are before taxes.")
    note(" ".join(notes_))

# --------------------------------------------------------------------------- #
# Tab 6 — Research report
# --------------------------------------------------------------------------- #
with tabs[5], safe("Research report"):
    if nar_err:
        st.info(nar_err)
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        section("Investment thesis")
        H("".join(f'<p class="prose">{esc(p_)}</p>' for p_ in nar["thesis"]))
        H(f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px">'
          f'<div class="case bull"><b>BULL CASE</b>{esc(nar["bull"])}</div><div class="case bear"><b>BEAR CASE</b>{esc(nar["bear"])}</div></div>')
    with c2:
        section("Key catalysts")
        H('<ul class="list">' + "".join(f"<li>{esc(c_)}</li>" for c_ in nar["catalysts"]) + "</ul>")
        section("Key risks")
        H('<ul class="list">' + "".join(f"<li>{esc(r_)}</li>" for r_ in nar["risks"]) + "</ul>")
    st.divider()
    section("Institutional research note, three pages")
    note("Page 1: executive summary and financial snapshot. Page 2: valuation bridge, charts, sensitivity and peers. "
         f"Page 3: DuPont, forensic quality, INR macro, risks and disclosures. Narrative engine: {nar['engine']}.")
    st.download_button("Download research note (PDF)", data=make_pdf, file_name=pdf_name, mime="application/pdf",
                       type="primary", key="dl_tab")
