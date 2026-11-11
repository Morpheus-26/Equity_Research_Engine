"""
report_generator.py — three-page institutional equity research PDF (ReportLab + Matplotlib).

A4 portrait, 0.5-inch margins, Helvetica typography, navy/slate institutional palette.
No OS-level dependencies: charts are rendered by Matplotlib (Agg) at 300 DPI and the
rupee glyph comes from the DejaVu Sans font bundled with Matplotlib. Each page body is
wrapped in a shrink-to-fit frame, so long company names or LLM-written prose can never
overflow, overlap or spill onto a fourth page.
"""
from __future__ import annotations

import io
import os
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepInFrame, PageBreak, PageTemplate,  # noqa: E402
                                Paragraph, Spacer, Table, TableStyle)

import config  # noqa: E402
from config import fmt_cr, fmt_inr, fmt_num, fmt_pct, fmt_x, indian_number  # noqa: E402

NAVY, SLATE, GRAY, TINT = (colors.HexColor(c) for c in (config.NAVY, config.SLATE, config.GRAY, config.TINT))
GREEN, RED, AMBER = (colors.HexColor(c) for c in (config.GREEN, config.RED, config.AMBER))
INK, RULE = colors.HexColor("#1A202C"), colors.HexColor("#CBD5E0")
PAGE_W, PAGE_H = A4
MARGIN = 36.0                      # 0.5 inch
BANNER_H, FOOTER_H = 22.0, 16.0
BODY_W = PAGE_W - 2 * MARGIN
BODY_H = PAGE_H - 2 * MARGIN - BANNER_H - FOOTER_H - 14
RATING_COLOR = {"BUY": GREEN, "NEUTRAL": AMBER, "SELL": RED}
TOTAL_PAGES = 3

# ---- Unicode fallback font (₹, →, −) from Matplotlib's bundled DejaVu Sans -------
_UNI = _UNI_B = None
try:
    _ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(_ttf, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(_ttf, "DejaVuSans-Bold.ttf")))
    _UNI, _UNI_B = "DejaVuSans", "DejaVuSans-Bold"
except Exception:  # noqa: BLE001 - fall back to ASCII substitutes
    pass
_ASCII = {"₹": "Rs ", "→": "to", "−": "-", "≥": ">=", "≤": "<=", "″": '"', "✓": "Y", "✗": "N"}


def mk(text, bold: bool = False) -> str:
    """Escape text for a Paragraph and route non-WinAnsi glyphs to the Unicode font."""
    out = []
    for ch in escape(str(text)):
        try:
            ch.encode("cp1252")
            out.append(ch)
        except UnicodeEncodeError:
            if _UNI:
                out.append(f'<font name="{_UNI_B if bold else _UNI}">{ch}</font>')
            else:
                out.append(_ASCII.get(ch, "?"))
    return "".join(out)


def _style(name, size=8.0, bold=False, color=INK, align=TA_LEFT, leading=None, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                          leading=leading or size * 1.28, textColor=color, alignment=align, **kw)


ST = {
    "body": _style("body", 8.0, leading=10.4, spaceAfter=4.5),
    "thesis": _style("thesis", 8.9, leading=11.9, spaceAfter=6.5),
    "bullet_l": _style("bullet_l", 8.7, leading=11.4, leftIndent=9, firstLineIndent=-9, spaceAfter=3.2),
    "case": _style("case", 7.9, leading=10.2),
    "small": _style("small", 6.6, color=GRAY, leading=8.2),
    "note": _style("note", 7.2, color=GRAY, leading=9.2),
    "title": _style("title", 15.5, bold=True, color=NAVY, leading=18),
    "sub": _style("sub", 8.2, color=GRAY, leading=10.5),
    "h": _style("h", 8.6, bold=True, color=NAVY, leading=11),
    "bullet": _style("bullet", 7.8, leading=10.0, leftIndent=9, firstLineIndent=-9, spaceAfter=2.2),
    "cl": _style("cl", 7.2, leading=9.0),
    "cr": _style("cr", 7.2, align=TA_RIGHT, leading=9.0),
    "cc": _style("cc", 7.2, align=TA_CENTER, leading=9.0),
    "clb": _style("clb", 7.2, bold=True, leading=9.0),
    "crb": _style("crb", 7.2, bold=True, align=TA_RIGHT, leading=9.0),
    "ccb": _style("ccb", 7.2, bold=True, align=TA_CENTER, leading=9.0),
    "hl": _style("hl", 7.0, bold=True, color=colors.white, leading=8.8),
    "hr": _style("hr", 7.0, bold=True, color=colors.white, align=TA_RIGHT, leading=8.8),
    "hc": _style("hc", 7.0, bold=True, color=colors.white, align=TA_CENTER, leading=8.8),
    "kl": _style("kl", 6.1, color=GRAY, align=TA_CENTER, leading=8),
    "kv": _style("kv", 11.5, bold=True, align=TA_CENTER, leading=14),
}


def P(text, style="body", bold=None, color=None) -> Paragraph:
    st = ST[style]
    is_bold = st.fontName.endswith("Bold") if bold is None else bold
    body = mk(text, is_bold)
    if bold and not st.fontName.endswith("Bold"):
        body = f"<b>{body}</b>"
    if color is not None:
        body = f'<font color="{color.hexval() if hasattr(color, "hexval") else color}">{body}</font>'
    return Paragraph(body, st)


def heading(text: str, width: float) -> Table:
    t = Table([[P(text.upper(), "h")]], colWidths=[width])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.0, NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 0)]))
    return t


def grid(rows, widths, header=True, bold_rows=(), tint_rows=(), first_left=True, pad=2.2, extra=()) -> Table:
    """Striped institutional table. ``rows`` is a list of lists of strings."""
    data = []
    for i, r in enumerate(rows):
        line = []
        for j, c in enumerate(r):
            left = first_left and j == 0
            if header and i == 0:
                line.append(P(c, "hl" if left else "hr"))
            elif i in bold_rows:
                line.append(P(c, "clb" if left else "crb"))
            else:
                line.append(P(c, "cl" if left else "cr"))
        data.append(line)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), pad),
            ("BOTTOMPADDING", (0, 0), (-1, -1), pad), ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE)]
    start = 1 if header else 0
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for i in range(start, len(rows)):
        if (i - start) % 2 == 1:
            cmds.append(("BACKGROUND", (0, i), (-1, i), TINT))
    for i in tint_rows:
        cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#E3ECF6")))
    for i in bold_rows:
        cmds.append(("LINEABOVE", (0, i), (-1, i), 0.6, NAVY))
    t.setStyle(TableStyle(cmds + list(extra)))
    return t


# --------------------------------------------------------------------------- #
# Charts (Matplotlib, 300 DPI)
# --------------------------------------------------------------------------- #
def _mpl_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#A0AEC0")
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(labelsize=5.8, colors=config.GRAY, length=2, width=0.5)


def _image(fig, w: float, h: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w, height=h)


def chart_revenue_margin(ctx: dict, w: float, h: float) -> Image:
    t, pr, fin = ctx["fund"]["table"], ctx["projection"], ctx["fund"]["financial"]
    labels = list(t.index) + list(pr.index)
    rev = list(t["Revenue"].astype(float)) + list(pr["Revenue"].astype(float))
    unit, div = ("₹ lakh Cr", 1e5) if np.nanmax(rev) >= 1e5 else ("₹ '000 Cr", 1e3)
    line_name = "RoE %" if fin else "EBITDA margin %"
    line = list(t["ROE %"].astype(float)) + list(pr["ROE %"].astype(float)) if fin else \
        list(t["EBITDA Margin %"].astype(float)) + list(pr["EBITDA Margin %"].astype(float))
    fig, ax = plt.subplots(figsize=(w / 72, h / 72))
    x = np.arange(len(labels))
    cols = [config.NAVY] * len(t) + ["#8FB3D9"] * len(pr)
    ax.bar(x, np.array(rev) / div, color=cols, width=0.62, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Revenue ({unit})", fontsize=5.8, color=config.GRAY)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.5, zorder=0)
    _mpl_style(ax)
    ax2 = ax.twinx()
    ax2.plot(x, line, color="#C05621", marker="o", markersize=2.6, linewidth=1.1, zorder=3)
    for xi, v in zip(x, line):
        if np.isfinite(v):
            ax2.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 5), ha="center",
                         fontsize=5.2, color="#9C4221", zorder=5,
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    finite = [v for v in line if np.isfinite(v)]
    if finite:
        pad = max((max(finite) - min(finite)) * 0.6, 2.0)
        ax2.set_ylim(min(finite) - pad, max(finite) + pad)
    ax2.set_ylabel(line_name, fontsize=5.8, color="#9C4221")
    _mpl_style(ax2)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#A0AEC0")
    ax2.spines["right"].set_linewidth(0.6)
    ax.set_title(f"Revenue (₹ Cr) & {line_name.replace(' %', '')} trajectory  |  light bars = projected",
                 fontsize=6.6, color=config.NAVY, fontweight="bold", loc="left", pad=5)
    fig.tight_layout(pad=0.5)
    return _image(fig, w, h)


def chart_relative_performance(ctx: dict, w: float, h: float) -> Image:
    px = ctx["prices"]
    s, b = px[ctx["ticker"]].dropna(), px[config.BENCHMARK].dropna()
    start = s.index[-1] - pd.Timedelta(days=365)
    df = pd.concat([s, b], axis=1, keys=["s", "b"]).dropna()
    df = df[df.index >= start]
    df = df / df.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(w / 72, h / 72))
    ax.plot(df.index, df["s"], color=config.NAVY, linewidth=1.15, label=f"{ctx['ticker']} ({df['s'].iloc[-1] - 100:+.1f}%)")
    ax.plot(df.index, df["b"], color="#A0AEC0", linewidth=1.0, label=f"NIFTY 50 ({df['b'].iloc[-1] - 100:+.1f}%)")
    ax.axhline(100, color="#CBD5E0", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
    _mpl_style(ax)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b-%y"))
    ax.set_ylabel("Rebased to 100", fontsize=5.8, color=config.GRAY)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, handlelength=1.6)
    ax.set_title("1-year price performance vs NIFTY 50 (^NSEI), rebased", fontsize=6.6, color=config.NAVY,
                 fontweight="bold", loc="left", pad=5)
    fig.tight_layout(pad=0.5)
    return _image(fig, w, h)


def chart_fx(ctx: dict, w: float, h: float) -> Image:
    fxa = ctx["fx"]
    usd = fxa["usd_inr"].dropna()
    fig, ax = plt.subplots(figsize=(w / 72, h / 72))
    ax.plot(usd.index, usd.values, color=config.NAVY, linewidth=1.0)
    ax.set_ylabel("USD/INR", fontsize=5.8, color=config.NAVY)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
    _mpl_style(ax)
    corr = fxa.get("corr_series", pd.Series(dtype=float))
    if len(corr):
        ax2 = ax.twinx()
        ax2.plot(corr.index, corr.values, color="#C05621", linewidth=0.8)
        ax2.axhline(0, color="#C05621", linewidth=0.4, linestyle=":")
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("90D rolling correlation", fontsize=5.8, color="#9C4221")
        _mpl_style(ax2)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b-%y"))
    ax.set_title(f"USD/INR and 90-day correlation with {ctx['ticker']}", fontsize=6.6, color=config.NAVY,
                 fontweight="bold", loc="left", pad=5)
    fig.tight_layout(pad=0.5)
    return _image(fig, w, h)


# --------------------------------------------------------------------------- #
# Page builders
# --------------------------------------------------------------------------- #
def _n(x, d=0):
    return "n.m." if x is None or pd.isna(x) else indian_number(x, d)


def _title_block(ctx: dict) -> Table:
    tr = ctx["rating"]
    up_col = GREEN if (tr["upside"] or 0) >= 0 else RED
    kpis = Table(
        [[P("CMP", "kl"), P("TARGET PRICE", "kl"), P("UPSIDE/DOWNSIDE", "kl"), P("RATING", "kl")],
         [P(fmt_inr(ctx["price"], 0), "kv"), P(fmt_inr(tr["target"], 0), "kv", color=NAVY),
          P(fmt_pct((tr["upside"] or np.nan) * 100, 1, True), "kv", color=up_col),
          P(tr["rating"], "kv", color=colors.white)]],
        colWidths=[BODY_W * 0.10, BODY_W * 0.115, BODY_W * 0.135, BODY_W * 0.11], rowHeights=[11, 24])
    kpis.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 1),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 1), ("TOPPADDING", (0, 0), (-1, -1), 1),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                              ("BACKGROUND", (3, 1), (3, 1), RATING_COLOR.get(tr["rating"], GRAY)),
                              ("BOX", (0, 0), (2, 1), 0.5, RULE), ("LINEAFTER", (0, 0), (1, 1), 0.5, RULE)]))
    left = [P(ctx["name"], "title"),
            P(f"NSE: {ctx['ticker']}   |   {ctx['sector']}   |   Price as of {ctx['as_of']:%d %b %Y}", "sub")]
    t = Table([[left, kpis]], colWidths=[BODY_W * 0.54, BODY_W * 0.46])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("LINEBELOW", (0, 0), (-1, -1), 1.2, SLATE)]))
    return t


def _case_box(title: str, text: str, color, width: float) -> Table:
    t = Table([[P(title, "hl")], [P(text, "case")]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), color), ("BACKGROUND", (0, 1), (-1, 1), TINT),
                           ("BOX", (0, 0), (-1, -1), 0.5, color), ("LEFTPADDING", (0, 0), (-1, -1), 5),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def _fin_summary(ctx: dict) -> Table:
    t, pr = ctx["fund"]["table"].iloc[-4:], ctx["projection"]
    cols = list(t.index) + list(pr.index)
    def row(label, key, f):
        return [label] + [f(t[key].get(c)) for c in t.index] + [f(pr[key].get(c)) if key in pr else "n.m." for c in pr.index]
    pct = lambda v: "n.m." if v is None or pd.isna(v) else f"{v:.1f}%"  # noqa: E731
    mult = lambda v: "n.m." if v is None or pd.isna(v) else f"{v:.1f}x"  # noqa: E731
    if ctx["fund"]["financial"]:
        rows = [["₹ Cr (FY end as labelled)"] + cols,
                row("Net total income", "Revenue", _n), row("YoY growth %", "Revenue Growth %", pct),
                row("PAT", "PAT", _n), row("PAT margin %", "PAT Margin %", pct), row("EPS (₹)", "EPS", lambda v: _n(v, 1)),
                row("Net worth", "Equity", _n), row("Book value per share (₹)", "BVPS", lambda v: _n(v, 1)),
                row("RoE %", "ROE %", pct)]
        lw = BODY_W * 0.25
        cw = (BODY_W - lw) / len(cols)
        n_hist = len(t.index)
        extra = [("BACKGROUND", (1 + n_hist, 0), (-1, 0), SLATE),
                 ("LINEBEFORE", (1 + n_hist, 0), (1 + n_hist, -1), 0.8, SLATE)]
        return grid(rows, [lw] + [cw] * len(cols), bold_rows=(1, 3), extra=extra, pad=4.6)
    rows = [["₹ Cr (FY end as labelled)"] + cols,
            row("Revenue", "Revenue", _n), row("YoY growth %", "Revenue Growth %", pct),
            row("EBITDA", "EBITDA", _n), row("EBITDA margin %", "EBITDA Margin %", pct),
            row("PAT", "PAT", _n), row("PAT margin %", "PAT Margin %", pct), row("EPS (₹)", "EPS", lambda v: _n(v, 1)),
            row("FCF", "FCF", _n), row("RoE %", "ROE %", pct), row("Net debt (net cash if negative)", "Net Debt", _n),
            row("Net debt / EBITDA", "Net Debt/EBITDA", mult)]
    lw = BODY_W * 0.25
    cw = (BODY_W - lw) / len(cols)
    n_hist = len(t.index)
    extra = [("BACKGROUND", (1 + n_hist, 0), (-1, 0), SLATE),
             ("LINEBEFORE", (1 + n_hist, 0), (1 + n_hist, -1), 0.8, SLATE)]
    return grid(rows, [lw] + [cw] * len(cols), bold_rows=(1, 5), extra=extra, pad=4.6)


def _page1(ctx: dict, nar: dict) -> list:
    lw, rw = BODY_W * 0.65, BODY_W * 0.35
    inner_l, inner_r = lw - 10, rw
    left = [heading("Investment Thesis", inner_l), Spacer(1, 4)]
    left += [P(p, "thesis") for p in nar["thesis"]]
    half = (inner_l - 6) / 2
    cases = Table([[_case_box("BULL CASE", nar["bull"], GREEN, half), _case_box("BEAR CASE", nar["bear"], RED, half)]],
                  colWidths=[half + 3, half + 3])
    cases.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (0, 0), 6), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    left += [cases, heading("Key Catalysts", inner_l), Spacer(1, 4)]
    left += [P(f"• {c}", "bullet_l") for c in nar["catalysts"]]

    m, d, f = ctx["market"], ctx["dcf"], ctx["fund"]
    last = f["table"].iloc[-1]
    fxb = ctx["fx"].get("fx_beta")
    data = [["Market cap", fmt_cr(m["mcap"])], ["52W high / low", f"{fmt_inr(m['high_52w'], 0)} / {fmt_inr(m['low_52w'], 0)}"],
            ["Beta (vs NIFTY 50)", fmt_num(ctx["beta"]["adjusted"])],
            ["Shares / free float", f"{_n(m['shares_cr'], 1)} Cr / {fmt_pct(m['free_float'] * 100, 0)}"],
            [d["rate_label"], fmt_pct(d["coc"]["wacc"] * 100)],
            ["Implied EV/EBITDA (DCF)", fmt_x(d["implied_ev_ebitda"])],
            ["P/E (latest FY)", fmt_x(m["pe"])], ["P/B", fmt_x(m["pb"])], ["RoE", fmt_pct(m["roe"])],
            ["Dividend yield", fmt_pct(m["div_yield"] * 100)], ["USD/INR beta (2Y)", fmt_num(fxb)],
            ["1Y return vs NIFTY 50", f"{fmt_pct(m['ret_1y'] * 100, 1, True)} / {fmt_pct(m['bench_ret_1y'] * 100, 1, True)}"]]
    box = grid([["Key market & valuation data", ""]] + data, [inner_r * 0.56, inner_r * 0.44], pad=3.6)
    fs, z = f["piotroski"], f["altman"]
    q2 = [["RoA", fmt_pct(last["ROA %"], 2)], ["Assets / equity", fmt_x(last["Total Assets"] / last["Equity"])]] if f["financial"] else \
        [["RoIC", fmt_pct(last["ROIC %"])], ["FCF conversion (FCF/PAT)", fmt_pct(last["FCF Conversion %"], 0)]]
    qual = [["Quality scorecard", ""], *q2,
            ["Piotroski F-Score", f"{fs['score']} / {fs['evaluated']}" if fs else "n.a."],
            ["Altman Z-Score", f"{z['z']:.2f} ({z['zone']})" if z["applicable"] and pd.notna(z["z"]) else "n.a. (bank)"],
            [f"{f['cagr']['span']}Y revenue / PAT CAGR",
             f"{fmt_pct(f['cagr']['Revenue CAGR'] * 100)} / {fmt_pct(f['cagr']['PAT CAGR'] * 100)}"]]
    right = [box, Spacer(1, 10), grid(qual, [inner_r * 0.56, inner_r * 0.44], pad=3.6), Spacer(1, 6),
             P(f"Target price basis: {ctx['rating']['basis']}. Rating scale: BUY above +15% upside, NEUTRAL −10% to +15%, "
               f"SELL below −10%.", "small")]
    two = Table([[left, right]], colWidths=[lw, rw])
    two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (0, 0), 10), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                             ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    out = [_title_block(ctx), Spacer(1, 10), two, Spacer(1, 14),
           heading("Financial Summary & Projections (₹ Cr)", BODY_W), Spacer(1, 5), _fin_summary(ctx)]
    foot = "E = model projection consistent with the valuation growth path."
    if f["financial"]:
        foot += " Banks are shown on net total income, net worth and book value; EBITDA, FCF and net debt are not meaningful."
    out += [Spacer(1, 3), P(foot, "small")]
    return out


def _sens_table(ctx: dict, width: float) -> Table:
    d, price = ctx["dcf"], ctx["price"]
    s = d["sens"]
    head = [f"{'Ke' if d['financial'] else 'WACC'} \\ g"] + [f"{g * 100:.1f}%" for g in s.columns]
    rows = [head] + [[f"{r * 100:.1f}%"] + [fmt_inr(v, 0) for v in s.loc[r]] for r in s.index]
    data = []
    for i, r in enumerate(rows):
        line = []
        for j, c in enumerate(r):
            if i == 0:
                line.append(P(c, "hc"))
            elif j == 0:
                line.append(P(c, "ccb"))
            elif i == 3 and j == 3:
                line.append(P(c, "ccb", color=colors.white))
            else:
                v = s.iloc[i - 1, j - 1]
                line.append(P(c, "cc", color=(GREEN if pd.notna(v) and v >= price else RED)))
        data.append(line)
    fw = width * 0.22
    t = Table(data, colWidths=[fw] + [(width - fw) / 5] * 5)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("BACKGROUND", (0, 1), (0, -1), TINT),
                           ("BACKGROUND", (3, 3), (3, 3), SLATE), ("GRID", (0, 0), (-1, -1), 0.25, RULE),
                           ("BOX", (0, 0), (-1, -1), 0.5, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return t


def _page2(ctx: dict, nar: dict) -> list:
    d, pv, tr, p = ctx["dcf"], ctx["peers"], ctx["rating"], ctx["params"]
    coc = d["coc"]
    cw = (BODY_W - 10) / 2
    ch = cw * 0.70
    charts = Table([[chart_revenue_margin(ctx, cw, ch), chart_relative_performance(ctx, cw, ch)]], colWidths=[cw + 5, cw + 5])
    charts.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), 10),
                                ("RIGHTPADDING", (1, 0), (1, 0), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    lw, rw = BODY_W * 0.47, BODY_W * 0.50
    a = [["Assumptions & valuation bridge", ""], ["Method", d["method"]],
         ["Risk-free rate (10Y G-Sec)", fmt_pct(p["rf"] * 100, 2)], ["Equity risk premium", fmt_pct(p["erp"] * 100, 2)],
         ["Beta (Blume-adjusted)", fmt_num(coc["beta"])], ["Cost of equity (CAPM)", fmt_pct(coc["ke"] * 100, 2)]]
    if not d["financial"]:
        a += [["Post-tax cost of debt", fmt_pct(coc["kd_post"] * 100, 2)],
              ["Equity / debt weight", f"{coc['we'] * 100:.0f}% / {coc['wd'] * 100:.0f}%"],
              ["WACC", fmt_pct(coc["wacc"] * 100, 2)], ["Normalised base FCF", fmt_cr(d["base_fcf"])]]
    a += [["Growth: year 1 → terminal", f"{d['g0'] * 100:.1f}% → {d['g'] * 100:.1f}%"]]
    n_assump = len(a)
    for label, v in d["bridge"]:
        a.append([label, f"{_n(v, 1)}" if "shares" in label.lower() else fmt_cr(v)])
    a += [["Intrinsic value per share", fmt_inr(d["intrinsic"], 0)],
          ["Upside / downside vs CMP", fmt_pct(d["upside"] * 100, 1, True)]]
    bold = tuple(i for i, r in enumerate(a) if r[0] in ("Enterprise value", "Equity value", "Intrinsic value per share", "WACC"))
    bridge = grid(a, [lw * 0.56, lw * 0.44], bold_rows=bold, tint_rows=(len(a) - 2,),
                  extra=[("LINEABOVE", (0, n_assump), (-1, n_assump), 0.8, SLATE)])
    tp = [["Target price derivation", "₹ / share", "Weight"],
          [d["method"].split(" (")[0], fmt_inr(d["intrinsic"], 0), f"{p['dcf_weight'] * 100:.0f}%"],
          [f"Peer-relative ({', '.join(pv['implied'].keys()) or 'n.a.'})", fmt_inr(pv["implied_price"], 0), f"{(1 - p['dcf_weight']) * 100:.0f}%"],
          ["Target price", fmt_inr(tr["target"], 0), tr["rating"]],
          ["Current market price", fmt_inr(ctx["price"], 0), fmt_pct(tr["upside"] * 100, 1, True)]]
    right = [heading(f"Sensitivity: {'Ke' if d['financial'] else 'WACC'} vs Terminal Growth (₹ / share)", rw), Spacer(1, 4), _sens_table(ctx, rw),
             Spacer(1, 3), P("Base case highlighted. Green = above current price, red = below.", "small"), Spacer(1, 8),
             grid(tp, [rw * 0.56, rw * 0.22, rw * 0.22], bold_rows=(3,))]
    left = [heading("DCF Valuation Bridge & Assumptions" if not d["financial"] else "Excess-Return Valuation & Assumptions", lw),
            Spacer(1, 4), bridge]
    mid = Table([[left, right]], colWidths=[lw + BODY_W * 0.03, rw])
    mid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (0, 0), BODY_W * 0.03), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                             ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    pt = pv["table"]
    rows = [["Company", "MCap (₹ Cr)", "P/E", "EV/EBITDA", "P/B", "RoE %", "3Y PAT CAGR"]]
    for idx, r in pt.iterrows():
        nm = r["Company"] if idx == "MEDIAN" else f"{r['Company']} ({idx})"
        rows.append([nm, "—" if idx == "MEDIAN" else _n(r["MCap (₹ Cr)"]), fmt_x(r["P/E"]), fmt_x(r["EV/EBITDA"]), fmt_x(r["P/B"]),
                     fmt_pct(r["RoE %"]), fmt_pct(r["3Y PAT CAGR %"])])
    widths = [BODY_W * 0.34] + [BODY_W * 0.11] * 6
    peer = grid(rows, widths, bold_rows=(len(rows) - 1,), tint_rows=(1,))
    return [charts, Spacer(1, 10), mid, Spacer(1, 10), heading("NSE Sector Peer Comparison", BODY_W), Spacer(1, 4), peer,
            Spacer(1, 3), P("Multiples use the latest reported fiscal year and current prices; the first row is the "
                            "subject company; the median is taken across peers only.", "small"),
            Spacer(1, 8), heading("Valuation Commentary", BODY_W), Spacer(1, 4), P(nar["valuation"], "body")]


def _page3(ctx: dict, nar: dict) -> list:
    f, fxa = ctx["fund"], ctx["fx"]
    dp = f["dupont"]
    yrs = list(dp.index)
    fm = {0: lambda v: f"{v:.2f}x", 1: lambda v: f"{v:.2f}x", 2: lambda v: f"{v * 100:.1f}%", 3: lambda v: f"{v:.2f}x",
          4: lambda v: f"{v:.2f}x", 5: lambda v: f"{v * 100:.1f}%"}
    rows = [["Component"] + yrs]
    for i, c in enumerate(dp.columns):
        rows.append([c.replace("ROE (product)", "= Return on Equity")] + ["n.m." if pd.isna(v) else fm[i](v) for v in dp[c]])
    lw = BODY_W * 0.40
    dup = grid(rows, [lw] + [(BODY_W - lw) / len(yrs)] * len(yrs), bold_rows=(6,))
    out = [heading("5-Step DuPont RoE Decomposition", BODY_W), Spacer(1, 4), dup, Spacer(1, 4),
           P(f"Analyst interpretation: {nar['dupont']}", "body"), Spacer(1, 4)]

    fs, z = f["piotroski"], f["altman"]
    pw, zw = BODY_W * 0.60, BODY_W * 0.37
    if fs:
        prow = [[f"Piotroski F-Score: {fs['score']} / {fs['evaluated']} ({fs['year']})", "Result", "Evidence"]]
        cmds = []
        for i, c in enumerate(fs["criteria"], start=1):
            res = "Pass" if c["passed"] else ("n.a." if not c["available"] else "Fail")
            prow.append([c["criterion"], res, c["detail"]])
            cmds.append(("TEXTCOLOR", (1, i), (1, i), GREEN if c["passed"] else RED))
        ptab = grid(prow, [pw * 0.40, pw * 0.12, pw * 0.48], extra=cmds)
        for i, c in enumerate(fs["criteria"], start=1):   # colour the result text
            col = GREEN if c["passed"] else (GRAY if not c["available"] else RED)
            ptab._cellvalues[i][1] = P(prow[i][1], "crb", color=col)
    else:
        ptab = P("Piotroski F-Score requires two fiscal years of data.", "note")
    zrows = [[f"Altman Z-Score: {z['z']:.2f} ({z['zone']})" if z["applicable"] and pd.notna(z["z"]) else "Altman Z-Score: not applicable", ""]]
    for k, v in z["components"].items():
        zrows.append([k, fmt_num(v, 3)])
    if z["applicable"]:
        zrows.append(["Z''-EM (book-value variant)", fmt_num(z["z_em"])])
    ztab = grid(zrows, [zw * 0.76, zw * 0.24])
    fq = Table([[[ptab], [ztab, Spacer(1, 4), P(z["note"], "small")]]], colWidths=[pw + BODY_W * 0.03, zw])
    fq.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (0, 0), BODY_W * 0.03), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    out += [heading("Forensic Quality: Piotroski F-Score & Altman Z-Score", BODY_W), Spacer(1, 4), fq, Spacer(1, 8)]

    names = {"USDINR=X": "USD/INR", "EURINR=X": "EUR/INR", "GBPINR=X": "GBP/INR"}
    fxr = [["Pair", "Last", "1Y change", "52W range"]]
    for k, v in fxa["pairs"].items():
        fxr.append([names.get(k, k), f"{v['last']:.2f}", fmt_pct(v["chg_1y"] * 100, 1, True), f"{v['low']:.2f} – {v['high']:.2f}"])
    tw = BODY_W * 0.46
    fxt = grid(fxr, [tw * 0.22, tw * 0.22, tw * 0.24, tw * 0.32])
    prior = fxa["sector_prior"][0]
    st = [[f"{ctx['ticker']} sensitivity to USD/INR", ""],
          ["FX beta (2Y daily)", fmt_num(fxa.get("fx_beta"))], ["FX beta, NIFTY-neutral", fmt_num(fxa.get("fx_beta_neutral"))],
          ["t-statistic (NIFTY-neutral)", fmt_num(fxa.get("t_stat"), 1)], ["90-day rolling correlation", fmt_num(fxa.get("corr_90d"))],
          ["INR depreciation: sector effect", prior.capitalize()]]
    stt = grid(st, [tw * 0.66, tw * 0.34])
    cwid = BODY_W - tw - BODY_W * 0.03
    fxrow = Table([[[fxt, Spacer(1, 5), stt], chart_fx(ctx, cwid, cwid * 0.50)]], colWidths=[tw + BODY_W * 0.03, cwid])
    fxrow.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (0, 0), BODY_W * 0.03), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    out += [heading("INR Currency Macro Sensitivity", BODY_W), Spacer(1, 4), fxrow, Spacer(1, 4), P(nar["fx"], "body"),
            Spacer(1, 4), heading("Key Risks", BODY_W), Spacer(1, 4)]
    out += [P(f"• {r}", "bullet") for r in nar["risks"]]
    src = {"LIVE": "live Yahoo Finance data", "OFFLINE": "the bundled offline demo dataset",
           "FALLBACK": "the bundled offline demo dataset (live source unavailable)"}.get(ctx["status"], ctx["status"])
    disc = (f"Methodology and disclosures. This note was generated algorithmically by an open-source research engine using {src}; "
            f"prices as of {ctx['as_of']:%d %b %Y}. Narrative engine: {nar.get('engine', 'deterministic')}. "
            "Valuation: five-year FCFF DCF (excess-return model for banks) with CAPM cost of equity, India 10-year G-Sec risk-free "
            "rate, India equity risk premium and Section 115BAA tax rate; target price blends intrinsic and peer-relative value. "
            "Projections are mechanical extrapolations, not management guidance or consensus. "
            + ("The offline dataset contains approximate financials and simulated daily price paths anchored to periodic market "
               "levels; figures are illustrative. " if ctx["status"] != "LIVE" else "")
            + "This document is for educational and demonstration purposes only. It is not investment advice, not a "
            "recommendation to buy or sell any security, and is not issued by a SEBI-registered research analyst or by any "
            "brokerage or bank. Past performance does not indicate future results.")
    out += [Spacer(1, 6), heading("Disclosures", BODY_W), Spacer(1, 3), P(disc, "small")]
    return out


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #
def _decorate(ctx: dict):
    def draw(canvas, doc):
        canvas.saveState()
        top = PAGE_H - MARGIN
        canvas.setFillColor(NAVY)
        canvas.rect(MARGIN, top - BANNER_H, BODY_W, BANNER_H, stroke=0, fill=1)
        canvas.setFillColor(SLATE)
        canvas.rect(MARGIN, top - BANNER_H - 2.5, BODY_W, 2.5, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 8.6)
        canvas.drawString(MARGIN + 8, top - BANNER_H + 7.5, "INDIA EQUITY RESEARCH  |  INSTITUTIONAL COVERAGE NOTE")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN - 8, top - BANNER_H + 7.5, f"{ctx['report_date']}  |  {ctx['sector'].upper()}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, MARGIN + FOOTER_H - 4, PAGE_W - MARGIN, MARGIN + FOOTER_H - 4)
        canvas.setFillColor(GRAY)
        canvas.setFont("Helvetica", 6.4)
        canvas.drawString(MARGIN, MARGIN + 2, f"{ctx['name']} ({ctx['ticker']})  |  Model-generated; for educational use only; not investment advice.")
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN + 2, f"Page {doc.page} of {TOTAL_PAGES}")
        canvas.restoreState()
    return draw


def build_report(ctx: dict, narrative: dict) -> bytes:
    """Render the three-page research note and return the PDF bytes."""
    ctx = dict(ctx)
    ctx.setdefault("report_date", pd.Timestamp.today().strftime("%d %b %Y"))
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
                          title=f"{ctx['name']} ({ctx['ticker']}) — India Equity Research Note",
                          author="Institutional Indian Equity Research Engine", subject=f"{ctx['sector']} | {ctx['rating']['rating']}")
    frame = Frame(MARGIN, MARGIN + FOOTER_H + 2, BODY_W, BODY_H, leftPadding=0, rightPadding=0, topPadding=0,
                  bottomPadding=0, id="body")
    doc.addPageTemplates([PageTemplate(id="note", frames=[frame], onPage=_decorate(ctx))])
    story = []
    for i, page in enumerate((_page1(ctx, narrative), _page2(ctx, narrative), _page3(ctx, narrative))):
        story.append(KeepInFrame(BODY_W, BODY_H - 1, page, mode="shrink", hAlign="LEFT", vAlign="TOP"))
        if i < TOTAL_PAGES - 1:
            story.append(PageBreak())
    doc.build(story)
    return buf.getvalue()


def report_filename(ctx: dict) -> str:
    return f"{ctx['ticker'].replace('.NS', '').replace('&', 'and')}_NSE_Research_Note_{ctx['as_of']:%Y%m%d}.pdf"
