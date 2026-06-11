# -*- coding: utf-8 -*-
"""Render the working paper -> an SSRN-grade PDF (ReportLab; zero native deps, proxy-safe).
Front matter (title page, author, affiliation, abstract, JEL, keywords, AI disclosure) is composed
from META; the body is rendered from volbook_working_paper.md; an auto-generated Appendix reads the
headline numbers straight from results.json (render-from-data: the PDF cannot drift from the code).

    python build_paper_pdf.py        # -> <DATA_DIR>/ssrn_submission/timing_is_the_product.pdf
"""
import os, re, html, json
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
PKG = r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab"
FONTS = r"C:\Windows\Fonts"

META = dict(
    title="Timing Is the Product: A Friction-Realistic, Look-Ahead-Free "
          "Multi-Market Equity-Volatility Book",
    author="Zhengshao Zhao",
    affiliation="Independent Researcher",
    email="willzhao2024@outlook.com",
    date_written="June 2026",
    keywords=["variance risk premium", "volatility timing", "delta-hedged straddle", "walk-forward",
              "transaction costs", "backtest overfitting", "deflated Sharpe ratio", "regime gating"],
    jel=["G13", "G12", "G17", "G14", "C58", "C22", "C53", "G11"],
    classifications=["Financial Economics Network — Derivatives",
                     "Capital Markets: Asset Pricing & Valuation",
                     "Econometrics: Econometric & Statistical Methods"],
    ai_disclosure="The author conducted this research using Claude (Anthropic) as a coding and "
                  "analysis assistant under the author's direction; all methodological choices, "
                  "verification, and conclusions are the author's responsibility.",
    license="Posted under SSRN default terms.",
    abstract=(
        "We study a regime-gated, long-short equity-volatility strategy across eight equity-index / "
        "volatility-index pairs (SPX, NQ, EEM, DAX, SX5E, N225, HSI, NIFTY) over 2005–2026. The "
        "strategy forecasts next-day realized variance with two causal regime signals and expresses "
        "the view through a one-day-to-expiry, delta-hedged at-the-money straddle — a near-pure gamma "
        "instrument calibrated to a measured short-dated implied-volatility ratio. Per-market sleeves "
        "are combined into a book by a selection-free weighting (inverse trailing volatility times a "
        "cost-coverage gate) that uses no return information and therefore has no overfitting surface. "
        "Under a six-component, stress-aware execution-friction model — widening spreads, asymmetric "
        "long-volatility bid-ask, discrete strikes, floors, assignment, and margin funding — the book "
        "earns a Sharpe ratio of 1.26 on deployed days and 1.01 on a calendar basis, with a 20% "
        "maximum drawdown; a moving-block bootstrap places the 95% confidence interval for the "
        "calendar Sharpe at [0.58, 1.53]. Our central finding qualifies the variance-risk-premium "
        "literature: at realistic short-dated pricing the unconditional premium does not survive — an "
        "always-on short-volatility version of the same book loses money (Sharpe −0.97). All of the "
        "value is timing, and gross-leg attribution shows the long-volatility legs out-earning the "
        "short legs. We subject the result to multiple-testing correction (only four of eight markets "
        "survive a Benjamini–Hochberg false-discovery control), a structural-break test (no "
        "significant break; the edge is not one crisis), and a machine-checked no-look-ahead "
        "guarantee. We report in full the negative results and the artifacts we caught and retracted. "
        "Parameter selection is walk-forward throughout; no global fit is used. The contribution is "
        "not a high Sharpe ratio but a calibrated, falsifiable estimate of how much of the variance "
        "risk premium remains harvestable once short-dated execution is modeled honestly, and a "
        "reproducible discipline for arriving at it."),
)

REFERENCES = [
    "Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection "
    "bias, backtest overfitting, and non-normality. Journal of Portfolio Management, 40(5), 94–107.",
    "Bauer, M. D., & Swanson, E. T. (2023). A reassessment of monetary policy surprises and "
    "high-frequency identification. NBER Macroeconomics Annual, 37, 87–155.",
    "Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia. "
    "Review of Financial Studies, 22(11), 4463–4492.",
    "Bollerslev, T., Todorov, V., & Xu, L. (2015). Tail risk premia and return predictability. "
    "Journal of Financial Economics, 118(1), 113–134.",
    "Carr, P., & Wu, L. (2009). Variance risk premiums. Review of Financial Studies, 22(3), 1311–1341.",
    "Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. Review of "
    "Financial Studies, 29(1), 5–68.",
    "Lucca, D. O., & Moench, E. (2015). The pre-FOMC announcement drift. Journal of Finance, 70(1), "
    "329–371.",
    "Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of "
    "return. Journal of Business, 53(1), 61–65.",
]

GLYPH = {"√": "sqrt ", "∈": " in ", "⅛": "1/8", "≈": "~", "↔": "<->"}   # belt-and-suspenders


def _fonts():
    pdfmetrics.registerFont(TTFont("body", os.path.join(FONTS, "cambria.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("body-b", os.path.join(FONTS, "cambriab.ttf")))
    pdfmetrics.registerFont(TTFont("body-i", os.path.join(FONTS, "cambriai.ttf")))
    pdfmetrics.registerFont(TTFont("body-bi", os.path.join(FONTS, "cambriaz.ttf")))
    pdfmetrics.registerFont(TTFont("mono", os.path.join(FONTS, "consola.ttf")))
    pdfmetrics.registerFontFamily("body", normal="body", bold="body-b", italic="body-i",
                                  boldItalic="body-bi")


def _inline(s):
    for k, v in GLYPH.items():
        s = s.replace(k, v)
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r'<font face="mono" size=9.5>\1</font>', s)
    return s


def _styles():
    S = {}
    S["title"] = ParagraphStyle("title", fontName="body-b", fontSize=18, leading=22,
                                alignment=1, spaceAfter=14)
    S["author"] = ParagraphStyle("author", fontName="body", fontSize=12, leading=16, alignment=1)
    S["sub"] = ParagraphStyle("sub", fontName="body-i", fontSize=10, leading=13, alignment=1,
                              textColor=colors.HexColor("#444441"))
    S["h2"] = ParagraphStyle("h2", fontName="body-b", fontSize=13, leading=16, spaceBefore=14,
                             spaceAfter=5)
    S["abhead"] = ParagraphStyle("abhead", fontName="body-b", fontSize=11, leading=14, spaceBefore=8,
                                 spaceAfter=4)
    S["body"] = ParagraphStyle("body", fontName="body", fontSize=10.5, leading=15, spaceAfter=7,
                               alignment=4)
    S["abstract"] = ParagraphStyle("abstract", fontName="body", fontSize=10, leading=14,
                                   alignment=4, leftIndent=18, rightIndent=18)
    S["li"] = ParagraphStyle("li", parent=S["body"], leftIndent=16, spaceAfter=4)
    S["small"] = ParagraphStyle("small", fontName="body", fontSize=8.5, leading=11,
                                textColor=colors.HexColor("#444441"))
    S["ref"] = ParagraphStyle("ref", fontName="body", fontSize=9.5, leading=13, spaceAfter=5,
                              leftIndent=18, firstLineIndent=-18)
    S["cap"] = ParagraphStyle("cap", fontName="body-i", fontSize=9, leading=12, alignment=1,
                              textColor=colors.HexColor("#444441"), spaceAfter=8)
    return S


def _table(rows, S):
    data = [[Paragraph(_inline(c), ParagraphStyle("tc", parent=S["small"], fontSize=8.5)) for c in r]
            for r in rows]
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "body-b"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1EFE8")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#888780")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#D3D1C7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _parse_body(md, S):
    lines = md.split("\n")
    # body starts at section 1; drop title/disclaimer/abstract (in front matter)
    start = next(i for i, l in enumerate(lines) if l.startswith("## 1."))
    flow, i = [], start
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("## "):
            flow.append(Paragraph(_inline(ln[3:]), S["h2"]))
        elif ln.startswith("|") and i + 1 < len(lines) and set(lines[i+1].replace("|", "").strip()) <= set("-: "):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if set("".join(cells)) <= set("-: "):
                    i += 1; continue
                rows.append(cells); i += 1
            flow.append(_table(rows, S)); flow.append(Spacer(1, 6)); continue
        elif ln.strip() == "---":
            flow.append(Spacer(1, 4))
        elif re.match(r"^\s*-\s+", ln):
            flow.append(Paragraph("•&nbsp;&nbsp;" + _inline(re.sub(r"^\s*-\s+", "", ln)), S["li"]))
        elif re.match(r"^\d+\.\s+", ln):
            m = re.match(r"^(\d+)\.\s+(.*)", ln)
            flow.append(Paragraph(f"{m.group(1)}.&nbsp;&nbsp;" + _inline(m.group(2)), S["li"]))
        elif ln.strip() == "":
            pass
        else:
            flow.append(Paragraph(_inline(ln.strip()), S["body"]))
        i += 1
    return flow


def _appendix(S):
    with open(os.path.join(DATA_DIR, "results.json"), encoding="utf-8") as f:
        R = json.load(f)
    fl, cm, inf, rob = R["floor"], R["comparison"], R["inference"], R["robustness"]
    pct = lambda x: f"{x*100:.0f}%"
    rows = [["metric", "value"],
            ["Sharpe — active days", f"{fl['sharpe_active']:.2f}"],
            ["Sharpe — calendar basis", f"{fl['sharpe_calendar']:.2f}"],
            ["95% bootstrap CI (calendar)", f"[{inf['boot_ci_calendar'][0]:.2f}, {inf['boot_ci_calendar'][1]:.2f}]"],
            ["max drawdown", pct(fl['maxdd'])],
            ["days deployed", f"{fl['deployed_pct']:.0f}%"],
            ["sleeves past BH-FDR", f"{inf['fdr_survive']} / {R['meta']['n_markets']}"],
            ["P(Sharpe > 0), calendar", f"{inf['psr_cal_gt0']:.2f}"],
            ["baseline equal-risk Sharpe", f"{cm['baseline_equalrisk_sharpe']:.2f}"],
            ["static short-vol Sharpe", f"{cm['static_shortvol_sharpe']:.2f}"],
            ["robustness window 1000/200 to 1500/252", f"{rob['window_1000_200']:.2f} to {rob['window_1500_252']:.2f}"],
            ["adverse corner (short window + 5% funding)", f"{rob['adverse_corner']:.2f}"],
            ["data through", f"{R['meta']['data_through']}"]]
    return [Paragraph("Appendix A. Key results (auto-generated from results.json)", S["h2"]),
            _table(rows, S),
            Paragraph("These figures are emitted by build_results.py and read directly into this "
                      "document; the prose above is consistent with them by construction.", S["cap"])]


def build(meta=META, out_pdf=None):
    _fonts(); S = _styles()
    out_pdf = out_pdf or os.path.join(DATA_DIR, "ssrn_submission", "timing_is_the_product.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    paper_md = os.path.join(PKG, "volbook_working_paper.md")
    if not os.path.exists(paper_md):
        paper_md = os.path.join(DATA_DIR, "volbook_working_paper.md")
    with open(paper_md, encoding="utf-8") as f:
        md = f.read()

    story = []
    story += [Spacer(1, 0.5*inch), Paragraph(_inline(meta["title"]), S["title"]), Spacer(1, 6),
              Paragraph(meta["author"], S["author"]),
              Paragraph(f'{meta["affiliation"]} · {meta["email"]}', S["sub"]),
              Paragraph(f'Working paper · draft of {meta["date_written"]} · research/educational, '
                        f'not investment advice', S["sub"]), Spacer(1, 16),
              HRFlowable(width="60%", thickness=0.5, color=colors.HexColor("#B4B2A9")), Spacer(1, 10),
              Paragraph("Abstract", S["abhead"]), Paragraph(_inline(meta["abstract"]), S["abstract"]),
              Spacer(1, 10),
              Paragraph("<b>Keywords:</b> " + "; ".join(meta["keywords"]), S["small"]), Spacer(1, 3),
              Paragraph("<b>JEL classification:</b> " + ", ".join(meta["jel"]), S["small"]),
              Spacer(1, 3),
              Paragraph("<b>AI disclosure.</b> " + _inline(meta["ai_disclosure"]), S["small"]),
              PageBreak()]
    story += _parse_body(md, S)
    story += [Spacer(1, 8), Paragraph("References", S["h2"])]
    story += [Paragraph(_inline(r), S["ref"]) for r in REFERENCES]
    story += [Spacer(1, 10)] + _appendix(S)

    def footer(c, d):
        c.saveState(); c.setFont("body", 8); c.setFillGray(0.45)
        c.drawString(inch, 0.55*inch, "Timing Is the Product — Z. Zhao")
        c.drawRightString(letter[0]-inch, 0.55*inch, str(d.page)); c.restoreState()

    doc = SimpleDocTemplate(out_pdf, pagesize=letter, leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch, title=meta["title"],
                            author=meta["author"])
    doc.build(story, onLaterPages=footer)
    return out_pdf, len(story)


if __name__ == "__main__":
    p, n = build()
    print(f"PDF -> {p}  ({os.path.getsize(p)/1024:.0f} KB, {n} flowables)")
