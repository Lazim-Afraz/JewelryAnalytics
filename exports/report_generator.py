"""
PDF Report Generator
exports/report_generator.py

Generates a branded PDF analytics report using reportlab.
Consumes data exclusively from AnalyticsService.get_report_data().

Usage:
    from exports.report_generator import ReportGenerator
    gen = ReportGenerator(service=svc)
    pdf_bytes = gen.generate_pdf_bytes()   # returns bytes for st.download_button

Dependencies:
    pip install reportlab
"""

import io
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Colour palette (mirrors app.py) ──────────────────────────────────────────
GOLD        = (0.788, 0.659, 0.298)   # #C9A84C  as 0-1 RGB
DARK_BG     = (0.051, 0.051, 0.078)   # #0D0D14
CARD_BG     = (0.075, 0.075, 0.122)   # #13131F
TEXT_LIGHT  = (0.941, 0.902, 0.800)   # #F0E6CC
TEXT_MUTED  = (0.541, 0.541, 0.667)   # #8A8AAA
ACCENT_TEAL = (0.298, 0.788, 0.659)   # #4CC9A8
ACCENT_RED  = (0.788, 0.298, 0.298)   # #C94C4C
WHITE       = (1, 1, 1)


class ReportGenerator:
    """
    Builds a multi-page PDF analytics report.

    Pages:
        1. Cover page  — title, generated timestamp, portfolio KPIs
        2. Top Branches — ranked table (top 10 by SALE_COUNT)
        3. Cluster Analysis — cluster profiles + quality scores
        4. Attribute Intelligence — top purity / finish / theme (global)
    """

    def __init__(self, service):
        """
        Args:
            service: AnalyticsService instance (data must be loaded).
        """
        self.service = service

    # =========================================================================
    # Public API
    # =========================================================================

    def generate_pdf_bytes(self) -> bytes:
        """
        Build the full PDF and return raw bytes.

        Returns:
            PDF as bytes (safe to pass to st.download_button).

        Raises:
            ImportError: if reportlab is not installed.
            RuntimeError: if data is not loaded.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, PageBreak,
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError as e:
            raise ImportError(
                "reportlab is required. Install it with: pip install reportlab"
            ) from e

        if not self.service.is_data_loaded():
            raise RuntimeError("Data not loaded. Call service.load_data() first.")

        data = self.service.get_report_data()
        buf  = io.BytesIO()

        # ── Document setup ────────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2.2 * cm,
            bottomMargin=2 * cm,
            title="Jewelry Portfolio Analytics Report",
            author="Jewelry Analytics Platform",
        )

        rl_gold       = colors.Color(*GOLD)
        rl_dark       = colors.Color(*DARK_BG)
        rl_card       = colors.Color(*CARD_BG)
        rl_text       = colors.Color(*TEXT_LIGHT)
        rl_muted      = colors.Color(*TEXT_MUTED)
        rl_teal       = colors.Color(*ACCENT_TEAL)
        rl_red        = colors.Color(*ACCENT_RED)
        rl_white      = colors.white

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=rl_gold,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=rl_muted,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=rl_gold,
            spaceBefore=18,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=rl_text,
            spaceAfter=4,
            leading=14,
        )
        kpi_label_style = ParagraphStyle(
            "KpiLabel",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=rl_muted,
            alignment=TA_CENTER,
        )
        kpi_value_style = ParagraphStyle(
            "KpiValue",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=rl_gold,
            alignment=TA_CENTER,
        )

        story = []

        # ── TABLE STYLE HELPERS ───────────────────────────────────────────────
        def _header_style():
            return TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), rl_gold),
                ("TEXTCOLOR",    (0, 0), (-1, 0), rl_dark),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0), 9),
                ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
                ("BOTTOMPADDING",(0, 0), (-1, 0), 6),
                ("TOPPADDING",   (0, 0), (-1, 0), 6),
                ("BACKGROUND",   (0, 1), (-1, -1), rl_card),
                ("TEXTCOLOR",    (0, 1), (-1, -1), rl_text),
                ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",     (0, 1), (-1, -1), 8),
                ("ALIGN",        (1, 1), (-1, -1), "CENTER"),
                ("ALIGN",        (0, 1), (0, -1),  "LEFT"),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1),
                 [rl_card, colors.Color(0.09, 0.09, 0.15)]),
                ("GRID",         (0, 0), (-1, -1), 0.3,
                 colors.Color(0.165, 0.165, 0.243)),
                ("TOPPADDING",   (0, 1), (-1, -1), 5),
                ("BOTTOMPADDING",(0, 1), (-1, -1), 5),
                ("LEFTPADDING",  (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])

        ts = _header_style()

        # =================================================================
        # PAGE 1 — COVER
        # =================================================================
        story.append(Spacer(1, 1.5 * cm))

        story.append(Paragraph("💎 Jewelry Portfolio Analytics", title_style))
        story.append(Paragraph("PERFORMANCE INTELLIGENCE REPORT", subtitle_style))
        story.append(Spacer(1, 0.3 * cm))

        now = datetime.now().strftime("%d %B %Y  ·  %H:%M")
        story.append(Paragraph(f"Generated: {now}", subtitle_style))

        story.append(HRFlowable(
            width="100%", thickness=1,
            color=rl_gold, spaceAfter=18, spaceBefore=12,
        ))

        # KPI grid on cover
        dash = data.get("dashboard", {})

        def _kpi_cell(label, value):
            return [
                Paragraph(str(value), kpi_value_style),
                Paragraph(label, kpi_label_style),
            ]

        def _fmt(n):
            try:
                n = float(n)
                if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
                if n >= 1_000:     return f"{n/1_000:.1f}K"
                return f"{n:,.0f}"
            except Exception:
                return str(n)

        kpi_data = [
            [
                _kpi_cell("Total Sales",       _fmt(dash.get("total_sales",  0))),
                _kpi_cell("Total Stock",        _fmt(dash.get("total_stock",  0))),
                _kpi_cell("Branches",           _fmt(dash.get("total_branches", 0))),
                _kpi_cell("Regions",            _fmt(dash.get("total_regions",  0))),
            ],
            [
                _kpi_cell("Efficiency",
                          f"{dash.get('overall_efficiency', 0):.3f}"),
                _kpi_cell("Sell-Through",
                          f"{dash.get('overall_sell_through', 0):.1%}"),
                _kpi_cell("Local Heroes",
                          _fmt(dash.get("total_local_heroes", 0))),
                _kpi_cell("Clusters",
                          _fmt(dash.get("cluster_count",     0))),
            ],
        ]

        col_w = (doc.width / 4)
        for row in kpi_data:
            kpi_tbl = Table(
                [[cell] for cell in row],
                colWidths=[col_w] * 4,
                rowHeights=None,
            )
            # Reformat: each cell is a list of two Paragraphs
            flat_row = [cell for cell in row]
            kpi_tbl  = Table(
                [flat_row],
                colWidths=[col_w] * 4,
            )
            kpi_tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, -1), rl_card),
                ("BOX",          (0, 0), (-1, -1), 0.5,
                 colors.Color(0.165, 0.165, 0.243)),
                ("INNERGRID",    (0, 0), (-1, -1), 0.3,
                 colors.Color(0.165, 0.165, 0.243)),
                ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",   (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
            ]))
            story.append(kpi_tbl)
            story.append(Spacer(1, 0.3 * cm))

        # Highlights
        top_branch = dash.get("top_branch", "—")
        top_region = dash.get("top_region", "—")
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(
            f'<font color="#{int(GOLD[0]*255):02X}{int(GOLD[1]*255):02X}'
            f'{int(GOLD[2]*255):02X}">Top Branch:</font>  {top_branch}  '
            f'&nbsp;&nbsp;&nbsp;  '
            f'<font color="#{int(ACCENT_TEAL[0]*255):02X}{int(ACCENT_TEAL[1]*255):02X}'
            f'{int(ACCENT_TEAL[2]*255):02X}">Top Region:</font>  {top_region}',
            body_style,
        ))

        story.append(PageBreak())

        # =================================================================
        # PAGE 2 — TOP BRANCHES
        # =================================================================
        story.append(Paragraph("Top 10 Branches by Sales", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=rl_gold, spaceAfter=10))

        top_df = data.get("top_branches")
        if top_df is not None and not top_df.empty:
            headers = ["#", "Branch", "Region", "Sales",
                       "Stock", "Efficiency", "Sell-Through"]
            rows    = [headers]

            for _, row in top_df.iterrows():
                rows.append([
                    str(int(row.get("rank", 0))),
                    str(row.get("BRANCHNAME", "—")),
                    str(row.get("REGION", "—")),
                    _fmt(row.get("SALE_COUNT", 0)),
                    _fmt(row.get("STOCK_COUNT", 0)),
                    f"{row.get('avg_efficiency', 0):.3f}",
                    f"{row.get('branch_sell_through', 0):.1%}",
                ])

            col_widths = [
                0.5 * cm, 4.5 * cm, 3.0 * cm,
                1.8 * cm, 1.8 * cm, 2.0 * cm, 2.4 * cm,
            ]
            tbl = Table(rows, colWidths=col_widths, repeatRows=1)
            tbl.setStyle(ts)
            story.append(tbl)
        else:
            story.append(Paragraph("No branch data available.", body_style))

        story.append(PageBreak())

        # =================================================================
        # PAGE 3 — CLUSTER ANALYSIS
        # =================================================================
        story.append(Paragraph("Branch Cluster Analysis", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=rl_gold, spaceAfter=10))

        ca  = data.get("cluster_analysis", {})
        qs  = ca.get("quality_scores", {})
        n_c = ca.get("n_clusters", 0)

        # Quality scores row
        qs_data = [[
            Paragraph(f"<b>{n_c}</b><br/>Clusters",         kpi_value_style),
            Paragraph(f"<b>{qs.get('silhouette', 0):.3f}</b><br/>Silhouette",
                      kpi_value_style),
            Paragraph(f"<b>{qs.get('calinski', 0):,.0f}</b><br/>Calinski",
                      kpi_value_style),
            Paragraph(f"<b>{qs.get('inertia', 0):,.0f}</b><br/>Inertia",
                      kpi_value_style),
        ]]
        qs_tbl = Table(qs_data, colWidths=[doc.width / 4] * 4)
        qs_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), rl_card),
            ("BOX",          (0, 0), (-1, -1), 0.5,
             colors.Color(0.165, 0.165, 0.243)),
            ("INNERGRID",    (0, 0), (-1, -1), 0.3,
             colors.Color(0.165, 0.165, 0.243)),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ]))
        story.append(qs_tbl)
        story.append(Spacer(1, 0.5 * cm))

        # Cluster profiles table
        summary  = ca.get("summary", {})
        clusters = summary.get("clusters", [])

        if clusters:
            story.append(Paragraph("Cluster Profiles", section_style))
            c_headers = ["Cluster", "Tier", "Branches",
                         "Avg Sales", "Avg Efficiency", "Regions"]
            c_rows    = [c_headers]

            tier_colors = {
                "Elite":          colors.Color(0.788, 0.659, 0.298),
                "Strong":         colors.Color(0.298, 0.788, 0.659),
                "Average":        colors.Color(0.541, 0.541, 0.667),
                "Underperforming":colors.Color(0.788, 0.298, 0.298),
            }

            for c in clusters:
                am  = c.get("avg_metrics", {})
                c_rows.append([
                    c.get("label", "—"),
                    c.get("performance_tier", "—"),
                    str(c.get("num_branches", 0)),
                    f"{am.get('SALE_COUNT', 0):.1f}",
                    f"{am.get('efficiency_ratio', 0):.3f}",
                    ", ".join(c.get("regions", [])) or "—",
                ])

            col_w_c = [
                2.5 * cm, 3.0 * cm, 2.0 * cm,
                2.5 * cm, 2.8 * cm, 4.2 * cm,
            ]
            c_tbl = Table(c_rows, colWidths=col_w_c, repeatRows=1)
            base_ts = _header_style()

            # Colour-code tier column (col 1, rows 1+)
            for i, c in enumerate(clusters, start=1):
                tier  = c.get("performance_tier", "")
                tclr  = tier_colors.get(tier, rl_muted)
                base_ts.add("TEXTCOLOR", (1, i), (1, i), tclr)
                base_ts.add("FONTNAME",  (1, i), (1, i), "Helvetica-Bold")

            c_tbl.setStyle(base_ts)
            story.append(c_tbl)

        story.append(PageBreak())

        # =================================================================
        # PAGE 4 — ATTRIBUTE INTELLIGENCE
        # =================================================================
        story.append(Paragraph("Top Attribute Combinations", section_style))
        story.append(Paragraph(
            "Global ranking of product attributes by total sales volume.",
            body_style,
        ))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=rl_gold, spaceAfter=10))

        # Pull top-5 global recommendations
        try:
            recs = self.service.get_recommendations(top_n=5)
            attr_sections = [
                ("by_purity",    "PURITY",    "Purity"),
                ("by_finish",    "FINISH",    "Finish"),
                ("by_theme",     "THEME",     "Theme"),
                ("by_shape",     "SHAPE",     "Shape"),
            ]

            for key, col, label in attr_sections:
                items = recs.get(key, [])
                if not items:
                    continue

                story.append(Paragraph(f"Top {label} Values", section_style))
                a_headers = ["#", label, "Total Sales", "Sell-Through"]
                a_rows    = [a_headers]

                for idx, item in enumerate(items, start=1):
                    a_rows.append([
                        str(idx),
                        str(item.get(col, item.get("attribute_value", "—"))),
                        _fmt(item.get("total_sales", 0)),
                        f"{item.get('sell_through', 0):.1%}",
                    ])

                a_tbl = Table(
                    a_rows,
                    colWidths=[1.0 * cm, 5.0 * cm, 4.0 * cm, 4.0 * cm],
                    repeatRows=1,
                )
                a_tbl.setStyle(_header_style())
                story.append(a_tbl)
                story.append(Spacer(1, 0.4 * cm))

        except Exception as e:
            logger.warning(f"Attribute intelligence section failed: {e}")
            story.append(Paragraph(
                "Attribute data could not be loaded for this report.",
                body_style,
            ))

        # ── Build ─────────────────────────────────────────────────────────────
        doc.build(
            story,
            onFirstPage=self._page_footer,
            onLaterPages=self._page_footer,
        )

        pdf_bytes = buf.getvalue()
        buf.close()

        logger.info(f"✅ PDF generated — {len(pdf_bytes):,} bytes")
        return pdf_bytes

    # =========================================================================
    # Page decoration
    # =========================================================================

    @staticmethod
    def _page_footer(canvas, doc):
        """Draw gold footer line + page number on every page."""
        try:
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.units import cm

            rl_gold  = rl_colors.Color(*GOLD)
            rl_muted = rl_colors.Color(*TEXT_MUTED)

            canvas.saveState()

            # Gold rule
            canvas.setStrokeColor(rl_gold)
            canvas.setLineWidth(0.5)
            canvas.line(
                doc.leftMargin,
                doc.bottomMargin - 0.3 * cm,
                doc.width + doc.leftMargin,
                doc.bottomMargin - 0.3 * cm,
            )

            # Page number
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(rl_muted)
            canvas.drawRightString(
                doc.width + doc.leftMargin,
                doc.bottomMargin - 0.7 * cm,
                f"Page {doc.page}  ·  Jewelry Portfolio Analytics",
            )

            # Left: timestamp
            canvas.drawString(
                doc.leftMargin,
                doc.bottomMargin - 0.7 * cm,
                datetime.now().strftime("%d %b %Y"),
            )

            canvas.restoreState()

        except Exception:
            pass   # footer failure must never crash report generation
