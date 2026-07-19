#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def status_text(validation_passed: bool) -> str:
    return "PASS" if validation_passed else "FAIL"


def build_metric_rows(clickstream: dict[str, Any], products: dict[str, Any]) -> list[list[str]]:
    return [
        ["Rows", str(clickstream.get("total_rows", 0)), str(products.get("total_rows", 0))],
        ["Columns", str(clickstream.get("total_columns", 0)), str(products.get("total_columns", 0))],
        ["Missing values", str(clickstream.get("missing_values_total", 0)), str(products.get("missing_values_total", 0))],
        ["Duplicate rows", str(clickstream.get("duplicate_rows", 0)), str(products.get("duplicate_rows", 0))],
        ["Quality score (%)", f"{clickstream.get('quality_score', 0):.2f}", f"{products.get('quality_score', 0):.2f}"],
        ["Validation status", status_text(clickstream.get("validation_passed", False)), status_text(products.get("validation_passed", False))],
    ]


def issue_lines(report: dict[str, Any]) -> list[str]:
    issues = report.get("issues", [])
    if not issues:
        return ["No issues detected."]
    return [f"• {issue}" for issue in issues]


def make_table(data: list[list[str]], col_widths: list[int]) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_pdf(clickstream: dict[str, Any], products: dict[str, Any], output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Section", fontSize=13, leading=16, spaceAfter=8, textColor=colors.HexColor("#1f4e79")))

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=landscape(letter),
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    story: list[Any] = []
    story.append(Paragraph("Data Quality Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
            styles["Small"],
        )
    )
    story.append(Spacer(1, 12))

    summary_data = [["Metric", "Clickstream", "Products"]] + build_metric_rows(clickstream, products)
    story.append(Paragraph("Overall Summary", styles["Section"]))
    story.append(make_table(summary_data, [160, 170, 170]))
    story.append(Spacer(1, 14))

    # Clickstream section
    story.append(Paragraph("Clickstream Validation", styles["Section"]))
    click_meta = [
        f"Source file: {clickstream.get('file_path', '')}",
        f"Checked at UTC: {clickstream.get('checked_at_utc', '')}",
        f"Missing columns: {clickstream.get('missing_columns', []) or 'None'}",
        f"Extra columns: {clickstream.get('extra_columns', []) or 'None'}",
        f"Valid rows estimate: {clickstream.get('valid_rows_estimate', 0)}",
    ]
    for line in click_meta:
        story.append(Paragraph(line, styles["Small"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Issues", styles["Small"]))
    for line in issue_lines(clickstream):
        story.append(Paragraph(line, styles["Small"]))
    story.append(Spacer(1, 10))

    click_rule_rows = [["Rule", "Count"]] + [[k, str(v)] for k, v in clickstream.get("rule_violations", {}).items()]
    story.append(make_table(click_rule_rows, [300, 100]))

    story.append(PageBreak())

    # Products section
    story.append(Paragraph("Product Catalog Validation", styles["Section"]))
    prod_meta = [
        f"Source file: {products.get('file_path', '')}",
        f"Checked at UTC: {products.get('checked_at_utc', '')}",
        f"Missing columns: {products.get('missing_columns', []) or 'None'}",
        f"Extra columns: {products.get('extra_columns', []) or 'None'}",
        f"Valid rows estimate: {products.get('valid_rows_estimate', 0)}",
    ]
    for line in prod_meta:
        story.append(Paragraph(line, styles["Small"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Issues", styles["Small"]))
    for line in issue_lines(products):
        story.append(Paragraph(line, styles["Small"]))
    story.append(Spacer(1, 10))

    prod_rule_rows = [["Rule", "Count"]] + [[k, str(v)] for k, v in products.get("rule_violations", {}).items()]
    story.append(make_table(prod_rule_rows, [300, 100]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Notes", styles["Section"]))
    story.append(
        Paragraph(
            "This report is generated from JSON summaries produced by the validation scripts. "
            "The quality score is an approximate indicator based on the number of detected issues.",
            styles["Small"],
        )
    )

    doc.build(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PDF data quality report from validation JSON files.")
    parser.add_argument("--clickstream-json", required=True, help="Path to clickstream validation JSON")
    parser.add_argument("--products-json", required=True, help="Path to products validation JSON")
    parser.add_argument(
        "--output-pdf",
        default="validation/reports/data_quality_report.pdf",
        help="Output PDF report path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clickstream = load_report(Path(args.clickstream_json))
    products = load_report(Path(args.products_json))
    build_pdf(clickstream, products, Path(args.output_pdf))
    print(f"PDF report written to: {args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())