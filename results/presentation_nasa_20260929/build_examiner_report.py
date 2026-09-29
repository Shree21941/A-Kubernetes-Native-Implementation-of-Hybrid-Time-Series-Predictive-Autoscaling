"""Render EXAMINER_GUIDE.md and its verified figures to a PDF."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "EXAMINER_GUIDE.md"
OUTPUT = HERE / "EXAMINER_REPORT.pdf"
INK = colors.HexColor("#182A3A")
TEAL = colors.HexColor("#007378")
PALE = colors.HexColor("#EEF4F4")
MUTED = colors.HexColor("#5D6C77")
GOLD = colors.HexColor("#C27124")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=25, leading=31, textColor=INK, alignment=TA_LEFT, spaceAfter=14))
styles.add(ParagraphStyle(
    name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=19, leading=23, textColor=INK, spaceBefore=7, spaceAfter=12,
    keepWithNext=True))
styles.add(ParagraphStyle(
    name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=14, leading=17, textColor=TEAL, spaceBefore=8, spaceAfter=7,
    keepWithNext=True))
styles.add(ParagraphStyle(
    name="H3Custom", parent=styles["Heading3"], fontName="Helvetica-Bold",
    fontSize=11, leading=14, textColor=INK, spaceBefore=5, spaceAfter=4,
    keepWithNext=True))
styles.add(ParagraphStyle(
    name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9, leading=12.4, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(
    name="SmallCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=7.6, leading=10, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(
    name="TableCellCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=7.3, leading=9, textColor=INK))
styles.add(ParagraphStyle(
    name="TableHeaderCustom", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=7.4, leading=9, textColor=colors.white))
styles.add(ParagraphStyle(
    name="QuoteCustom", parent=styles["BodyText"], fontName="Helvetica-Oblique",
    fontSize=9, leading=12.3, textColor=GOLD, leftIndent=12, rightIndent=12,
    spaceBefore=5, spaceAfter=7))
styles.add(ParagraphStyle(
    name="CodeCustom", parent=styles["Code"], fontName="Courier",
    fontSize=7.6, leading=10, textColor=INK, backColor=PALE,
    borderPadding=6, spaceBefore=4, spaceAfter=8))


def inline_markup(text: str) -> str:
    text = escape(text, quote=False)
    code_spans = []

    def protect_code(match):
        code_spans.append(match.group(1))
        return f"CODESPAN{len(code_spans) - 1}TOKEN"

    text = re.sub(r"`([^`]+)`", protect_code, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    for index, value in enumerate(code_spans):
        text = text.replace(f"CODESPAN{index}TOKEN", f'<font name="Courier">{value}</font>')
    return text


def make_table(rows, width):
    if not rows:
        return Spacer(1, 1)
    count = max(len(row) for row in rows)
    col_widths = [width / count] * count
    data = []
    for row_index, row in enumerate(rows):
        style = "TableHeaderCustom" if row_index == 0 else "TableCellCustom"
        row = row + [""] * (count - len(row))
        data.append([Paragraph(inline_markup(cell), styles[style]) for cell in row])
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5D8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), INK),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 1:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    table.setStyle(TableStyle(commands))
    return table


def parse_markdown(path: Path):
    story = []
    lines = path.read_text().splitlines()
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(s.strip() for s in paragraph_lines)
            story.append(Paragraph(inline_markup(text), styles["BodyCustom"]))
            paragraph_lines = []

    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        rows = []
        for line in table_lines:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                continue
            rows.append(cells)
        story.append(make_table(rows, 7.2 * inch))
        story.append(Spacer(1, 6))
        table_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["CodeCustom"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = []
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines.append(stripped)
            i += 1
            continue
        flush_table()
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_paragraph()
            caption, target = image_match.groups()
            image_path = (path.parent / target).resolve()
            if image_path.exists():
                image = Image(str(image_path))
                scale = min((7.15 * inch) / image.drawWidth,
                            (5.2 * inch) / image.drawHeight, 1.0)
                image.drawWidth *= scale
                image.drawHeight *= scale
                story.append(image)
                story.append(Paragraph(inline_markup(caption), styles["SmallCustom"]))
                story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(f"Missing image: {escape(target)}", styles["SmallCustom"]))
            i += 1
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), styles["CoverTitle"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[3:]), styles["H1Custom"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), styles["H2Custom"]))
            i += 1
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), styles["QuoteCustom"]))
            i += 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]),
                                   styles["BodyCustom"], bulletText="-"))
            i += 1
            continue
        if re.match(r"^\d+\. ", stripped):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), styles["BodyCustom"]))
            i += 1
            continue
        if not stripped or stripped == "---":
            flush_paragraph()
            i += 1
            continue
        paragraph_lines.append(stripped)
        i += 1

    flush_paragraph()
    flush_table()
    if in_code and code_lines:
        story.append(Preformatted("\n".join(code_lines), styles["CodeCustom"]))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E0E2"))
    canvas.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, 0.35 * inch,
                      "Examiner report | Project evidence cutoff: 29 September 2026")
    canvas.drawRightString(7.85 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter,
        rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.62 * inch, bottomMargin=0.75 * inch,
        title="Examiner-Ready Project Report: Kubernetes Autoscaling",
        author="Project team",
    )
    story = [
        Spacer(1, 1.1 * inch),
        Paragraph("Deep Learning-Based Kubernetes Autoscaling", styles["CoverTitle"]),
        Paragraph("Paper reproduction and cold-start-aware planning optimization",
                  styles["H2Custom"]),
        Spacer(1, 0.3 * inch),
        Paragraph("Examiner-ready technical report", styles["BodyCustom"]),
        Paragraph("Prepared 29 September 2026", styles["SmallCustom"]),
        Spacer(1, 0.4 * inch),
        Paragraph("Student details were not present in the supplied project materials.",
                  styles["SmallCustom"]),
        Spacer(1, 1.1 * inch),
        Paragraph("Evidence status: the NASA baseline was rerun; the controlled KIND pilot "
                  "shows a consistent two-pair direction. It is preliminary (n=2), not a "
                  "statistical or publication-grade result.", styles["QuoteCustom"]),
        PageBreak(),
    ]
    story.extend(parse_markdown(SOURCE))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
