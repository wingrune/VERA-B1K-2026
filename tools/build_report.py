#!/usr/bin/env python3
"""Render the editable technical report to PDF. Requires reportlab."""
from html import escape
from pathlib import Path
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, XPreformatted

ROOT = Path(__file__).resolve().parents[1]
FONT_ROOT = Path('/usr/share/fonts/truetype/dejavu')
if FONT_ROOT.is_dir():
    for name, filename in [('Body', 'DejaVuSans.ttf'), ('Body-Bold', 'DejaVuSans-Bold.ttf'),
                           ('Title', 'DejaVuSerif.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / filename)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body', boldItalic='Body-Bold')
else:
    # ReportLab's built-in fonts keep rendering available on non-Linux hosts.
    for alias, base in [('Body', 'Helvetica'), ('Body-Bold', 'Helvetica-Bold'), ('Title', 'Times-Roman'), ('Mono', 'Courier')]:
        pdfmetrics.registerFont(pdfmetrics.Font(alias, base, 'WinAnsiEncoding'))

INK = colors.HexColor('#252820')
ORANGE = colors.HexColor('#ce4a25')
MUTED = colors.HexColor('#62685a')
STYLES = {
    'body': ParagraphStyle('body', fontName='Body', fontSize=9, leading=14, textColor=INK, spaceAfter=10, splitLongWords=True),
    'title': ParagraphStyle('title', fontName='Title', fontSize=25, leading=30, textColor=INK, spaceAfter=15),
    'h2': ParagraphStyle('h2', fontName='Body-Bold', fontSize=14, leading=19, textColor=ORANGE, spaceBefore=10, spaceAfter=12, keepWithNext=True),
    'h3': ParagraphStyle('h3', fontName='Body-Bold', fontSize=10, leading=15, textColor=INK, spaceBefore=7, spaceAfter=7, keepWithNext=True),
    'cell': ParagraphStyle('cell', fontName='Body', fontSize=7.5, leading=11, textColor=INK),
    'code': ParagraphStyle('code', fontName='Mono', fontSize=7.1, leading=11, backColor=colors.HexColor('#eeeee5'), borderPadding=9, spaceBefore=5, spaceAfter=14),
}


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" color="#b34423">\1</a>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Mono" size="8">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return text


def render():
    lines = (ROOT / 'docs/technical-report.md').read_text().splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('<!-- pagebreak'):
            story.append(PageBreak())
            i += 1
        elif line.startswith('#'):
            depth = len(line) - len(line.lstrip('#'))
            style = 'title' if depth == 1 else 'h2' if depth == 2 else 'h3'
            story.append(Paragraph(inline(line.lstrip('#').strip()), STYLES[style]))
            i += 1
        elif line.startswith('```'):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith('```'):
                block.extend(textwrap.wrap(lines[i], width=88, replace_whitespace=False, drop_whitespace=False) or [''])
                i += 1
            story.append(XPreformatted(escape('\n'.join(block)), STYLES['code']))
            i += 1
        elif line.startswith('|'):
            table_rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', c) for c in cells):
                    table_rows.append([Paragraph(inline(c), STYLES['cell']) for c in cells])
                i += 1
            widths = [233, 112, 150] if 'Successful' in ''.join(str(c) for c in table_rows[0]) else [150, 55, 290]
            table = Table(table_rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6e9df')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7), ('LINEBELOW', (0, 0), (-1, -1), .4, colors.HexColor('#d8d8cc'))]))
            story.extend([table, Spacer(1, 13)])
        else:
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '```', '<!--')):
                para.append(lines[i]); i += 1
            story.append(Paragraph(inline(' '.join(para)), STYLES['body']))
    out = ROOT / 'docs/technical-report.pdf'
    def furniture(canvas, doc):
        w, h = A4
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d8d8cc'))
        canvas.line(50, h - 38, w - 50, h - 38)
        canvas.setFont('Body-Bold', 7)
        canvas.setFillColor(ORANGE)
        canvas.drawString(50, h - 29, 'VERA / BEHAVIOR CHALLENGE 2026')
        canvas.setFont('Body', 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(50, 29, 'VERA-Volve | System and release report | November 9, 2026')
        canvas.drawRightString(w - 50, 29, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=53, bottomMargin=48,
        title='VERA-Volve for the 2026 BEHAVIOR Challenge', author='VERA team', subject='Architecture and pinned public validation evidence')
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
    print(out)


if __name__ == '__main__':
    render()
