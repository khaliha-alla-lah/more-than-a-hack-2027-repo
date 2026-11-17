"""Build the upload PDF from the reviewed technical Markdown (optional ReportLab)."""
import argparse
import html
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak


def build(source, output):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyRelease', fontName='Helvetica', fontSize=10,
                              leading=14, spaceAfter=8, alignment=TA_LEFT))
    styles['Title'].fontSize = 20
    styles['Title'].leading = 25
    styles['Heading2'].textColor = colors.HexColor('#123c56')
    story = []
    chunks = source.read_text(encoding='utf-8').split('\n\n')
    for chunk in chunks:
        if chunk.startswith('## Validation'):
            story.append(PageBreak())
        chunk = chunk.replace('\u2014', ' - ').replace('\u2013', '-')
        if chunk.startswith('# '):
            text, style = chunk[2:], styles['Title']
        elif chunk.startswith('## '):
            text, style = chunk[3:], styles['Heading2']
        else:
            text, style = ' '.join(chunk.splitlines()), styles['BodyRelease']
        text = html.escape(text)
        text = re.sub(r'`([^`]+)`', r'<font name="Courier" size="8">\1</font>', text)
        story.append(Paragraph(text, style))
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#52606d'))
        canvas.drawString(18*mm, 12*mm, 'khaliha aala lah | More Than A Hack 2027')
        canvas.drawRightString(192*mm, 12*mm, str(document.page))
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=(210*mm,297*mm), leftMargin=18*mm,
                      rightMargin=18*mm, topMargin=16*mm, bottomMargin=20*mm,
                      title='khaliha aala lah - technical approach', author='khaliha aala lah').build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'docs/TECHNICAL.md')
    parser.add_argument('--output', type=Path, default=root/'submission-output/khaliha aala lah.pdf')
    args = parser.parse_args()
    build(args.source, args.output)
