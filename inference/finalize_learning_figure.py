"""Apply paper-layout fixes to the generated learning-curve PDF."""

from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from reportlab.pdfgen import canvas


FIGURE = Path("article/latex/fig_learning_curves.pdf")
OVERLAY = Path("/private/tmp/fig_learning_curves_legend.pdf")
PAGE_WIDTH = 504.0
PAGE_HEIGHT = 212.4
LEGEND = (
    ("GRU-L1", "#4C78A8", (1.0, 1.8)),
    ("GRU-L2", "#4C78A8", (5.0, 2.2)),
    ("GRU-L3", "#4C78A8", ()),
    ("MoSAIC-L2C4", "#F58518", (5.0, 2.2)),
    ("MoSAIC-L3C4", "#F58518", ()),
    ("Transformer-256", "#54A24B", (6.0, 2.0, 1.0, 2.0)),
)


def draw_legend():
    pdf = canvas.Canvas(str(OVERLAY), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, PAGE_WIDTH, 31, stroke=0, fill=1)

    positions = ((78, 21), (243, 21), (408, 21), (66, 10), (239, 10), (395, 10))
    for (label, color, dash), (x, y) in zip(LEGEND, positions):
        red = int(color[1:3], 16) / 255
        green = int(color[3:5], 16) / 255
        blue = int(color[5:7], 16) / 255
        pdf.setStrokeColorRGB(red, green, blue)
        pdf.setLineWidth(1.5)
        pdf.setDash(dash)
        pdf.line(x - 24, y + 2, x - 7, y + 2)
        pdf.setDash()
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica", 7.2)
        pdf.drawString(x, y, label)
    pdf.save()


def main():
    draw_legend()
    source = PdfReader(str(FIGURE)).pages[0]
    source.add_transformation(Transformation().scale(sx=0.975, sy=1.0))
    page = PageObject.create_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.merge_page(source)
    page.merge_page(PdfReader(str(OVERLAY)).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    with FIGURE.open("wb") as output:
        writer.write(output)


if __name__ == "__main__":
    main()
