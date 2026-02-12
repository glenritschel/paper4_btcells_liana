"""
_plotting.py

Centralized matplotlib settings + helper to save figures as BOTH PNG and PDF.
PNG is always 300 DPI. PDF is vector.
"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt


def set_publication_style() -> None:
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"] = 10


def savefig_png_pdf(
    out_path_no_ext: str,
    *,
    dpi: int = 300,
    transparent: bool = False,
    close: bool = True,
) -> tuple[str, str]:
    """
    Save the current matplotlib figure to:
      - <out_path_no_ext>.png (raster, dpi)
      - <out_path_no_ext>.pdf (vector)

    Returns: (png_path, pdf_path)
    """
    out_dir = os.path.dirname(out_path_no_ext)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    png_path = f"{out_path_no_ext}.png"
    pdf_path = f"{out_path_no_ext}.pdf"

    plt.savefig(png_path, dpi=dpi, transparent=transparent)
    plt.savefig(pdf_path, transparent=transparent)

    if close:
        plt.close()

    return png_path, pdf_path

