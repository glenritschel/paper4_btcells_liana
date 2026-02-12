"""
_plotting.py

Centralized matplotlib settings for publication-quality figures.
All PNG outputs default to 300 DPI.
"""

import matplotlib.pyplot as plt

def set_publication_style():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"] = 10

