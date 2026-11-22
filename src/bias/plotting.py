import seaborn as sns

INK = "#242424"
BLUE = "#0072B2"
OCHRE = "#E69F00"
VERMILION = "#D55E00"
PALETTE = (INK, BLUE, OCHRE, VERMILION)
SEQUENTIAL = "rocket"
WIDE = (7, 2.55)


def use_style():
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.facecolor": "white",
            "axes.axisbelow": True,
            "axes.grid.axis": "y",
            "figure.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        },
    )
