"""Simplified 2-panel batch distribution-shift figure for slide A1."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = "."
ASSETS = "./figures_out"
PREV = "./figures_out"

for f in font_manager.findSystemFonts(fontpaths=None, fontext="otf"):
    if "texgyretermes" in f.lower().replace("-", ""):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": "#3c4540", "axes.linewidth": 1.1,
    "axes.labelsize": 16, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "axes.labelcolor": "#1f2723", "text.color": "#1f2723",
})
GREEN, AMBER = "#2e7d52", "#d97a1f"

df = pd.read_csv("./feature_matrix_full.csv", index_col=0)
g1 = df.index.str.startswith("b1")
feats = [("dq_variance", "log var($\\Delta Q_{100-10}$)"),
         ("chargetime_cycle100", "charge time, cycle 100")]

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=200)
for ax, (col, lab) in zip(axes, feats):
    a, b = df.loc[g1, col].values, df.loc[~g1, col].values
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    bins = np.linspace(lo, hi, 16)
    ax.hist(a, bins=bins, color=GREEN, alpha=0.55, label="Batch 1/2 (35)")
    ax.hist(b, bins=bins, color=AMBER, alpha=0.55, label="Batch 3 (40)")
    ax.set_xlabel(lab)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("cells")
axes[0].legend(fontsize=13, frameon=False, loc="upper left")
fig.suptitle("The two strongest features shift between batch groups",
             fontsize=17.5, fontweight="bold", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(ASSETS + "/ppt_shift.png", facecolor="white"); plt.close(fig)

print('DONE')
