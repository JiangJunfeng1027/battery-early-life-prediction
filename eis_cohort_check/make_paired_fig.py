"""Cohort-wide with/without-EIS comparison figure. Publication style (matches thesis mechanism figs)."""
import numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts(fontpaths=None, fontext="otf"):
    if "texgyretermes" in f.lower().replace("-", ""):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": "#3c4540", "axes.linewidth": 1.1,
    "axes.labelsize": 16, "xtick.labelsize": 13.5, "ytick.labelsize": 12,
    "xtick.color": "#3c4540", "ytick.color": "#3c4540",
    "axes.labelcolor": "#1f2723", "text.color": "#1f2723",
})
GREEN, AMBER, RED, GRAY = "#B42B40", "#7E1E2C", "#7E1E2C", "#9AA0A6"  # EIS house palette: crimson=with-EIS, gray=without
d = np.load("./eis_ridge_loocv.npz", allow_pickle=True)
cells, cap, eis = d["cells"], d["cap"], d["eis"]
fig = plt.figure(figsize=(12.6, 5.2), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.24,
                      left=0.065, right=0.975, top=0.78, bottom=0.175)
fig.text(0.065, 0.955, "Adding EIS improves the held-out error for every one of 24 cells",
         fontsize=20, fontweight="bold", ha="left")
fig.text(0.065, 0.895, "Same ridge setup as the single-cell illustration, now leave-one-cell-out across the whole cohort "
         "(Jones et al. 2022 public data, 2,866 cycle-points)", fontsize=13.5, color="#5a665f", ha="left")
# Panel A: parity scatter
ax = fig.add_subplot(gs[0, 0])
lim = (1.2, 5.4)
ax.plot(lim, lim, ls=(0, (5, 4)), color="#3c4540", lw=1.6)
ax.scatter(cap, eis, s=68, color=GREEN, edgecolor="white", linewidth=0.9, zorder=3)
ax.fill_between(lim, lim, lim[0], color=GREEN, alpha=0.04)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Per-cell MAE, capacity/usage features only (cycles)")
ax.set_ylabel("Per-cell MAE, + EIS spectrum (cycles)")
ax.text(4.55, 1.52, "all 24 cells fall below\nthe parity line", fontsize=13.5, color=GREEN,
        fontweight="bold", ha="center")
ax.annotate("no-change line", xy=(4.55, 4.55), xytext=(3.6, 4.9), fontsize=11.5, color="#3c4540",
            arrowprops=dict(arrowstyle="-|>", color="#3c4540", lw=1.2))
# Panel B: sorted dumbbells
ax2 = fig.add_subplot(gs[0, 1])
order = np.argsort(cap)[::-1]
yy = np.arange(len(order))
for i, j in enumerate(order):
    ax2.plot([eis[j], cap[j]], [i, i], color="#d5d9d4", lw=2.2, zorder=1)
ax2.scatter(cap[order], yy, s=42, color=GRAY, zorder=2, label="capacity/usage only")
ax2.scatter(eis[order], yy, s=46, color=GREEN, zorder=3, label="+ EIS spectrum")
ax2.set_yticks(yy); ax2.set_yticklabels(cells[order], fontsize=9.5)
ax2.set_ylim(-0.8, len(order) - 0.2); ax2.invert_yaxis()
ax2.set_xlabel("Held-out MAE per cell (cycles)")
ax2.legend(loc="lower right", bbox_to_anchor=(0.99, 0.30), fontsize=12.5, frameon=False)
ax2.text(0.985, 0.055,
         "mean MAE 3.13 → 2.20 cycles\nmedian improvement −27%\nWilcoxon signed-rank  $p = 1.2\\times10^{-7}$",
         transform=ax2.transAxes, fontsize=13, color="#1f2723", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.45", fc="#faf3f4", ec=GREEN, lw=1.3))
for a in (ax, ax2):
    a.grid(True, color="#e4e7e3", lw=0.7); a.set_axisbelow(True)
fig.text(0.065, 0.022, "Corroborated by a local rerun of the original forecasting pipeline on the same 24 cells: "
         "median error 8.3% → 6.0%, R² 0.81 → 0.91.", fontsize=11.5, color="#5a665f", style="italic")
out = "./35_eis_paired_loocv__fig5x.png"
fig.savefig(out, facecolor="white"); print("saved", out)
