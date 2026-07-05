"""Frequency-band importance of EIS for ageing information. Publication style."""
import numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr

ROOT = "."
ASSETS = "."
PREV = "."

for f in font_manager.findSystemFonts(fontpaths=None, fontext="otf"):
    if "texgyretermes" in f.lower().replace("-", ""):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": "#3c4540", "axes.linewidth": 1.1,
    "axes.labelsize": 17, "xtick.labelsize": 14.5, "ytick.labelsize": 14.5,
    "xtick.color": "#3c4540", "ytick.color": "#3c4540",
    "axes.labelcolor": "#1f2723", "text.color": "#1f2723",
})
GREEN, AMBER, RED = "#55606A", "#B42B40", "#7E1E2C"  # EIS house palette: slate Re, crimson Im, dark-red literature lines

z = np.load(ROOT + "/tmp/natcomm2022_capacity_vs_eis/variable_discharge_features.npz",
            allow_pickle=True)
y = z["y"]; X = z["data_2"]
freq = 10 ** np.linspace(-1.66, 3.9, 100)
rho_re = np.array([abs(spearmanr(X[:, i], y).statistic) for i in range(100)])
rho_im = np.array([abs(spearmanr(X[:, 100 + i], y).statistic) for i in range(100)])
print("peak Re:", freq[rho_re.argmax()], rho_re.max())
print("peak Im:", freq[rho_im.argmax()], rho_im.max())

fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=200)
# physical regions
ax.axvspan(freq[0], 1.0, color="#8a8f94", alpha=0.10)
ax.axvspan(1.0, 200.0, color="#B42B40", alpha=0.05)
ax.axvspan(200.0, freq[-1], color="#8a8f94", alpha=0.10)
for x, lab in ((0.12, "low $f$ · diffusion"), (14, "mid $f$ · charge transfer\n+ interfaces"), (1450, "high $f$ · ohmic\n+ SEI")):
    ax.text(x, 1.005, lab, fontsize=12.5, color="#5a665f", ha="center", va="bottom")
ax.plot(freq, rho_re, color=GREEN, lw=3.0, label="Re $Z$ ($f$)")
ax.plot(freq, rho_im, color=AMBER, lw=3.0, label="$-$Im $Z$ ($f$)")
# Zhang 2020 canonical frequencies
for fz in (2.16, 17.8):
    ax.axvline(fz, color=RED, lw=1.8, ls=(0, (5, 4)), alpha=0.85)
ax.annotate("2.16 & 17.8 Hz — the two predictive\nfrequencies of Zhang et al. (2020)",
            xy=(2.05, 0.10), xytext=(0.14, 0.095),
            fontsize=12.5, color=RED, ha="center", va="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                            connectionstyle="arc3,rad=0.12"))
i_pk = int(rho_im.argmax())
ax.annotate("ageing information\nconcentrates here",
            xy=(freq[i_pk], rho_im[i_pk]), xytext=(0.42, 0.74),
            fontsize=14.5, color="#1f2723", ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#3c4540", lw=1.7,
                            connectionstyle="arc3,rad=-0.15"))
ax.set_xscale("log")
ax.set_xlim(freq[0], freq[-1]); ax.set_ylim(0, 1.0)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Spearman $\\rho$| with capacity")
ax.legend(fontsize=13.5, loc="center right", frameon=False)

for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(ASSETS + "/34_eis_band_importance__fig5x.png", facecolor="white"); plt.close(fig)

print('DONE')
