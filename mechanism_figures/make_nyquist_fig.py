"""Nyquist schematic, EIS house palette (crimson/gray). Thesis fig 33_."""
import numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts(fontpaths=None, fontext="otf"):
    if "texgyretermes" in f.lower().replace("-", ""):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Times New Roman","DejaVu Serif"],
    "axes.edgecolor":"#55606A","axes.linewidth":1.2,"text.color":"#1f2723","axes.labelcolor":"#1f2723"})
CRIM,CRIMD,SLATE,GRAYB = "#B42B40","#7E1E2C","#55606A","#9AA0A6"
fig, ax = plt.subplots(figsize=(9.4, 4.9), dpi=200)
x0 = 1.0; D = 4.2
th = np.linspace(0, np.pi, 200)
xa = x0 + D/2 + (D/2)*np.cos(th[::-1]); ya = (D/2)*np.sin(th[::-1])
xt = np.linspace(x0+D, x0+D+2.6, 60); yt = (xt-(x0+D))*0.95 + ya[-1]
# region shading
ax.axvspan(0.0, x0+0.55, color="#e9ebee", alpha=0.9)
ax.axvspan(x0+0.55, x0+D-0.15, color="#f2eff0", alpha=0.9)
ax.axvspan(x0+D-0.15, 8.6, color="#eef1f0", alpha=0.9)
for xx, lab in ((0.62,"high $f$\nohmic + SEI"), (x0+D/2,"mid $f$\ncharge transfer + interfaces"), (x0+D+1.6,"low $f$\ndiffusion")):
    ax.text(xx, 3.62, lab, ha="center", va="top", fontsize=13.5, color="#5c646b")
# spectrum
ax.plot(xa, ya, color=SLATE, lw=3.4, solid_capstyle="round")
ax.plot(xt, yt, color=SLATE, lw=3.4, solid_capstyle="round")
ax.annotate("the full spectrum resolves ohmic + SEI,\ncharge-transfer and diffusion regions",
            xy=(x0+D+1.55, yt[36]), xytext=(6.15, 2.72), fontsize=13, color="#1f2723", ha="center",
            arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.5, connectionstyle="arc3,rad=-0.15"))
# single IR point
ax.scatter([x0],[0.02], s=150, color=CRIM, zorder=5, edgecolor="white", linewidth=1.4)
ax.annotate("a per-cycle internal-resistance summary\nsamples only this one point",
            xy=(x0+0.06, 0.10), xytext=(3.12, 0.52), fontsize=12.5, color=CRIM, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-|>", color=CRIM, lw=1.7, connectionstyle="arc3,rad=-0.18"))
ax.set_xlim(0, 8.6); ax.set_ylim(0, 3.8)
ax.set_xlabel("Re $Z$ (a.u.)", fontsize=14.5); ax.set_ylabel("$-$Im $Z$ (a.u.)", fontsize=14.5)
ax.set_xticks([]); ax.set_yticks([])
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.075, right=0.98, top=0.97, bottom=0.13)
out = "./33_nyquist_schematic__fig5x.png"
fig.savefig(out, facecolor="white"); print("saved", out)
