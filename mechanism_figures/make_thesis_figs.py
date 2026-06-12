"""Thesis-style variants of the four mechanism figures (no in-figure titles)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

ROOT = "."
FIGS = "./figures_out"
PREV = "./figures_out"

for f in font_manager.findSystemFonts(fontpaths=None, fontext="otf"):
    if "texgyretermes" in f.lower().replace("-", ""):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": "#222", "axes.linewidth": 0.9,
    "axes.labelsize": 13, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
})
GREEN, AMBER, RED = "#2e7d52", "#d97a1f", "#b3422e"

z = np.load(ROOT + "/tmp/figcache_2018.npz", allow_pickle=True)
keys, cl, q10, q100, qd = z["keys"], z["cl"], z["q10"], z["q100"], z["qd"]
EXC = {f"b12c{i}" for i in (8, 10, 12, 13, 22)} | {f"b3c{i}" for i in (2, 23, 32, 37, 42, 43)}
m = np.array([k not in EXC for k in keys])
keys, cl, q10, q100 = keys[m], cl[m], q10[m], q100[m]
qd = [qd[i] for i in np.where(m)[0]]
V = np.linspace(3.5, 2.0, q10.shape[1])
dq = q100 - q10
cmap = plt.cm.viridis
norm = LogNorm(vmin=cl.min(), vmax=cl.max())

# ---------- 31 · knee window ----------
def smooth(x, w=9):
    k = np.ones(w) / w
    return np.convolve(np.pad(np.asarray(x, float), (w // 2, w - 1 - w // 2), mode="edge"), k, "valid")

def clean_curve(i):
    q = np.asarray(qd[i], float).flatten()
    q = q[(q > 0.7) & (q < 1.25)]
    if len(q) < 120: return None
    qs = smooth(q)
    if np.max(np.abs(np.diff(qs[:150]))) > 0.010: return None
    return qs

fig, ax = plt.subplots(figsize=(7.0, 4.1), dpi=300)
targets = np.quantile(cl, np.linspace(0.04, 0.96, 9))
picked, used = [], set()
for t in targets:
    for i in np.argsort(np.abs(cl - t)):
        if i in used: continue
        qs = clean_curve(i)
        if qs is not None:
            picked.append((i, qs)); used.add(i); break
picked = sorted(picked, key=lambda p: cl[p[0]])[:7]
knees = []
for i, qs in picked:
    c = np.arange(1, len(qs) + 1)
    ax.plot(c, qs, color=cmap(norm(cl[i])), lw=1.6, alpha=0.95, zorder=2)
    if len(qs) > 220:
        chord = qs[0] + (qs[-1] - qs[0]) * (c - 1) / (len(qs) - 1)
        k = 100 + int(np.argmax((qs - chord)[100:len(qs) - 10]))
        if qs[k] > 0.9:
            knees.append((c[k], qs[k]))
            ax.plot(c[k], qs[k], "o", ms=6, mfc="white", mec=RED, mew=1.8, zorder=4)
ax.axvspan(0, 100, color=GREEN, alpha=0.13, zorder=0)
ax.text(52, 0.952, "observation\nwindow\n(first 100\ncycles)", fontsize=10,
        color=GREEN, ha="center", va="center", fontweight="bold")
ax.axhline(0.88, color="#444", lw=0.9, ls=(0, (5, 4)))
kx, ky = max(knees, key=lambda t: t[0])
ax.annotate("knee onsets ($\\circ$) emerge only hundreds\nof cycles after the window closes",
            xy=(kx - 8, ky + 0.005), xytext=(kx * 0.52, 1.103),
            fontsize=11, color=RED, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3,
                            connectionstyle="arc3,rad=0.16"))
cmax = max(len(qs) for _, qs in picked)
ax.text(cmax * 1.02, 0.869, "end of life (80%)", fontsize=10, color="#444",
        ha="right", va="top")
ax.set_xlim(-15, cmax * 1.04); ax.set_ylim(0.852, 1.125)
ax.set_xlabel("Cycle number"); ax.set_ylabel("Discharge capacity (Ah)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIGS + "/31_knee_window__fig3x.png", facecolor="white"); plt.close(fig)

# ---------- 32 · dQ mechanism map ----------
fig, ax = plt.subplots(figsize=(8.4, 4.3), dpi=300)
for i in np.argsort(cl)[::-1]:
    ax.plot(V, dq[i], color=cmap(norm(cl[i])), lw=1.0, alpha=0.85)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=ax, pad=0.012, ticks=[500, 1000, 1800])
cb.ax.set_yticklabels(["500", "1000", "1800"]); cb.ax.minorticks_off()
cb.set_label("cycle life", fontsize=12); cb.ax.tick_params(labelsize=10.5)
spread = dq.std(axis=0)
j = int(np.argmax(spread)); vj = V[j]
ax.axvspan(vj - 0.12, vj + 0.12, color=AMBER, alpha=0.14, zorder=0)
ax.annotate("cell-to-cell spread peaks here —\nthe var($\\Delta Q$) feature reads this window",
            xy=(vj + 0.045, 0.0105), xytext=(3.475, 0.0175),
            fontsize=11, color="#8a5310", ha="left", va="top",
            arrowprops=dict(arrowstyle="-|>", color="#8a5310", lw=1.2,
                            connectionstyle="arc3,rad=-0.12"))
jmin = int(np.argmin(dq.min(axis=0)))
ax.annotate("deepest loss at the plateau edge —\nconsistent with LLI / SEI growth",
            xy=(V[jmin] - 0.03, dq.min()), xytext=(2.62, dq.min() * 0.93),
            fontsize=11, color="#1f2723", ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2))
ax.text(2.72, 0.013, "long-lived cells stay flat ($\\Delta Q \\approx 0$)",
        fontsize=11, color=GREEN, ha="left", va="bottom")
ax.axhline(0, color="#444", lw=0.7, ls=(0, (4, 3)), alpha=0.6)
ax.set_xlim(2.0, 3.5); ax.invert_xaxis()
ax.set_ylim(dq.min() * 1.14, 0.029)
ax.set_xlabel("Voltage (V)"); ax.set_ylabel("$\\Delta Q_{100-10}(V)$  (Ah)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIGS + "/32_dq_mechanism_map__fig4x.png", facecolor="white"); plt.close(fig)

# ---------- 33 · Nyquist schematic ----------
fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=300)
R0, Rct = 1.0, 2.4
th = np.linspace(np.pi, 0, 300)
xs = R0 + Rct / 2 * (1 + np.cos(th)); ys = Rct / 2 * np.sin(th)
t = np.linspace(0, 1, 140)
xt = R0 + Rct + 2.45 * t; yt = 2.30 * t ** 0.93
ax.plot(np.r_[xs, xt], np.r_[ys, yt], color="#27563f", lw=2.6,
        solid_capstyle="round", zorder=2)
ax.plot([R0], [0], "o", ms=10, mfc=RED, mec="white", mew=1.8, zorder=4, clip_on=False)
ax.annotate("what a single internal-resistance\nnumber sees: one point",
            xy=(R0 - 0.02, 0.06), xytext=(0.52, 1.62), fontsize=11.5, color=RED,
            ha="left", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
ax.annotate("", xy=(R0 + Rct + 2.25, 2.62), xytext=(1.15, 2.62),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))
ax.text((R0 + Rct + 2.25 + 1.15) / 2, 2.72, "what EIS sees: the whole internal state",
        fontsize=12, color=GREEN, ha="center", fontweight="bold")
for x, lab in ((R0 + 0.28, "high $f$\nohmic + SEI"),
               (R0 + Rct / 2, "mid $f$\ncharge transfer"),
               (R0 + Rct + 1.45, "low $f$\ndiffusion")):
    ax.text(x, -0.30, lab, fontsize=10.5, color="#555", ha="center", va="top")
ax.set_xlabel("Re $Z$  (a.u.)"); ax.set_ylabel("$-$Im $Z$  (a.u.)")
ax.set_xlim(0.35, R0 + Rct + 2.75); ax.set_ylim(-0.72, 3.0)
ax.set_xticks([]); ax.set_yticks([])
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIGS + "/33_nyquist_schematic__fig5x.png", facecolor="white"); plt.close(fig)

# ---------- 34 · EIS band importance ----------
zb = np.load(ROOT + "/tmp/natcomm2022_capacity_vs_eis/variable_discharge_features.npz",
             allow_pickle=True)
yb = zb["y"]; X = zb["data_2"]
freq = 10 ** np.linspace(-1.66, 3.9, 100)
rho_re = np.array([abs(spearmanr(X[:, i], yb).statistic) for i in range(100)])
rho_im = np.array([abs(spearmanr(X[:, 100 + i], yb).statistic) for i in range(100)])
fig, ax = plt.subplots(figsize=(7.0, 4.1), dpi=300)
ax.axvspan(freq[0], 1.0, color="#7d6aa8", alpha=0.07)
ax.axvspan(1.0, 200.0, color=GREEN, alpha=0.08)
ax.axvspan(200.0, freq[-1], color="#5a87b0", alpha=0.07)
for x, lab in ((0.12, "low $f$ · diffusion"), (14, "mid $f$ · charge transfer\n+ interfaces"), (1450, "high $f$ · ohmic\n+ SEI")):
    ax.text(x, 1.005, lab, fontsize=9.5, color="#555", ha="center", va="bottom")
ax.plot(freq, rho_re, color=GREEN, lw=2.2, label="Re $Z$ ($f$)")
ax.plot(freq, rho_im, color=AMBER, lw=2.2, label="$-$Im $Z$ ($f$)")
for fz in (2.16, 17.8):
    ax.axvline(fz, color=RED, lw=1.3, ls=(0, (5, 4)), alpha=0.85)
ax.annotate("2.16 & 17.8 Hz — the two predictive\nfrequencies identified by\nZhang et al. (2020)",
            xy=(2.05, 0.115), xytext=(0.16, 0.105),
            fontsize=10, color=RED, ha="center", va="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1,
                            connectionstyle="arc3,rad=0.12"))
i_pk = int(rho_im.argmax())
ax.annotate("ageing information\nconcentrates here",
            xy=(freq[i_pk], rho_im[i_pk]), xytext=(0.07, 0.83),
            fontsize=11, color="#1f2723", ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2,
                            connectionstyle="arc3,rad=-0.15"))
ax.set_xscale("log")
ax.set_xlim(freq[0], freq[-1]); ax.set_ylim(0, 1.0)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Spearman $\\rho$| with capacity")
ax.legend(fontsize=10.5, loc="center right", frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIGS + "/34_eis_band_importance__fig5x.png", facecolor="white"); plt.close(fig)

print('DONE')
