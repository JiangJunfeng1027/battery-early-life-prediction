"""Four-arm pilot figures: thesis 3-panel + PPT singles. Publication style."""
import numpy as np, matplotlib, os, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts(fontpaths=None, fontext="otf"):
    if "texgyretermes" in f.lower().replace("-", ""):
        font_manager.fontManager.addfont(f)
BASE = {"font.family":"serif","font.serif":["TeX Gyre Termes","Times New Roman","DejaVu Serif"],
        "axes.edgecolor":"#3c4540","axes.linewidth":1.1,"xtick.color":"#3c4540","ytick.color":"#3c4540",
        "axes.labelcolor":"#1f2723","text.color":"#1f2723"}
plt.rcParams.update(BASE)
CRIM,CRIML,CRIMD,GRAYB = "#B42B40","#CE5468","#7E1E2C","#9AA0A6"
GREEN,GREEN2,AMBER,RED,GRAY = CRIM,CRIML,CRIMD,GRAYB,"#8a938d"  # EIS house palette
D = np.load("./deep_suite.npz", allow_pickle=True)
arms = {k[4:]: D[k] for k in D.files if k.startswith("arm_")}
KS, ECON = D["econ_ks"], D["econ"]
SH = json.loads(str(D["shift_json"]))
cap_m, full_m = arms["cap"].mean(), arms["cap+eis"].mean()

def panel_arms(ax, fs=1.0):
    rows = [("capacity/usage only (14 D)", arms["cap"].mean(), GRAYB, "—"),
            ("+ EIS spectrum (214 D)", arms["cap+eis"].mean(), GREEN, "24/24 ↓"),
            ("+ 3-band summary (20 D)", arms["cap+bands6"].mean(), GREEN2, "23/24 ↓"),
            ("+ 3 literature freqs (20 D)", arms["cap+lit3"].mean(), GREEN2, "24/24 ↓"),
            ("EIS alone (200 D)", arms["eis_only"].mean(), AMBER, "")]
    yy = np.arange(len(rows))[::-1]
    for (lab, v, c, tag), y in zip(rows, yy):
        if lab.startswith("EIS alone"):
            ax.barh(y, v, height=0.62, color="white", edgecolor=c, hatch="///", lw=1.6)
        else:
            ax.barh(y, v, height=0.62, color=c, alpha=0.92 if c==GRAYB else 1.0)
        ax.text(v+0.07, y, f"{v:.2f}", va="center", fontsize=13*fs, fontweight="bold", color=c)
        ax.text(-0.12, y, lab, va="center", ha="right", fontsize=11.5*fs)
        ax.text(7.05, y, tag, va="center", ha="right", fontsize=10.5*fs, color=c, fontweight="bold")
    ax.set_xlim(0, 7.15); ax.set_ylim(-0.95, 4.6); ax.set_yticks([])
    ax.set_xlabel("Per-cell held-out MAE (cycles), LOOCV mean", fontsize=12*fs)
    ax.axvline(cap_m, color=GRAY, lw=1.2, ls=(0,(4,4)), alpha=0.8, zorder=0)
    ax.text(3.6, -0.62, "worse alone → a complement, not a replacement", fontsize=11*fs,
            color=AMBER, ha="center", fontweight="bold")
def panel_econ(ax, fs=1.0):
    x = np.arange(len(KS))
    ax.axhline(cap_m, color=GRAY, lw=1.6, ls=(0,(4,4))); ax.text(len(KS)-.6, cap_m+0.03, "capacity only 3.13", fontsize=10.5*fs, color=GRAY, ha="right")
    ax.axhline(full_m, color=GREEN, lw=1.6, ls=(0,(4,4))); ax.text(len(KS)-3.3, full_m+0.035, "full spectrum 2.20", fontsize=10.5*fs, color=GREEN, ha="right")
    ax.plot(x, ECON, "-o", color=CRIM, lw=2.6, ms=7, mfc="white", mew=2)
    for i, k in enumerate(KS):
        rec = (cap_m-ECON[i])/(cap_m-full_m)*100
        if k in (1,3):
            ax.annotate(f"{rec:.0f}% of the gain", xy=(x[i],ECON[i]), xytext=(x[i]+0.35,ECON[i]+0.24-(0.10 if k==3 else 0)),
                        fontsize=11*fs, color="#1f2723", fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color="#3c4540", lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(KS, fontsize=10.5*fs)
    ax.set_xlabel("Number of frequencies kept (selected inside each fold)", fontsize=12*fs)
    ax.set_ylabel("MAE (cycles)", fontsize=12*fs); ax.set_ylim(2.05, 3.35)
    ax.text(0.03, 0.03, "most-picked 1st frequency: 6.5–7.4 Hz (19/24 folds)",
            transform=ax.transAxes, fontsize=9.8*fs, color=CRIMD, style="italic")
def panel_shift(ax, fs=1.0, headline=True):
    groups = [("train 16\n→ test 8", SH["cap"][0], SH["cap+eis"][0]), ("train 8\n→ test 16", SH["cap"][1], SH["cap+eis"][1])]
    xc = [0, 1]; w = 0.32
    for x0, (lab, a, b) in zip(xc, groups):
        ax.bar(x0-w/2, a, w, color=RED, alpha=0.85); ax.text(x0-w/2, a+0.08, f"{a:.2f}", ha="center", fontsize=12.5*fs, fontweight="bold", color=RED)
        ax.bar(x0+w/2, b, w, color=GREEN); ax.text(x0+w/2, b+0.08, f"{b:.2f}", ha="center", fontsize=12.5*fs, fontweight="bold", color=GREEN)
        pa = (a/cap_m-1)*100; pb = (b/full_m-1)*100
        ax.text(x0-w/2, a/2, f"+{pa:.0f}%", ha="center", fontsize=10.5*fs, color="#3a4045", fontweight="bold")
        ax.text(x0+w/2, b/2, f"+{pb:.0f}%", ha="center", fontsize=10.5*fs, color="white", fontweight="bold")
    ax.axhline(cap_m, color=RED, lw=1.3, ls=(0,(4,4)), alpha=0.7); ax.text(1.72, cap_m+0.09, "no shift: 3.13", fontsize=9.5*fs, color=RED)
    ax.axhline(full_m, color=GREEN, lw=1.3, ls=(0,(4,4)), alpha=0.7); ax.text(1.72, full_m+0.09, "no shift: 2.20", fontsize=9.5*fs, color=GREEN)
    ax.set_xticks(xc); ax.set_xticklabels([g[0] for g in groups], fontsize=10.5*fs)
    ax.set_xlim(-0.62, 2.3); ax.set_ylim(0, 6.35)
    ax.set_ylabel("Held-out MAE (cycles)", fontsize=12*fs)
    if headline:
        ax.text(0.02, 0.965, "cohort-shift penalty ≈ halved with EIS", transform=ax.transAxes, fontsize=12*fs, fontweight="bold", color=GREEN, va="top")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED,label="capacity/usage only"),Patch(color=GREEN,label="+ EIS spectrum")],
              loc="upper right", bbox_to_anchor=(1.0, 0.99), fontsize=10*fs, frameon=False)
# ---- thesis 3-panel
fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.5), dpi=200, gridspec_kw=dict(width_ratios=[1.28,1,1], wspace=0.34, left=0.145, right=0.985, top=0.86, bottom=0.155))
panel_arms(axs[0]); panel_econ(axs[1]); panel_shift(axs[2], headline=False)
for ax, t in zip(axs, ["(a) The four-arm comparison, executed","(b) How much spectrum is needed?","(c) Does EIS survive a cohort shift?"]):
    ax.set_title(t, fontsize=13.5, fontweight="bold", pad=10)
    ax.grid(True, color="#e4e7e3", lw=0.6); ax.set_axisbelow(True)
fig.savefig("./36_eis_fourarm_pilot__fig5x.png", facecolor="white")
print("thesis fig saved")
# ---- ppt singles
for name, fn, size in [("ppt_fourarm", panel_arms, (8.6,5.2)), ("ppt_econ", panel_econ, (8.6,5.2)), ("ppt_shifteis", panel_shift, (8.6,5.2))]:
    f2, ax2 = plt.subplots(figsize=size, dpi=200)
    fn(ax2, fs=1.35)
    ax2.grid(True, color="#e4e7e3", lw=0.7); ax2.set_axisbelow(True)
    f2.subplots_adjust(left=0.40 if name=="ppt_fourarm" else 0.14, right=0.97, top=0.95, bottom=0.16)
    f2.savefig(f"./{name}.png", facecolor="white")
    plt.close(f2)
    print(name, "saved")
