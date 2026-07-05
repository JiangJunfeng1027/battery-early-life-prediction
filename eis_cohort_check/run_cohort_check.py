"""Cohort-wide with/without-EIS comparison (thesis Fig 5.6 / Section 5, future-work de-risking).

Input: variable_discharge_features.npz — derived feature file produced by the official
code release of Jones et al. (2022), "Impedance-based forecasting of lithium-ion battery
performance amid uneven usage", Nat. Commun. (public variable-discharge dataset,
24 cells with both capacity/usage descriptors and EIS spectra; 2,866 cell-cycle points).
Place it in ./data/ (not redistributed here — obtain via the original release).

Two arms, identical otherwise:
  cap      = capacity/usage descriptors only (14 dims)
  cap+eis  = same + raw impedance spectrum (200 dims: 100 Re + 100 -Im)
Ridge(alpha=10) on standardized features, leave-one-cell-out over 24 cells.
Reports per-cell MAE, paired Wilcoxon, and writes eis_ridge_loocv.npz for the figure script.
"""
import numpy as np, os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

SRC = os.environ.get("JONES_NPZ", "./data/variable_discharge_features.npz")
z = np.load(SRC, allow_pickle=True)
cells, y = z["cell"], z["y"]
blocks = [z["data_0"].reshape(-1, 1), z["data_1"].reshape(-1, 1), z["data_3"],
          z["data_4"].reshape(-1, 1), z["data_5"].reshape(-1, 1),
          z["data_6"], z["data_7"], z["data_8"]]
Xcap = np.hstack(blocks)
Xeis = np.hstack([Xcap, z["data_2"]])
uc = sorted(set(cells.tolist()))
res = {}
for name, X in [("cap", Xcap), ("cap+eis", Xeis)]:
    per = {}
    for c in uc:
        te = cells == c; tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
        per[c] = float(np.mean(np.abs(y[te] - m.predict(sc.transform(X[te])))))
    res[name] = per
    v = np.array(list(per.values()))
    print(f"{name}: per-cell MAE mean {v.mean():.2f} | median {np.median(v):.2f} cycles")
a = np.array([res["cap"][c] for c in uc]); b = np.array([res["cap+eis"][c] for c in uc])
w = wilcoxon(a, b); imp = (a - b) / a * 100
print(f"improved: {(b < a).sum()} / {len(uc)} cells | median improvement {np.median(imp):.1f}% | Wilcoxon p = {w.pvalue:.1e}")
np.savez("eis_ridge_loocv.npz", cells=np.array(uc), cap=a, eis=b)
print("saved eis_ridge_loocv.npz (input for make_paired_fig.py)")
