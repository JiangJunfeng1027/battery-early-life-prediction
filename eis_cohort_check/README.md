# Cohort-wide with/without-EIS check (thesis Fig. 6.5)

Future-work de-risking for the EIS extension: the same ridge recipe trained with and
without the impedance spectrum, leave-one-cell-out across all 24 cells of the public
variable-discharge dataset of Jones et al. (2022, Nat. Commun.) — same experimental
platform as Zhang et al. (2020).

Result: adding EIS improves the held-out MAE for **all 24 of 24 cells**
(mean 3.13 → 2.20 cycles; median improvement 27%; Wilcoxon signed-rank p = 1.2×10⁻⁷).
A local rerun of the dataset's original forecasting pipeline on the same cells shows the
same direction (median error 8.3% → 6.0%, R² 0.81 → 0.91).

- `run_cohort_check.py` — the two-arm LOOCV comparison + significance test (thesis Fig. 6.5).
- `make_paired_fig.py` — publication figure for the two-arm comparison (thesis Fig. 6.5).
- `deep_suite.py` — the four-arm pilot (thesis Fig. 6.7): capacity / EIS-only / combined /
  compressed arms; leakage-aware frequency-economy scan (selection inside each fold);
  structured campaign holdout (PJ097–112 ↔ PJ145–152). Key results: EIS alone is *worse*
  than capacity features alone (complement, not replacement); 1 frequency ≈ 71% and
  3 ≈ 79% of the full-spectrum gain (most-picked 6.5–7.4 Hz); the cohort-shift penalty
  roughly halves with the spectrum (+39/+56% → +16/+29%).
- `make_deep_figs.py` — publication figure for the pilot (thesis Fig. 6.7).
- Input data are **not redistributed**: derive `variable_discharge_features.npz` from the
  official Jones et al. (2022) code/data release and place it under `./data/`.
