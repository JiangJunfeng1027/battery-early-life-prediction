# Mechanism figures (thesis Figs 3.2, 4.7, 5.3, 5.6 and slide variants)

Scripts that generate the mechanism-oriented figures added in the v2 draft.

- `extract_figdata.py` — selective extraction of dQ(V) curves and fade summaries
  from the public .mat batch files (place them under `./data/MIT/`); caches to `./cache/`.
- `make_thesis_figs.py` — knee-window figure, dQ mechanism map, Nyquist schematic,
  EIS band-importance figure (thesis styling).
- `make_shift_fig.py` — batch distribution-shift comparison
  (uses the included derived `feature_matrix_full.csv`, 75 eligible cells).
- `make_eisband_fig.py` — per-frequency ageing-information scan. Requires
  `variable_discharge_features.npz`, the feature cache produced by the public code of
  Jones et al. 2022 (https://doi.org/10.1038/s41467-022-32422-w) on their
  variable-discharge dataset; place it under `./cache/`.

Outputs are written to `./figures_out/`.
