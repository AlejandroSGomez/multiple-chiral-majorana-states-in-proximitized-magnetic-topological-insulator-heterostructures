# Figure reproducibility package

This directory contains the figure code and compact data release for the
manuscript *Multiple chiral Majorana states in proximitized magnetic
topological insulator heterostructures*. It is designed to reproduce the plots
from the bundled processed data without access to the authors' original
working directory.

The paper figures are preserved in `figures/published/`. Running the scripts
writes new files to `figures/generated/`; it does not overwrite the published
versions.

## Quick start

A working LaTeX installation is required for the Computer Modern labels used
by the plotting scripts.

```bash
conda env create -f environment.yml
conda activate sc-hcn-reproducibility
make figures
```

`make verify` checks the release contents, file-size limit, checksums, and
absence of personal absolute paths. `make all` generates the scripted outputs
and then runs those checks.

To generate one figure, use the command in the table below from this
directory.

## Figure map

| Figure | Command | Main processed inputs | Reproduction status |
| --- | --- | --- | --- |
| Fig. 1 | `python scripts/plot_figure_01_panel_b.py` | `fig01_bands.npz` | Panel (b) is scripted. The final multi-panel layout is manually composed in Keynote. |
| Fig. 2 | `python scripts/plot_figures.py Fig2` | `fig02_gap_map.npz`, `fig02_ribbon.npz` | Plot reproduced from bundled processed data. |
| Fig. 3 | `python scripts/plot_figures.py Fig3` | `fig03_phase_map.npz`, `fig03_slab.npz` | Plot reproduced from bundled processed data. |
| Fig. 4 | `python scripts/plot_figures.py Fig4` | `fig04_pair_map.npz`, `fig04_spectrum_*.npz`, `fig04_states_*.npz` | Plot reproduced from bundled processed data. |
| Fig. 5 | `python scripts/plot_figures.py Fig5` | `fig05_*.npz` | Plot reproduced from bundled processed data; the full raw disorder ensemble is not included. |
| Fig. 6 | `python scripts/plot_figure_06_component.py` | `fig06_pair_map.npz`, `fig06_spectra.npz`, `fig06_spectrum_4pi.npz` | Computational panels are scripted. The schematic and final layout are manually composed in Keynote. |
| Fig. 7 | `python scripts/plot_figure_07.py` | `fig07_spectrum.npz`, `fig07_spatial_mp.npz` | Plot reproduced from bundled processed data. |
| Fig. A1 | `python scripts/plot_figures.py FigA1` | `figA1_*.npz` | Plot reproduced from bundled processed data. |
| Fig. A2 | `python scripts/plot_figures.py FigA2` | `figA2_pair_map.npz`, `figA2_spectrum_*.npz`, `figA2_states_*.npz` | Plot reproduced from bundled processed data. |
| Fig. A3 | `python scripts/plot_figures.py FigA3` | `figA3_gap_*.npz`, `figA3_pair_map.npz`, `figA3_width_scan.npz` | Plot reproduced from bundled processed data. |

## Contents

- `scripts/`: plotting entry points.
- `data/processed/`: compact, plot-ready NumPy archives.
- `figures/published/`: the versions used by the manuscript.
- `figures/generated/`: outputs created locally.
- `compute/`: readable source for the main numerical workflows.
- `manual_sources/`: editable Keynote sources and schematic assets for
  Figs. 1 and 6.

## Scope and limitations

The bundled processed data are sufficient to regenerate all scripted plots.
This is distinct from recomputing every numerical result from raw data.
Several calculations require substantial memory, runtime, or an HPC
environment, and the `compute/` directory covers the main workflows rather
than a complete one-command reconstruction of every archive.

In particular, the full raw disorder data underlying Fig. 5 are not available
locally and are therefore not distributed. The winding-number cache used in
Fig. 5(f) records the provenance
`fallback_from_EvolucionWinding_Wa_svg`: those values were recovered from the
previous SVG output, not recomputed from the missing raw ensemble.

Figures 1 and 6 include manual Keynote composition. Their editable sources,
schematic assets, and final PDFs are included so this boundary remains
explicit. No claim of bitwise-identical numerical recomputation is made.

## Citation

Please cite the associated manuscript and this repository. Machine-readable
citation metadata are provided in `CITATION.cff`.

## License

Source code is released under the MIT License. Processed data, figures,
documentation, and manual composition sources are released under the Creative
Commons Attribution 4.0 International License (CC BY 4.0). See `LICENSE` for
the file-level scope and attribution requirements.
