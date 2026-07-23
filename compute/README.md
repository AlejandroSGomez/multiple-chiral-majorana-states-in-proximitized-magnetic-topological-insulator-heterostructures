# Numerical source

This directory contains readable Python implementations of the main numerical
workflows behind the processed figure data. They are provided for model
transparency and for targeted recomputation. They are not a complete
one-command pipeline from raw calculations to every archive in
`data/processed/`.

Install the compute dependencies with the supplied Conda environment, or add
them to an existing plotting environment:

```bash
python -m pip install -r requirements-compute.txt
```

Most scripts expose their parameters through `--help`. Production-size sparse
diagonalizations and parameter sweeps can be expensive and were designed to
run as independent rows or jobs on HPC resources.

## Workflow map

| Source | Purpose | Related figures |
| --- | --- | --- |
| `ribbon_model.py` | Confined-mode superconducting slab bands in the \(B=0\) limit. | Fig. 1 |
| `linear_b_scan.py`, `collate_linear_b_scan.py` | Linear-\(B\) thickness scans, representative spectra, and pair-map collation. | Fig. 4, Fig. A2 |
| `josephson_model.py` | Josephson-phase spectrum and eigenstate calculation for a finite slab. | Fig. 6, Fig. A3 |
| `josephson_pair_scan.py` | Phase/thickness Majorana-pair maps. | Fig. 6, Fig. A3 |
| `josephson_parity_scan.py` | Phase-dependent gap and Pfaffian-parity diagnostics. | Fig. A3 |
| `josephson_width_scan.py` | Josephson minigap as a function of slab width. | Fig. A3 |
| `trilayer_spectrum.py`, `trilayer_spatial_mp.py` | Trilayer spectra and spatial Majorana-polarization fields. | Fig. 7 |
| `disorder_2d_scan.py`, `disorder_3d_scan.py` | Representative disorder calculations in two and three dimensions. | Fig. 5 |

The disorder scripts document the model and available calculations, but the
full raw disorder ensembles used to form the paper averages are not present.
Consequently, Fig. 5 can be replotted from the bundled processed data but
cannot be fully recomputed from this release alone. The winding counts in
Fig. 5(f) use an explicitly marked SVG-derived fallback.

Use small grids first when adapting a workflow. Output names from a new
numerical run should be reviewed before replacing any curated archive in
`data/processed/`.

