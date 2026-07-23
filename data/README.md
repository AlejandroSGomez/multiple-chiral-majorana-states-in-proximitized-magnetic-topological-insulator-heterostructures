# Processed figure data

`processed/` contains compact NumPy `.npz` archives consumed directly by the
plotting scripts. Energies and lengths follow the dimensionless manuscript
units, normally \(\Delta\) and \(c\), unless an archive stores explicit
metadata to the contrary.

The archives can be inspected without Python object deserialization:

```python
import numpy as np

with np.load("data/processed/fig03_slab.npz", allow_pickle=False) as data:
    print(data.files)
```

## Data groups

| Files | Contents |
| --- | --- |
| `fig01_bands.npz` | Low-energy ribbon bands and the parameters used for Fig. 1(b). |
| `fig02_*.npz` | Bulk gap map, ribbon spectrum, charge, edge densities, and selected-state Majorana polarization. |
| `fig03_*.npz` | Bulk phase map and compact slab-spectrum data for the selected bands and densities. |
| `fig04_*.npz` | Finite-momentum pair map, representative spectra, charges, densities, and Majorana-polarization diagnostics. |
| `fig05_*.npz` | Disorder averages, representative 2D states, a 3D density, and winding-count summary. |
| `fig06_*.npz` | Josephson pair map and phase-dependent spectra used in the scripted Fig. 6 component. |
| `fig07_*.npz` | Trilayer spectrum and spatial density/Majorana-polarization fields. |
| `figA1_*.npz` | Three analytical-parameter cuts used in the first appendix figure. |
| `figA2_*.npz` | The Fig. 4 diagnostics repeated at exactly \(k_x=0\). |
| `figA3_*.npz` | Josephson gaps, Pfaffian signs, thickness map, and width scan. |

`fig03_slab.npz` is intentionally compact: it retains the plotted spectrum
and only the selected-state density and Majorana-polarization values. The
large raw eigenvector archive is not distributed, so arbitrary states cannot
be reconstructed from this file.

The Fig. 6 and Fig. A3 pair maps consolidate independently calculated
thickness rows onto the phase grids used by the paper. The endpoint columns
were checked for \(2\pi\) periodicity during preparation.

## Disorder-data limitation

The complete raw disorder ensembles for Fig. 5 require tens of gigabytes and
are not available in the local project. The release contains the processed
statistics and representative states needed to reproduce the plotted panels.
The `source` field in `fig05_winding_counts.npz` is
`fallback_from_EvolucionWinding_Wa_svg`, documenting that the Fig. 5(f) values
were transcribed from the earlier SVG when the raw calculation was
unavailable.

SHA-256 hashes for distributed files are stored in
`../checksums.sha256`. Run `make verify` to check them.

