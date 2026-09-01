# Multiple chiral Majorana states in proximitized magnetic topological insulator heterostructures

Code, processed data, and figures associated with the manuscript *Multiple
chiral Majorana states in proximitized magnetic topological insulator
heterostructures*.

## Requirements

Tested with:

- Python 3.12
- NumPy 2.3
- SciPy 1.16
- Matplotlib 3.10
- Kwant 1.5

A LaTeX installation is also required for the figure labels.

The environment can be installed with:

```bash
conda env create -f environment.yml
conda activate sc-hcn-reproducibility
```

## Figures

Run

```bash
make figures
```

to generate the figures from the processed data. The output is written to
`figures/generated/`, while the versions used in the manuscript are stored in
`figures/published/`.

Figures 2--5 and A1--A3 are generated with `scripts/plot_figures.py`. Figures
1, 6, and 7 have their own plotting scripts. The final layouts of Figures 1
and 6 were assembled in Keynote, and the corresponding source files are
included in `manual_sources/`.

## Repository contents

- `scripts/`: scripts used to plot the paper figures.
- `compute/`: numerical calculations for the main results.
- `data/processed/`: processed data used by the plotting scripts.
- `figures/`: generated figures and the final versions used in the paper.
- `manual_sources/`: Keynote sources for Figures 1 and 6.

Run `make verify` to check the files and data included in the repository.
