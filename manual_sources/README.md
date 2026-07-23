# Manual figure sources

Figures 1 and 6 combine scripted scientific panels with schematic artwork and
were assembled manually in Apple Keynote. This directory preserves the
editable sources and extracted schematic assets:

| Figure | Keynote source | Schematic asset | Scripted component |
| --- | --- | --- | --- |
| Fig. 1 | `Figure_01.key` | `Figure_01_schematic.pdf` | `../scripts/plot_figure_01_panel_b.py` |
| Fig. 6 | `Figure_06.key` | `Figure_06_schematic.pdf` | `../scripts/plot_figure_06_component.py` |

The corresponding manuscript PDFs are
`../figures/published/Fig1.pdf` and `../figures/published/Fig6.pdf`.
Regenerating a scripted component does not automatically update either
Keynote file or its final PDF; the replacement and export must be performed
manually.

This manual boundary is intentional and should be retained in any public
release rather than presenting the final compositions as headless outputs.

