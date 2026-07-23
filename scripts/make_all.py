#!/usr/bin/env python3
"""Generate every computational figure and component in the release."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(SCRIPTS / script), *arguments]
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    run("plot_figure_01_panel_b.py")
    run("plot_figures.py")
    run("plot_figure_06_component.py")
    run("plot_figure_07.py")


if __name__ == "__main__":
    main()
