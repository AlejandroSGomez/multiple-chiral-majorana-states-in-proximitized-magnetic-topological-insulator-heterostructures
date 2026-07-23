#!/usr/bin/env python3
"""Validate the local figure-reproducibility release."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = ROOT / "checksums.sha256"
GITHUB_FILE_LIMIT = 100 * 1024 * 1024

PUBLISHED_FILES = tuple(
    f"figures/published/{name}.pdf"
    for name in (
        "Fig1",
        "Fig2",
        "Fig3",
        "Fig4",
        "Fig5",
        "Fig6",
        "Fig7",
        "FigA1",
        "FigA2",
        "FigA3",
    )
)

GENERATED_FILES = (
    "figures/generated/Fig1_panel_b.pdf",
    "figures/generated/Fig1_panel_b.png",
    "figures/generated/Fig1_panel_b.svg",
    "figures/generated/Fig2.pdf",
    "figures/generated/Fig2.svg",
    "figures/generated/Fig3.pdf",
    "figures/generated/Fig3.svg",
    "figures/generated/Fig4.pdf",
    "figures/generated/Fig4.svg",
    "figures/generated/Fig5.pdf",
    "figures/generated/Fig5.svg",
    "figures/generated/Fig6_component.pdf",
    "figures/generated/Fig6_component.png",
    "figures/generated/Fig6_component.svg",
    "figures/generated/Fig7.pdf",
    "figures/generated/Fig7.png",
    "figures/generated/FigA1.pdf",
    "figures/generated/FigA1.svg",
    "figures/generated/FigA2.pdf",
    "figures/generated/FigA2.svg",
    "figures/generated/FigA3.pdf",
    "figures/generated/FigA3.svg",
)

DATA_FILES = tuple(
    f"data/processed/{name}.npz"
    for name in (
        "fig01_bands",
        "fig02_gap_map",
        "fig02_ribbon",
        "fig03_phase_map",
        "fig03_slab",
        "fig04_pair_map",
        "fig04_spectrum_A",
        "fig04_spectrum_B",
        "fig04_states_A",
        "fig04_states_B",
        "fig05_density_3d",
        "fig05_disorder_summary",
        "fig05_state_A",
        "fig05_state_B",
        "fig05_winding_counts",
        "fig06_pair_map",
        "fig06_spectra",
        "fig06_spectrum_4pi",
        "fig07_spatial_mp",
        "fig07_spectrum",
        "figA1_exchange_fixed",
        "figA1_mixing",
        "figA1_mu_fixed",
        "figA2_pair_map",
        "figA2_spectrum_A",
        "figA2_spectrum_B",
        "figA2_states_A",
        "figA2_states_B",
        "figA3_gap_Lz10",
        "figA3_gap_Lz15",
        "figA3_gap_Lz20",
        "figA3_pair_map",
        "figA3_width_scan",
    )
)

MANUAL_FILES = (
    "manual_sources/Figure_01.key",
    "manual_sources/Figure_01_schematic.pdf",
    "manual_sources/Figure_06.key",
    "manual_sources/Figure_06_schematic.pdf",
)

REQUIRED_FILES = PUBLISHED_FILES + GENERATED_FILES + DATA_FILES + MANUAL_FILES
CHECKSUMMED_ASSETS = PUBLISHED_FILES + DATA_FILES + MANUAL_FILES

TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".gitignore", "Makefile"}
IGNORED_TEXT_DIRECTORIES = {
    ".git",
    ".matplotlib-cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}

PERSONAL_PATH_PATTERNS = (
    ("macOS user path", re.compile(re.escape("/" + "Users" + "/"))),
    ("Linux home path", re.compile(r"/home/[A-Za-z0-9._-]+(?:/[^ \t\"']*)?")),
    (
        "Windows user path",
        re.compile(r"[A-Za-z]:" + re.escape("\\") + r"Users" + re.escape("\\")),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_missing_files() -> list[str]:
    return [relative for relative in REQUIRED_FILES if not (ROOT / relative).is_file()]


def find_oversized_files() -> list[tuple[str, int]]:
    oversized = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink() or ".git" in path.parts:
            continue
        size = path.stat().st_size
        if size >= GITHUB_FILE_LIMIT:
            oversized.append((path.relative_to(ROOT).as_posix(), size))
    return sorted(oversized)


def is_text_source(path: Path) -> bool:
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES


def find_personal_paths() -> list[tuple[str, int, str]]:
    matches = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.is_symlink()
            or any(part in IGNORED_TEXT_DIRECTORIES for part in path.parts)
            or not is_text_source(path)
        ):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for label, pattern in PERSONAL_PATH_PATTERNS:
                if pattern.search(line):
                    matches.append(
                        (path.relative_to(ROOT).as_posix(), line_number, label)
                    )
    return matches


def parse_checksums() -> tuple[dict[str, str], list[str]]:
    if not CHECKSUM_FILE.is_file():
        return {}, ["checksums.sha256 is missing"]

    entries = {}
    errors = []
    for line_number, line in enumerate(
        CHECKSUM_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
        if not match:
            errors.append(f"checksums.sha256:{line_number}: invalid entry")
            continue
        expected, relative = match.groups()
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"checksums.sha256:{line_number}: unsafe path")
            continue
        if relative in entries:
            errors.append(f"checksums.sha256:{line_number}: duplicate path")
            continue
        entries[relative] = expected.lower()
    if not entries:
        errors.append("checksums.sha256 contains no valid entries")
    return entries, errors


def verify_checksums() -> list[str]:
    entries, errors = parse_checksums()
    for relative in CHECKSUMMED_ASSETS:
        if relative not in entries:
            errors.append(f"checksum missing for {relative}")

    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"checksummed file is missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def write_checksums() -> None:
    missing = [
        relative
        for relative in CHECKSUMMED_ASSETS
        if not (ROOT / relative).is_file()
    ]
    if missing:
        for relative in missing:
            print(f"ERROR: cannot checksum missing file: {relative}", file=sys.stderr)
        raise SystemExit(1)

    lines = [
        f"{sha256(ROOT / relative)}  {relative}"
        for relative in sorted(CHECKSUMMED_ASSETS)
    ]
    CHECKSUM_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to checksums.sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="replace the manifest with checksums for immutable release assets",
    )
    args = parser.parse_args()

    if args.write_checksums:
        write_checksums()
        return

    errors = []
    errors.extend(f"required file is missing: {path}" for path in find_missing_files())
    errors.extend(
        f"file reaches the 100 MiB GitHub limit: {path} ({size / 1024**2:.1f} MiB)"
        for path, size in find_oversized_files()
    )
    errors.extend(
        f"personal absolute path: {path}:{line_number} ({label})"
        for path, line_number, label in find_personal_paths()
    )
    errors.extend(verify_checksums())

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    print(
        "Release verified: "
        f"{len(REQUIRED_FILES)} required files, "
        f"{len(CHECKSUMMED_ASSETS)} checksummed assets, "
        "all files below 100 MiB, and no personal absolute paths."
    )


if __name__ == "__main__":
    main()
