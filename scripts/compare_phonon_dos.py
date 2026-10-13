#!/usr/bin/env python3
"""
Compare a DFT total phonon DOS with one or more model DOS curves: normalize each by its number of
atoms (states / THz / atom), plot DFT solid vs ML dashed on one axis, and report the MAE between the
curves in a frequency window (default 0-5 THz, the acoustic + interlayer region) and over the full range.

Input files are phonopy-style 2-column total_dos.dat files (frequency THz, DOS states/THz/cell) with a
"# natoms = N" header line as written by workflow/phonopy/write_phonon_dos.py and by
scripts/eval_twisted_phonons.py --dft-results. --natoms-dft / --natoms-ml override the header.

Usage (from nequix/):
  uv run python scripts/compare_phonon_dos.py \\
      --dft phonon_comparisons/<run>/dos/<name>_dft_dos.dat \\
      --ml  phonon_comparisons/<run>/dos/<name>_ft_dos.dat phonon_comparisons/<run>/dos/<name>_base_dos.dat \\
      --ml-labels "fine-tuned (v5+D3)" "omat-1 zero-shot" \\
      --output phonon_comparisons/<run>/plots/<name>_dos.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_dos(path: Path, natoms: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (frequency_THz, dos_per_cell, natoms). natoms from the '# natoms = N' header unless given."""
    header_natoms = None
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            m = re.search(r"natoms\s*=\s*(\d+)", line)
            if m:
                header_natoms = int(m.group(1))
            continue
        parts = line.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))
    if natoms is None:
        natoms = header_natoms
    if natoms is None:
        raise ValueError(f"{path}: no '# natoms = N' header; pass --natoms-dft/--natoms-ml")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], natoms


def dos_mae(f_ref: np.ndarray, d_ref: np.ndarray, f_ml: np.ndarray, d_ml: np.ndarray,
            fmin: float | None = None, fmax: float | None = None) -> float:
    """Mean |DOS_ml - DOS_ref| on the reference grid (ML interpolated onto it) within [fmin, fmax]."""
    d_ml_on_ref = np.interp(f_ref, f_ml, d_ml, left=0.0, right=0.0)
    mask = np.ones_like(f_ref, dtype=bool)
    if fmin is not None:
        mask &= f_ref >= fmin
    if fmax is not None:
        mask &= f_ref <= fmax
    return float(np.abs(d_ml_on_ref[mask] - d_ref[mask]).mean())


def compare(dft_path: Path, ml_paths: list[Path], ml_labels: list[str], fmin: float, fmax: float,
            natoms_dft: int | None = None, natoms_ml: list[int] | None = None) -> dict:
    f_dft, d_dft, n_dft = load_dos(dft_path, natoms_dft)
    d_dft = d_dft / n_dft
    out = {
        "dft": {"path": str(dft_path), "natoms": n_dft, "integral_states_per_atom": float(getattr(np, "trapezoid", np.trapz)(d_dft, f_dft))},
        "mae_range_THz": [fmin, fmax],
        "models": [],
        "curves": {"freq": f_dft, "dft": d_dft},
    }
    for i, (p, lab) in enumerate(zip(ml_paths, ml_labels)):
        f_ml, d_ml, n_ml = load_dos(p, natoms_ml[i] if natoms_ml else None)
        d_ml = d_ml / n_ml
        out["models"].append({
            "label": lab, "path": str(p), "natoms": n_ml,
            "integral_states_per_atom": float(getattr(np, "trapezoid", np.trapz)(d_ml, f_ml)),
            "mae_window_THz": dos_mae(f_dft, d_dft, f_ml, d_ml, fmin, fmax),
            "mae_all_THz": dos_mae(f_dft, d_dft, f_ml, d_ml),
        })
        out["curves"][lab] = np.interp(f_dft, f_ml, d_ml, left=0.0, right=0.0)
    return out


def plot(result: dict, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = result["curves"]["freq"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(f, result["curves"]["dft"], color="black", lw=1.2, label="DFT (PBE+D3)")
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    fmin, fmax = result["mae_range_THz"]
    for i, m in enumerate(result["models"]):
        ax.plot(f, result["curves"][m["label"]], color=colors[i % len(colors)], lw=1.0, ls="--",
                label=f"{m['label']}  (MAE {fmin:g}-{fmax:g} THz: {m['mae_window_THz']:.4f})")
    ax.axvspan(fmin, fmax, color="gray", alpha=0.08, lw=0)
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("DOS (states / THz / atom)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dft", type=Path, required=True, help="DFT total_dos.dat")
    ap.add_argument("--ml", type=Path, nargs="+", required=True, help="one or more model total_dos.dat files")
    ap.add_argument("--ml-labels", nargs="*", default=None, help="legend labels, one per --ml file")
    ap.add_argument("--natoms-dft", type=int, default=None)
    ap.add_argument("--natoms-ml", type=int, nargs="*", default=None)
    ap.add_argument("--mae-range", type=float, nargs=2, default=[0.0, 5.0], metavar=("FMIN", "FMAX"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--output", type=Path, required=True, help="PNG path; a .json with the numbers is written next to it")
    args = ap.parse_args()

    labels = args.ml_labels or [p.stem.replace("_dos", "") for p in args.ml]
    if len(labels) != len(args.ml):
        ap.error("--ml-labels must have one entry per --ml file")
    if args.natoms_ml and len(args.natoms_ml) != len(args.ml):
        ap.error("--natoms-ml must have one entry per --ml file")

    res = compare(args.dft, args.ml, labels, args.mae_range[0], args.mae_range[1], args.natoms_dft, args.natoms_ml)
    fmin, fmax = args.mae_range
    print(f"DFT {args.dft}: natoms {res['dft']['natoms']}, integral {res['dft']['integral_states_per_atom']:.3f} states/atom")
    print(f"{'model':<28}{'natoms':>7}{'integral':>10}{f'MAE {fmin:g}-{fmax:g}':>14}{'MAE all':>10}")
    for m in res["models"]:
        print(f"{m['label']:<28}{m['natoms']:>7}{m['integral_states_per_atom']:>10.3f}{m['mae_window_THz']:>14.5f}{m['mae_all_THz']:>10.5f}")

    plot(res, args.title or f"{args.dft.stem.replace('_dft_dos', '')}: phonon DOS, DFT vs Nequix", args.output)
    summary = {k: v for k, v in res.items() if k != "curves"}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
