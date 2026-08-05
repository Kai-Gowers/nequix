#!/usr/bin/env python3
"""
Compare DFT phonon dispersions (from a workflow/ staticpoint directory) against
Nequix-predicted dispersions, using the model's analytical Hessian
(NequixCalculator.get_hessian) in place of phonopy's finite-displacement
FORCE_CONSTANTS.

Usage:
  uv run python scripts/compare_phonon_dispersion.py --monolayer WSe2 \
      --finetuned-model wandb/offline-run-20260730_145505-lfqpdt1u/files/checkpoint.nqx

  uv run python scripts/compare_phonon_dispersion.py --bilayer WSe2_bilayer_3R \
      --finetuned-model checkpoints/nequix-healthy-2d-pft.pkl \
      --baseline-model nequix-mp-1

  # No-DFT model-only screen (e.g. out-of-distribution chemistry with no DFT
  # reference yet): builds the supercell from the workflow's own monolayer
  # generator (or a structure file path) and skips the DFT comparison entirely.
  uv run python scripts/compare_phonon_dispersion.py --structure silicene \
      --finetuned-model checkpoints/nequix-healthy-2d-pft.pkl \
      --baseline-model nequix-mp-1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import phonopy
from ase import Atoms

ROOT = Path(__file__).parent.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"

# Generic hexagonal-lattice Gamma-K-M-Gamma path, matching the BAND/BAND_LABELS
# convention used throughout workflow/'s own band.conf files. Used as a stand-in
# q-path for --structure mode, where no band.conf exists.
HEX_BAND_LINE = "0 0 0  0.6667 0.3333 0  0.5 0 0  0 0 0"
HEX_BAND_LABELS = "Γ K M Γ"


def phonopy_atoms_to_ase(patoms) -> Atoms:
    return Atoms(numbers=patoms.numbers, positions=patoms.positions, cell=patoms.cell, pbc=True)


def band_path_from_spec(band_line: str, labels_line: str = None):
    segments = [seg.strip() for seg in band_line.split(",")]
    all_labels = labels_line.split() if labels_line else None

    paths, labels_out = [], []
    label_idx = 0
    N = 101
    for seg in segments:
        vals = [float(x) for x in seg.split()]
        waypoints = np.array(vals).reshape(-1, 3)
        for i in range(len(waypoints) - 1):
            q0, q1 = waypoints[i], waypoints[i + 1]
            paths.append(np.array([q0 + (q1 - q0) * t / (N - 1) for t in range(N)]))
        if all_labels:
            labels_out.extend(all_labels[label_idx : label_idx + len(waypoints)])
            label_idx += len(waypoints)
    return paths, (labels_out or None)


def parse_band_conf(band_conf: Path):
    band_line = None
    labels_line = None
    for line in band_conf.read_text().splitlines():
        if line.strip().upper().startswith("BAND ="):
            band_line = line.split("=", 1)[1].strip()
        elif line.strip().upper().startswith("BAND_LABELS"):
            labels_line = line.split("=", 1)[1].strip()
    if band_line is None:
        raise ValueError(f"No BAND line found in {band_conf}")
    return band_path_from_spec(band_line, labels_line)


def band_frequencies(ph, paths) -> np.ndarray:
    ph.run_band_structure(paths)
    return np.vstack(ph.get_band_structure_dict()["frequencies"])


def build_structure_only_phonopy(material: str, supercell: str):
    """Build a Phonopy object with no DFT displacement/force-set step - only used
    to get a supercell + primitive-cell definition for a model-Hessian-only phonon
    prediction, for materials with no DFT reference available yet."""
    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp

    structure_path = Path(material)
    if not structure_path.exists():
        monolayer_dir = WORKFLOW_ROOT / "relaxation" / "monolayer"
        sys.path.insert(0, str(monolayer_dir))
        from generate_monolayer_poscar import generate_poscar

        out_dir = ROOT / "phonon_comparisons" / "ood_screening" / "_structures"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = generate_poscar(material, out_dir, filename=f"POSCAR_{material}")
        structure_path = result["output_path"]

    unitcell = read_vasp(str(structure_path))
    supercell_matrix = np.diag([int(x) for x in supercell.split()])
    ph = Phonopy(unitcell, supercell_matrix=supercell_matrix, primitive_matrix="auto")
    paths, labels = band_path_from_spec(HEX_BAND_LINE, HEX_BAND_LABELS)
    return ph, paths, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monolayer", action="store_true")
    parser.add_argument("--bilayer", action="store_true")
    parser.add_argument(
        "--structure",
        action="store_true",
        help="No-DFT mode: build the supercell from a workflow material name or a "
        "structure file path, skip the DFT reference entirely (for materials with "
        "no DFT calculation yet, e.g. an out-of-distribution screen)",
    )
    parser.add_argument("material", help="Material name, e.g. WSe2, WSe2_bilayer_3R, or (with --structure) silicene")
    parser.add_argument("--finetuned-model", required=True, help="Path to fine-tuned .nqx/.pkl checkpoint")
    parser.add_argument("--baseline-model", default="nequix-mp-1", help="Pretrained model name for comparison")
    parser.add_argument(
        "--supercell",
        default="4 4 1",
        help="Supercell matrix diagonal, --structure mode only (default: '4 4 1')",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output plot path (default: phonon_comparisons/<material>_phonon_compare.png, "
        "or phonon_comparisons/ood_screening/<material>_phonon_predict.png with --structure)",
    )
    args = parser.parse_args()

    if sum([args.monolayer, args.bilayer, args.structure]) > 1:
        parser.error("Specify at most one of --monolayer / --bilayer / --structure")

    if args.structure:
        print(f"Building structure-only Phonopy object for {args.material} (no DFT reference) ...")
        ph, paths, labels = build_structure_only_phonopy(args.material, args.supercell)
        dft_freqs = None
    else:
        kind = "phonopy_bilayer_examples" if args.bilayer else "phonopy_monolayer_examples"
        staticpoint_dir = WORKFLOW_ROOT / kind / f"{args.material}_staticpoint"
        if not staticpoint_dir.exists():
            print(f"Error: staticpoint directory not found: {staticpoint_dir}", file=sys.stderr)
            sys.exit(1)

        phonopy_yaml = staticpoint_dir / "phonopy.yaml"
        fc_file = staticpoint_dir / "FORCE_CONSTANTS"
        band_conf = staticpoint_dir / "band.conf"

        print(f"Loading DFT reference from {staticpoint_dir} ...")
        ph = phonopy.load(
            phonopy_yaml=str(phonopy_yaml),
            force_constants_filename=str(fc_file),
            is_compact_fc=True,
            log_level=0,
        )
        paths, labels = parse_band_conf(band_conf)
        dft_freqs = band_frequencies(ph, paths)

    supercell_ase = phonopy_atoms_to_ase(ph.supercell)
    print(f"Supercell: {len(supercell_ase)} atoms, formula {supercell_ase.get_chemical_formula()}")

    from nequix.calculator import NequixCalculator

    print(f"Computing Hessian with fine-tuned model: {args.finetuned_model}")
    calc_ft = NequixCalculator(model_path=args.finetuned_model, backend="jax", use_kernel=False)
    hessian_ft = calc_ft.get_hessian(supercell_ase)
    ph.force_constants = hessian_ft
    ft_freqs = band_frequencies(ph, paths)

    print(f"Computing Hessian with baseline model: {args.baseline_model}")
    if Path(args.baseline_model).exists():
        calc_base = NequixCalculator(model_path=args.baseline_model, backend="jax", use_kernel=False)
    else:
        calc_base = NequixCalculator(model_name=args.baseline_model, backend="jax", use_kernel=False)
    hessian_base = calc_base.get_hessian(supercell_ase)
    ph.force_constants = hessian_base
    base_freqs = band_frequencies(ph, paths)

    print(f"\n{'model':<20}{'RMSE (THz)':>14}{'MAE (THz)':>14}{'min freq (THz)':>18}")
    if dft_freqs is not None:
        rmse_ft = np.sqrt(np.mean((ft_freqs - dft_freqs) ** 2))
        rmse_base = np.sqrt(np.mean((base_freqs - dft_freqs) ** 2))
        mae_ft = np.mean(np.abs(ft_freqs - dft_freqs))
        mae_base = np.mean(np.abs(base_freqs - dft_freqs))
        print(f"{'DFT (reference)':<20}{'':>14}{'':>14}{dft_freqs.min():>18.4f}")
        print(f"{'fine-tuned':<20}{rmse_ft:>14.4f}{mae_ft:>14.4f}{ft_freqs.min():>18.4f}")
        print(f"{'baseline (' + args.baseline_model + ')':<20}{rmse_base:>14.4f}{mae_base:>14.4f}{base_freqs.min():>18.4f}")
    else:
        print(f"{'fine-tuned':<20}{'':>14}{'':>14}{ft_freqs.min():>18.4f}")
        print(f"{'baseline (' + args.baseline_model + ')':<20}{'':>14}{'':>14}{base_freqs.min():>18.4f}")

    import matplotlib.pyplot as plt

    n_q = dft_freqs.shape[0] if dft_freqs is not None else ft_freqs.shape[0]
    q = np.arange(n_q)
    fig, ax = plt.subplots(figsize=(8, 5))
    if dft_freqs is not None:
        ax.plot(q, dft_freqs[:, 0], color="black", lw=1.5, label="DFT (reference)")
        ax.plot(q, dft_freqs[:, 1:], color="black", lw=1.5)
    ax.plot(q, base_freqs[:, 0], color="tab:red", lw=1.0, ls="--", label=f"baseline ({args.baseline_model})")
    ax.plot(q, base_freqs[:, 1:], color="tab:red", lw=1.0, ls="--")
    ax.plot(q, ft_freqs[:, 0], color="tab:blue", lw=1.0, label="fine-tuned")
    ax.plot(q, ft_freqs[:, 1:], color="tab:blue", lw=1.0)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Frequency (THz)")
    title_suffix = "DFT vs Nequix" if dft_freqs is not None else "Nequix prediction (no DFT reference)"
    ax.set_title(f"{args.material} phonon dispersion: {title_suffix}")
    if labels:
        n_seg = len(paths)
        n_per_seg = n_q // n_seg
        tick_pos = [i * n_per_seg for i in range(n_seg)] + [n_q - 1]
        # collapse consecutive duplicate labels at segment joins
        tick_labels = labels[: len(tick_pos)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        for pos in tick_pos:
            ax.axvline(pos, color="gray", lw=0.5)
    ax.legend()
    fig.tight_layout()

    if args.output:
        output = Path(args.output)
    else:
        if args.structure:
            output_dir = ROOT / "phonon_comparisons" / "ood_screening"
            output = output_dir / f"{args.material}_phonon_predict.png"
        else:
            output_dir = ROOT / "phonon_comparisons"
            output = output_dir / f"{args.material}_phonon_compare.png"
        output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"\nSaved comparison plot to {output}")


if __name__ == "__main__":
    main()
