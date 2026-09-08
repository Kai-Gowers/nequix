#!/usr/bin/env python3
"""
Non-Nequix baseline for the publish-readiness ablation: evaluate an out-of-the-box
MACE-MP foundation model (no fine-tuning, no phonon-specific training of any kind)
against DFT on all workflow/FINAL_RESULTS_HEALTHY/ materials, using the same RMSE
metric as eval_healthy2d_finetune.py's baseline column. Purpose: show whether
nequix-mp-1's raw-phonon weakness (which Phonon Fine-Tuning fixes) is a general
foundation-model property or something nequix-specific.

MACE has no analytical Hessian (unlike Nequix's JAX backend), so this uses the
standard finite-displacement route instead: phonopy generates a fresh symmetry-
reduced displacement set for the same primitive/supercell definition used for the
DFT calculation, MACE computes forces on each displaced supercell, phonopy
assembles the force constants from that.

Must run in the separate MACE venv (mace_baseline/.venv) with SSL_CERT_FILE set to
its certifi bundle, e.g.:

  SSL_CERT_FILE=$(mace_baseline/.venv/bin/python -c "import certifi; print(certifi.where())") \\
    mace_baseline/.venv/bin/python scripts/eval_mace_baseline.py --materials-dir ../workflow/FINAL_RESULTS_HEALTHY
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"
FINAL_RESULTS_HEALTHY = WORKFLOW_ROOT / "FINAL_RESULTS_HEALTHY"

sys.path.insert(0, str(ROOT / "scripts"))
from eval_healthy2d_finetune import (  # noqa: E402
    build_material_groups,
    load_dft_reference,
    phonopy_atoms_to_ase,
    run_qpoints_frequencies,
    branch_rmse_mae,
)


def evaluate_material(material: str, calc, distance: float, plot_dir: Path = None, label: str = "MACE-MP") -> dict:
    ph, q_list, dft_freqs = load_dft_reference(material)
    n_bands = dft_freqs.shape[1]

    ph.generate_displacements(distance=distance)
    force_sets = []
    for sc in ph.supercells_with_displacements:
        atoms = phonopy_atoms_to_ase(sc)
        atoms.calc = calc
        force_sets.append(atoms.get_forces())
    ph.forces = force_sets
    ph.produce_force_constants()
    mace_freqs = run_qpoints_frequencies(ph, q_list)

    m = branch_rmse_mae(mace_freqs, dft_freqs)
    row = {
        "material": material,
        "n_atoms_primitive": len(ph.primitive),
        "n_bands": n_bands,
        "n_displacements": len(force_sets),
        "rmse_mace": m["rmse"],
        "mae_mace": m["mae"],
        "rmse_mace_acoustic": m["rmse_acoustic"],
        "rmse_mace_optical": m["rmse_optical"],
    }

    if plot_dir is not None:
        try:
            import matplotlib.pyplot as plt

            n_q = dft_freqs.shape[0]
            q = np.arange(n_q)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(q, dft_freqs[:, 0], color="black", lw=1.5, label="DFT (reference)")
            ax.plot(q, dft_freqs[:, 1:], color="black", lw=1.5)
            ax.plot(q, mace_freqs[:, 0], color="tab:green", lw=1.0, label=f"{label}, RMSE={m['rmse']:.3f}")
            ax.plot(q, mace_freqs[:, 1:], color="tab:green", lw=1.0)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_ylabel("Frequency (THz)")
            ax.set_title(f"{material} phonon dispersion: DFT vs {label}")
            ax.legend()
            fig.tight_layout()
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_dir / f"{material}_phonon_compare.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  (plot failed for {material}: {e})", file=sys.stderr)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="medium", help="MACE-MP model size (small/medium/large) -- ignored if --checkpoint-path is given")
    parser.add_argument("--checkpoint-path", default=None, help="Path to a local fine-tuned MACE checkpoint (.model) to evaluate instead of the zero-shot MACE-MP model")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--distance", type=float, default=0.01, help="Finite-displacement distance (Angstrom)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--materials", nargs="*", default=None, help="Restrict to these material names (default: all in FINAL_RESULTS_HEALTHY)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "phonon_comparisons" / "mace_mp_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = None if args.no_plots else output_dir / "plots"

    if args.materials:
        materials = args.materials
    else:
        materials = sorted(
            p.name for p in FINAL_RESULTS_HEALTHY.iterdir()
            if p.is_dir() and (p / "phonopy.yaml").exists()
        )

    v2_groups = build_material_groups(WORKFLOW_ROOT / "nequix_datasets" / "v2_73healthy_split")
    v3_groups = build_material_groups(WORKFLOW_ROOT / "nequix_datasets" / "v3_2h3r")
    v4_groups = build_material_groups(WORKFLOW_ROOT / "nequix_datasets" / "v4_tmd_only")

    if args.checkpoint_path:
        from mace.calculators import MACECalculator

        print(f"Loading fine-tuned MACE checkpoint {args.checkpoint_path} (device={args.device}) ...")
        calc = MACECalculator(model_paths=args.checkpoint_path, device=args.device, default_dtype="float32")
    else:
        from mace.calculators import mace_mp

        print(f"Loading MACE-MP ({args.model}, device={args.device}) ...")
        calc = mace_mp(model=args.model, device=args.device, default_dtype="float32")

    label = "MACE fine-tuned (v4_tmd_only forces)" if args.checkpoint_path else "MACE-MP (no fine-tune)"

    rows = []
    for i, material in enumerate(materials):
        print(f"[{i + 1}/{len(materials)}] {material} ...")
        t0 = time.time()
        try:
            row = evaluate_material(material, calc, args.distance, plot_dir, label=label)
            row["v2_group"] = v2_groups.get(material, "n/a")
            row["v3_group"] = v3_groups.get(material, "n/a")
            row["v4_group"] = v4_groups.get(material, "n/a")
            rows.append(row)
            print(f"    RMSE={row['rmse_mace']:.4f} THz  ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  ✗ {material}: {e}", file=sys.stderr)

    if not rows:
        print("No materials evaluated successfully.", file=sys.stderr)
        sys.exit(1)

    csv_path = output_dir / "results.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-material results → {csv_path}")

    rmse = np.array([r["rmse_mace"] for r in rows])
    print(f"\nOverall (n={len(rows)}): mean={rmse.mean():.4f}  median={np.median(rmse):.4f} THz")


if __name__ == "__main__":
    main()
