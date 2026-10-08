#!/usr/bin/env python3
"""
Batch-evaluate a Nequix PFT fine-tune against baseline on a healthy-2D dataset split
under workflow/nequix_datasets/<split-dir>/ (--dataset-dir, default v2_73healthy_split):
val materials plus whatever generalization-test set that split defines — v2's holdout/
(14 novel-chemistry + 6 novel-pairing, tagged via holdout_novel_pairing.txt) or v3's
test/ (all "_3R" materials, tagged homobilayer vs heterobilayer by name).

Unlike scripts/compare_phonon_dispersion.py (which reads its DFT reference from the
transient, per-host workflow/phonopy_{mono,bi}layer_examples/<name>_staticpoint/
directories), this script sources everything directly from the permanent,
git-tracked workflow/FINAL_RESULTS_HEALTHY/<name>/ — phonopy.yaml + FORCE_CONSTANTS
for the structure, and band.yaml for the exact q-path + ground-truth frequencies
(no band.conf needed). This matters because some materials were computed on a
different host and only FINAL_RESULTS_HEALTHY syncs between hosts.

Usage:
  uv run python scripts/eval_healthy2d_finetune.py \\
      --finetuned-model wandb/offline-run-XXXX/files/checkpoint.nqx \\
      --baseline-model nequix-mp-1 \\
      --dataset-dir v3_2h3r

Geometry modes (--relax):
  none       (default) fixed-geometry benchmark: Hessian at the DFT-relaxed structure.
  positions  end-to-end: start from the workflow's pre-relaxation template
             (--start-dir, default workflow/template_structures/<material>/POSCAR), relax
             atomic positions with the model at the template cell, then Hessian + bands.
  cell       as above but also relax the in-plane cell (c stays fixed). Each model (fine-tuned,
             baseline) relaxes with its own forces/stress. Frequencies are always compared with
             DFT on the DFT band.yaml q-list (fractional, so still the same Gamma-K-M-Gamma path).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import phonopy
import yaml
from ase import Atoms

ROOT = Path(__file__).parent.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"
FINAL_RESULTS_HEALTHY = WORKFLOW_ROOT / "FINAL_RESULTS_HEALTHY"


def read_manifest(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def build_material_groups(dataset_dir: Path) -> dict[str, str]:
    """material name -> group label. 'val' always; the test-set label(s) depend on
    which generalization axis this dataset_dir's split holds out (v2: holdout/ with
    novel-chemistry/novel-pairing tags; v3+: test/ with a homobilayer/heterobilayer
    "_3R" split, tagged directly from the name rather than a separate tag file)."""
    val_names = read_manifest(dataset_dir / "val" / "materials.txt")
    groups = {name: "val" for name in val_names}

    train_names = read_manifest(dataset_dir / "train" / "materials.txt")
    for name in train_names:
        groups[name] = "train"

    if (dataset_dir / "holdout").exists():
        holdout_names = read_manifest(dataset_dir / "holdout" / "materials.txt")
        novel_pairing_path = dataset_dir / "holdout_novel_pairing.txt"
        novel_pairing_names = set(read_manifest(novel_pairing_path)) if novel_pairing_path.exists() else set()
        novel_compound_path = dataset_dir / "holdout_novel_compound.txt"
        novel_compound_names = set(read_manifest(novel_compound_path)) if novel_compound_path.exists() else set()
        for name in holdout_names:
            if name in novel_compound_names:
                groups[name] = "novel_compound"
            elif name in novel_pairing_names:
                groups[name] = "novel_pairing"
            else:
                groups[name] = "novel_chemistry"
    elif (dataset_dir / "test").exists():
        test_names = read_manifest(dataset_dir / "test" / "materials.txt")
        for name in test_names:
            groups[name] = "test_3R_homobilayer" if name.endswith("_bilayer_3R") else "test_3R_heterobilayer"
    else:
        raise FileNotFoundError(f"Neither 'holdout' nor 'test' split found under {dataset_dir}")

    return groups


def phonopy_atoms_to_ase(patoms) -> Atoms:
    return Atoms(numbers=patoms.numbers, positions=patoms.positions, cell=patoms.cell, pbc=True)


def load_dft_reference(material: str):
    """Load phonopy structure + DFT q-path/frequencies from FINAL_RESULTS_HEALTHY."""
    material_dir = FINAL_RESULTS_HEALTHY / material
    ph = phonopy.load(
        phonopy_yaml=str(material_dir / "phonopy.yaml"),
        force_constants_filename=str(material_dir / "FORCE_CONSTANTS"),
        is_compact_fc=True,
        log_level=0,
    )
    band_yaml = yaml.safe_load((material_dir / "band.yaml").read_text())
    phonon = band_yaml["phonon"]
    q_list = [entry["q-position"] for entry in phonon]
    dft_freqs = np.array([[b["frequency"] for b in entry["band"]] for entry in phonon])
    return ph, q_list, dft_freqs


def run_qpoints_frequencies(ph, q_list) -> np.ndarray:
    ph.run_qpoints(np.array(q_list))
    return ph.get_qpoints_dict()["frequencies"]


def surface_gap(atoms) -> float:
    """Surface-to-surface interlayer gap (largest z-gap) in Angstrom; 0 for a monolayer."""
    if len(atoms) <= 3:
        return 0.0
    return float(np.diff(np.sort(atoms.positions[:, 2])).max())


def relax_unitcell(atoms: Atoms, calc, mode: str):
    """Relax a 1x1 unit cell with the model. mode="positions": atoms only; mode="cell": atoms +
    in-plane cell (Voigt mask xx, yy, xy; c fixed). FixSymmetry keeps the hexagonal 1x1
    symmetry against float noise (DFT relaxed the same 1x1 cell). Returns (relaxed, info)."""
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from ase.units import GPa

    atoms = atoms.copy()
    atoms.pbc = True
    atoms.calc = calc
    scaled0 = atoms.get_scaled_positions().copy()
    a0 = float(atoms.cell.lengths()[0])
    forces0 = atoms.get_forces()
    stress0 = atoms.get_stress(voigt=True)
    info = {
        "fmax0": float(np.linalg.norm(forces0, axis=1).max()),
        "stress0_GPa": float(np.mean(stress0[:2]) / GPa),
        "gap0_A": surface_gap(atoms),
    }
    atoms.set_constraint(FixSymmetry(atoms))
    target = FrechetCellFilter(atoms, mask=[1, 1, 0, 0, 0, 1]) if mode == "cell" else atoms
    opt = BFGS(target, logfile=None, maxstep=0.1)
    converged = bool(opt.run(fmax=1e-3, steps=1000))
    atoms.set_constraint()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)
    pos_ref = scaled0 @ atoms.cell[:]  # template positions carried along with any cell change
    a1 = float(atoms.cell.lengths()[0])
    info.update({
        "steps": int(opt.get_number_of_steps()),
        "converged": converged,
        "fmax_final": float(np.linalg.norm(forces, axis=1).max()),
        "stress_final_GPa": float(np.mean(stress[:2]) / GPa),
        "max_disp_A": float(np.linalg.norm(atoms.get_positions() - pos_ref, axis=1).max()),
        "a_final": a1,
        "da_pct": 100.0 * (a1 - a0) / a0,
        "gap_A": surface_gap(atoms),
    })
    atoms.calc = None
    return atoms, info


def phonopy_from_ase(atoms: Atoms, supercell_matrix, primitive_matrix):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell[:],
        scaled_positions=atoms.get_scaled_positions(),
    )
    return Phonopy(unitcell, supercell_matrix=supercell_matrix, primitive_matrix=primitive_matrix, log_level=0)


def branch_rmse_mae(pred: np.ndarray, ref: np.ndarray) -> dict:
    n_bands = ref.shape[1]
    n_acoustic = min(3, n_bands)
    diff = pred - ref
    acoustic = diff[:, :n_acoustic]
    optical = diff[:, n_acoustic:]
    out = {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse_acoustic": float(np.sqrt(np.mean(acoustic**2))),
        "mae_acoustic": float(np.mean(np.abs(acoustic))),
    }
    if optical.size > 0:
        out["rmse_optical"] = float(np.sqrt(np.mean(optical**2)))
        out["mae_optical"] = float(np.mean(np.abs(optical)))
    else:
        out["rmse_optical"] = float("nan")
        out["mae_optical"] = float("nan")
    return out


def evaluate_material(
    material: str,
    group: str,
    finetuned_model: str,
    baseline_model: str,
    plot_dir: Path = None,
    relax_mode: str = "none",
    start_dir: Path = None,
    relaxed_dir: Path = None,
    add_d3: str = "none",
) -> dict:
    ph, q_list, dft_freqs = load_dft_reference(material)
    supercell_ase = phonopy_atoms_to_ase(ph.supercell)
    n_bands = dft_freqs.shape[1]

    from nequix.calculator import NequixCalculator

    calc_ft = NequixCalculator(model_path=finetuned_model, backend="jax", use_kernel=False)
    if Path(baseline_model).exists():
        calc_base = NequixCalculator(model_path=baseline_model, backend="jax", use_kernel=False)
    else:
        calc_base = NequixCalculator(model_name=baseline_model, backend="jax", use_kernel=False)

    # Explicit D3 hook for models trained on bare-PBE targets (explicit_dispersion/): the wrapper adds
    # the VASP-matched D3(BJ) energy/forces/stress during relaxation and the analytic D3 Hessian on top
    # of the model Hessian at whatever geometry the relaxation ends at (so no cached FC_D3 is used).
    if add_d3 != "none":
        sys.path.insert(0, str(ROOT / "explicit_dispersion"))
        from d3 import NequixD3Calculator  # noqa: E402

        if add_d3 in ("finetuned", "both"):
            calc_ft = NequixD3Calculator(calc_ft)
        if add_d3 in ("baseline", "both"):
            calc_base = NequixD3Calculator(calc_base)

    extra = {"d3_added_to": add_d3}
    if relax_mode == "none":
        ph.force_constants = calc_ft.get_hessian(supercell_ase)
        ft_freqs = run_qpoints_frequencies(ph, q_list)
        ph.force_constants = calc_base.get_hessian(supercell_ase)
        base_freqs = run_qpoints_frequencies(ph, q_list)
    else:
        from ase.io import read as ase_read
        from ase.io import write as ase_write

        template = ase_read(str(start_dir / material / "POSCAR"))
        template.pbc = True

        def model_bands(calc, tag):
            relaxed, info = relax_unitcell(template, calc, relax_mode)
            ph_m = phonopy_from_ase(relaxed, ph.supercell_matrix, ph.primitive_matrix)
            ph_m.force_constants = calc.get_hessian(phonopy_atoms_to_ase(ph_m.supercell))
            if relaxed_dir is not None:
                relaxed_dir.mkdir(parents=True, exist_ok=True)
                ase_write(str(relaxed_dir / f"{material}_{tag}.vasp"), relaxed, format="vasp", direct=True)
            for k, v in info.items():
                extra[f"{tag}_{k}"] = v
            print(
                f"    relax[{tag}] steps={info['steps']} conv={info['converged']} fmax0={info['fmax0']:.3f}"
                f" stress0={info['stress0_GPa']:+.2f}GPa da={info['da_pct']:+.2f}% gap={info['gap0_A']:.3f}->{info['gap_A']:.3f}"
            )
            return run_qpoints_frequencies(ph_m, q_list)

        ft_freqs = model_bands(calc_ft, "ft")
        base_freqs = model_bands(calc_base, "base")
        if relax_mode == "positions":
            # how bad is "no relaxation at all" (fine-tuned model at the raw template)?
            ph_t = phonopy_from_ase(template, ph.supercell_matrix, ph.primitive_matrix)
            ph_t.force_constants = calc_ft.get_hessian(phonopy_atoms_to_ase(ph_t.supercell))
            unrelaxed_freqs = run_qpoints_frequencies(ph_t, q_list)
            extra["rmse_ft_template_unrelaxed"] = branch_rmse_mae(unrelaxed_freqs, dft_freqs)["rmse"]

    m_ft = branch_rmse_mae(ft_freqs, dft_freqs)
    m_base = branch_rmse_mae(base_freqs, dft_freqs)

    row = {
        "material": material,
        "group": group,
        "n_atoms_primitive": len(ph.primitive),
        "n_bands": n_bands,
        "rmse_ft": m_ft["rmse"], "rmse_base": m_base["rmse"],
        "mae_ft": m_ft["mae"], "mae_base": m_base["mae"],
        "rmse_ft_acoustic": m_ft["rmse_acoustic"], "rmse_base_acoustic": m_base["rmse_acoustic"],
        "rmse_ft_optical": m_ft["rmse_optical"], "rmse_base_optical": m_base["rmse_optical"],
        "min_freq_ft": float(ft_freqs.min()), "min_freq_base": float(base_freqs.min()),
        "relax_mode": relax_mode,
    }
    row.update(extra)

    if plot_dir is not None:
        try:
            import matplotlib.pyplot as plt

            n_q = dft_freqs.shape[0]
            q = np.arange(n_q)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(q, dft_freqs[:, 0], color="black", lw=1.5, label="DFT (reference)")
            ax.plot(q, dft_freqs[:, 1:], color="black", lw=1.5)
            ax.plot(q, base_freqs[:, 0], color="tab:red", lw=1.0, ls="--", label=f"baseline ({baseline_model})")
            ax.plot(q, base_freqs[:, 1:], color="tab:red", lw=1.0, ls="--")
            ax.plot(q, ft_freqs[:, 0], color="tab:blue", lw=1.0, label="fine-tuned")
            ax.plot(q, ft_freqs[:, 1:], color="tab:blue", lw=1.0)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_ylabel("Frequency (THz)")
            geom = "DFT structure" if relax_mode == "none" else f"template start, model-relaxed {relax_mode}"
            ax.set_title(f"{material} ({group}) phonon dispersion: DFT vs Nequix ({geom})")
            ax.legend()
            fig.tight_layout()
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_dir / f"{material}_phonon_compare.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  (plot failed for {material}: {e})", file=sys.stderr)

    return row


def print_group_summary(rows: list[dict], group: str) -> None:
    group_rows = [r for r in rows if r["group"] == group]
    if not group_rows:
        return
    rmse_ft = np.array([r["rmse_ft"] for r in group_rows])
    rmse_base = np.array([r["rmse_base"] for r in group_rows])
    n_improved = int(np.sum(rmse_ft < rmse_base))
    print(f"\n== {group} (n={len(group_rows)}) ==")
    print(f"  RMSE (THz)   fine-tuned: mean={rmse_ft.mean():.4f} median={np.median(rmse_ft):.4f}")
    print(f"               baseline:   mean={rmse_base.mean():.4f} median={np.median(rmse_base):.4f}")
    print(f"  fine-tune improves on baseline: {n_improved}/{len(group_rows)} materials")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finetuned-model", required=True, help="Path to fine-tuned .nqx checkpoint")
    parser.add_argument("--baseline-model", default="nequix-mp-1", help="Pretrained model name or path for comparison")
    parser.add_argument("--dataset-dir", default="v2_73healthy_split", help="Split dir name under workflow/nequix_datasets/ (default: v2_73healthy_split)")
    parser.add_argument("--output-dir", default=None, help="Where to write results.csv, summary, and plots (default: phonon_comparisons/<dataset-dir>_eval)")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating per-material dispersion PNGs")
    parser.add_argument("--materials", nargs="*", default=None, help="Restrict to these material names (default: all val + test/holdout)")
    parser.add_argument(
        "--relax",
        choices=["none", "positions", "cell"],
        default="none",
        help="none: Hessian at the DFT-relaxed structure (default). positions/cell: start from the "
        "workflow template (--start-dir) and relax with each model before the Hessian",
    )
    parser.add_argument(
        "--start-dir",
        default=str(WORKFLOW_ROOT / "template_structures"),
        help="Directory of <material>/POSCAR template starts for --relax positions|cell "
        "(default: workflow/template_structures, from scripts/generate_template_start_structures.py)",
    )
    parser.add_argument(
        "--add-d3",
        choices=["none", "finetuned", "baseline", "both"],
        default="none",
        help="Wrap the named model(s) in explicit_dispersion.d3.NequixD3Calculator (bare-PBE models: "
        "D3 forces/stress during relaxation + analytic D3 Hessian). Default none.",
    )
    args = parser.parse_args()

    dataset_dir = WORKFLOW_ROOT / "nequix_datasets" / args.dataset_dir
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "phonon_comparisons" / f"{args.dataset_dir}_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = None if args.no_plots else output_dir / "plots"

    groups = build_material_groups(dataset_dir)
    materials = args.materials if args.materials else sorted(groups.keys())

    rows = []
    for i, material in enumerate(materials):
        group = groups.get(material)
        if group is None:
            print(f"  ✗ {material}: not in val/test manifests, skipping", file=sys.stderr)
            continue
        print(f"[{i + 1}/{len(materials)}] {material} ({group}) ...")
        try:
            row = evaluate_material(
                material, group, args.finetuned_model, args.baseline_model, plot_dir,
                relax_mode=args.relax, start_dir=Path(args.start_dir),
                relaxed_dir=None if args.relax == "none" else output_dir / "relaxed_structures",
                add_d3=args.add_d3,
            )
            rows.append(row)
            print(f"    RMSE ft={row['rmse_ft']:.4f}  base={row['rmse_base']:.4f}  THz")
        except Exception as e:
            print(f"  ✗ {material}: {e}", file=sys.stderr)

    if not rows:
        print("No materials evaluated successfully.", file=sys.stderr)
        sys.exit(1)

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-material results → {csv_path}")

    group_order = {"val": 0, "novel_pairing": 1, "novel_chemistry": 2, "test_3R_homobilayer": 1, "test_3R_heterobilayer": 2}
    for group in sorted({r["group"] for r in rows}, key=lambda g: group_order.get(g, 99)):
        print_group_summary(rows, group)


if __name__ == "__main__":
    main()
