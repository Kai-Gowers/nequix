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
        novel_pairing_names = set(read_manifest(dataset_dir / "holdout_novel_pairing.txt"))
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


def evaluate_material(material: str, group: str, finetuned_model: str, baseline_model: str, plot_dir: Path = None) -> dict:
    ph, q_list, dft_freqs = load_dft_reference(material)
    supercell_ase = phonopy_atoms_to_ase(ph.supercell)
    n_bands = dft_freqs.shape[1]

    from nequix.calculator import NequixCalculator

    calc_ft = NequixCalculator(model_path=finetuned_model, backend="jax", use_kernel=False)
    hessian_ft = calc_ft.get_hessian(supercell_ase)
    ph.force_constants = hessian_ft
    ft_freqs = run_qpoints_frequencies(ph, q_list)

    if Path(baseline_model).exists():
        calc_base = NequixCalculator(model_path=baseline_model, backend="jax", use_kernel=False)
    else:
        calc_base = NequixCalculator(model_name=baseline_model, backend="jax", use_kernel=False)
    hessian_base = calc_base.get_hessian(supercell_ase)
    ph.force_constants = hessian_base
    base_freqs = run_qpoints_frequencies(ph, q_list)

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
    }

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
            ax.set_title(f"{material} ({group}) phonon dispersion: DFT vs Nequix")
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
            row = evaluate_material(material, group, args.finetuned_model, args.baseline_model, plot_dir)
            rows.append(row)
            print(f"    RMSE ft={row['rmse_ft']:.4f}  base={row['rmse_base']:.4f}  THz")
        except Exception as e:
            print(f"  ✗ {material}: {e}", file=sys.stderr)

    if not rows:
        print("No materials evaluated successfully.", file=sys.stderr)
        sys.exit(1)

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-material results → {csv_path}")

    group_order = {"val": 0, "novel_pairing": 1, "novel_chemistry": 2, "test_3R_homobilayer": 1, "test_3R_heterobilayer": 2}
    for group in sorted({r["group"] for r in rows}, key=lambda g: group_order.get(g, 99)):
        print_group_summary(rows, group)


if __name__ == "__main__":
    main()
