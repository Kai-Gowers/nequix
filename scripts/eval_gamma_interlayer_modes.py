#!/usr/bin/env python3
"""
Isolate the Gamma-point rigid-layer modes (2 degenerate interlayer shear + 1
breathing, bands 4-6 after the 3 acoustic bands) for bilayer test/val materials
and compare fine-tuned vs baseline vs DFT there specifically.

Motivation: scripts/eval_healthy2d_finetune.py's acoustic/optical RMSE split
lumps these 2-3 low-frequency (<2 THz) interlayer bands in with the entire
rest of the optical manifold (which can run up to 8+ THz), so a real,
localized error confined to just the interlayer modes could be diluted in
that aggregate and go unnoticed.

2026-10-26: modes are identified by overlap with the rigid-layer shear/breathing patterns (signed
frequencies, imaginary shear pairs are reported as negative instead of being mis-sorted below the
acoustic zeros), --add-d3 wraps bare-PBE models in the explicit D3 calculator, --all-bilayers includes
the train split (needed for the registry-scan error correlation).

Usage:
  uv run python scripts/eval_gamma_interlayer_modes.py \\
      --finetuned-model wandb/offline-run-XXXX/files/checkpoint.nqx \\
      --baseline-model nequix-mp-1 \\
      --dataset-dir v3_2h3r
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(ROOT / "scripts"))
from eval_healthy2d_finetune import (  # noqa: E402
    WORKFLOW_ROOT,
    build_material_groups,
    load_dft_reference,
    phonopy_atoms_to_ase,
)

sys.path.insert(0, str(WORKFLOW_ROOT / "common"))
from structural_families import STACKING_SUFFIXES  # noqa: E402


def is_bilayer(name: str) -> bool:
    return any(name.endswith("_" + suffix) for suffix in STACKING_SUFFIXES)


def find_gamma_index(q_list) -> int:
    for i, q in enumerate(q_list):
        if np.allclose(q, [0.0, 0.0, 0.0], atol=1e-6):
            return i
    raise ValueError("No Gamma point found in q_list")


def rigid_layer_modes(ph) -> np.ndarray:
    """Mass-weighted rigid-layer displacement patterns of the primitive cell: (3, 3n) for the two layers
    moving against each other along x, y (shear) and z (breathing). Layers = split by median z."""
    prim = ph.primitive
    z = prim.scaled_positions[:, 2] - np.round(prim.scaled_positions[:, 2] - prim.scaled_positions[0, 2])
    top = z > np.median(z)
    m = np.asarray(prim.masses)
    sign = np.where(top, 1.0 / m[top].sum(), -1.0 / m[~top].sum())  # equal and opposite layer momenta
    modes = []
    for d in range(3):
        v = np.zeros((len(m), 3))
        v[:, d] = sign * np.sqrt(m)  # phonopy eigenvectors are mass-weighted (sqrt(m) * displacement)
        modes.append((v / np.linalg.norm(v)).ravel())
    return np.array(modes)


def gamma_interlayer(ph, q_gamma, rigid) -> tuple[np.ndarray, np.ndarray]:
    """Signed Gamma frequencies of the modes with the largest overlap with the rigid shear-x, shear-y and
    breathing patterns (after enforcing the acoustic sum rule). Sorting by frequency alone mislabels an
    imaginary shear pair, since it sorts below the three acoustic zeros."""
    ph.symmetrize_force_constants()
    ph.run_qpoints(q_gamma, with_eigenvectors=True)
    qd = ph.get_qpoints_dict()
    freqs, vecs = qd["frequencies"][0], qd["eigenvectors"][0]  # vecs[:, k] = mode k
    overlaps = np.abs(rigid.conj() @ vecs) ** 2  # (3, nmodes)
    il = []
    for d in range(3):
        k = int(np.argmax(overlaps[d]))
        il.append(float(freqs[k]))
    il = np.array(il)
    return np.array([min(il[0], il[1]), max(il[0], il[1]), il[2]]), overlaps.max(axis=1)


def evaluate_material(material: str, group: str, finetuned_model: str, baseline_model: str,
                      add_d3: str = "none") -> dict | None:
    ph, q_list, dft_freqs = load_dft_reference(material)
    n_bands = dft_freqs.shape[1]
    if n_bands < 6:
        return None  # not enough bands for 3 acoustic + 3 interlayer modes

    gamma_idx = find_gamma_index(q_list)
    dft_g = np.sort(dft_freqs[gamma_idx])

    supercell_ase = phonopy_atoms_to_ase(ph.supercell)
    q_gamma = np.array([q_list[gamma_idx]])
    rigid = rigid_layer_modes(ph)
    dft_il, dft_ov = gamma_interlayer(ph, q_gamma, rigid)  # DFT force constants are loaded in ph

    from nequix.calculator import NequixCalculator

    calc_ft = NequixCalculator(model_path=finetuned_model, backend="jax", use_kernel=False)
    if Path(baseline_model).exists():
        calc_base = NequixCalculator(model_path=baseline_model, backend="jax", use_kernel=False)
    else:
        calc_base = NequixCalculator(model_name=baseline_model, backend="jax", use_kernel=False)
    # Explicit D3 hook (bare-PBE models, explicit_dispersion/): analytic D3 Hessian added to the model Hessian,
    # same as eval_healthy2d_finetune.py --add-d3.
    if add_d3 != "none":
        sys.path.insert(0, str(ROOT / "explicit_dispersion"))
        from d3 import NequixD3Calculator  # noqa: E402

        if add_d3 in ("finetuned", "both"):
            calc_ft = NequixD3Calculator(calc_ft)
        if add_d3 in ("baseline", "both"):
            calc_base = NequixD3Calculator(calc_base)

    ph.force_constants = calc_ft.get_hessian(supercell_ase)
    ft_il, ft_ov = gamma_interlayer(ph, q_gamma, rigid)
    ph.force_constants = calc_base.get_hessian(supercell_ase)
    base_il, base_ov = gamma_interlayer(ph, q_gamma, rigid)

    gap_within = float(dft_il[-1] - dft_il[0])
    gap_after = float(dft_g[6] - dft_il[-1]) if n_bands > 6 else float("nan")

    row = {
        "material": material,
        "group": group,
        "n_bands": n_bands,
        "dft_shear1": round(float(dft_il[0]), 4),
        "dft_shear2": round(float(dft_il[1]), 4),
        "dft_breathing": round(float(dft_il[2]), 4),
        "ft_shear1": round(float(ft_il[0]), 4),
        "ft_shear2": round(float(ft_il[1]), 4),
        "ft_breathing": round(float(ft_il[2]), 4),
        "base_shear1": round(float(base_il[0]), 4),
        "base_shear2": round(float(base_il[1]), 4),
        "base_breathing": round(float(base_il[2]), 4),
        "rmse_ft_interlayer": float(np.sqrt(np.mean((ft_il - dft_il) ** 2))),
        "rmse_base_interlayer": float(np.sqrt(np.mean((base_il - dft_il) ** 2))),
        "gap_ratio": round(gap_after / gap_within, 2) if gap_within > 1e-6 else float("nan"),
        "dft_min_overlap": round(float(dft_ov.min()), 3),
        "ft_min_overlap": round(float(ft_ov.min()), 3),
        "base_min_overlap": round(float(base_ov.min()), 3),
        "d3_added_to": add_d3,
    }
    return row


def print_summary(rows: list[dict]) -> None:
    for group in sorted({r["group"] for r in rows}):
        group_rows = [r for r in rows if r["group"] == group]
        rmse_ft = np.array([r["rmse_ft_interlayer"] for r in group_rows])
        rmse_base = np.array([r["rmse_base_interlayer"] for r in group_rows])
        n_improved = int(np.sum(rmse_ft < rmse_base))
        print(f"\n== {group} interlayer modes (n={len(group_rows)}) ==")
        print(f"  RMSE (THz)   fine-tuned: mean={rmse_ft.mean():.4f} median={np.median(rmse_ft):.4f}")
        print(f"               baseline:   mean={rmse_base.mean():.4f} median={np.median(rmse_base):.4f}")
        print(f"  fine-tune improves on baseline: {n_improved}/{len(group_rows)} materials")

    low_gap = [r for r in rows if r["gap_ratio"] < 1.0]
    if low_gap:
        print(f"\nWARNING: {len(low_gap)} material(s) where the gap to the next optical branch "
              f"is smaller than the spread within the 3 interlayer bands (gap_ratio < 1) — "
              f"mode identification there is less clean, inspect individually:")
        for r in low_gap:
            print(f"  {r['material']}: gap_ratio={r['gap_ratio']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finetuned-model", required=True, help="Path to fine-tuned .nqx checkpoint")
    parser.add_argument("--baseline-model", default="nequix-mp-1", help="Pretrained model name or path for comparison")
    parser.add_argument("--dataset-dir", default="v2_73healthy_split", help="Split dir name under workflow/nequix_datasets/")
    parser.add_argument("--output-dir", default=None, help="Where to write results.csv (default: phonon_comparisons/<dataset-dir>_gamma_interlayer)")
    parser.add_argument("--materials", nargs="*", default=None, help="Restrict to these material names (default: all bilayers in val + test/holdout)")
    parser.add_argument("--all-bilayers", action="store_true", help="Also include the train-split bilayers (every bilayer in the manifests)")
    parser.add_argument("--add-d3", choices=["none", "finetuned", "baseline", "both"], default="none",
                        help="Wrap the named model(s) in explicit_dispersion.d3.NequixD3Calculator (bare-PBE models such as "
                             "the -baresub fine-tunes have no interlayer binding on their own)")
    args = parser.parse_args()

    dataset_dir = WORKFLOW_ROOT / "nequix_datasets" / args.dataset_dir
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "phonon_comparisons" / f"{args.dataset_dir}_gamma_interlayer"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = build_material_groups(dataset_dir)
    if args.materials:
        materials = args.materials
    elif args.all_bilayers:
        materials = sorted(name for name in groups if is_bilayer(name))
    else:
        materials = sorted(name for name in groups if is_bilayer(name) and groups[name] != "train")
    print(f"D3 added to: {args.add_d3}; {len(materials)} bilayers; dataset {args.dataset_dir}")

    rows = []
    for i, material in enumerate(materials):
        group = groups.get(material)
        if group is None:
            print(f"  x {material}: not in val/test manifests, skipping", file=sys.stderr)
            continue
        print(f"[{i + 1}/{len(materials)}] {material} ({group}) ...")
        try:
            row = evaluate_material(material, group, args.finetuned_model, args.baseline_model, args.add_d3)
            if row is None:
                print(f"    skipped: fewer than 6 bands (not a bilayer with resolvable interlayer modes)")
                continue
            rows.append(row)
            print(f"    interlayer RMSE ft={row['rmse_ft_interlayer']:.4f}  base={row['rmse_base_interlayer']:.4f}  THz"
                  f"  (gap_ratio={row['gap_ratio']})")
        except Exception as e:
            print(f"  x {material}: {e}", file=sys.stderr)

    if not rows:
        print("No materials evaluated successfully.", file=sys.stderr)
        sys.exit(1)

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-material results -> {csv_path}")

    print_summary(rows)


if __name__ == "__main__":
    main()
