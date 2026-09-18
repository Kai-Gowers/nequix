"""
Phonon-band evaluation with the explicit D3 term added back to the model Hessian.

Same outputs and metrics as scripts/eval_healthy2d_finetune.py (results.csv, plots/, group
summaries) — its helpers are imported, not copied — with one difference: for the model(s)
selected by --add-d3, the force constants handed to phonopy are

    H_total = H_nequix(supercell) + FC_D3

where FC_D3 is the per-material cached D3 force-constant matrix that strip_d3_dataset.py
subtracted from the training targets (explicit_dispersion/d3_cache/<mat>.npz). If no cache entry
exists, the analytic D3 Hessian is computed on the fly (identical to ~1e-6 eV/A^2).

Usage (from nequix/):
  uv run python explicit_dispersion/eval_explicit_dispersion.py \
      --finetuned-model checkpoints/nequix-healthy-2d-pft-v4-tmd-omat-bared3.nqx \
      --baseline-model  checkpoints/nequix-healthy-2d-pft-v4-tmd-omat.nqx \
      --add-d3 finetuned --dataset-dir v4_tmd_only \
      --output-dir phonon_comparisons/v4_tmd_only_omat_bared3_eval

--reference bare compares against bare-PBE bands instead of the PBE+D3 band.yaml: the reference
force constants become FC_ref - FC_D3 (same cached FC_D3), evaluated on the same q-list, and are
saved to <output-dir>/reference_bare/<mat>.npz. This is a fixed-geometry check at the PBE+D3-relaxed
structure, which is not a stationary point of bare PBE (|F_D3| ~ 0.1-0.17 eV/A there), so the bare
bands are the curvature of the bare-PBE surface at that point rather than physical phonons. Stage-1
check of "train on bare PBE, verify, then add D3":
  uv run python explicit_dispersion/eval_explicit_dispersion.py \
      --finetuned-model checkpoints/nequix-healthy-2d-pft-v4-tmd-omat-bared3.nqx \
      --baseline-model nequix-omat-1 --add-d3 none --reference bare \
      --output-dir phonon_comparisons/v4_tmd_only_omat_bared3_stage1_bare_eval
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from phonopy.harmonic.force_constants import compact_fc_to_full_fc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from d3 import NequixD3Calculator, d3_hessian_analytic, load_d3_cache  # noqa: E402
from eval_healthy2d_finetune import (  # noqa: E402
    branch_rmse_mae, build_material_groups, load_dft_reference, phonopy_atoms_to_ase,
    print_group_summary, run_qpoints_frequencies,
)


def make_nequix_calc(model: str):
    from nequix.calculator import NequixCalculator

    if Path(model).exists():
        return NequixCalculator(model_path=model, backend="jax", use_kernel=False)
    return NequixCalculator(model_name=model, backend="jax", use_kernel=False)


def model_hessian(model: str, supercell_ase, add_d3: bool, fc_d3):
    calc = make_nequix_calc(model)
    if not add_d3:
        return calc.get_hessian(supercell_ase)
    return NequixD3Calculator(calc).get_hessian(supercell_ase, fc_d3=fc_d3)


def full_fc(ph) -> np.ndarray:
    fc = ph.force_constants
    if fc.shape[0] != fc.shape[1]:
        fc = compact_fc_to_full_fc(ph.primitive, fc)
    return np.asarray(fc)


def evaluate_material(material, group, finetuned_model, baseline_model, add_d3: str, reference: str,
                      cache_dir: Path, plot_dir, ref_dir):
    ph, q_list, pbed3_freqs = load_dft_reference(material)
    supercell_ase = phonopy_atoms_to_ase(ph.supercell)

    cache = cache_dir / f"{material}.npz"
    if cache.exists():
        fc_d3 = load_d3_cache(cache)["fc_d3"]
        d3_source = "cache"
    else:
        fc_d3 = d3_hessian_analytic(supercell_ase)
        d3_source = "analytic"

    extra = {}
    if reference == "bare":
        ph.force_constants = full_fc(ph) - fc_d3
        ref_freqs = run_qpoints_frequencies(ph, q_list)
        extra = {
            "rmse_bare_ref_vs_pbed3_ref": branch_rmse_mae(ref_freqs, pbed3_freqs)["rmse"],
            "min_freq_ref": float(ref_freqs.min()),
        }
        if ref_dir is not None:
            ref_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ref_dir / f"{material}.npz", q_positions=np.array(q_list),
                                frequencies_thz=ref_freqs, pbed3_frequencies_thz=pbed3_freqs)
    else:
        ref_freqs = pbed3_freqs

    hessian_ft = model_hessian(finetuned_model, supercell_ase, add_d3 in ("finetuned", "both"), fc_d3)
    ph.force_constants = hessian_ft
    ft_freqs = run_qpoints_frequencies(ph, q_list)

    hessian_base = model_hessian(baseline_model, supercell_ase, add_d3 in ("baseline", "both"), fc_d3)
    ph.force_constants = hessian_base
    base_freqs = run_qpoints_frequencies(ph, q_list)

    m_ft, m_base = branch_rmse_mae(ft_freqs, ref_freqs), branch_rmse_mae(base_freqs, ref_freqs)
    row = {
        "material": material, "group": group,
        "n_atoms_primitive": len(ph.primitive), "n_bands": ref_freqs.shape[1],
        "rmse_ft": m_ft["rmse"], "rmse_base": m_base["rmse"],
        "mae_ft": m_ft["mae"], "mae_base": m_base["mae"],
        "rmse_ft_acoustic": m_ft["rmse_acoustic"], "rmse_base_acoustic": m_base["rmse_acoustic"],
        "rmse_ft_optical": m_ft["rmse_optical"], "rmse_base_optical": m_base["rmse_optical"],
        "min_freq_ft": float(ft_freqs.min()), "min_freq_base": float(base_freqs.min()),
        "reference": reference, "d3_added_to": add_d3, "fc_d3_source": d3_source,
        "fc_d3_max": float(np.abs(fc_d3).max()),
    }
    row.update(extra)

    if plot_dir is not None:
        try:
            import matplotlib.pyplot as plt

            q = np.arange(ref_freqs.shape[0])
            ft_label = f"fine-tuned ({Path(finetuned_model).stem})" + (" + D3" if add_d3 in ("finetuned", "both") else "")
            base_label = f"baseline ({Path(baseline_model).stem})" + (" + D3" if add_d3 in ("baseline", "both") else "")
            ref_label = "DFT bare PBE (PBE+D3 FCs − FC_D3)" if reference == "bare" else "DFT PBE+D3 (reference)"
            fig, ax = plt.subplots(figsize=(8, 5))
            if reference == "bare":
                ax.plot(q, pbed3_freqs[:, 0], color="gray", lw=0.8, ls=":", label="DFT PBE+D3 (context)")
                ax.plot(q, pbed3_freqs[:, 1:], color="gray", lw=0.8, ls=":")
            ax.plot(q, ref_freqs[:, 0], color="black", lw=1.5, label=ref_label)
            ax.plot(q, ref_freqs[:, 1:], color="black", lw=1.5)
            ax.plot(q, base_freqs[:, 0], color="tab:red", lw=1.0, ls="--", label=base_label)
            ax.plot(q, base_freqs[:, 1:], color="tab:red", lw=1.0, ls="--")
            ax.plot(q, ft_freqs[:, 0], color="tab:blue", lw=1.0, label=ft_label)
            ax.plot(q, ft_freqs[:, 1:], color="tab:blue", lw=1.0)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_ylabel("Frequency (THz)")
            ref_short = "bare PBE" if reference == "bare" else "PBE+D3"
            ax.set_title(f"{material} ({group}): DFT {ref_short} vs Nequix, fixed DFT geometry")
            ax.legend(fontsize=8)
            fig.tight_layout()
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_dir / f"{material}_phonon_compare.png", dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"  (plot failed for {material}: {e})", file=sys.stderr)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finetuned-model", required=True, help="Path or alias of the model shown as 'fine-tuned'")
    p.add_argument("--baseline-model", required=True, help="Path or alias of the comparison model")
    p.add_argument("--add-d3", choices=["finetuned", "baseline", "both", "none"], default="finetuned",
                   help="Which model(s) get the explicit D3 Hessian added (default: finetuned)")
    p.add_argument("--reference", choices=["pbed3", "bare"], default="pbed3",
                   help="pbed3: DFT band.yaml (default). bare: FC_ref - FC_D3 on the same q-list (bare-PBE curvature at the PBE+D3 geometry)")
    p.add_argument("--dataset-dir", default="v4_tmd_only")
    p.add_argument("--cache-dir", type=Path, default=HERE / "d3_cache")
    p.add_argument("--output-dir", type=Path, required=True, help="Explicit output dir (never defaults, to avoid overwrites)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--materials", nargs="*", default=None)
    args = p.parse_args()

    dataset_dir = WORKFLOW_ROOT / "nequix_datasets" / args.dataset_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = None if args.no_plots else args.output_dir / "plots"
    ref_dir = args.output_dir / "reference_bare" if args.reference == "bare" else None
    groups = build_material_groups(dataset_dir)
    materials = args.materials or sorted(groups)

    rows = []
    for i, material in enumerate(materials):
        group = groups.get(material)
        if group is None:
            print(f"  x {material}: not in manifests, skipping", file=sys.stderr)
            continue
        print(f"[{i + 1}/{len(materials)}] {material} ({group}) ...", flush=True)
        try:
            row = evaluate_material(material, group, args.finetuned_model, args.baseline_model, args.add_d3,
                                    args.reference, args.cache_dir, plot_dir, ref_dir)
            rows.append(row)
            msg = f"    RMSE ft={row['rmse_ft']:.4f}  base={row['rmse_base']:.4f}  THz  (ref={row['reference']}, D3 -> {row['d3_added_to']}, FC_D3 from {row['fc_d3_source']})"
            if args.reference == "bare":
                msg += f"  |bare-vs-PBE+D3 ref {row['rmse_bare_ref_vs_pbed3_ref']:.4f}, min ref {row['min_freq_ref']:+.3f}"
            print(msg, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  x {material}: {e}", file=sys.stderr)

    if not rows:
        sys.exit("No materials evaluated successfully.")
    csv_path = args.output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-material results -> {csv_path}")
    order = {"val": 0, "novel_pairing": 1, "train": 2}
    for g in sorted({r["group"] for r in rows}, key=lambda g: order.get(g, 99)):
        print_group_summary(rows, g)


if __name__ == "__main__":
    main()
