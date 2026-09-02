#!/usr/bin/env python3
"""
One-off diagnostic for the publish-readiness from-scratch ablation: full band-path
DFT vs baseline vs fine-tuned vs from-scratch plots, for hand-picked materials that
illustrate the ablation's headline finding (from-scratch ties/beats fine-tune on the
v3 structural-generalization split, but catastrophically fails on v2's novel-chemistry
holdout). Neither existing per-model eval script overlays fine-tune and from-scratch
together (each only compares one "finetuned-model" against baseline), so this pulls
both checkpoints into one plot per material.

Usage:
  uv run python scripts/plot_finetune_vs_fromscratch.py \\
      --finetuned-model wandb/offline-run-XXXX/files/checkpoint.nqx \\
      --fromscratch-model wandb/offline-run-YYYY/files/checkpoint.nqx \\
      --baseline-model nequix-mp-1 \\
      --materials MoTe2_bilayer_3R MoTe2_WTe2_3R \\
      --output-dir phonon_comparisons/finetune_vs_fromscratch
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_healthy2d_finetune import load_dft_reference, phonopy_atoms_to_ase, run_qpoints_frequencies  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument("--fromscratch-model", required=True)
    parser.add_argument("--baseline-model", default="nequix-mp-1")
    parser.add_argument("--materials", nargs="+", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "phonon_comparisons" / "finetune_vs_fromscratch"
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    from nequix.calculator import NequixCalculator

    calc_ft = NequixCalculator(model_path=args.finetuned_model, backend="jax", use_kernel=False)
    calc_scratch = NequixCalculator(model_path=args.fromscratch_model, backend="jax", use_kernel=False)
    if Path(args.baseline_model).exists():
        calc_base = NequixCalculator(model_path=args.baseline_model, backend="jax", use_kernel=False)
    else:
        calc_base = NequixCalculator(model_name=args.baseline_model, backend="jax", use_kernel=False)

    for material in args.materials:
        print(f"{material} ...")
        ph, q_list, dft_freqs = load_dft_reference(material)
        supercell_ase = phonopy_atoms_to_ase(ph.supercell)

        ph.force_constants = calc_ft.get_hessian(supercell_ase)
        ft_freqs = run_qpoints_frequencies(ph, q_list)

        ph.force_constants = calc_scratch.get_hessian(supercell_ase)
        scratch_freqs = run_qpoints_frequencies(ph, q_list)

        ph.force_constants = calc_base.get_hessian(supercell_ase)
        base_freqs = run_qpoints_frequencies(ph, q_list)

        rmse_ft = float(np.sqrt(np.mean((ft_freqs - dft_freqs) ** 2)))
        rmse_scratch = float(np.sqrt(np.mean((scratch_freqs - dft_freqs) ** 2)))
        rmse_base = float(np.sqrt(np.mean((base_freqs - dft_freqs) ** 2)))

        n_q = dft_freqs.shape[0]
        q = np.arange(n_q)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(q, dft_freqs[:, 0], color="black", lw=1.5, label="DFT (reference)")
        ax.plot(q, dft_freqs[:, 1:], color="black", lw=1.5)
        ax.plot(q, base_freqs[:, 0], color="tab:red", lw=1.0, ls="--", label=f"baseline ({args.baseline_model}), RMSE={rmse_base:.3f}")
        ax.plot(q, base_freqs[:, 1:], color="tab:red", lw=1.0, ls="--")
        ax.plot(q, ft_freqs[:, 0], color="tab:blue", lw=1.2, label=f"fine-tuned, RMSE={rmse_ft:.3f}")
        ax.plot(q, ft_freqs[:, 1:], color="tab:blue", lw=1.2)
        ax.plot(q, scratch_freqs[:, 0], color="tab:orange", lw=1.2, ls=":", label=f"from-scratch (no pretrain), RMSE={rmse_scratch:.3f}")
        ax.plot(q, scratch_freqs[:, 1:], color="tab:orange", lw=1.2, ls=":")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel("Frequency (THz)")
        ax.set_title(f"{material}: fine-tuned vs from-scratch vs DFT")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{material}_finetune_vs_fromscratch.png", dpi=150)
        plt.close(fig)
        print(f"  RMSE  ft={rmse_ft:.4f}  scratch={rmse_scratch:.4f}  base={rmse_base:.4f}")


if __name__ == "__main__":
    main()
