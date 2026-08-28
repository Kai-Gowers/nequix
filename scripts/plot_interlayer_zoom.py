#!/usr/bin/env python3
"""
One-off diagnostic: full band-path DFT vs baseline vs fine-tuned plots, zoomed to
the low-frequency (<2.5 THz) interlayer-mode window, for the materials flagged by
eval_gamma_interlayer_modes.py as having a fine-tuned regression specifically at
Gamma. Makes the Gamma-vs-rest-of-path visual comparison actually legible (the
0.1-0.4 THz gaps involved are invisible on the full 0-14 THz scale plots from
eval_healthy2d_finetune.py).

Usage:
  uv run python scripts/plot_interlayer_zoom.py \\
      --finetuned-model wandb/offline-run-XXXX/files/checkpoint.nqx \\
      --baseline-model nequix-mp-1 \\
      --materials MoS2_bilayer_3R WSe2_TiS2_3R ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_healthy2d_finetune import load_dft_reference, phonopy_atoms_to_ase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument("--baseline-model", default="nequix-mp-1")
    parser.add_argument("--materials", nargs="+", required=True)
    parser.add_argument("--ylim", type=float, default=2.5, help="Zoomed y-axis upper bound (THz)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "phonon_comparisons" / "v3_2h3r_gamma_interlayer" / "zoom_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from nequix.calculator import NequixCalculator

    for material in args.materials:
        print(f"{material} ...")
        ph, q_list, dft_freqs = load_dft_reference(material)
        supercell_ase = phonopy_atoms_to_ase(ph.supercell)
        q_arr = np.array(q_list)

        calc_ft = NequixCalculator(model_path=args.finetuned_model, backend="jax", use_kernel=False)
        ph.force_constants = calc_ft.get_hessian(supercell_ase)
        ph.run_qpoints(q_arr)
        ft_freqs = ph.get_qpoints_dict()["frequencies"]

        if Path(args.baseline_model).exists():
            calc_base = NequixCalculator(model_path=args.baseline_model, backend="jax", use_kernel=False)
        else:
            calc_base = NequixCalculator(model_name=args.baseline_model, backend="jax", use_kernel=False)
        ph.force_constants = calc_base.get_hessian(supercell_ase)
        ph.run_qpoints(q_arr)
        base_freqs = ph.get_qpoints_dict()["frequencies"]

        gamma_indices = [i for i, q in enumerate(q_list) if np.allclose(q, [0.0, 0.0, 0.0], atol=1e-6)]

        n_q = dft_freqs.shape[0]
        q_axis = np.arange(n_q)
        fig, ax = plt.subplots(figsize=(9, 5))
        for gi in gamma_indices:
            ax.axvline(gi, color="gray", lw=0.7, ls=":", zorder=0)
        ax.plot(q_axis, dft_freqs[:, 0], color="black", lw=2.0, label="DFT (reference)")
        ax.plot(q_axis, dft_freqs[:, 1:], color="black", lw=2.0)
        ax.plot(q_axis, base_freqs[:, 0], color="tab:red", lw=1.3, ls="--", label=f"baseline ({args.baseline_model})")
        ax.plot(q_axis, base_freqs[:, 1:], color="tab:red", lw=1.3, ls="--")
        ax.plot(q_axis, ft_freqs[:, 0], color="tab:blue", lw=1.3, label="fine-tuned")
        ax.plot(q_axis, ft_freqs[:, 1:], color="tab:blue", lw=1.3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylim(-0.3, args.ylim)
        ax.set_ylabel("Frequency (THz)")
        ax.set_xlabel("q-path index (dotted lines = Gamma)")
        ax.set_title(f"{material}: low-frequency (interlayer mode) window, zoomed")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{material}_zoom.png", dpi=150)
        plt.close(fig)
        print(f"  -> {output_dir / f'{material}_zoom.png'}")


if __name__ == "__main__":
    main()
