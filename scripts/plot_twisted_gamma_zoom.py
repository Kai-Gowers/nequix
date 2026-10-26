#!/usr/bin/env python3
"""Low-frequency zoom of the twisted-bilayer DFT-vs-Nequix band overlays (the Gamma interlayer modes).

The full-range overlays (0-14 THz) hide the one thing that is wrong with the fine-tuned model: its Gamma
shear pair is imaginary (plotted as negative) where grid-tight DFT gives a small positive value. This
reads the per-cell band .dat files of a `workflow/share/twisted_dft_vs_nequix_v*` export and draws a
2 x 3 grid (rows: fixed DFT geometry / model-relaxed positions; columns: cells) zoomed to the
interlayer window, annotating the Gamma shear-pair and breathing frequencies of every curve.

  uv run python scripts/plot_twisted_gamma_zoom.py \
      --export ../workflow/share/twisted_dft_vs_nequix_v2 --output phonon_comparisons/twisted/gamma_zoom_v5.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS = [
    ("MoS2_twist_m1_near0", "MoS2 21.8°"),
    ("MoS2_WS2_twist_m1_near0", "MoS2/WS2 21.8°"),
    ("MoS2_twist_m2_near0", "MoS2 13.2°"),
]
RUNS = [("fixed", "DFT geometry (fixed)"), ("relaxpos", "model-relaxed positions")]
CURVES = [
    ("DFT_PBE-D3", "DFT (PBE+D3, grid-tight + projected)", dict(color="#1f1f1f", lw=1.6, ls="-")),
    ("Nequix_finetuned_v5_D3", "fine-tuned v5 + D3", dict(color="#1f77b4", lw=1.3, ls="-")),
    ("Nequix_omat1_zeroshot", "omat-1 zero-shot", dict(color="#d62728", lw=1.1, ls="--")),
]
TICKS = ["Γ", "K", "M", "Γ"]


def gamma_modes(freqs_at_gamma):
    """Shear pair (two lowest non-acoustic modes, sign kept) and breathing from the 6 lowest Gamma modes."""
    g = np.sort(freqs_at_gamma)
    # the 3 acoustic modes sit at |f| < 1e-3 after the ASR; the two shear modes may be negative (imaginary)
    non_ac = [f for f in g[:8] if abs(f) > 1e-3]
    return non_ac[0], non_ac[2] if len(non_ac) > 2 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ymax", type=float, default=1.6)
    ap.add_argument("--ymin", type=float, default=-0.3)
    a = ap.parse_args()

    fig, axes = plt.subplots(len(RUNS), len(CELLS), figsize=(13, 7.2), sharey=True, constrained_layout=True)
    for r, (run, run_label) in enumerate(RUNS):
        for c, (cell, cell_label) in enumerate(CELLS):
            ax = axes[r, c]
            lines = []
            for curve, label, style in CURVES:
                f = a.export / cell / f"bands_{run}_{curve}.dat"
                d = np.loadtxt(f)
                x, bands = d[:, 0], d[:, 1:]
                for b in range(bands.shape[1]):
                    (ln,) = ax.plot(x, bands[:, b], **style, zorder=3 if curve.startswith("DFT") else 2)
                lines.append((ln, label))
                shear, breathing = gamma_modes(bands[0])
                ax.annotate(
                    f"Γ shear {shear:+.3f}, breathing {breathing:.2f} THz",
                    xy=(0.02, 0.97 - 0.075 * CURVES.index((curve, label, style))),
                    xycoords="axes fraction", fontsize=8.5, color=style["color"], va="top", zorder=10,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85),
                )
            n = len(x)
            # the exported path has 3 equal segments (Γ-K, K-M, M-Γ)
            seg = [x[0], x[n // 3], x[2 * n // 3], x[-1]]
            for s in seg[1:-1]:
                ax.axvline(s, color="0.8", lw=0.8, zorder=0)
            ax.axhline(0, color="0.5", lw=0.8, zorder=0)
            ax.set_xticks(seg)
            ax.set_xticklabels(TICKS)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(a.ymin, a.ymax)
            ax.set_title(f"{cell_label} — {run_label}", fontsize=10)
            if c == 0:
                ax.set_ylabel("frequency (THz)")
    fig.legend([l for l, _ in lines], [n for _, n in lines], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Twisted bilayers, low-frequency window: the fine-tuned model's Γ shear pair is imaginary "
                 "(negative) where DFT is small and positive", fontsize=11)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160, bbox_inches="tight")
    fig.savefig(a.output.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", a.output, "and .pdf")


if __name__ == "__main__":
    main()
