#!/usr/bin/env python3
"""
Model-only phonon dispersions for twisted (moire) bilayer supercells: relax the structure with
the model (positions only), compute the analytical Hessian on a phonopy supercell, apply the
acoustic sum rule, and plot the model's bands along Gamma-K-M-Gamma of the moire Brillouin zone.

There is no DFT reference here by design (structures come from
workflow/twisted/build_twisted_bilayer.py). Plots show the model's bands alone.

Conventions
  * Structures are 60-deg hexagonal cells; phonopy uses the identity primitive matrix, in which
    (2/3, 1/3, 0) is the K point (verified numerically) and (1/2, 0, 0) is M.
  * Phonopy supercell diag(k, k, 1); k defaults to twist.json's phonopy_k_recommended
    (ceil(25 A / a_moire), capped at 900 supercell atoms). k = 1 is Gamma-exact but off-Gamma
    approximate (force constants beyond half the moire cell are folded).
  * is_symmetry=False (low-symmetry moire cells); phonopy's symmetrize_force_constants() applies
    the translational/permutation acoustic sum rule. The three Gamma acoustic frequencies are
    reported before and after.

Usage (from nequix/):
  uv run python scripts/eval_twisted_phonons.py \\
      --model checkpoints/nequix-healthy-2d-pft-v4-tmd-omat.nqx \\
      --output-dir phonon_comparisons/twisted_mos2_near0 \\
      --structures ../workflow/twisted/structures/MoS2_twist_m{1,5,4,2,3}_near0
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from compare_phonon_dispersion import HEX_BAND_LABELS, HEX_BAND_LINE, band_path_from_spec  # noqa: E402
from eval_healthy2d_finetune import phonopy_atoms_to_ase, surface_gap  # noqa: E402

PHONOPY_TARGET_LENGTH_A = 25.0
MAX_SUPERCELL_ATOMS = 900


def default_k(atoms: Atoms) -> int:
    a_m = float(np.linalg.norm(atoms.cell[0]))
    k = max(1, int(np.ceil(PHONOPY_TARGET_LENGTH_A / a_m)))
    while k > 1 and len(atoms) * k * k > MAX_SUPERCELL_ATOMS:
        k -= 1
    return k


def layer_slide(pos0: np.ndarray, pos1: np.ndarray, z0: np.ndarray) -> float:
    """Rigid in-plane displacement of the top layer relative to the bottom layer (A): difference of the
    two layers' mean in-plane displacements. Detects interlayer sliding that a symmetric start hides."""
    top = z0 > z0.mean()
    d = pos1[:, :2] - pos0[:, :2]
    return float(np.linalg.norm(d[top].mean(axis=0) - d[~top].mean(axis=0)))


def relax_positions(atoms: Atoms, calc, fmax: float, max_steps: int, rattle: float = 0.0, seed: int = 0):
    """Positions-only BFGS relaxation (cell fixed: the moire lattice is set by commensurability).
    No FixSymmetry -- the moire cell has little symmetry and the model may lower it further.
    `rattle` (A, Gaussian std) breaks the exact point symmetry of the as-built cell so the optimizer can
    leave a symmetry-protected saddle (e.g. rigid interlayer sliding, whose force vanishes by symmetry at
    the rotation-centre registry)."""
    from ase.optimize import BFGS
    from ase.units import GPa

    atoms = atoms.copy()
    atoms.pbc = True
    pos0 = atoms.get_positions().copy()
    z0 = pos0[:, 2].copy()
    if rattle > 0:
        atoms.rattle(stdev=rattle, seed=seed)
    atoms.calc = calc
    forces0 = atoms.get_forces()
    info = {"rattle_A": rattle, "fmax0": float(np.linalg.norm(forces0, axis=1).max()), "gap0_A": surface_gap(atoms)}
    opt = BFGS(atoms, logfile=None, maxstep=0.1)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)
    info.update(
        {
            "steps": int(opt.get_number_of_steps()),
            "converged": converged,
            "fmax_final": float(np.linalg.norm(forces, axis=1).max()),
            "stress_inplane_GPa": float(np.mean(stress[:2]) / GPa),
            "max_disp_A": float(np.linalg.norm(atoms.get_positions() - pos0, axis=1).max()),
            "layer_slide_A": layer_slide(pos0, atoms.get_positions(), z0),
            "gap_A": surface_gap(atoms),
            "energy_per_atom_eV": float(atoms.get_potential_energy() / len(atoms)),
        }
    )
    atoms.calc = None
    return atoms, info


def phonopy_bands(atoms: Atoms, calc, k: int):
    """Returns (distances, frequencies[nq, nbands], segment_edges, gamma_before, gamma_after, n_super)."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:], scaled_positions=atoms.get_scaled_positions()
    )
    ph = Phonopy(unitcell, supercell_matrix=np.diag([k, k, 1]), primitive_matrix=None, is_symmetry=False, log_level=0)
    supercell = phonopy_atoms_to_ase(ph.supercell)
    ph.force_constants = calc.get_hessian(supercell)

    ph.run_qpoints([[0, 0, 0]])
    gamma_before = np.sort(ph.get_qpoints_dict()["frequencies"][0])
    ph.symmetrize_force_constants()
    ph.run_qpoints([[0, 0, 0]], with_eigenvectors=True)
    qd = ph.get_qpoints_dict()
    order = np.argsort(qd["frequencies"][0])
    gamma_after = qd["frequencies"][0][order]
    # phonopy eigenvectors are mass-weighted; convert the softest mode to a real displacement pattern
    masses = np.array(unitcell.masses)
    softest = qd["eigenvectors"][0][:, order[0]].real.reshape(-1, 3) / np.sqrt(masses)[:, None]

    paths, _labels = band_path_from_spec(HEX_BAND_LINE, HEX_BAND_LABELS)
    ph.run_band_structure(paths, with_eigenvectors=False, is_band_connection=False)
    bs = ph.get_band_structure_dict()
    distances = np.concatenate(bs["distances"])
    freqs = np.concatenate(bs["frequencies"], axis=0)
    edges = np.cumsum([0] + [len(d) for d in bs["distances"]]) - 1
    edges[0] = 0
    return distances, freqs, edges, gamma_before, gamma_after, len(supercell), softest


def displace_along_mode(atoms: Atoms, pattern: np.ndarray, amplitude: float) -> Atoms:
    """Displace atoms along a (real) mode pattern so the largest atomic displacement is `amplitude` A."""
    out = atoms.copy()
    out.positions += pattern * (amplitude / np.linalg.norm(pattern, axis=1).max())
    return out


def plot_bands(distances, freqs, edges, labels, title, out_path, extra=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(distances, freqs, color="tab:blue", lw=0.8)
    if extra is not None:
        ax.plot(distances, extra["freqs"], color="tab:red", lw=0.6, ls="--")
        ax.plot([], [], color="tab:red", ls="--", label=extra["label"])
        ax.plot([], [], color="tab:blue", label="fine-tuned")
        ax.legend()
    ax.axhline(0, color="gray", lw=0.5)
    for e in edges:
        ax.axvline(distances[e], color="gray", lw=0.5)
    ax.set_xticks([distances[e] for e in edges])
    ax.set_xticklabels(labels.split()[: len(edges)])
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_structure(struct_dir: Path, calcs: dict, out_dir: Path, relax: str, fmax: float, max_steps: int,
                       k_override: int | None, plots: bool, rattle: float = 0.0, max_follow: int = 3,
                       follow_threshold: float = 0.02, follow_amplitude: float = 0.3,
                       follow_fmax_factor: float = 0.2) -> dict:
    name = struct_dir.name
    atoms = ase_read(str(struct_dir / "POSCAR"), format="vasp")
    atoms.pbc = True
    meta = {}
    twist_json = struct_dir / "twist.json"
    if twist_json.exists():
        meta = json.loads(twist_json.read_text())
    k = k_override or meta.get("phonopy_k_recommended") or default_k(atoms)

    row = {
        "name": name,
        "family": meta.get("family", ""),
        "m": meta.get("m", ""),
        "theta_deg": meta.get("theta_deg", ""),
        "n_atoms": len(atoms),
        "a_moire_A": float(np.linalg.norm(atoms.cell[0])),
        "k": k,
        "relax_mode": relax,
        "rattle_A": rattle,
    }
    print(f"\n=== {name}: {len(atoms)} atoms, a_m={row['a_moire_A']:.2f} A, theta={row['theta_deg']}, k={k} "
          f"({len(atoms) * k * k} supercell atoms), relax={relax}")

    bands_for_plot = {}
    for tag, calc in calcs.items():
        t0 = time.perf_counter()
        t_relax = 0.0
        n_follow = 0
        start = atoms
        while True:
            if relax == "positions":
                tr = time.perf_counter()
                # follow rounds: tighter fmax -- the sliding direction is so flat (~0.05 THz) that 1e-3 eV/A
                # cannot resolve its minimum
                relaxed, info = relax_positions(start, calc, fmax if n_follow == 0 else fmax * follow_fmax_factor,
                                                max_steps, rattle=rattle if n_follow == 0 else 0.0)
                t_relax += time.perf_counter() - tr
                info["layer_slide_A"] = layer_slide(atoms.get_positions(), relaxed.get_positions(), atoms.positions[:, 2])
                info["max_disp_A"] = float(np.linalg.norm(relaxed.get_positions() - atoms.get_positions(), axis=1).max())
                print(f"  relax[{tag}] steps={info['steps']} conv={info['converged']} fmax {info['fmax0']:.3f}->"
                      f"{info['fmax_final']:.4f} eV/A  max_disp={info['max_disp_A']:.3f} A  layer_slide={info['layer_slide_A']:.3f} A"
                      f"  gap {info['gap0_A']:.3f}->{info['gap_A']:.3f} A  stress_inplane={info['stress_inplane_GPa']:+.3f} GPa"
                      f"  E/atom={info['energy_per_atom_eV']:.5f} eV")
            else:
                relaxed, info = start.copy(), {}
            t1 = time.perf_counter()
            distances, freqs, edges, g_before, g_after, n_super, softest = phonopy_bands(relaxed, calc, k)
            t2 = time.perf_counter()
            n_imag = int((g_after < -follow_threshold).sum())
            if relax != "positions" or n_imag == 0 or n_follow >= max_follow:
                break
            # symmetry-protected saddle (e.g. rigid interlayer sliding): kick along the softest Gamma mode and re-relax
            n_follow += 1
            print(f"  follow[{tag}] {n_imag} imaginary Gamma mode(s) below {-follow_threshold} THz (softest "
                  f"{g_after[0]:.4f}); displacing {follow_amplitude} A along it and re-relaxing (round {n_follow})")
            start = displace_along_mode(relaxed, softest, follow_amplitude)
        if relax == "positions":
            (out_dir / "relaxed_structures").mkdir(parents=True, exist_ok=True)
            ase_write(str(out_dir / "relaxed_structures" / f"{name}_{tag}.vasp"), relaxed, format="vasp", direct=True)
        info["n_follow_rounds"] = n_follow
        n_ac = 3
        print(f"  phonons[{tag}] supercell {n_super} atoms, Hessian+bands {t2 - t1:.0f} s; "
              f"Gamma acoustic before ASR {np.round(g_before[:n_ac], 4)} -> after {np.round(g_after[:n_ac], 4)} THz; "
              f"lowest Gamma optical {np.round(g_after[n_ac:n_ac + 3], 3)} THz; min {freqs.min():.4f}, max {freqs.max():.3f} THz")
        for kk, v in info.items():
            row[f"{tag}_{kk}"] = v
        row.update({
            f"{tag}_supercell_atoms": n_super,
            f"{tag}_gamma_acoustic_before_asr_max_abs_THz": float(np.abs(g_before[:n_ac]).max()),
            f"{tag}_gamma_acoustic_after_asr_max_abs_THz": float(np.abs(g_after[:n_ac]).max()),
            f"{tag}_gamma_optical_lowest3_THz": ";".join(f"{x:.4f}" for x in g_after[n_ac:n_ac + 3]),
            f"{tag}_min_freq_THz": float(freqs.min()),
            f"{tag}_max_freq_THz": float(freqs.max()),
            f"{tag}_n_imag_below_m0p05": int((freqs < -0.05).sum()),
            f"{tag}_t_relax_s": round(t_relax, 1),
            f"{tag}_t_phonons_s": round(t2 - t1, 1),
        })
        (out_dir / "bands").mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "bands" / f"{name}_{tag}.npz", distances=distances, frequencies=freqs, edges=edges,
                 gamma_before_asr=g_before, gamma_after_asr=g_after, labels=np.array(HEX_BAND_LABELS.split()))
        bands_for_plot[tag] = (distances, freqs, edges)

    if plots:
        d, f, e = bands_for_plot["ft"]
        extra = None
        if "base" in bands_for_plot:
            extra = {"freqs": bands_for_plot["base"][1], "label": "baseline"}
        theta = f"theta={row['theta_deg']:.2f} deg, " if row["theta_deg"] != "" else ""
        title = (f"{name}: Nequix fine-tuned phonons ({theta}{len(atoms)} atoms, phonopy {k}x{k}x1, "
                 f"model-relaxed {relax})")
        plot_bands(d, f, e, HEX_BAND_LABELS, title, out_dir / "plots" / f"{name}_bands.png", extra=extra)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Fine-tuned .nqx checkpoint (plotted as the model)")
    parser.add_argument("--baseline-model", default=None,
                        help="Optional pretrained model name/path to run as well (dashed in plots); off by default")
    parser.add_argument("--structures", nargs="+", required=True, type=Path,
                        help="Structure dirs, each with POSCAR (+ optional twist.json)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--relax", choices=["positions", "none"], default="positions")
    parser.add_argument("--fmax", type=float, default=1e-3, help="BFGS force threshold, eV/A (default 1e-3)")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--supercell-k", type=int, default=None, help="Override phonopy k for all structures")
    parser.add_argument("--rattle", type=float, default=0.0,
                        help="Gaussian rattle (A) applied before relaxation to break the as-built point symmetry so "
                             "symmetry-protected saddles (rigid interlayer sliding) can relax (default 0 = off)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-follow", type=int, default=3,
                        help="Max mode-following rounds: if Gamma has non-acoustic imaginary modes after the ASR, "
                             "displace along the softest one and re-relax (default 3; 0 = off)")
    parser.add_argument("--follow-threshold", type=float, default=0.02, help="THz; modes below -threshold trigger following")
    parser.add_argument("--follow-fmax-factor", type=float, default=0.2,
                        help="fmax multiplier for follow-round relaxations (default 0.2 -> 2e-4 eV/A)")
    parser.add_argument("--follow-amplitude", type=float, default=0.3, help="A; max atomic displacement of the kick")
    args = parser.parse_args()

    from nequix.calculator import NequixCalculator

    calcs = {"ft": NequixCalculator(model_path=args.model, backend="jax", use_kernel=False)}
    if args.baseline_model:
        if Path(args.baseline_model).exists():
            calcs["base"] = NequixCalculator(model_path=args.baseline_model, backend="jax", use_kernel=False)
        else:
            calcs["base"] = NequixCalculator(model_name=args.baseline_model, backend="jax", use_kernel=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    rows = []
    for struct_dir in args.structures:
        row = evaluate_structure(struct_dir, calcs, args.output_dir, args.relax, args.fmax, args.max_steps,
                                 args.supercell_k, not args.no_plots, rattle=args.rattle, max_follow=args.max_follow,
                                 follow_threshold=args.follow_threshold, follow_amplitude=args.follow_amplitude,
                                 follow_fmax_factor=args.follow_fmax_factor)
        rows.append(row)
        keys = sorted({kk for r in rows for kk in r}, key=lambda s: (s not in row, s))
        keys = list(row.keys()) + [kk for kk in keys if kk not in row]
        with open(csv_path, "w", newline="") as f:  # rewritten after every structure so partial runs survive
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'name':<26}{'theta':>7}{'atoms':>7}{'k':>3}{'steps':>6}{'conv':>6}{'fol':>4}{'slide':>7}{'gap':>7}{'minF':>9}{'G-ASR':>9}{'t_ph(s)':>9}")
    for r in rows:
        theta = f"{r['theta_deg']:.2f}" if r["theta_deg"] != "" else "-"
        print(f"{r['name']:<26}{theta:>7}{r['n_atoms']:>7}{r['k']:>3}{str(r.get('ft_steps', '-')):>6}"
              f"{str(r.get('ft_converged', '-')):>6}{str(r.get('ft_n_follow_rounds', '-')):>4}{r.get('ft_layer_slide_A', float('nan')):>7.3f}{r.get('ft_gap_A', float('nan')):>7.3f}"
              f"{r['ft_min_freq_THz']:>9.4f}{r['ft_gamma_acoustic_after_asr_max_abs_THz']:>9.4f}{r['ft_t_phonons_s']:>9.0f}")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
