#!/usr/bin/env python3
"""Model-side interlayer landscape: energy / forces / stress / rigid-layer curvatures of a Nequix model on the
registry x gap configurations of the workflow registry scan (the model half of the landscape-error characterisation).

For every bilayer, the top layer of workflow/FINAL_RESULTS_HEALTHY/<name>/POSCAR is shifted exactly as
workflow/scripts/registry_gap_scan.py does (same generator functions, same config tags), and at each configuration
the script records, for the fine-tuned model (bare), the explicit D3 term, their sum (the deployed model) and the
zero-shot baseline:
  * energy relative to the equilibrium registry (u=v=dz=0) of the same model, max |F|, the net force on the top
    layer, in-plane stress;
  * rigid-layer curvatures from the analytic Hessian: K_ab = sum_{i,j in top} H_ij^ab, i.e. d2E/(d_a d_b) for a rigid
    displacement d of the top layer with the bottom layer fixed (the same K_prim as the twisted invariance-projection
    json and the sliding scans; K_xx/K_yy = shear stiffness, K_zz = breathing stiffness, eV/A^2 per cell).
Per bilayer a summary row gives the stacking-fault corrugation (max-min energy over the registry grid at dz=0), the
curvatures at the equilibrium registry and the gap offset that minimises E along the breathing curve.

Once the DFT scan (bare PBE, same configs) has run, `results.csv` joins on (material, tag) for the point-by-point
comparison; the DFT curvatures come from fitting the same grid.

  sbatch --job-name=landscape_v5 sbatch/model_registry_landscape.sbatch \
      checkpoints/nequix-healthy-2d-pft-v5-tmd-cellrelax-omat-baresub-e20.nqx phonon_comparisons/registry_landscape/v5 --all
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "explicit_dispersion"))

from registry_gap_scan import (  # noqa: E402  (workflow generator: single source of the configurations)
    DEFAULT_BREATHING, DEFAULT_DZ, PILOT, RESULTS, configs, read_poscar, split_layers, tag, tmd_bilayers,
)

EV_A3_TO_GPA = 160.21766208


def make_atoms(cell, species, counts, pos):
    symbols = [s for s, c in zip(species, counts) for _ in range(c)]
    return Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)


def efs(calc, atoms):
    a = atoms.copy()
    a.calc = calc
    return float(a.get_potential_energy()), np.asarray(a.get_forces()), np.asarray(a.get_stress()) * EV_A3_TO_GPA


def layer_curvatures(hessian, top):
    """K_ab = sum over top-layer atom pairs of the (n, n, 3, 3) Hessian block (eV/A^2 per cell)."""
    h = np.asarray(hessian)
    blk = h[np.ix_(top, top)].sum(axis=(0, 1))
    return blk


def evaluate_bilayer(name, calcs, hess_fns, grid, dz_list, breathing, log):
    comment, cell, species, counts, pos = read_poscar(RESULTS / name / "POSCAR")
    top, gap = split_layers(pos)
    a1, a2 = cell[0], cell[1]
    cfgs = configs(grid, dz_list, breathing)
    rows, e_ref = [], {}
    for u, v, dz in cfgs:
        p = pos.copy()
        p[top] += u * a1 + v * a2 + np.array([0.0, 0.0, dz])
        atoms = make_atoms(cell, species, counts, p)
        row = {"material": name, "tag": tag(u, v, dz), "u": u, "v": v, "dz_A": dz, "gap_A": gap + dz}
        for label, calc in calcs.items():
            e, f, s = efs(calc, atoms)
            if (u, v, dz) == (0.0, 0.0, 0.0):
                e_ref[label] = e
            row[f"{label}_E_eV"] = e
            row[f"{label}_Fmax_eV_A"] = float(np.abs(f).max())
            ft = f[top].sum(axis=0)
            row[f"{label}_Ftop_x"], row[f"{label}_Ftop_y"], row[f"{label}_Ftop_z"] = map(float, ft)
            row[f"{label}_sxx_GPa"], row[f"{label}_syy_GPa"], row[f"{label}_sxy_GPa"] = float(s[0]), float(s[1]), float(s[5])
        for label, hfn in hess_fns.items():
            k = layer_curvatures(hfn(atoms), top)
            row[f"{label}_Kxx"], row[f"{label}_Kyy"], row[f"{label}_Kzz"], row[f"{label}_Kxy"] = (
                float(k[0, 0]), float(k[1, 1]), float(k[2, 2]), float(k[0, 1]))
        rows.append(row)
        log.write(f"  {row['tag']}  " + "  ".join(f"{l} E={row[f'{l}_E_eV']:.4f}" for l in calcs) + "\n")
        log.flush()
    for row in rows:
        for label in calcs:
            row[f"{label}_Erel_meV"] = (row[f"{label}_E_eV"] - e_ref[label]) * 1e3
    return rows, gap


def summarise(name, rows, gap, labels):
    """labels: model labels present in rows (ft, d3, base, ftd3); all carry E/F/stress and K columns."""
    calcs = hess_fns = labels
    out = {"material": name, "gap_dft_A": gap}
    at_eq = next(r for r in rows if r["tag"] == tag(0.0, 0.0, 0.0))
    grid0 = [r for r in rows if r["dz_A"] == 0.0]
    bre = sorted([r for r in rows if r["u"] == 0.0 and r["v"] == 0.0], key=lambda r: r["dz_A"])
    for label in calcs:
        e = np.array([r[f"{label}_Erel_meV"] for r in grid0])
        out[f"{label}_corrugation_meV"] = float(e.max() - e.min())
        out[f"{label}_Emax_registry"] = grid0[int(e.argmax())]["tag"]
        # breathing curve: parabola through the three points around the minimum -> gap offset
        dzs = np.array([r["dz_A"] for r in bre])
        eb = np.array([r[f"{label}_Erel_meV"] for r in bre])
        i = int(eb.argmin())
        if 0 < i < len(dzs) - 1:
            c = np.polyfit(dzs[i - 1:i + 2], eb[i - 1:i + 2], 2)
            out[f"{label}_dz_min_A"] = float(-c[1] / (2 * c[0]))
            out[f"{label}_Kzz_fit_eV_A2"] = float(2 * c[0] * 1e-3)
        else:
            out[f"{label}_dz_min_A"] = float("nan")
            out[f"{label}_Kzz_fit_eV_A2"] = float("nan")
        out[f"{label}_Ftop_z_eq"] = at_eq[f"{label}_Ftop_z"]
    for label in hess_fns:
        out[f"{label}_Kshear_eq"] = 0.5 * (at_eq[f"{label}_Kxx"] + at_eq[f"{label}_Kyy"])
        out[f"{label}_Kzz_eq"] = at_eq[f"{label}_Kzz"]
        ks = np.array([0.5 * (r[f"{label}_Kxx"] + r[f"{label}_Kyy"]) for r in grid0])
        out[f"{label}_Kshear_min_registry"] = grid0[int(ks.argmin())]["tag"]
        out[f"{label}_Kshear_min"] = float(ks.min())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="fine-tuned bare-PBE .nqx checkpoint")
    ap.add_argument("--baseline-model", default="models/nequix-omat-1.nqx")
    ap.add_argument("--output-dir", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--materials", nargs="+")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--all", action="store_true", help="all 42 TMD bilayers of the v5 manifests")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--dz", type=float, nargs="+", default=DEFAULT_DZ)
    ap.add_argument("--breathing", type=float, nargs="+", default=DEFAULT_BREATHING)
    a = ap.parse_args()
    names = PILOT if a.pilot else tmd_bilayers() if a.all else a.materials
    if not names:
        ap.error("give --materials, --pilot or --all")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from nequix.calculator import NequixCalculator
    from d3 import D3Calculator, d3_hessian_analytic  # noqa: E402

    ft = NequixCalculator(model_path=a.model, backend="jax", use_kernel=False)
    base = (NequixCalculator(model_path=a.baseline_model, backend="jax", use_kernel=False)
            if Path(a.baseline_model).exists() else NequixCalculator(model_name=a.baseline_model, backend="jax", use_kernel=False))
    d3 = D3Calculator()

    calcs = {"ft": ft, "d3": d3, "base": base}
    hess_fns = {"ft": ft.get_hessian, "d3": d3_hessian_analytic, "base": base.get_hessian}

    all_rows, summary = [], []
    with open(out / "progress.log", "a") as log:
        for i, name in enumerate(names):
            print(f"[{i + 1}/{len(names)}] {name}", flush=True)
            log.write(f"== {name}\n")
            rows, gap = evaluate_bilayer(name, calcs, hess_fns, a.grid, a.dz, a.breathing, log)
            # deployed model = ft + D3 (energies, forces, stress and curvatures are all additive, as in NequixD3Calculator)
            additive = ("E_eV", "Ftop_x", "Ftop_y", "Ftop_z", "sxx_GPa", "syy_GPa", "sxy_GPa", "Kxx", "Kyy", "Kzz", "Kxy", "Erel_meV")
            for r in rows:
                for kk in additive:
                    r[f"ftd3_{kk}"] = r[f"ft_{kk}"] + r[f"d3_{kk}"]
                r["ftd3_Fmax_eV_A"] = float("nan")  # per-atom forces are not stored; see Ftop_*
            s = summarise(name, rows, gap, ["ft", "d3", "base", "ftd3"])
            summary.append(s)
            all_rows.extend(rows)
            print(f"    corrugation meV ft+D3 {s['ftd3_corrugation_meV']:.1f} base {s['base_corrugation_meV']:.1f} | "
                  f"K_shear eq ft+D3 {s['ftd3_Kshear_eq']:+.4f} base {s['base_Kshear_eq']:+.4f} | "
                  f"K_zz eq ft+D3 {s['ftd3_Kzz_eq']:+.4f} | dz_min ft+D3 {s['ftd3_dz_min_A']:+.3f} base {s['base_dz_min_A']:+.3f} A", flush=True)
            for fname, data in (("results.csv", all_rows), ("summary.csv", summary)):
                with open(out / fname, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                    w.writeheader()
                    w.writerows(data)
    print(f"wrote {out}/results.csv ({len(all_rows)} rows) and summary.csv ({len(summary)} bilayers)")


if __name__ == "__main__":
    main()
