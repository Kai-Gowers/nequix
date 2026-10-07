"""
Build the bare-PBE training set from workflow/FINAL_RESULTS_BARE_PBE_SUBTRACT with REAL stress labels.

Differences from strip_d3_dataset.py (the 2026-09-15 explicit-dispersion set):
  * hessian = FORCE_CONSTANTS of FINAL_RESULTS_BARE_PBE_SUBTRACT/<mat>/ (FORCE_SETS-level D3 subtraction
    with the equilibrium D3 force removed, hiphive rotational-sum-rule refit on 25/48) instead of
    FC_ref - FC_D3 at the force-constant level.
  * forces = 0: the eq-force removal in the subtraction pipeline makes the PBE+D3 geometry the
    equilibrium of the bare targets by construction.
  * stress = the REAL bare-PBE unit-cell stress, sigma_PBE+D3(relaxation OUTCAR, ISIF=2) - sigma_D3,
    as a (3,3) eV/A^3 tensor in ASE sign convention. Every earlier fine-tune wrote stress = 0 (or
    -sigma_D3) although the cells were never relaxed, so the model was trained toward zero stress at
    cells that carry a few kbar of in-plane stress.
  * energy = bare unit-cell energy (E_OUTCAR - E_D3) scaled to the supercell (n_sc/n_uc); energy_weight
    stays 0 in the config, stored for completeness only.

Usage (from nequix/, needs the uv env for simple-dftd3):
  uv run python explicit_dispersion/build_bare_subtract_dataset.py [--dataset-dir v4_tmd_only]
      [--output-root explicit_dispersion/data/v4_tmd_only_bare_subtract]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import ase.db
import numpy as np
import phonopy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress
from phonopy.harmonic.force_constants import compact_fc_to_full_fc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"
BARE_SUBTRACT = WORKFLOW_ROOT / "FINAL_RESULTS_BARE_PBE_SUBTRACT"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from d3 import asr_residual, d3_efs, to_training_layout  # noqa: E402
from eval_healthy2d_finetune import read_manifest  # noqa: E402

EV_A3_TO_GPA = 160.21766


def relaxation_dir(name: str) -> Path:
    for d in (WORKFLOW_ROOT / "monolayer_examples" / name, WORKFLOW_ROOT / "bilayer_examples" / name):
        if (d / "OUTCAR").exists():
            return d
    sys.exit(f"{name}: no relaxation OUTCAR on this host")


def unitcell_bare_efs(name: str, poscar: Path) -> dict:
    """Bare-PBE unit-cell energy and (3,3) stress from the relaxation OUTCAR minus D3, after checking
    the OUTCAR's final geometry is the shipped POSCAR (same check as export_bare_pbe_subtract_dataset.py)."""
    out = read(relaxation_dir(name) / "OUTCAR", index=-1)
    ref = read(poscar)
    if len(out) != len(ref) or np.abs(out.cell.array - ref.cell.array).max() > 1e-3 or \
            np.abs(out.get_scaled_positions() - ref.get_scaled_positions()).max() > 1e-4:
        sys.exit(f"{name}: relaxation OUTCAR geometry != FINAL_RESULTS POSCAR")
    e_ref = out.get_potential_energy()
    s_ref = voigt_6_to_full_3x3_stress(out.get_stress())
    d = d3_efs(ref)
    return {
        "n_uc": len(ref),
        "energy_bare": float(e_ref - d["energy"]),
        "stress_bare": np.asarray(s_ref) - np.asarray(d["stress"]),
        "stress_pbe_d3": np.asarray(s_ref),
        "stress_d3": np.asarray(d["stress"]),
    }


def process_material(material: str, db) -> dict:
    mdir = BARE_SUBTRACT / material
    ph = phonopy.load(str(mdir / "phonopy.yaml"), force_constants_filename=str(mdir / "FORCE_CONSTANTS"), log_level=0)
    fc = ph.force_constants
    if fc.shape[0] != fc.shape[1]:
        fc = compact_fc_to_full_fc(ph.primitive, fc)
    fc = np.asarray(fc)
    sc = ph.supercell
    atoms = Atoms(numbers=sc.numbers, positions=sc.positions, cell=sc.cell, pbc=True)
    n = len(atoms)

    uc = unitcell_bare_efs(material, mdir / "POSCAR")
    scale = n / uc["n_uc"]
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(uc["energy_bare"] * scale),
        forces=np.zeros((n, 3), dtype=np.float32),
        stress=uc["stress_bare"].astype(np.float32),
    )
    atoms.info["hessian"] = to_training_layout(fc).astype(np.float32)
    atoms.info["material"] = material
    db.write(atoms, data=atoms.info)

    s = uc["stress_bare"] * EV_A3_TO_GPA
    return {
        "material": material, "n_atoms_supercell": n, "n_atoms_unitcell": uc["n_uc"],
        "fc_max": float(np.abs(fc).max()), "asr": asr_residual(fc),
        "energy_bare_uc_eV": uc["energy_bare"],
        "stress_bare_xx_GPa": float(s[0, 0]), "stress_bare_yy_GPa": float(s[1, 1]), "stress_bare_zz_GPa": float(s[2, 2]),
        "stress_bare_offdiag_max_GPa": float(max(abs(s[0, 1]), abs(s[0, 2]), abs(s[1, 2]))),
        "stress_pbe_d3_xx_GPa": float(uc["stress_pbe_d3"][0, 0] * EV_A3_TO_GPA),
        "stress_d3_xx_GPa": float(uc["stress_d3"][0, 0] * EV_A3_TO_GPA),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="v4_tmd_only")
    p.add_argument("--output-root", type=Path, default=HERE / "data" / "v4_tmd_only_bare_subtract")
    p.add_argument("--splits", nargs="*", default=["train", "val", "holdout"])
    args = p.parse_args()

    split_root = WORKFLOW_ROOT / "nequix_datasets" / args.dataset_dir
    rows = []
    for split in args.splits:
        materials = read_manifest(split_root / split / "materials.txt")
        out = args.output_root / split / "dataset.aselmdb"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        print(f"\n== {split}: {len(materials)} materials -> {out}", flush=True)
        with ase.db.connect(str(out)) as db:
            for mat in materials:
                r = process_material(mat, db)
                r["split"] = split
                rows.append(r)
                print(f"  ok {mat:20s} n_sc {r['n_atoms_supercell']:3d}  |FC|max {r['fc_max']:6.2f}  ASR {r['asr']:.1e}"
                      f"  sigma_bare xx/zz {r['stress_bare_xx_GPa']:+.3f}/{r['stress_bare_zz_GPa']:+.3f} GPa"
                      f"  (PBE+D3 xx {r['stress_pbe_d3_xx_GPa']:+.3f}, D3 xx {r['stress_d3_xx_GPa']:+.3f})", flush=True)
        with open(out.parent / "materials.txt", "w") as f:
            f.write(f"# Nequix dataset (bare PBE from FINAL_RESULTS_BARE_PBE_SUBTRACT, real unit-cell stress) — {len(materials)} materials\n")
            f.write(f"# Source split: {split_root / split}\n")
            f.write(f"# Generated: uv run python {' '.join(sys.argv)}\n")
            f.write(f"# Count: {len(materials)}\n\n")
            for m in sorted(materials):
                f.write(f"{m}\n")

    summary = args.output_root / "summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} materials; summary -> {summary}")


if __name__ == "__main__":
    main()
