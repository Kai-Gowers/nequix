"""
Build the "bare PBE" training set for explicit-dispersion delta learning.

For each material in a workflow split (default v4_tmd_only: 39 train / 3 val / 6 holdout), read
the reference PBE+D3 force constants from workflow/FINAL_RESULTS_HEALTHY/<mat>/ and write one
row per material — same format as workflow/scripts/convert_to_nequix.py — but with the D3
contribution removed from every target:

    hessian_bare = FC_ref - FC_D3          (FC_D3 built from D3 forces on the same displaced
                                            supercells VASP ran, so the subtraction is exact)
    forces_bare  = 0 - F_D3(equilibrium)   (the structure is a PBE+D3 minimum, not a PBE one)
    stress_bare  = 0 - sigma_D3
    energy_bare  = 0 - E_D3                (energy_weight is 0 in every PFT config; kept for consistency)

Per material, the D3 terms are cached in d3_cache/<mat>.npz so inference adds back *exactly*
what was subtracted. A summary.csv records the sanity checks.

Diagnostic: rebuilding FCs from FORCE_SETS with this phonopy reproduces the shipped FORCE_CONSTANTS
for ~28/48 materials (max diff ~1e-4..1e-3 eV/A^2); the rest were hiphive rotational-sum-rule
corrected after the build and are flagged (ref_is_linear_rebuild=False in summary.csv). Either way
FC_bare + FC_D3 == FC_ref exactly, so inference with D3 added back is self-consistent.

Usage (from nequix/):
  uv run python explicit_dispersion/strip_d3_dataset.py [--dataset-dir v4_tmd_only]
      [--output-root explicit_dispersion/data/v4_tmd_only_bare_d3] [--cache-dir explicit_dispersion/d3_cache]
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
from phonopy.harmonic.force_constants import compact_fc_to_full_fc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"
FINAL_RESULTS_HEALTHY = WORKFLOW_ROOT / "FINAL_RESULTS_HEALTHY"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from d3 import (  # noqa: E402
    asr_residual, d3_efs, d3_force_constants_phonopy, d3_hessian_analytic, save_d3_cache, to_training_layout,
)
from eval_healthy2d_finetune import read_manifest  # noqa: E402

FC_REPRODUCE_TOL = 2e-3  # eV/A^2; plain phonopy rebuilds land at 1e-4..1.5e-3 (|FC|max ~20-29); hiphive-corrected refs at 1e-2..5e-1


def load_reference(material: str):
    mdir = FINAL_RESULTS_HEALTHY / material
    ph_ref = phonopy.load(str(mdir / "phonopy.yaml"), force_constants_filename=str(mdir / "FORCE_CONSTANTS"),
                          is_compact_fc=True, log_level=0)
    fc_ref = ph_ref.force_constants
    if fc_ref.shape[0] != fc_ref.shape[1]:
        fc_ref = compact_fc_to_full_fc(ph_ref.primitive, fc_ref)
    ph_fs = phonopy.load(str(mdir / "phonopy.yaml"), force_sets_filename=str(mdir / "FORCE_SETS"), log_level=0)
    return ph_ref, np.asarray(fc_ref), ph_fs


def process_material(material: str, db, cache_dir: Path) -> dict:
    ph_ref, fc_ref, ph_fs = load_reference(material)

    # 1. sanity: FORCE_SETS -> FC reproduces the shipped FORCE_CONSTANTS
    ph_chk = phonopy.load(str(FINAL_RESULTS_HEALTHY / material / "phonopy.yaml"),
                          force_sets_filename=str(FINAL_RESULTS_HEALTHY / material / "FORCE_SETS"), log_level=0)
    ph_chk.produce_force_constants()
    ph_chk.symmetrize_force_constants()
    fc_chk = ph_chk.force_constants
    if fc_chk.shape[0] != fc_chk.shape[1]:
        fc_chk = compact_fc_to_full_fc(ph_chk.primitive, fc_chk)
    reproduce_err = float(np.abs(fc_chk - fc_ref).max())
    ref_is_linear_rebuild = reproduce_err <= FC_REPRODUCE_TOL
    if not ref_is_linear_rebuild:
        # The shipped FORCE_CONSTANTS were post-processed after the plain phonopy build — in this
        # project that is the hiphive rotational-sum-rule correction (workflow/phonopy/hiphive_*.py),
        # applied to ~20 of the 48 TMDs. The subtraction below is still exactly self-consistent
        # (FC_bare + FC_D3 == FC_ref by construction, and FC_D3 is itself translationally and
        # rotationally invariant), it just is no longer literally "phonopy on PBE-only forces".
        print(f"  note {material}: FORCE_CONSTANTS is not a plain rebuild of FORCE_SETS "
              f"(max |diff| {reproduce_err:.2e} eV/A^2) -> hiphive-corrected reference", flush=True)

    # 2. D3 terms
    fc_d3 = d3_force_constants_phonopy(ph_fs)
    sc = ph_ref.supercell
    atoms = Atoms(numbers=sc.numbers, positions=sc.positions, cell=sc.cell, pbc=True)
    efs = d3_efs(atoms)
    fc_d3_analytic = d3_hessian_analytic(atoms)

    fc_bare = fc_ref - fc_d3
    n = len(atoms)

    # 3. cache
    save_d3_cache(cache_dir / f"{material}.npz", fc_d3=fc_d3, energy=efs["energy"], forces=efs["forces"], stress=efs["stress"])

    # 4. write the row exactly like convert_to_nequix.py, with bare targets
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(-efs["energy"]),
        forces=(-efs["forces"]).astype(np.float32),
        stress=(-efs["stress"]).astype(np.float32),
    )
    atoms.info["hessian"] = to_training_layout(fc_bare).astype(np.float32)
    atoms.info["material"] = material
    db.write(atoms, data=atoms.info)

    return {
        "material": material, "n_atoms_supercell": n,
        "fc_reproduce_maxdiff": reproduce_err, "ref_is_linear_rebuild": ref_is_linear_rebuild,
        "fc_ref_max": float(np.abs(fc_ref).max()), "fc_d3_max": float(np.abs(fc_d3).max()),
        "fc_d3_vs_analytic_maxdiff": float(np.abs(fc_d3 - fc_d3_analytic).max()),
        "asr_ref": asr_residual(fc_ref), "asr_d3": asr_residual(fc_d3), "asr_bare": asr_residual(fc_bare),
        "E_d3_eV": efs["energy"], "F_d3_max": float(np.abs(efs["forces"]).max()),
        "stress_d3_max": float(np.abs(efs["stress"]).max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="v4_tmd_only")
    p.add_argument("--output-root", type=Path, default=HERE / "data" / "v4_tmd_only_bare_d3")
    p.add_argument("--cache-dir", type=Path, default=HERE / "d3_cache")
    p.add_argument("--splits", nargs="*", default=["train", "val", "holdout"])
    args = p.parse_args()

    split_root = WORKFLOW_ROOT / "nequix_datasets" / args.dataset_dir
    summary_rows = []
    for split in args.splits:
        materials = read_manifest(split_root / split / "materials.txt")
        out = args.output_root / split / "dataset.aselmdb"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        print(f"\n== {split}: {len(materials)} materials -> {out}")
        written = []
        with ase.db.connect(str(out)) as db:
            for mat in materials:
                row = process_material(mat, db, args.cache_dir)
                row["split"] = split
                summary_rows.append(row)
                written.append(mat)
                print(f"  ok {mat:20s} FC repro {row['fc_reproduce_maxdiff']:.1e}  |FC_D3|max {row['fc_d3_max']:.4f}"
                      f"  FD-vs-analytic {row['fc_d3_vs_analytic_maxdiff']:.1e}  ASR bare {row['asr_bare']:.1e}"
                      f"  E_D3 {row['E_d3_eV']:9.4f} eV  |F_D3|max {row['F_d3_max']:.4f}")
        with open(out.parent / "materials.txt", "w") as f:
            f.write(f"# Nequix dataset (bare PBE = reference PBE+D3 minus explicit D3) — {len(written)} materials\n")
            f.write(f"# Source split: {split_root / split}\n")
            f.write(f"# Generated: uv run python {' '.join(sys.argv)}\n")
            f.write(f"# Count: {len(written)}\n\n")
            for m in sorted(written):
                f.write(f"{m}\n")

    summary = args.output_root / "summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} materials; summary -> {summary}")


if __name__ == "__main__":
    main()
