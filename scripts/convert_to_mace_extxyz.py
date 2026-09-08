#!/usr/bin/env python3
"""
Standard (forces-only) MACE fine-tuning counterpart to Nequix's Phonon Fine-Tuning:
converts the same underlying DFT data used to build v4_tmd_only's Hessian-based PFT
training set into extxyz files MACE's own `--foundation_model` fine-tuning CLI can
consume directly.

NOT the same recipe as Nequix's PFT (which trains on the DFT Hessian/force-constant
matrix, energy_weight=0). MACE's stock training loop has no Hessian loss, so this
uses the raw per-displacement DFT forces instead (`FORCE_SETS`, already permanently
retained per material under workflow/FINAL_RESULTS_HEALTHY/) with MACE's
`--loss forces_only` — the closest apples-to-apples "does fine-tuning help MACE too"
comparison achievable without building a custom Hessian loss for MACE.

Only the train+val materials from --dataset-dir go into the output (matching
v4_tmd_only's 39/3 split) — the 6 holdout materials are reserved for phonon-RMSE
evaluation via eval_mace_baseline.py, exactly as with the Nequix PFT checkpoint.

Usage (run in nequix's main venv — only needs phonopy+ase, not mace itself):
  uv run python scripts/convert_to_mace_extxyz.py --dataset-dir v4_tmd_only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import phonopy
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

ROOT = Path(__file__).parent.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"
FINAL_RESULTS_HEALTHY = WORKFLOW_ROOT / "FINAL_RESULTS_HEALTHY"

sys.path.insert(0, str(ROOT / "scripts"))
from eval_healthy2d_finetune import read_manifest, phonopy_atoms_to_ase  # noqa: E402


def material_displacement_atoms(material: str) -> list:
    """One ASE Atoms per symmetry-inequivalent DFT displacement, forces attached."""
    material_dir = FINAL_RESULTS_HEALTHY / material
    ph = phonopy.load(
        phonopy_yaml=str(material_dir / "phonopy.yaml"),
        force_sets_filename=str(material_dir / "FORCE_SETS"),
        log_level=0,
    )
    out = []
    for disp, supercell in zip(ph.dataset["first_atoms"], ph.supercells_with_displacements):
        atoms = phonopy_atoms_to_ase(supercell)
        forces = np.array(disp["forces"])
        atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=forces)
        atoms.info["material"] = material
        out.append(atoms)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="v4_tmd_only")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    dataset_dir = WORKFLOW_ROOT / "nequix_datasets" / args.dataset_dir
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "mace_baseline" / "data" / f"{args.dataset_dir}_forces"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        materials = read_manifest(dataset_dir / split / "materials.txt")
        all_atoms = []
        for material in materials:
            all_atoms.extend(material_displacement_atoms(material))
        out_path = output_dir / f"{split}.xyz"
        write(str(out_path), all_atoms, format="extxyz")
        print(f"{split}: {len(materials)} materials, {len(all_atoms)} displacement structures -> {out_path}")


if __name__ == "__main__":
    main()
