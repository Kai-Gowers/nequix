"""
Build the v5 bare-PBE training set: v4's rows, but the 16 cellrelax materials at their ISIF=4-refined cells.

Thin additive wrapper over build_bare_subtract_dataset.py (unchanged): same labels and layout
(hessian = bare FORCE_CONSTANTS, forces = -F_D3(eq), stress = sigma_OUTCAR - sigma_D3, energy = bare
unit-cell energy x n_sc/n_uc), with two per-material redirections for the 16 materials listed in
workflow/cellrelax/materials_16.txt:

  * bare-PBE source   workflow/FINAL_RESULTS_BARE_PBE_SUBTRACT/<mat>  -> .../FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX/<mat>
  * relaxation OUTCAR workflow/{monolayer,bilayer}_examples/<mat>     -> .../{monolayer,bilayer}_examples_cellrelax/<mat>
    (the production relaxation dirs still hold the OLD-cell relaxation; the geometry check in
    unitcell_bare_efs would fail against the refined POSCAR, which is exactly the guard we want)

The other 32 materials are read from the production dirs, so their rows are byte-for-byte the v4 ones.

Usage (from nequix/, uv env for simple-dftd3):
  uv run python explicit_dispersion/build_bare_subtract_dataset_cellrelax.py
      [--dataset-dir v5_tmd_cellrelax] [--output-root explicit_dispersion/data/v5_tmd_cellrelax_bare_subtract]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_bare_subtract_dataset as B  # noqa: E402

W = B.WORKFLOW_ROOT
CELLRELAX_SUBTRACT = W / "FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX"
PRODUCTION_SUBTRACT = B.BARE_SUBTRACT
M16 = {l.split()[0] for l in (W / "cellrelax" / "materials_16.txt").read_text().splitlines()
       if l.strip() and not l.startswith("#")}
assert len(M16) == 16, M16

_orig_relaxation_dir = B.relaxation_dir
_orig_process_material = B.process_material


def relaxation_dir(name: str) -> Path:
    if name in M16:
        for d in (W / "monolayer_examples_cellrelax" / name, W / "bilayer_examples_cellrelax" / name):
            if (d / "OUTCAR").exists():
                return d
        sys.exit(f"{name}: no cellrelax relaxation OUTCAR on this host")
    return _orig_relaxation_dir(name)


def process_material(material: str, db) -> dict:
    B.BARE_SUBTRACT = CELLRELAX_SUBTRACT if material in M16 else PRODUCTION_SUBTRACT
    row = _orig_process_material(material, db)
    row["source"] = "cellrelax" if material in M16 else "v4"
    return row


B.relaxation_dir = relaxation_dir
B.process_material = process_material

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--dataset-dir" not in argv:
        argv += ["--dataset-dir", "v5_tmd_cellrelax"]
    if "--output-root" not in argv:
        argv += ["--output-root", str(HERE / "data" / "v5_tmd_cellrelax_bare_subtract")]
    sys.argv = [sys.argv[0]] + argv
    B.main()
