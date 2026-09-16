"""
Validate the explicit D3 term against VASP's own DFT-D3(BJ) energies.

For every TMD material that still has its VASP static-point directory on this host
(workflow/phonopy_{monolayer,bilayer}_examples/<mat>_staticpoint/disp-00N/OUTCAR), parse the
`Edisp (eV)` line VASP prints for IVDW=12 and compare it with `d3.d3_efs()` evaluated on the
same displaced supercell (POSCAR-00N). Agreement to ~1 meV per cell means the damping
parameters and real-space cutoffs used here reproduce what went into the reference data.

Usage (from nequix/):
  uv run python explicit_dispersion/check_vs_vasp.py [--dataset-dir v4_tmd_only] [--out explicit_dispersion/check_vs_vasp.csv]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKFLOW_ROOT = ROOT.parent / "workflow"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from d3 import d3_efs  # noqa: E402
from eval_healthy2d_finetune import read_manifest  # noqa: E402

EDISP_RE = re.compile(r"Edisp \(eV\)\s+([-+0-9.Ee]+)")


def staticpoint_dir(material: str) -> Path | None:
    for sub in ("phonopy_monolayer_examples", "phonopy_bilayer_examples"):
        d = WORKFLOW_ROOT / sub / f"{material}_staticpoint"
        if (d / "disp-001" / "OUTCAR").exists():
            return d
    return None


def vasp_edisp(outcar: Path) -> float | None:
    text = outcar.read_text(errors="ignore")
    matches = EDISP_RE.findall(text)
    return float(matches[-1]) if matches else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="v4_tmd_only")
    p.add_argument("--out", type=Path, default=HERE / "check_vs_vasp.csv")
    args = p.parse_args()

    split_root = WORKFLOW_ROOT / "nequix_datasets" / args.dataset_dir
    materials = sorted({m for s in ("train", "val", "holdout") for m in read_manifest(split_root / s / "materials.txt")})

    rows = []
    for mat in materials:
        sp = staticpoint_dir(mat)
        if sp is None:
            print(f"  - {mat}: no staticpoint dir on this host, skipped")
            continue
        for disp in sorted(sp.glob("disp-*")):
            outcar = disp / "OUTCAR"
            # the POSCAR inside disp-00N is what VASP actually ran; the top-level POSCAR-00N
            # can be stale (WTe2: 4x4x1 POSCAR-00N left over next to a 5x5x1 rerun)
            poscar = disp / "POSCAR"
            if not poscar.exists():
                poscar = sp / f"POSCAR-{disp.name.split('-')[1]}"
            e_vasp = vasp_edisp(outcar) if outcar.exists() else None
            if e_vasp is None:
                print(f"  - {mat}/{disp.name}: no Edisp line in OUTCAR, skipped")
                continue
            try:
                atoms = read(poscar)
                e_d3 = d3_efs(atoms)["energy"]
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {mat}/{disp.name}: D3 evaluation failed: {exc}")
                continue
            diff_mev = (e_d3 - e_vasp) * 1000.0
            rows.append({"material": mat, "disp": disp.name, "n_atoms": len(atoms),
                         "E_vasp_eV": e_vasp, "E_d3_eV": e_d3, "diff_meV": diff_mev,
                         "diff_meV_per_atom": diff_mev / len(atoms)})
            print(f"  {mat:22s} {disp.name}  VASP {e_vasp:12.5f}  ours {e_d3:12.5f}  diff {diff_mev:+8.3f} meV")

    if not rows:
        sys.exit("No comparisons made — no staticpoint OUTCARs found for this split.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    diffs = [abs(r["diff_meV"]) for r in rows]
    print(f"\n{len(rows)} displaced supercells over {len({r['material'] for r in rows})} materials")
    print(f"|E_d3 - Edisp|: max {max(diffs):.3f} meV, mean {sum(diffs)/len(diffs):.3f} meV  -> {args.out}")


if __name__ == "__main__":
    main()
