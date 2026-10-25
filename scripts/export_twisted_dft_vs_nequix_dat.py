#!/usr/bin/env python3
"""Export the twisted-cell DFT-vs-Nequix phonon comparisons as plain-text data for replotting (PI request).

Reproduces the hand-packed share/twisted_mos2_m1_dft_vs_nequix_v1 layout (2026-10-16), one subfolder per cell:
  <cell>/bands_<run>_<curve>.dat   path coordinate (1/A) + all branch frequencies (THz, ascending), 153 q on G-K-M-G
  <cell>/dos_<run>_<curve>.dat     DOS files as written by the eval (header: natoms, mesh, sigma/tetrahedron, grid)
  <cell>/figure_<run>_{bands_vs_dft,dos}.png
  README.md (top level) with per-cell headline numbers from results.csv
Sources: phonon_comparisons/twisted/<cell>/<run>/{bands/<cell>_vs_dft.npz, dos/*.dat, plots/*.png, results.csv};
cell metadata from ../workflow/twisted/structures/<cell>/twist.json and the DFT phonopy.yaml supercell.
Usage: uv run python scripts/export_twisted_dft_vs_nequix_dat.py [--cells ...] [--name twisted_dft_vs_nequix_v2]
"""
import argparse, csv, hashlib, json, shutil, zipfile
from datetime import date
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT.parent / "workflow"
CURVES = {"dft": "DFT_PBE-D3", "ft": "Nequix_finetuned_v5_D3", "base": "Nequix_omat1_zeroshot"}
CURVE_DESC = {
    "dft": "VASP PBE+D3(BJ), phonopy +-0.01 A finite displacements, grid-tight statics (LREAL=F, ADDGRID, EDIFF 1E-8), "
           "force constants projected onto exact translational/rotational/Huang invariance",
    "ft": "Nequix fine-tuned on the 48 Mo/W TMD references (nequix-healthy-2d-pft-v5-tmd-cellrelax-omat-baresub-e20) + explicit D3, analytical Hessian",
    "base": "Nequix nequix-omat-1 base model, zero-shot, no fine-tuning, no D3",
}
RUNS = {"fixed": "ML Hessian at the DFT-relaxed geometry", "relaxpos": "ML model relaxes the atomic positions first (deployment protocol)"}


def export_cell(cell: str, out: Path) -> dict:
    src = ROOT / "phonon_comparisons" / "twisted" / cell
    tw = json.loads((WF / "twisted" / "structures" / cell / "twist.json").read_text())
    py = yaml.safe_load((WF / "FINAL_RESULTS_TWISTED" / cell / "phonopy.yaml").read_text())
    dim = np.array(py["supercell_matrix"]).diagonal()
    natoms = len(py["unit_cell"]["points"])
    formula = cell.split("_twist")[0].replace("_", "/")
    desc = f"{formula} twisted bilayer, {tw['theta_deg']:.1f} deg (commensurate m={tw['m']} 'near-0' cell, {natoms} atoms, a = {tw['a_layer_A']:.4f} A)"
    d = out / cell; d.mkdir(parents=True)
    info = {"cell": cell, "formula": formula, "theta": tw["theta_deg"], "natoms": natoms, "dim": "x".join(map(str, dim)), "runs": {}}
    for run in RUNS:
        z = np.load(src / run / "bands" / f"{cell}_vs_dft.npz")
        dist, edges = z["distances"], z["edges"]; nb = z["dft"].shape[1]
        for key, curve in CURVES.items():
            arr = {"dft": z["dft"], "ft": z["model_ft"], "base": z["model_base"]}[key]
            hdr = (f"# {desc}, phonon dispersion along G-K-M-G\n"
                   f"# column 1: path coordinate (1/A, cumulative |dq|); columns 2..{nb + 1}: the {nb} branch frequencies in THz, sorted ascending\n"
                   f"# high-symmetry point indices (0-based rows): G={edges[0]}, K={edges[1]}, M={edges[2]}, G={edges[3]}\n"
                   f"# run: {run} = {RUNS[run]}\n# curve: {CURVE_DESC[key]}" + (f", {dim[0]}x{dim[1]}x{dim[2]} supercell" if key == "dft" else ""))
            np.savetxt(d / f"bands_{run}_{curve}.dat", np.column_stack([dist, np.sort(arr, axis=1)]), fmt="%12.6f", header=hdr, comments="")
            shutil.copy2(src / run / "dos" / f"{cell}_{key}_dos.dat", d / f"dos_{run}_{curve}.dat")
        shutil.copy2(src / run / "plots" / f"{cell}_vs_dft.png", d / f"figure_{run}_bands_vs_dft.png")
        shutil.copy2(src / run / "plots" / f"{cell}_dos.png", d / f"figure_{run}_dos.png")
        r = next(csv.DictReader(open(src / run / "results.csv")))
        info["runs"][run] = {k: float(r[k]) for k in ("ft_dft_rmse_THz", "base_dft_rmse_THz", "ft_dft_rmse_below2THz_THz", "base_dft_rmse_below2THz_THz",
                                                      "ft_dos_rmse_0to5_THz", "base_dos_rmse_0to5_THz", "ft_dos_mae_0to5_THz", "base_dos_mae_0to5_THz")}
        def gamma3(arr):  # Gamma row: drop the 3 modes closest to zero (acoustic), report the next 3 lowest (shear pair + breathing)
            g = np.sort(arr[edges[0]]); keep = np.sort(np.argsort(np.abs(g))[3:]); return " / ".join(f"{x:+.3f}" for x in g[keep][:3])
        info["runs"][run]["gamma"] = (gamma3(z["model_ft"]), gamma3(z["dft"]))
        info["nb"] = nb
        dos_hdr = [l for l in (src / run / "dos" / f"{cell}_dft_dos.dat").read_text().splitlines() if l.startswith("#")]
        info["dos_mesh"] = next(l.split("=")[1].strip() for l in dos_hdr if "mesh" in l)
    return info


def readme(infos: list[dict]) -> str:
    cells = "\n".join(f"| `{i['cell']}` | {i['formula']} | {i['theta']:.1f}° | {i['natoms']} | {i['nb']} | {i['dim']} | {i['dos_mesh']} |" for i in infos)
    rows = "\n".join(
        f"| `{i['cell']}` | {run} | {v['ft_dft_rmse_THz']:.3f} / {v['base_dft_rmse_THz']:.3f} | {v['ft_dft_rmse_below2THz_THz']:.3f} / {v['base_dft_rmse_below2THz_THz']:.3f} | "
        f"{v['ft_dos_rmse_0to5_THz']:.3f} / {v['base_dos_rmse_0to5_THz']:.3f} | {v['ft_dos_mae_0to5_THz']:.4f} / {v['base_dos_mae_0to5_THz']:.4f} | {v['gamma'][0]} (DFT {v['gamma'][1]}) |"
        for i in infos for run, v in i["runs"].items())
    return f"""# Twisted TMD bilayers: DFT vs Nequix phonons (data for replotting) — v2

Kai Gowers, {date.today().isoformat()}. Supersedes `twisted_mos2_m1_dft_vs_nequix_v1` (2026-10-16): the DFT references were
recomputed with grid-converged settings and sum-rule-projected force constants (see below), the DOS is now the sharp
tetrahedron DOS, and two more cells are included.

| cell | material | θ | atoms | branches | DFT phonon supercell | DOS mesh |
|---|---|---|---|---|---|---|
{cells}

## Files (one folder per cell)
- `bands_<run>_<curve>.dat` — phonon dispersion along Γ–K–M–Γ of the moiré Brillouin zone. Column 1 = path coordinate
  (1/Å), the remaining columns = all branches in THz (ascending; 3 × atoms of them). All curves of a cell share the same
  153 q-points, so they overlay directly. High-symmetry rows: 0 (Γ), 50 (K), 101 (M), 152 (Γ).
- `dos_<run>_<curve>.dat` — phonon DOS per cell on a Γ-centred mesh, **linear tetrahedron method (no smearing)**, 0.01 THz
  pitch; two columns (frequency THz, states/THz/cell), header lines start with #, `natoms` in the header for per-atom scaling.
- `figure_<run>_bands_vs_dft.png`, `figure_<run>_dos.png` — the plots these data come from.

Curves: `DFT_PBE-D3` (VASP PBE+D3(BJ), phonopy finite displacements), `Nequix_finetuned_v5_D3` (Nequix fine-tuned on
48 Mo/W-TMD mono/bilayer phonon references — no twisted or strained cells in training — plus explicit D3 dispersion),
`Nequix_omat1_zeroshot` (the untuned nequix-omat-1 base model, no D3).
Runs: `relaxpos` = the ML model relaxes the atomic positions itself before the Hessian (deployment protocol, headline
numbers); `fixed` = ML Hessian at the DFT geometry.

## Headline numbers (fine-tuned / zero-shot)

| cell | run | band RMSE all (THz) | band RMSE < 2 THz | DOS RMSE 0–5 THz (per atom) | DOS MAE 0–5 THz | fine-tuned Γ shear pair / breathing (THz) |
|---|---|---|---|---|---|---|
{rows}

## What changed in the DFT reference since v1 (please read before zooming near Γ)
1. v1's Γ layer-shear doublet (+0.230 THz) and breathing mode (1.235 THz) were finite-displacement artifacts of the VASP
   real-space grid. All statics were rerun with LREAL=.FALSE., ADDGRID, EDIFF=1E-8; the layer-shear stiffness now agrees
   with an independent rigid-sliding DFT energy scan in every cell (MoS2 m1: Γ doublet +0.028 THz, breathing 1.126 THz).
2. The remaining ~−0.05 THz dip of the flexural (ZA) branch near Γ was a violation of the rotational sum rule by the
   finite-displacement force constants (not an instability, not stress). The force constants were projected onto exact
   translational + rotational + Huang invariance with the Γ interlayer modes held fixed; optical branches moved < 0.002 THz.
   Every DFT branch in these files is ≥ 0 THz.
3. Consequence for the comparison: the fine-tuned model's imaginary Γ shear doublet (−0.13…−0.17 THz in the 21.8° cells)
   at the symmetric registry is now a genuine model error, not a reference artifact — DFT finds a near-free but stable
   sliding (phason) mode at +0.03…+0.05 THz. The model's breathing mode is 0.05–0.25 THz soft. Above ~1.5 THz the
   agreement is at the 0.03 THz level. The same shear error is present in the zero-shot base model, i.e. it is inherited
   from pretraining; the fix (registry/gap augmentation of the training set) is planned.
4. DOS: v1 used a 0.1 THz Gaussian smearing; v2 uses the tetrahedron method, so DOS error numbers are not comparable
   between versions (sharp van Hove peaks penalise small frequency shifts more).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="*", default=["MoS2_twist_m1_near0", "MoS2_WS2_twist_m1_near0", "MoS2_twist_m2_near0"])
    ap.add_argument("--name", default="twisted_dft_vs_nequix_v2")
    a = ap.parse_args()
    out = WF / "share" / a.name
    if out.exists():
        shutil.rmtree(out)
    infos = [export_cell(c, out) for c in a.cells]
    (out / "README.md").write_text(readme(infos))
    with (out / "SHA256SUMS").open("w") as fh:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n")
    zp = out.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zf.write(p, Path(out.name) / p.relative_to(out))
    print(f"{len(infos)} cells -> {out}\nzip: {zp} ({zp.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
