# Twisted-bilayer phonon evaluations

One folder per twisted cell, named exactly like the structure in `workflow/twisted/structures/` and the DFT
reference in `workflow/FINAL_RESULTS_TWISTED/` (e.g. `MoS2_twist_m1_near0` = MoS2 homobilayer, (m, m+1) = (1, 2),
θ = 21.8°, 3R-seeded "near 0°" family). Inside each:

| subfolder | what it is |
|---|---|
| `fixed/` | model Hessian at the DFT-relaxed POSCAR (`--relax none`): force-constant accuracy only |
| `relaxpos/` | template start (`workflow/twisted/structures/<name>/POSCAR`), positions relaxed by the model, `--max-follow 0` so the symmetric registry DFT kept is preserved: the deployment number |

Both runs: model = `checkpoints/nequix-healthy-2d-pft-v5-tmd-cellrelax-omat-baresub-e20.nqx`
+ explicit D3 (`--add-d3 finetuned`), baseline = `models/nequix-omat-1.nqx` zero-shot (no D3), phonopy supercell = the DFT one
(`--supercell-k 2` for m1; m2/m3 use 1 1 1).

**References (since 2026-10-24):** `workflow/FINAL_RESULTS_TWISTED/<name>` = grid-tight statics (LREAL=F, ADDGRID, EDIFF 1E-8, egg-box
fix for the Γ shear pair) + invariance-projected force constants (ZA dip removed; `workflow/phonopy/project_invariant_force_constants.py`).
The earlier MoS2 m1 comparison against the production-INCAR reference (Γ shear pair +0.23 THz, an artifact) was deleted (nequix b71c19b).

## Results vs the grid-tight projected references (2026-10-24, jobs 3220338–43)

| cell | run | ft band RMSE (THz) | ft < 2 THz | ft DOS RMSE 0–5 THz (per atom) | base band RMSE | base < 2 THz | base DOS RMSE 0–5 | ft Γ shear / breathing (DFT) |
|---|---|---|---|---|---|---|---|---|
| MoS2 m1 21.8° | fixed | 0.036 | 0.082 | 0.0088 | 0.207 | 0.103 | 0.0225 | −0.17 / 0.87 (0.028 / 1.126) |
| MoS2 m1 21.8° | relaxpos | 0.034 | 0.036 | 0.0070 | 0.224 | 0.140 | 0.0198 | −0.14 / 1.07 |
| MoS2_WS2 m1 21.8° | fixed | 0.032 | 0.066 | 0.0112 | 0.145 | 0.088 | 0.0443 | −0.16 / 0.80 (0.050 / 1.032) |
| MoS2_WS2 m1 21.8° | relaxpos | 0.030 | 0.025 | 0.0100 | 0.175 | 0.118 | 0.0397 | −0.13 / 1.00 |
| MoS2 m2 13.2° | fixed | 0.035 | 0.054 | 0.0079 | 0.197 | 0.088 | 0.0215 | +0.09 / 0.90 (0.017 / 1.103) |
| MoS2 m2 13.2° | relaxpos | 0.034 | 0.035 | 0.0067 | 0.215 | 0.091 | 0.0180 | −0.06 / 1.07 |

ft = v5 cellrelax baresub-e20 + D3; base = omat-1 zero-shot. Band RMSE on the DFT q-path (Γ–K–M–Γ, 153 points, all bands);
DOS RMSE per atom on the DFT grid (σ = 0.1 THz). relaxpos gaps: ft 3.095 / 3.096 / 3.064 Å vs DFT 3.192 / 3.198 / 3.151; base 3.96–3.98 Å.
Open discrepancy (unchanged from 10-13): the fine-tuned model gives an imaginary Γ shear pair (−0.13…−0.17 THz) at the symmetric registry
in the 2×2×1 m1 cells where grid-tight DFT and the sliding scan give +0.03…+0.05 (near-free phason); its breathing mode is 0.05–0.25 THz
soft. This is inherited from omat-1 (base −0.19 at fixed geometry) and is what the `< 2 THz` column measures.

Each run folder holds `results.csv` (one row; RMSE vs DFT on the DFT q-path, DOS MAE + RMSE, Γ modes, relaxation info),
`plots/<name>_bands.png` (model only), `plots/<name>_vs_dft.png` (band overlay), `plots/<name>_dos.png` + `.json`
(per-atom DOS, DFT solid / ML dashed, MAE 0–5 THz), `dos/<name>_{dft,ft,base}_dos.dat` (same grid, header has natoms/mesh/σ),
`relaxed_structures/` (relaxpos only) and `bands/*.npz` (ignored, regenerable).

`demo_model_only_a3.1922/{near0,near60}/`: the 2026-09-17 model-only pilot (m = 1..5, both families, `-pft-v4-tmd-omat`, old
MoS2 a = 3.1922 Å, no DFT). Kept for the record; superseded by the per-cell folders above.

## Producing a new cell
```bash
# 1. DFT DOS into the result dir (workflow, phonon-env)
TWIST_VARIANT=twisted python3 phonopy/write_phonon_dos.py FINAL_RESULTS_TWISTED/<name>
# 2. the two evals (nequix; SBATCH_EXPORT=none on the login node -> pass the env var explicitly)
COMMON="--add-d3 finetuned --baseline-model models/nequix-omat-1.nqx --dft-results ../workflow/FINAL_RESULTS_TWISTED --supercell-k <k>"
sbatch --export=ALL,TWISTED_EXTRA_ARGS="$COMMON --relax none" --job-name=twisted_<name>_fixed \
  sbatch/eval_twisted_phonons.sbatch checkpoints/nequix-healthy-2d-pft-v5-tmd-cellrelax-omat-baresub-e20.nqx \
  phonon_comparisons/twisted/<name>/fixed ../workflow/FINAL_RESULTS_TWISTED/<name>
sbatch --export=ALL,TWISTED_EXTRA_ARGS="$COMMON --relax positions --max-follow 0" --job-name=twisted_<name>_relaxpos \
  sbatch/eval_twisted_phonons.sbatch checkpoints/nequix-healthy-2d-pft-v5-tmd-cellrelax-omat-baresub-e20.nqx \
  phonon_comparisons/twisted/<name>/relaxpos ../workflow/twisted/structures/<name>
# 3. DOS plot per run folder
uv run python scripts/compare_phonon_dos.py --dft <run>/dos/<name>_dft_dos.dat --ml <run>/dos/<name>_ft_dos.dat <run>/dos/<name>_base_dos.dat \
  --ml-labels "fine-tuned v5 + D3" "omat-1 zero-shot" --output <run>/plots/<name>_dos.png
```
