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

## Results vs the grid-tight projected references (2026-10-24, jobs 3220379–84)

| cell | run | ft band RMSE (THz) | ft < 2 THz | ft DOS RMSE 0–5 THz (per atom) | ft DOS MAE 0–5 | base band RMSE | base DOS RMSE 0–5 | ft Γ shear / breathing (DFT) |
|---|---|---|---|---|---|---|---|---|
| MoS2 m1 21.8° | fixed | 0.036 | 0.082 | 0.025 | 0.012 | 0.207 | 0.035 | −0.17 / 0.87 (0.028 / 1.126) |
| MoS2 m1 21.8° | relaxpos | 0.034 | 0.036 | 0.023 | 0.011 | 0.224 | 0.033 | −0.14 / 1.07 |
| MoS2_WS2 m1 21.8° | fixed | 0.032 | 0.066 | 0.062 | 0.021 | 0.145 | 0.098 | −0.16 / 0.80 (0.050 / 1.032) |
| MoS2_WS2 m1 21.8° | relaxpos | 0.030 | 0.025 | 0.059 | 0.020 | 0.175 | 0.087 | −0.13 / 1.00 |
| MoS2 m2 13.2° | fixed | 0.035 | 0.054 | 0.030 | 0.014 | 0.197 | 0.040 | +0.09 / 0.90 (0.017 / 1.103) |
| MoS2 m2 13.2° | relaxpos | 0.034 | 0.035 | 0.024 | 0.013 | 0.215 | 0.033 | −0.06 / 1.07 |
| MoS2 m3 9.4° | fixed | 0.033 | 0.047 | 0.036 | 0.017 | 0.193 | 0.043 | +0.10 / 0.91 (0.018 / 1.085) |
| MoS2 m3 9.4° | relaxpos | 0.033 | 0.037 | 0.027 | 0.016 | 0.212 | 0.032 | +0.05 / 1.08 |

ft = v5 cellrelax baresub-e20 + D3; base = omat-1 zero-shot. Band RMSE on the DFT q-path (Γ–K–M–Γ, 153 points, all bands).
**DOS = linear tetrahedron method (no smearing) since 2026-10-24** (`write_phonon_dos.py` default: mesh length 400 Å → 55×55×1 / 34×34×1,
pitch 0.01 THz; the PI found the earlier σ = 0.1 THz Gaussian DOS too smooth). DOS errors are per atom on the DFT grid and are
3–6× larger than with σ = 0.1 (same runs: 0.007–0.011 RMSE) because sharp van Hove peaks penalise small frequency shifts;
the band RMSE is unaffected. relaxpos gaps: ft 3.095 / 3.096 / 3.064 / 3.006 Å vs DFT 3.192 / 3.198 / 3.151 / 3.074 (m1, MoS2_WS2 m1, m2, m3); base 3.95–3.98 Å.
MoS2 m3 (9.4°, 222 atoms, 1×1×1; evals 2026-10-26, jobs 3227750/1): no imaginary mode — the ft shear pair is positive but 3–6× too
stiff (+0.10 fixed / +0.05 relaxpos vs DFT +0.018), i.e. the same interlayer-landscape error as the imaginary pair at 21.8°, with the
opposite sign; breathing 0.91 (fixed) / 1.08 (relaxpos) vs 1.085.
Open discrepancy (unchanged from 10-13): the fine-tuned model gives an imaginary Γ shear pair (−0.13…−0.17 THz) at the symmetric registry
in the 2×2×1 m1 cells where grid-tight DFT and the sliding scan give +0.03…+0.05 (near-free phason); its breathing mode is 0.05–0.25 THz
soft. This is inherited from omat-1 (base −0.19 at fixed geometry) and is what the `< 2 THz` column measures.

**Figure for the Γ interlayer modes:** `gamma_zoom_v5.png` / `.pdf` (2026-10-26, `scripts/plot_twisted_gamma_zoom.py` on the
`workflow/share/twisted_dft_vs_nequix_v2` export) — the 0–1.6 THz window of all three cells × {fixed, relaxpos} with the Γ
shear-pair / breathing values annotated; the full-range `plots/<name>_vs_dft.png` overlays hide the imaginary shear pair.
`results.csv` Γ columns (fixed 2026-10-27): `*_gamma_optical_lowest3_THz` / `*_dft_path_gamma_optical_lowest3_THz` now list
the optical modes after removing the 3 Γ modes closest to zero (the ASR-pinned acoustic triple), so an imaginary shear pair shows
up as e.g. `-0.1682;-0.1682;0.8692`. Before the fix the script took modes 3..5 of the *sorted* list, which showed 0;0;breathing
whenever the pair was imaginary and reported the pair's magnitude under `*_gamma_acoustic_after_asr_max_abs_THz`. All CSVs in
this directory were recomputed from the saved `bands/*.npz` Γ data (no re-run; the band data, DOS, plots and README numbers
were never affected). `*_min_freq_THz` still gives the path minimum.

Each run folder holds `results.csv` (one row; RMSE vs DFT on the DFT q-path, DOS MAE + RMSE, Γ modes, relaxation info),
`plots/<name>_bands.png` (model only), `plots/<name>_vs_dft.png` (band overlay), `plots/<name>_dos.png` + `.json`
(per-atom DOS, DFT solid / ML dashed, MAE 0–5 THz), `dos/<name>_{dft,ft,base}_dos.dat` (same grid, header has natoms/mesh/σ),
`relaxed_structures/` (relaxpos only) and `bands/*.npz` (ignored, regenerable).

`demo_model_only_a3.1922/{near0,near60}/`: the 2026-09-17 model-only pilot (m = 1..5, both families, `-pft-v4-tmd-omat`, old
MoS2 a = 3.1922 Å, no DFT). Kept for the record; superseded by the per-cell folders above.

## Producing a new cell
```bash
# 1. DFT DOS into the result dir (nequix uv env; tetrahedron, mesh length 400 Å by default)
uv run python ../workflow/phonopy/write_phonon_dos.py ../workflow/FINAL_RESULTS_TWISTED/<name>
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
