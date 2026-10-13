# Twisted-bilayer phonon evaluations

One folder per twisted cell, named exactly like the structure in `workflow/twisted/structures/` and the DFT
reference in `workflow/FINAL_RESULTS_TWISTED/` (e.g. `MoS2_twist_m1_near0` = MoS2 homobilayer, (m, m+1) = (1, 2),
θ = 21.8°, 3R-seeded "near 0°" family). Inside each:

| subfolder | what it is |
|---|---|
| `fixed/` | model Hessian at the DFT-relaxed POSCAR (`--relax none`): force-constant accuracy only |
| `relaxpos/` | template start (`workflow/twisted/structures/<name>/POSCAR`), positions relaxed by the model, `--max-follow 0` so the symmetric registry DFT kept is preserved: the deployment number |
| `*_v4baseline/` | same two runs but with the fine-tuned implicit-D3 `-pft-v4-tmd-omat` as the baseline (secondary, fine-tune vs fine-tune) |

`fixed/` and `relaxpos/` are the canonical pair: model = `checkpoints/nequix-healthy-2d-pft-v5-tmd-cellrelax-omat-baresub-e20.nqx`
+ explicit D3 (`--add-d3 finetuned`), baseline = `models/nequix-omat-1.nqx` zero-shot (no D3), phonopy supercell = the DFT one
(`--supercell-k 2` for m1; m2/m3 use 1 1 1).

Each run folder holds `results.csv` (one row; RMSE vs DFT on the DFT q-path, DOS MAE, Γ modes, relaxation info),
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
