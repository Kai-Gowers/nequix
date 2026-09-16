# Explicit dispersion (delta learning) for Nequix phonon fine-tuning

**Question.** The DFT reference (VASP PBE + DFT-D3(BJ), `IVDW = 12`) includes a dispersion term the
pretrained Nequix models never saw. So far the fine-tunes learned it implicitly. Here the D3 term is
**subtracted from the training targets** so the network learns bare PBE, and **added back analytically
at inference**. Same 48-TMD data and 39/3/6 split as `v4_tmd_only`, same base model (`nequix-omat-1`,
itself trained on PBE(+U) without D3), same hyperparameters — only the targets change.

## What is subtracted / added back

Training rows are one equilibrium supercell per material with a force-constant Hessian target
(`energy_weight = 0`, `hessian_weight = 100`), so the directions "E_bare = E - E_D3, F_bare = F - F_D3"
translate to:

| target | reference (PBE+D3) row | bare-PBE row written here |
|---|---|---|
| hessian (supercell FCs) | `FC_ref` from `FORCE_CONSTANTS` | `FC_ref - FC_D3` |
| forces | 0 (PBE+D3 minimum) | `-F_D3` |
| stress | 0 (placeholder, cell not relaxed) | `-sigma_D3` (same placeholder assumption) |
| energy | 0 | `-E_D3` (weight 0, kept for consistency) |

`FC_D3` is built exactly the way the reference FCs were: D3 forces on phonopy's displaced supercells
from `FORCE_SETS` (the displacements VASP actually ran) -> `produce_force_constants` ->
`symmetrize_force_constants`. FC construction is linear in the forces, so for a plain phonopy reference
`FC_ref - FC_D3` *is* the FC phonopy would give from PBE-only forces. It agrees with the analytic D3
Hessian from the library to 2.2e-06 eV/A^2 and is translationally invariant (max ASR residual of
`FC_bare`: 3.9e-13).

At inference (`eval_explicit_dispersion.py`) the **same cached `FC_D3`** (`d3_cache/<mat>.npz`) is added
to the model Hessian before phonopy, so `FC_bare + FC_D3 == FC_ref` holds by construction and any residual
difference between this D3 implementation and VASP's cancels. `d3.NequixD3Calculator` does the same for
energy/forces/stress (`E = E_nequix + E_D3`, ...) as a plain ASE calculator.

## Parameter match with VASP

`d3.py` uses simple-dftd3 (`dftd3` PyPI package, optional extra `dispersion`) with the values VASP
prints in this dataset's OUTCARs: s6 = 1.0, s8 = 0.7875, a1 = 0.4289, a2 = 4.4407, two-body cutoff
50.2022 A, CN cutoff 21.1671 A, no three-body term. Checked against VASP's own `Edisp (eV)` line on
every displaced supercell still on this host (`check_vs_vasp.py` -> `check_vs_vasp.csv`):

| displaced supercells | materials | max diff | mean diff | max diff per atom |
|---|---|---|---|---|
| 390 | 40 | 0.85 meV | 0.49 meV | 0.0048 meV |

(With the library's default cutoffs the MoS2 error is 4.4 meV; the VASP cutoffs bring it to 0.14 meV.)

## Caveats found while building the data

- **22/48 reference `FORCE_CONSTANTS` are not a plain phonopy rebuild of `FORCE_SETS`** (max
  difference 1e-2 .. 5e-1 eV/A^2 vs 1e-4 .. 1.5e-3 for the rest). Those were post-processed with the
  hiphive rotational-sum-rule correction in the workflow (a `backups/` dir in the staticpoint folder
  confirms it for e.g. WS2_WTe2_3R). The subtraction is still exactly self-consistent (see above); it
  just is no longer literally "phonopy on PBE-only forces" for these. Flagged per material in
  `data/v4_tmd_only_bare_d3/summary.csv` (`ref_is_linear_rebuild`). Materials: MoS2_MoSe2_2H, MoS2_MoTe2_2H, MoS2_MoTe2_3R, MoS2_WS2_3R, MoS2_WSe2_3R, MoS2_WTe2_2H, MoS2_WTe2_3R, MoSe2, MoSe2_MoTe2_2H, MoSe2_WS2_2H, MoSe2_WS2_3R, MoSe2_bilayer_2H, MoTe2_WS2_2H, MoTe2_WS2_3R, MoTe2_WSe2_2H, MoTe2_WSe2_3R, MoTe2_WTe2_2H, MoTe2_WTe2_3R, WS2, WS2_WTe2_2H, WS2_WTe2_3R, WSe2_WTe2_3R.
- **Stress target.** The reference rows use stress = 0 although cells were relaxed at fixed lattice
  (ISIF=2), so the true PBE+D3 stress is not zero. `-sigma_D3` inherits that placeholder assumption;
  same approximation as every earlier fine-tune (`stress_weight = 5` vs `hessian_weight = 100`).
- `FC_D3` is small: |FC_D3|max 0.021 .. 0.068 eV/A^2 against |FC_ref|max ~20-29, i.e.
  the D3 part of the force constants is ~0.1-0.3 % of the largest elements but concentrated in the soft
  interlayer / flexural blocks that set the low-frequency bands.
- `check_vs_vasp.py` reads `disp-00N/POSCAR`, not the top-level `POSCAR-00N`: WTe2's top-level files are
  a stale 4x4x1 supercell next to a 5x5x1 rerun.

## Files

| file | role |
|---|---|
| `d3.py` | VASP-matched D3 term: `d3_efs`, `d3_hessian_analytic`, `d3_force_constants_phonopy`, layout helpers, `D3Calculator`, `NequixD3Calculator` |
| `strip_d3_dataset.py` | builds `data/v4_tmd_only_bare_d3/{train,val,holdout}/dataset.aselmdb` + `summary.csv`, fills `d3_cache/` |
| `check_vs_vasp.py` | parameter-match check against VASP `Edisp` -> `check_vs_vasp.csv` |
| `eval_explicit_dispersion.py` | phonon-band eval adding `FC_D3` back (`--add-d3 finetuned|baseline|both|none`) |
| `../configs/nequix-healthy-2d-pft-v4-tmd-omat-bared3.yml` | training config (copy of `-v4-tmd-omat` with bare data paths) |
| `../sbatch/eval_explicit_dispersion.sbatch` | GPU eval wrapper |

## Commands (from `nequix/`)

```bash
uv sync --extra pft --extra dispersion            # or: uv add --optional dispersion dftd3 (already done)
uv run python explicit_dispersion/check_vs_vasp.py
uv run python explicit_dispersion/strip_d3_dataset.py
sbatch --job-name=nequix_pft_v4_tmd_omat_bared3 sbatch/train_healthy_2d_pft_v4_tmd_base.sbatch configs/nequix-healthy-2d-pft-v4-tmd-omat-bared3.yml
# main comparison: bare-PBE fine-tune + D3  vs  implicit-D3 fine-tune (same base, same split)
sbatch --job-name=nequix_eval_omat_bared3 sbatch/eval_explicit_dispersion.sbatch \
  checkpoints/nequix-healthy-2d-pft-v4-tmd-omat-bared3.nqx checkpoints/nequix-healthy-2d-pft-v4-tmd-omat.nqx finetuned \
  phonon_comparisons/v4_tmd_only_omat_bared3_eval
# zero-shot check: is omat-1's 0.155 THz error mostly missing dispersion?
sbatch --job-name=nequix_eval_omat_zeroshot_plusd3 sbatch/eval_explicit_dispersion.sbatch \
  nequix-omat-1 nequix-omat-1 finetuned phonon_comparisons/v4_tmd_only_omat_zeroshot_plusd3_eval
```

## Results

Filled in from `phonon_comparisons/v4_tmd_only_omat_bared3_eval/` and
`phonon_comparisons/v4_tmd_only_omat_zeroshot_plusd3_eval/` when the jobs finish; see `EXPERIMENTS.md`.
