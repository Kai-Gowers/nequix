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

Both jobs finished 2026-09-16 (train 3012446, evals 3012447 / 3013019). Whole-band RMSE vs DFT in THz,
`v4_tmd_only` split (39 train / 3 val / 6 held-out heterobilayers).

### Main comparison: bare-PBE fine-tune + explicit D3 vs implicit-D3 fine-tune (both from `nequix-omat-1`)

`phonon_comparisons/v4_tmd_only_omat_bared3_eval/results.csv` (`rmse_ft` = bared3 ft + analytic D3 Hessian,
`rmse_base` = `checkpoints/nequix-healthy-2d-pft-v4-tmd-omat.nqx`, identical to `v4_tmd_only_omat_eval` to 1e-6).

| group | n | bared3 ft + D3 | implicit ft | bared3 wins |
|---|---|---|---|---|
| all 48 | 48 | 0.0512 | 0.0506 | 23/48 |
| train | 39 | 0.0488 | 0.0482 | 20/39 |
| val | 3 | 0.0564 | 0.0542 | 1/3 |
| held-out | 6 | 0.0644 | 0.0648 | 2/6 |
| worst held-out | | 0.110 (MoSe2_WSe2_2H) | 0.107 (MoSe2_WSe2_2H) | |

Acoustic / optical split is equally flat (0.064/0.048 vs 0.063/0.047). Training-side val Hessian MAE
0.0040 (bared3) vs 0.0044 (implicit) — the bare-PBE target is marginally easier to fit, but that does
not translate into band-structure accuracy once D3 is added back.

**Conclusion: null result.** Delta-learning the dispersion explicitly neither helps nor hurts the
fine-tuned model on this split. The fine-tune already absorbs the (smooth, small: max |FC_D3| ≤ 0.07
eV/Å²) D3 contribution without difficulty. Not worth the extra inference dependency; keep the implicit
recipe as default.

### Zero-shot: is omat-1's 0.155 THz error mostly missing dispersion?

`phonon_comparisons/v4_tmd_only_omat_zeroshot_plusd3_eval/results.csv` (`rmse_ft` = omat-1 + D3,
`rmse_base` = plain omat-1).

| group | n | omat-1 + D3 | omat-1 | +D3 wins |
|---|---|---|---|---|
| all 48 | 48 | 0.1535 | 0.1548 | 46/48 |
| held-out | 6 | 0.1278 | 0.1288 | 5/6 |
| worst | | 0.227 (MoTe2) | 0.229 (MoTe2) | |

Adding D3 is consistently but negligibly better (~1%). **No** — omat-1's residual error on these TMDs is
not dominated by the missing vdW treatment (OMat24 is PBE without D3); the 0.155 THz residual is
elsewhere, and only the fine-tune (0.155 → 0.051) removes it.

### Stage 1 (2026-09-18): does the bare-PBE model reproduce bare PBE *before* D3 is added back?

Asked for by the PI ("train without D3 first, make sure the bare PBE model is correct, then add D3").
`--reference bare --add-d3 none`, job 3019406 → `phonon_comparisons/v4_tmd_only_omat_bared3_stage1_bare_eval/`.
Reference = `FC_ref - FC_D3` on the band.yaml q-list, i.e. the bare-PBE curvature **at the PBE+D3-relaxed
geometry** (not a bare-PBE minimum: |F_D3| there is 0.10–0.17 eV/Å). Baseline = zero-shot `nequix-omat-1`,
itself a PBE-without-D3 model. Fixed geometry throughout.

| group | n | bared3 ft (no D3) vs bare ref | omat-1 vs bare ref | ft wins | stage 2: bared3 ft + D3 vs PBE+D3 ref |
|---|---|---|---|---|---|
| all 48 | 48 | 0.0527 | 0.1670 | 48/48 | 0.0525 |
| train | 39 | 0.0504 | 0.1691 | 39/39 | 0.0502 |
| val | 3 | 0.0566 | 0.1770 | 3/3 | 0.0565 |
| held-out | 6 | 0.0654 | 0.1482 | 6/6 | 0.0653 |
| worst held-out | | 0.112 (MoSe2_WSe2_2H) | | | 0.111 |

Per material, stage 1 and stage 2 agree to −0.001..+0.002 THz. That is expected: `H_model + FC_D3` vs `FC_ref`
and `H_model` vs `FC_ref - FC_D3` are the same difference in force-constant space, so once the D3 term is added
*exactly* (not learned) the two stages are one test. The staged plan therefore closes without further training.

Diagnostics: bare and PBE+D3 references differ by 0.007–0.018 THz RMSE (most for the six monolayers). In 23/48
bare references the ZA branch dips slightly imaginary within |q| ≤ 0.04 of Γ (worst MoSe2 −0.065 THz, WS2
−0.059, both hiphive-corrected references): `FC_D3` from plain 4x4x1 finite differences is not exactly
rotationally invariant, so subtracting it undoes a little of the rotational-sum-rule correction. This is an
artifact of the reference, not physics; the fine-tuned model does not reproduce it (min ≈ −0.001 THz there).
`reference_bare/<mat>.npz` (q-list, bare and PBE+D3 frequencies) is regenerable and gitignored.
