# Model-side interlayer landscape (registry x gap)

`scripts/model_registry_landscape.py` on the configurations of `workflow/scripts/registry_gap_scan.py` (4x4 registry grid x
dz {-0.3, 0, +0.3} A + breathing curve, 53 per bilayer; same tags as the DFT run dirs). `v5_all/` = v5 cellrelax baresub-e20
(bare, `ft_*`), the D3 term (`d3_*`), their sum = the deployed model (`ftd3_*`), omat-1 zero-shot (`base_*`); job 3227925,
2026-10-26. `results.csv` one row per configuration, `summary.csv` one per bilayer.

## First reading (2026-10-26, before any DFT scan data)

Cross-checked against `phonon_comparisons/v5_tmd_cellrelax_gamma_interlayer_d3/` (Γ shear / breathing of the same model on the
48 references, overlap-identified, job 3227926): the equilibrium shear curvature `ftd3_Kshear_eq` reproduces K = μ(2πν)² from the
Γ shear frequency to ≈1 % in every bilayer, so the two tools agree.

* **The model's shear stiffness is nearly the same for every bilayer.** K_shear(ft+D3) spans 0.11–0.19 eV/Å² across the 42
  bilayers while the DFT references span 0.05–0.68 (corr 0.36). Γ shear: model 0.43–0.69 THz for DFT 0.34–1.07; mean error
  −0.17 THz, RMS 0.24 (base −0.20 / 0.27). The fine-tune moved K_shear only slightly from omat-1 and left the stacking-fault
  corrugation untouched (ft+D3 vs base within a few meV in all 42) — expected, since no training label depends on registry.
* Breathing is where the fine-tune helped: mean error +0.06 THz, RMS 0.15 (base +0.26 / 0.31); the model's breathing-curve
  minimum sits within +0.00…+0.04 Å of the DFT gap at the symmetric registry (`ftd3_dz_min_A`).
* So the twisted-cell defect (imaginary / too-stiff phason) is the untwisted-cell defect seen through a moiré: one generic
  sliding stiffness instead of a material- and stacking-dependent one. The registry scan labels exactly that quantity.
* Caveat: the DFT shear values are the hiphive / raw production-INCAR references (±0.3 THz on the Γ pair for some cells),
  but the model's compressed range is far outside that uncertainty.
