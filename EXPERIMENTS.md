# Experiments manifest

One row per Phonon Fine-Tuning (PFT) experiment run in this fork. Everything an experiment
produces lives in one of five fixed places; this table is the map between them.

| Place | What | Naming |
|---|---|---|
| `configs/` | training config | `nequix-healthy-2d-<exp>.yml` |
| `logs/` | SLURM stdout for train + eval jobs | `<job-name>-<jobid>.out` (older: `slurm-<jobid>.out`) |
| `checkpoints/` | deployable fine-tuned model (tracked) + resumable state (`.pkl`, ignored) | `nequix-healthy-2d-<exp>.nqx` |
| `wandb/` | raw offline run (ignored, random id); `files/checkpoint.nqx` is the original of the `checkpoints/` copy | `offline-run-<timestamp>-<id>` |
| `phonon_comparisons/` | eval outputs: `results.csv` (+ `plots/`) | `<split>_<evaltype>[_<variant>]` |

Training data for every experiment is a copy of a split built in the sibling repo:
`data/healthy_2d*/` here ↔ `../workflow/nequix_datasets/<split>/` (see crosswalk below).
Whole-band evals read the DFT reference straight from `../workflow/FINAL_RESULTS_HEALTHY/`.

## Experiments

Whole-band RMSE numbers are THz vs DFT over the full q-path; "base" = the pretrained model
before fine-tuning. Dates point at the daily log in `../logs/YYYY-MM-DD.md`.

| Experiment | Config `configs/nequix-healthy-2d-…` | Split (`data/` ↔ `../workflow/nequix_datasets/`) | Train job → `logs/` | Checkpoint `checkpoints/nequix-healthy-2d-…` | Eval job(s) → `phonon_comparisons/` | Headline |
|---|---|---|---|---|---|---|
| **v1** healthy-2D pilot | `-pft` | `healthy_2d` ↔ `v1_46materials` (41/5 random) | 2787865 → `slurm-2787865.out` | `-pft.nqx` (wandb `…lfqpdt1u`) | spot-checks → `v1_healthy2d_spotchecks/`, `ood_screening/` | Fits train set, but tied/lost on 5/5 OOD materials (catastrophic forgetting). 2026-07-30, 08-03 |
| **v2** chemistry generalization | `-pft-v2` | `healthy_2d_v2` ↔ `v2_73healthy_split` (47/6/20 holdout) | 2879630 → `slurm-2879630.out` | `-pft-v2.nqx` (`…nta6efhy`) | 2879707 → `v2_eval/` (26 val+holdout); 2948071 → `v2_73healthy_split_gamma_interlayer/`; 2948109 → `…_gamma_interlayer_traincheck/` | Held-out 20: 1.00 vs base 1.66, 16/20 wins; only pure GaSe/InSe regress. 2026-08-18 |
| v2 seed-1 rerun (full 73 eval) | `-pft-v2-seed1` | same as v2 | 2953567 | `-pft-v2-seed1.nqx` (`…ylv09dp2`) | 2953609 → `v2_73healthy_split_eval_seed1/` (all 73); 2953612 → `…_gamma_interlayer_seed1/` | Source of the 48-TMD publication numbers: all 48 pure Mo/W TMDs ≤ 0.134, mean 0.075. 2026-09-01, 09-05, 09-07 |
| v2 MoTe2 holdout | `-pft-v2-mote2` | `healthy_2d_v2_mote2` ↔ `v2_mote2_holdout` | 2958453 | `-pft-v2-mote2.nqx` (`…p0134oms`) | 2958767 → `v2_mote2_holdout_eval/` | Novel-compound test: 13/13 wins, 0.185 mean. 2026-09-02 |
| v2 Ti holdout | `-pft-v2-ti` | `healthy_2d_v2_ti_holdout` ↔ `v2_ti_holdout` | 2962197 | `-pft-v2-ti.nqx` (`…qd11tjyc`) | 2962247 → `v2_ti_holdout_eval/` | Novel-element test: 6/6 wins, zero regressions. 2026-09-03 |
| v2 from scratch | `-fromscratch-v2` | same as v2 | 2950486 | `-fromscratch-v2.nqx` (`…3khq9tb0`) | 2954261 → `v2_fromscratch_eval/`; 2954360 → `finetune_vs_fromscratch/` | Pretraining ablation; see 2026-09-01 |
| **v3** stacking generalization (2H→3R) | `-pft-v3` | `healthy_2d_v3` ↔ `v3_2h3r` (40/10/23 test = all `_3R`) | 2904232 → `slurm-2904232.out` | `-pft-v3.nqx` (`…hz81xqem`) | 2904708 → `v3_2h3r_eval/` (33); 2913029 → `v3_2h3r_gamma_interlayer/`; 2913152 → `…/zoom_plots/`; 2948108 → `…_gamma_interlayer_traincheck/` | 33/33 whole-band wins; 6/29 regress on Γ-interlayer modes. 2026-08-22, 08-23 |
| v3 seed 7 / seed 123 | `-pft-v3-seed7`, `-pft-v3-seed123` | `healthy_2d_v3_seed7`, `…_seed123` ↔ `v3_2h3r_seed7`, `…_seed123` | 2953565 (s7), 2953566 (s123) | `-pft-v3-seed7.nqx` (`…1t33qobv`), `-pft-v3-seed123.nqx` (`…5ukwx7dv`) | 2953607 → `v3_2h3r_eval_seed7/`; 2953608 → `v3_2h3r_eval_seed123/`; 2953610/11 → `…_gamma_interlayer_seed7/`, `…_seed123/` | Split-variance ablation. 2026-09-01 |
| v3 data scaling 50 % / 75 % | `-pft-v3-50pct`, `-pft-v3-75pct` | `healthy_2d_v3_scaling/train_{50,75}pct` ↔ `v3_2h3r_scaling` | 2948169 (50), 2948170 (75) | `-pft-v3-50pct.nqx` (`…2yabi7d8`), `-pft-v3-75pct.nqx` (`…bi3udlzf`) | 2948174 → `v3_2h3r_gamma_interlayer_50pct/`; 2948175 → `…_75pct/` (Γ-interlayer only, no whole-band eval) | Flat: 0.239 / 0.240 / 0.236 at 20/30/40 train materials; refutes data-scarcity hypothesis. 2026-08-28, 08-30 |
| v3 relative-Hessian loss | `-pft-v3-relhessian` | same as v3 | 2949564 | `-pft-v3-relhessian.nqx` (`…3lzh0s4j`) | 2949579 → `v3_2h3r_eval_relhessian/` (csv only); 2949568 → `v3_2h3r_gamma_interlayer_relhessian/` | Worse on Γ-interlayer (regression margin +90 % → +153 %). 2026-08-30 |
| v3 from scratch | `-fromscratch-v3` | same as v3 | 2950487 | `-fromscratch-v3.nqx` (`…ywv7l7xu`) | 2954262 → `v3_fromscratch_eval/`; 2954359 → `finetune_vs_fromscratch/` | Pretraining ablation; see 2026-09-01 |
| **v4** TMD-only retrain | `-pft-v4-tmd` | `healthy_2d_v4_tmd_only` ↔ `v4_tmd_only` (38/4/6 holdout) | 2979522 → `slurm-2979522.out` | `-pft-v4-tmd.nqx` (`…cue4k50o`) | 2979535 → `v4_tmd_only_eval/` (48) | Held-out 6 mean 0.090, worst MoSe2_WSe2_2H 0.135 — no tightening vs v2 seed-1. 2026-09-07 |
| v4 base = nequix-oam-1 | `-pft-v4-tmd-oam` | same as v4 | 3011412 → `slurm-3011412.out` | `-pft-v4-tmd-oam.nqx` (`…qglgzpe3`) | 3011636 → `v4_tmd_only_oam_eval/` (baseline `nequix-oam-1`) | Mean 0.054 (mp-1 ft 0.072), held-out 6 mean 0.065, worst 0.110; better than mp-1 ft on 44/48, ~tied with omat ft. Un-fine-tuned oam-1 baseline 0.80 mean. 2026-09-15 |
| v4 base = nequix-omat-1 | `-pft-v4-tmd-omat` | same as v4 | 3011413 → `slurm-3011413.out` | `-pft-v4-tmd-omat.nqx` (`…k6mjzlcb`) | 3011637 → `v4_tmd_only_omat_eval/` (baseline `nequix-omat-1`); **re-run 3017374 on the corrected Γ–K–M–Γ path (K = 1/3,1/3,0), same dir** | Mean 0.051 (mp-1 ft 0.072), held-out 6 mean 0.065, worst 0.107; better than mp-1 ft on 45/48. Un-fine-tuned omat-1 baseline already 0.155 mean vs mp-1 1.13. 2026-09-15. **True-K re-run 2026-09-17:** ft mean 0.052 (Δ +0.001), held-out 6 0.066, worst MoSe2_WSe2_2H 0.108, 48/48 wins; omat-1 zero-shot 0.168 (Δ +0.013, held-out 0.149). Max per-material ft shift 0.008 THz (MoTe2_WTe2_3R). The old-path numbers above stand within noise. |
| **v4 explicit-D3 (delta learning)** omat base | `-pft-v4-tmd-omat-bared3` | `explicit_dispersion/data/v4_tmd_only_bare_d3` (= `v4_tmd_only` with D3 stripped) | 3012446 → `nequix_pft_v4_tmd_omat_bared3-3012446.out` | `-pft-v4-tmd-omat-bared3.nqx` | 3013019 (afterok:3012446) → `v4_tmd_only_omat_bared3_eval/` (ft + D3 vs implicit-D3 omat ft); **re-run 3017375 on the corrected path, same dir** | **Null result.** All-48 mean 0.051 vs 0.051 THz (23/48 wins), held-out 6 0.064 vs 0.065, worst held-out MoSe2_WSe2_2H 0.110 vs 0.107; val Hessian MAE 0.0040 vs 0.0044. Explicit D3 neither helps nor hurts. See `explicit_dispersion/README.md`. 2026-09-15/16. **True-K re-run 2026-09-17:** 0.053 vs 0.052 (23/48 wins), held-out 6 0.065 vs 0.066, worst 0.111 vs 0.108 — null result unchanged. |
| zero-shot omat-1 + explicit D3 | — (no training) | eval on `v4_tmd_only` | — | `models/nequix-omat-1.nqx` | 3012447 → `v4_tmd_only_omat_zeroshot_plusd3_eval/` (omat-1 + D3 vs omat-1) | Negligible: all-48 mean 0.1535 vs 0.1548 THz (46/48 wins but ~1% gain), held-out 6 0.128 vs 0.129. omat-1's residual error is not missing dispersion. 2026-09-15/16 |
| **Stage-1 bare-PBE check** (bared3 ft without D3 vs a bare-PBE reference) | — (eval only, `-pft-v4-tmd-omat-bared3`) | eval on `v4_tmd_only`; reference = `FC_ref − FC_D3` (cached `explicit_dispersion/d3_cache/`) on the band.yaml q-list = bare-PBE curvature at the PBE+D3-relaxed geometry, **not** a bare-PBE minimum (residual \|F_D3\| 0.10–0.17 eV/Å) | — | `-pft-v4-tmd-omat-bared3.nqx` | 3019406 → `v4_tmd_only_omat_bared3_stage1_bare_eval/` (`eval_explicit_dispersion.py --add-d3 none --reference bare`, baseline = zero-shot `nequix-omat-1`; `reference_bare/*.npz` regenerable, ignored) | PI's staged plan: verify the bare-PBE model first, then add D3 (stage 2 = the delta-learning row above, already a null result). **DONE 2026-09-18 (job 3019406, 41 min).** Bare model vs bare reference: all-48 mean 0.0527 THz (train 39 0.050, val 3 0.057, held-out 6 0.065, worst MoSe2_WSe2_2H 0.112); beats zero-shot omat-1 (0.167) 48/48. Per material within −0.001..+0.002 THz of the stage-2 number (same model + `FC_D3` vs PBE+D3 reference, 0.0525): the exact additive D3 add-back is a pass-through, so stages 1 and 2 are the same test in force-constant space — the staged plan closes with no further training. Bare vs PBE+D3 references differ by 0.007–0.018 THz RMSE (largest for the six monolayers). Artifact: 23/48 bare references dip slightly imaginary on the ZA branch within \|q\| ≤ 0.04 of Γ (worst MoSe2 −0.065, WS2 −0.059, both hiphive-corrected references) because `FC_D3` from plain 4x4x1 finite differences is not exactly rotationally invariant; the model does not reproduce these dips (ft ≈ −0.001 there). |
| **v4-omat end-to-end, template start, relax positions** | — (eval only, `-pft-v4-tmd-omat`) | eval on `v4_tmd_only`; start = `workflow/template_structures/` (pre-relaxation template, overrides-first) | — | `-pft-v4-tmd-omat.nqx` | 3015438 → `v4_tmd_only_omat_template_relaxpos_eval/` (`--relax positions`; both models relax with their own forces; also `rmse_ft_template_unrelaxed`) | **Yes, 0.051 survives.** All-48 mean 0.052 (fixed-DFT 0.051), held-out 6 0.066 (0.065), worst held-out MoSe2_WSe2_2H 0.110; ft beats omat-1 48/48. omat-1 degrades 0.155 → 0.199 and shows imaginary modes (<−0.05 THz) on 10/48; ft on 0/48. All 96 relaxations converged (fmax 1e-3), ~11 BFGS steps for ft. Unrelaxed-template ft RMSE 0.32, so the relaxation step is doing real work. 2026-09-17 |
| **v4-omat end-to-end, template start, relax positions + in-plane cell** | — (eval only) | same | — | same | 3015439 → `v4_tmd_only_omat_template_relaxcell_eval/` (`--relax cell`, `FrechetCellFilter` xx/yy/xy, c fixed) | Modest degradation: all-48 mean 0.067, held-out 6 0.081, worst MoS2_bilayer_3R 0.165 (held-out worst MoSe2_WSe2_2H 0.141); ft still 48/48 vs omat-1 (0.243). ft `a` drift centred on zero (mean −0.01 %, range −0.8..+1.0 %, 18/48 beyond ±0.5 %); omat-1 drifts +0.64 % mean (up to +1.7 %, 32/48 beyond ±0.5 %). Imaginary modes: ft 0/48, omat-1 8/48. Caveat: templates carry DFT-derived override `a` (ISIF=4) so this measures drift *from* DFT, not a clean raw-MP start (candidate follow-up). 2026-09-17 |
| **Twisted MoS2 pilot (model-only, no DFT)** | — (eval only, `-pft-v4-tmd-omat`; `sbatch/eval_twisted_phonons.sbatch`) | `workflow/twisted/structures/MoS2_twist_m{1..5}_{near0,near60}` (commensurate (m,m+1), θ = 21.8/13.2/9.4/7.3/6.0°, 42–546 atoms; single a = 3.192, 3R override not applied) | — | `-pft-v4-tmd-omat.nqx` | 3017227 → `twisted_mos2_near0/`, 3017228 → `twisted_mos2_near60/` (`scripts/eval_twisted_phonons.py`: relax positions, Hessian on diag(k,k,1) k = 3,2,2,1,1, ASR, Γ-mode following) | **DONE 2026-09-17, 10/10 dynamically stable** (n_imag < -0.05 THz = 0 everywhere; residual min freq -0.015..0 THz is Γ-acoustic noise, max 13.87-13.88 THz). Runtime 38/31 min. Γ mode-following triggered on 3/10 (m1_near0 3 rounds → slide 0.57 Å; m4_near0 1 round → 0.51 Å; m2_near60 2 rounds → 0.42 Å), others relaxed straight to a minimum. Interlayer shear pair at Γ is 0.02-0.08 THz for every twist (near-free sliding, expected for moiré phasons); lowest Γ breathing/optical 0.86/0.88/0.92/0.75/0.49 THz (near0 m=1..5). Mean gap shrinks toward the untwisted value as θ→0: near0 3.125→2.894 Å, near60 3.133→2.842 Å (untwisted 3R/2H start 2.899/2.938). In-plane stress +0.43-0.45 GPa everywhere (positions-only relax at fixed a = 3.192). Smoke (3017179/3017207/3017210): pipeline reproduces the 1x1 route (M to 1e-5 THz); m1_near0 sits on a symmetry-protected sliding saddle (−0.17 THz shear pair) → mode following slides layer 2 by 0.57 Å (−1.7 meV/cell); m1_near60 clean; m3_near0 shear pair ≈ 0. 888-atom Hessian 11 min. 2026-09-17 |
| **v4 bare-PBE-subtract + real stress** omat base | `-pft-v4-tmd-omat-baresub` | `explicit_dispersion/data/v4_tmd_only_bare_subtract` (= `v4_tmd_only` split; Hessians from `workflow/FINAL_RESULTS_BARE_PBE_SUBTRACT` = FORCE_SETS-level D3 subtraction, eq-force removed, hiphive refit 25/48; **stress = real bare unit-cell stress** σ_OUTCAR − σ_D3 instead of the stress = 0 / −σ_D3 placeholder of every earlier run; **forces = −F_D3(eq)** (true bare force at the PBE+D3 geometry, 0.10–0.17 eV/Å; a first build with forces = 0 was cancelled as 3115178/3115179 — Gemini's "artificial zero force" point was right); energy = bare E_uc × n_sc/n_uc at weight 0) | 3115218 → `nequix_pft_v4_tmd_omat_baresub-3115218.out` | `-pft-v4-tmd-omat-baresub.nqx` | 3115219 (afterok:3115218) → `v4_tmd_only_omat_baresub_eval/` (`eval_explicit_dispersion.sbatch`, ft + FC_D3 vs implicit-D3 omat ft, PBE+D3 reference, fixed geometry) | Motivation: (a) train on the canonical bare-PBE dataset (D3 subtraction, 2026-10-06) rather than the FC-level strip; (b) first run with true DFT stress labels — in-plane bare σ is −0.2…−2.3 GPa (D3 alone contributes ≈ +2.1 GPa on bilayers), so stress_weight 5 was previously pulling toward a false zero. Fixed-geometry eval cannot show the stress effect; a template-start `--relax cell` eval with D3 added back is the follow-up (needs a D3 hook in `eval_healthy2d_finetune.py`). Submitted 2026-10-07, PENDING. |
| v4 bare-subtract, energy_weight 20 | `-pft-v4-tmd-omat-baresub-e20` | same data as the row above | 3115394 → `nequix_pft_v4_tmd_omat_baresub_e20-3115394.out` | `-pft-v4-tmd-omat-baresub-e20.nqx` | 3115395 (afterok:3115394) → `v4_tmd_only_omat_baresub_e20_eval/` (same eval as above) | Restores the upstream PFT default energy_weight 20 (ours has been 0 since July because rows had no energy label). Precondition checked 2026-10-07: omat-1 zero-shot energies are within +0.007…+0.024 eV/atom of our bare labels on 6 materials, so the frozen pretrained energy shift is compatible. Also the protocol shipped to the coworker in `workflow/share/tmd48_bare_pbe_subtract_v2` (README "Training protocol"). Submitted 2026-10-07, PENDING. |
| MACE-MP zero-shot baseline | — (`sbatch/eval_mace_baseline.sbatch`) | evaluated on all 73 | — | — (foundation model) | 2954482 → `mace_mp_baseline/` | Mean 1.52 / median 1.24 vs nequix-mp-1's 1.35 / 1.07: comparably weak. 2026-09-01 |
| MACE forces-only fine-tune | — (CLI flags in `sbatch/train_mace_finetune.sbatch`) | `mace_baseline/data/v4_tmd_only_forces/` (from `v4_tmd_only`) | 2980394 → `slurm-2980394.out` | `mace_baseline/models/mace_finetune_v4_tmd_only.model` (ignored) | 2980457 → `mace_finetune_v4_tmd_only/` | Held-out 6: 1.35 → 0.87 vs Nequix PFT 0.09; no Hessian loss available. 2026-09-07 |

Eval scripts: whole-band = `scripts/eval_healthy2d_finetune.py`; Γ-interlayer =
`scripts/eval_gamma_interlayer_modes.py`; MACE = `scripts/eval_mace_baseline.py`;
plots = `scripts/plot_finetune_vs_fromscratch.py`, `scripts/plot_interlayer_zoom.py`.

## Naming crosswalk

The same split is referred to by three names depending on where you are:

| Config / checkpoint suffix | `data/` dir | `../workflow/nequix_datasets/` | eval dir prefix |
|---|---|---|---|
| `-pft` (v1) | `healthy_2d` | `v1_46materials` | `ood_screening`, `v1_healthy2d_spotchecks` |
| `-pft-v2*`, `-fromscratch-v2` | `healthy_2d_v2` | `v2_73healthy_split` | `v2_eval`, `v2_73healthy_split_*`, `v2_fromscratch_eval` |
| `-pft-v2-mote2` | `healthy_2d_v2_mote2` | `v2_mote2_holdout` | `v2_mote2_holdout_eval` |
| `-pft-v2-ti` | `healthy_2d_v2_ti_holdout` | `v2_ti_holdout` | `v2_ti_holdout_eval` |
| `-pft-v3*`, `-fromscratch-v3` | `healthy_2d_v3` | `v3_2h3r` | `v3_2h3r_*` |
| `-pft-v3-seed7/-seed123` | `healthy_2d_v3_seed7/_seed123` | `v3_2h3r_seed7/_seed123` | `v3_2h3r_*_seed7/_seed123` |
| `-pft-v3-50pct/-75pct` | `healthy_2d_v3_scaling` | `v3_2h3r_scaling` | `v3_2h3r_gamma_interlayer_50pct/_75pct` |
| `-pft-v4-tmd*` | `healthy_2d_v4_tmd_only` | `v4_tmd_only` | `v4_tmd_only_*` |
| `-pft-v4-tmd-omat-bared3` | `explicit_dispersion/data/v4_tmd_only_bare_d3` | `v4_tmd_only` (targets minus D3) | `v4_tmd_only_omat_bared3_*`, `v4_tmd_only_omat_zeroshot_plusd3_*` |

`v2_eval/` (26 val+holdout materials, original run) and `v2_73healthy_split_eval_seed1/`
(all 73, seed-1 rerun) are the same experiment family; the latter is the one the
48-TMD publication numbers come from.

## Conventions for new experiments

- **Band path (fixed 2026-09-17):** DFT `band.yaml` q-points live in phonopy's 120° standardized primitive cell, where
  K = (1/3, 1/3, 0). Until 2026-09-17 every `band.conf` and `compare_phonon_dispersion.HEX_BAND_LINE` used the 60°-cell
  value (2/3, 1/3, 0), so all `phonon_comparisons/*` plots and RMSEs dated before then are on a Γ–(2/3 Γ→M')–M–Γ path
  with a mislabelled "K" tick. DFT-vs-model RMSEs were computed on identical q-lists, so they remain valid as
  comparisons but are not on the true Γ–K–M–Γ path. `workflow/FINAL_RESULTS*/*/band.yaml` have been regenerated on the
  correct path; evals re-run after this date pick the new q-list up automatically. Only the two omat TMD evals
  (`v4_tmd_only_omat_eval`, `v4_tmd_only_omat_bared3_eval`) were re-run (jobs 3017374/3017375; user decision: v2/v3 retired, TMD +
  omat only) — fine-tuned means shifted ≤0.0013 THz; every other `phonon_comparisons/*` dir is still old-path. Use
  `hex_band_line_for_cell(ph.primitive.cell)` for any new q-path (twisted 60° cells keep (2/3, 1/3, 0)).

**Geometry mode of an eval.** Unless the eval dir name says `template_relax*`, every phonon eval is
**fixed-geometry**: the model Hessian is evaluated at the DFT-relaxed `FINAL_RESULTS_HEALTHY` structure
(a Hessian-accuracy benchmark, not a deployment test). `template_relaxpos` / `template_relaxcell` evals
start from `workflow/template_structures/<mat>/POSCAR` (the workflow's pre-relaxation template) and let
each model relax positions (or positions + in-plane cell) before the Hessian — say which kind a number is.

0. **Default base model is `nequix-omat-1`** (`finetune_from: models/nequix-omat-1.nqx`), decided
   2026-09-15 after the base-model ablation above. Everything before that date used `nequix-mp-1`;
   only compare a new run against an mp-1 run if that is the explicit point of the experiment.
1. Config: `configs/nequix-healthy-2d-<exp>.yml`, with `state_path: checkpoints/nequix-healthy-2d-<exp>.pkl`.
   Training then writes both the state pickle and `checkpoints/nequix-healthy-2d-<exp>.nqx`.
2. Submit with an explicit job name so the log is self-describing:
   `sbatch --job-name=<exp> sbatch/<script>.sbatch <args>` → `logs/<exp>-<jobid>.out`.
3. Eval into an explicit, experiment-specific directory:
   `sbatch --job-name=<exp>_eval sbatch/eval_healthy2d_finetune.sbatch checkpoints/<exp>.nqx <split> [baseline, default nequix-omat-1] phonon_comparisons/<split>_<evaltype>_<variant>`.
   The eval script's default output dir is `<split>_eval`, which would overwrite an earlier run
   on the same split.
4. Add a row above when you submit, fill in the headline when results land, and note the day in
   `../logs/YYYY-MM-DD.md`.
