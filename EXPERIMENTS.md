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
| v4 base = nequix-omat-1 | `-pft-v4-tmd-omat` | same as v4 | 3011413 → `slurm-3011413.out` | `-pft-v4-tmd-omat.nqx` (`…k6mjzlcb`) | 3011637 → `v4_tmd_only_omat_eval/` (baseline `nequix-omat-1`) | Mean 0.051 (mp-1 ft 0.072), held-out 6 mean 0.065, worst 0.107; better than mp-1 ft on 45/48. Un-fine-tuned omat-1 baseline already 0.155 mean vs mp-1 1.13. 2026-09-15 |
| **v4 explicit-D3 (delta learning)** omat base | `-pft-v4-tmd-omat-bared3` | `explicit_dispersion/data/v4_tmd_only_bare_d3` (= `v4_tmd_only` with D3 stripped) | 3012446 → `nequix_pft_v4_tmd_omat_bared3-3012446.out` | `-pft-v4-tmd-omat-bared3.nqx` | 3013019 (afterok:3012446) → `v4_tmd_only_omat_bared3_eval/` (ft + D3 vs implicit-D3 omat ft) | pending. See `explicit_dispersion/README.md`. 2026-09-15 |
| zero-shot omat-1 + explicit D3 | — (no training) | eval on `v4_tmd_only` | — | `models/nequix-omat-1.nqx` | 3012447 → `v4_tmd_only_omat_zeroshot_plusd3_eval/` (omat-1 + D3 vs omat-1) | pending. 2026-09-15 |
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
