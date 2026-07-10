# HY273 Loss and Architecture Experiment Tracker

| Run ID | Milestone | Purpose | Variant | Init | Steps | Seeds | Priority | Status |
|---|---|---|---|---|---:|---|---|---|
| R-ARCHIVE | M0 | freeze old baseline | 300K EMA ODE32 val16 + run manifest | 300K | 0 | 3407 | MUST | DONE |
| R-B0-CAL | B0 | scratch gradient calibration | alpha=0.3247346107, lambda=0.07 | scratch | 0 | 3407 | MUST | DONE |
| R-B0-TRACE | B0 | paired input comparability | rank-wise crop/caption/heading/t/noise/drop hashes | scratch | 100 | 3407 | MUST | DONE |
| R-B0-RESTART | B0 | exact checkpoint continuation | step-100 save, cursor, resume to 101 | scratch | 101 | 3407 | MUST | DONE |
| R-B0-REVIEW | B0 | adversarial implementation review | gpt-5.6-sol max | scratch | 0 | fixed | MUST | GO |
| R-M0-EVAL | B0 | finish evaluator before model selection | EMA/raw, FK skate, contact, motion, text bridge | n/a | 0 | fixed eval | MUST | IN_PROGRESS |
| R-L2 | M1 | selected semantic recipe | weighted-block MSE, GPU0-3, DDP4, accum2 | scratch | 137K observed / 100K durable | 3407 | MUST | STOPPED_BY_USER |
| R-L3 | M1 | consistency increment | L2 + FK consistency, GPU4-7, DDP4, accum2 | scratch | 200K | 3407 | MUST | RUNNING |
| R-L3-V-PREFLIGHT | M1.5 | validate clean-head JiT v-loss | source-bound real-cache DDP4, exact 500-step log/trace gate | scratch | 500 | 3407 | MUST | DONE |
| R-L3-V | M1.5 | JiT-recipe comparator | L3 + x0 head/v-space loss + JiT t distribution, GPU0-3 | scratch | 200K | 3407 | MUST | READY_TO_LAUNCH |
| R-S-MULTI | deferred | multi-seed loss confirmation | provisional winner | scratch | 50K | 3407/3408/3409 | DEFERRED | NOT_SCHEDULED |
| R-A1 | M2 | architecture baseline | current one-stage + confirmed loss | scratch | 50K screen | 3407 | MUST | BLOCKED_BY_M1.5 |
| R-A2 | M2 | root-body hypothesis | interleaved two-stage + confirmed loss | scratch | 50K screen | 3407 | MUST | BLOCKED_BY_M1.5 |
| R-A3 | M2 | parameter control | parameter-matched one-stage | scratch | 50K screen | 3407 | MUST | BLOCKED_BY_M1.5 |
| R-A4 | M2 | compute control | FLOP-matched one-stage | scratch | 50K screen | 3407 | MUST | BLOCKED_BY_M1.5 |
| R-A2-CONF | M2 | multi-seed architecture confirmation | A2 to 100K | scratch | 100K | 3407/3408/3409 | MUST | BLOCKED_BY_M2_SCREEN |
| R-A3-CONF | M2 | parameter-control confirmation | A3 to 100K | scratch | 100K | 3407/3408/3409 | MUST | BLOCKED_BY_M2_SCREEN |
| R-A4-CONF | M2 | compute-control confirmation | A4 to 100K | scratch | 100K | 3407/3408/3409 | MUST | BLOCKED_BY_M2_SCREEN |
| R-C2 | M3 | two-stage control pilot | A2 + mixed control | A2-100K | 50K | 3407/3408/3409 | MUST | BLOCKED_BY_M2 |
| R-C3 | M3 | parameter-matched control pilot | A3 + mixed control | A3-100K | 50K | 3407/3408/3409 | MUST | BLOCKED_BY_M2 |
| R-C4 | M3 | compute-matched control pilot | A4 + mixed control | A4-100K | 50K | 3407/3408/3409 | MUST | BLOCKED_BY_M2 |
| R-FINAL | M4 | final model | selected architecture/loss | scratch | 1M total | selected | MUST | BLOCKED_BY_M3 |

## Promotion Record

| Decision | Predeclared endpoint | Result | Decision |
|---|---|---|---|
| B0 training launch | scratch calibration, init SHA, paired trace, restart, frozen launch, adversarial review | all checks pass; gpt-5.6-sol max verdict GO | approved |
| B0 model selection | evaluator audits pass before first 50K comparison | pending | training may proceed; selection blocked |
| M1 result | independent L2/L3 scratch runs reach 200K and pass paired gates | pending | blocked |
| M1 -> M2 | user reviews 200K scratch results | pending | blocked |
| M2 -> M3 | A2, A3 and A4 complete matched three-seed 100K runs | pending | blocked |
| M3 -> M4 | pre-clamp control + physical + text gates pass | pending | blocked |

## B0 Evidence

- Machine-readable preflight: `run_logs/hy273_l2_l3_scratch_preflight_report.json` (`passed: true`).
- Frozen launch bytes: `run_logs/hy273_l2_l3_scratch_source_manifest.sha256`; the launcher requires the report to bind this exact manifest.
- Scratch calibration: `run_logs/hy273_l2_l3_calibration_scratch_seed3407_n16_final.json`.
- Initial rank-0 model SHA: `808a1639f7ec134887ce07b3d2634849dccfe68e6441bac0addba4f4cc579e59`.
- L2 and L3 use separate random-initialized runs; neither resumes the archived 300K model.
- L2 allocation is GPU 0-3; L3 allocation is GPU 4-7. Both use DDP4, per-rank batch 16, accumulation 2, global batch 128.
- Each run targets 200K optimizer steps and saves at 50K, 100K, 150K, and 200K while atomically updating `latest.pt`.
- Production pair: `hy273_l2_scratch200k_ddp4_20260710_103509` and `hy273_l3_scratch200k_ddp4_20260710_103509`.
- Production first-100-step audit: both initial SHA values match calibration; all four paired rank traces are byte-identical.
- L3 clean-head JiT v-loss calibration: `run_logs/hy273_l3_vloss_jitpm08ps08_calibration_train_t_seed3407_n4096.json`.
- L3 clean-head JiT v-loss fixed-bin audit: `run_logs/hy273_l3_vloss_jitpm08ps08_calibration_bins_seed3407_n16_alpha0939702.json`.
- L3 clean-head JiT v-loss pilot gate: `run_logs/hy273_l3_vloss_jit_preflight_report.json` (`passed: true`, source manifest `0b5bce3116b0eb2e4b89ab78223b853e27473c631c78e036d7c80fed23a8a0e5`).
- Bound pilot: `hy273_l3_vloss_jit_bound_pilot500_v2`; all 500 train rows and four 100-row rank traces passed, with 0/500 clipping hits and flow mean decreasing from `0.20676698` (first 50) to `0.02048116` (last 50).
- Final adversarial launch review: `logs/codex_gpt56sol_max_l3_vloss_pilot_logging_r3.txt` (`gpt-5.6-sol`, `max`; GO with no P0/P1).
