# 273 Motion Raw Diffusion Plan

This repository is an audit package for the HY273 raw-space diffusion/control harness plan.

Current scope: the first implementation/training pass should use the HumanML3D Kimodo273 conversion as the primary training dataset. The primary generative objective is raw-space rectified flow / flow matching with ODE sampling, matching the current MoGeFlow CodeFlow backbone. MotionFix Kimodo273 remains in the package as a later edit/control dataset reference.

## Main Document

- `docs/HY273_raw_space_diffusion_confirm_plan.md`
- `docs/HY273_hytext_cache_integration_review.md`
- `refine-logs/EXPERIMENT_PLAN.md` (current L2/L3 scratch experiment protocol)
- `refine-logs/EXPERIMENT_TRACKER.md` (live experiment status)

This is the current implementation plan after checking:

- HY201 -> Kimodo273 converted data and semantic audit.
- HumanML3D Kimodo273 split/text coverage.
- MotionFix pair manifests and converted clip coverage.
- Kimodo raw representation, canonicalization, condition construction, endpoint/contact semantics.
- Existing MoGeFlow/codeflow DiT and training harness pieces that can be reused.
- The original raw-space training/control implementation note.

## Review Questions

- `REVIEW_QUESTIONS.md`

These are the remaining design choices that need human review before implementation.

## Evidence

- `evidence/humanml3d_audit_summary.json`
- `evidence/humanml3d_semantic_audit_summary.json`
- `evidence/humanml3d_conversion_summary.json`
- `evidence/semantic_audit_summary.json` (MotionFix, retained for reference)
- `evidence/conversion_summary.json` (MotionFix, retained for reference)
- `evidence/verification_summary.md`

The JSON files are copied from the checked local data conversion/evaluation outputs. They are small summaries only, not the dataset.

## References

- `references/HY273_raw_space_diffusion_training_control_impl.md`
- `references/HY273_raw_space_diffusion_harness_plan.md`
- `references/HY201_to_K273_README.md`
- `references/HY201_to_K273_REPORT.md`
- `references/code_inventory.md`

The reference docs capture the previous plan material and the source-code line references used by the confirmation plan.

## Implementation Snapshot

This repository now includes the current HY273 raw-flow implementation snapshot needed to review the HYText-cache change:

- `models/raw_motion/`
- `models/codeflow/dit_blocks.py`
- `data/kimodo273_datasets.py`
- `train_hy273_raw_flow.py`
- `sample_hy273_raw.py`
- `configs/raw_flow_hy273.yaml`
- `configs/raw_flow_hy273_hytext.yaml`
- `tools/cache_hy273_hytext_embeddings.py`
- `tools/check_hy273_hytext_cache_coverage.py`
- `scripts/launch/train_hy273_raw_flow_stage1_x0_hytext_ddp8.sh`
- `scripts/launch/train_hy273_raw_flow_stage2_x0_control_ddp8.sh`
- `configs/raw_flow_hy273_hytext_l2_l3_scratch.yaml`
- `scripts/launch/train_hy273_raw_flow_l2_l3_scratch_ddp4.sh`
- `tools/calibrate_hy273_l2_l3_losses.py`
- `tools/verify_hy273_l2_l3_preflight.py`
- `tests/test_hy273_constraints.py`
- `tests/test_kimodo273_dataset.py`
- `tests/test_raw_flow_model.py`
- `tests/test_raw_flow_sampling.py`

## L2/L3 Scratch Runs

The current loss experiment consists of two independent runs from random initialization:

- L2: semantic block-weighted clean-x0 MSE on four GPUs.
- L3: the same objective plus meter-space FK/position consistency with a 5K-step warmup, on a separate four GPUs.

Both runs target 200K optimizer steps with global batch 128 and checkpoints at 50K intervals. They share the same rank-0 initialization SHA and deterministic per-rank input trace; neither resumes the archived 300K model.

Launch evidence is versioned in:

- `run_logs/hy273_l2_l3_calibration_scratch_seed3407_n16_final.json`
- `run_logs/hy273_l2_l3_scratch_preflight_report.json`
- `run_logs/hy273_l2_l3_scratch_source_manifest.sha256`

The source manifest intentionally contains absolute hashes for the local audited dataset/cache metadata. Recreate those environment-specific entries before launching on a different machine.

The reviewer prompts for the HYText and Stage-2 control patches are in:

- `docs/HY273_hytext_cache_integration_review.md`
- `docs/HY273_stage2_kimodo_control_review.md`

## Data Not Included

This repository does not include raw motion `.npy` files, checkpoints, complete third-party repositories, or generated training outputs. It includes only the small hash-locked SMPL-X22 skeleton asset required to audit FK semantics.
