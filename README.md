# 273 Motion Raw Diffusion Plan

This repository is an audit package for the HY273 raw-space diffusion/control harness plan.

Current scope: the first implementation/training pass should use the HumanML3D Kimodo273 conversion as the primary training dataset. MotionFix Kimodo273 remains in the package as a later edit/control dataset reference.

## Main Document

- `docs/HY273_raw_space_diffusion_confirm_plan.md`

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

## Data Not Included

This repository does not include raw motion `.npy` files, checkpoints, third-party repositories, or generated training outputs.
