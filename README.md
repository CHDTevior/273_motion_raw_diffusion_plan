# 273 Motion Raw Diffusion Plan

This repository is an audit package for the HY273 raw-space diffusion/control harness plan.

## Main Document

- `docs/HY273_raw_space_diffusion_confirm_plan.md`

This is the current implementation plan after checking:

- HY201 -> Kimodo273 converted data and semantic audit.
- MotionFix pair manifests and converted clip coverage.
- Kimodo raw representation, canonicalization, condition construction, endpoint/contact semantics.
- Existing MoGeFlow/codeflow DiT and training harness pieces that can be reused.
- The original raw-space training/control implementation note.

## Review Questions

- `REVIEW_QUESTIONS.md`

These are the remaining design choices that need human review before implementation.

## Evidence

- `evidence/semantic_audit_summary.json`
- `evidence/conversion_summary.json`
- `evidence/verification_summary.md`

The JSON files are copied from the checked local data conversion/evaluation outputs. They are small summaries only, not the dataset.

## References

- `references/HY273_raw_space_diffusion_training_control_impl.md`
- `references/HY273_raw_space_diffusion_harness_plan.md`
- `references/code_inventory.md`

The reference docs capture the previous plan material and the source-code line references used by the confirmation plan.

## Data Not Included

This repository does not include raw motion `.npy` files, checkpoints, third-party repositories, or generated training outputs.
