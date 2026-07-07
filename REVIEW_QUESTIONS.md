# Review Questions

These are the remaining points that still need human confirmation before implementation.

## 1. Default Endpoint Preset

Recommendation: use `kimodo_ee` as the default evaluation/control preset, and keep `five_point` as a named config option.

```text
kimodo_ee:
  left_ankle(7), left_foot(10),
  right_ankle(8), right_foot(11),
  left_wrist(20), right_wrist(21)

five_point:
  head(15),
  left_wrist(20), right_wrist(21),
  left_foot(10), right_foot(11)
```

Reason: Kimodo's end-effector expansion for SMPLX22 maps foot controls to ankle+foot pairs, while the project phrase "five endpoints" may mean a separate application-level convention.

## 2. First Training Condition Mode

Recommendation: start with `condition_mode=null`, i.e. a null-text/unconditional raw motion prior plus synthetic control masks.

Reason: MotionFix instructions are source-to-target edit instructions, not target-motion captions. Feeding them directly as T2M captions would mix edit semantics into the base prior.

## 3. Kimodo-Style Sequence Transform Defaults

Recommendation: default on.

```text
root_origin_shift=true
random_first_heading=true
first_heading_angle/c_dir condition=true
```

Reason: Kimodo's constraint protocol is authored relative to a frame-0 canonical origin, and random first heading is passed as an explicit condition. This is a training-time transform only; the raw converted archive remains unchanged.

## 4. FK/Foot/Ground Loss Weights

Recommendation: implement the interfaces in the first version, then start with zero or small weights after shape/smoke checks.

Reason: foot skating and FK inconsistency are core failure modes, but the first smoke run should isolate data/model/sampler correctness before adding hard kinematic losses.

## 5. Postprocess Timing

Recommendation: baseline metrics should first be reported without postprocess; keep the postprocess interface from the start and enable it as a separate protocol.

Reason: exact anchors and low foot skate are system-level goals. The neural sampler, step-wise clamp, contact prediction, and optional contact-aware correction should be measured separately.
