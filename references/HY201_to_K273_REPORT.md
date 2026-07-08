# HY201 to Kimodo273 Conversion Report

Status: full MotionFix HY201 -> Kimodo273 conversion and QA completed.

## Inputs and References

Guide:

```text
/mnt/afs/UMO_debug/outside_material/HY201_to_kimodo273_guide.md
/mnt/afs/UMO_debug/outside_material/HY201_to_kimodo273_guide (1).md
/mnt/afs/UMO_debug/outside_material/HY201_to_kimodo273_guide.pdf
```

Reference Kimodo repo:

```text
/mnt/afs/UMO_debug/outside_material/kimodo
commit: 6bb58488037dd65360ff0c5d1692b403a23309f7
upstream: https://github.com/nv-tlabs/kimodo
```

Source HY201 dataset:

```text
/mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272
/mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272
```

Converted Kimodo273 dataset:

```text
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22
/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22
```

This conversion repo:

```text
/mnt/afs/UMO_debug/hy201_to_kimodo273
remote target: https://github.com/CHDTevior/HY201_to_K273.git
```

## Representation Verdict

Kimodo273 is built with official `KimodoMotionRep(SMPLXSkeleton22, fps=30)`.
Feature slices:

```text
[0:3]     smooth_root_pos
[3:5]     global_root_heading = [cos(theta), sin(theta)]
[5:71]    local_joints_positions, 22x3
[71:203]  global_rot_data, 22x6
[203:269] velocities, 22x3
[269:273] foot_contacts, 4
```

The Kimodo `global_root_heading` is computed from hips:

```text
right_hip - left_hip -> atan2(diff_z, -diff_x)
```

It is not copied from HY root yaw.

HY201 and Kimodo SMPLX22 use the same first-22 joint order and parent topology:

```text
pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2,
left_ankle, right_ankle, spine3, left_foot, right_foot, neck,
left_collar, right_collar, head, left_shoulder, right_shoulder,
left_elbow, right_elbow, left_wrist, right_wrist
```

Parent list:

```text
[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
```

Coordinate convention:

```text
Y-up
XZ ground
+Z forward
```

This is explicit in the official Kimodo code/docs, not only inferred from the
paper: `kimodo/exports/smplx.py` describes Kimodo output as Y-up and +Z
forward, and `docs/source/key_concepts/constraints.md` says
`first_heading_angle=0` faces +Z. The converter also follows Kimodo's official
heading definition: `global_root_heading` is computed from the hip vector by
`KimodoMotionRep`, not copied from HY root yaw.

Small rest-pose nuance: with identity local rotations, Kimodo SMPLX22 FK gives a
hip-based heading near zero but not exactly zero (`~0.049 rad`, about 2.8
degrees) because the neutral hip offsets are not perfectly symmetric in the
heading formula. This does not change the coordinate convention; it just means
we should not use exact hip-derived rest heading as the sole proof of facing.

Rest-pose caveat: HY WoodenMesh and Kimodo SMPLX22 neutral joints are not
identical skeleton assets. They share order/topology and canonical orientation,
but limb proportions and neutral joint offsets differ. Therefore the converter
uses HY201 rotations/translations as source motion and lets official Kimodo
SMPLX22 FK regenerate Kimodo positions, velocities, heading, smooth root, and
contacts.

## Raw Archive vs Kimodo Training Canonicalization

The converted dataset in this report is raw Kimodo273:

```text
KimodoMotionRep(..., to_canonicalize=False)
```

This preserves source HY201 root translations and source initial headings, which
is useful for representation-level auditing and side-by-side visual checks.

There are two different design choices that should not be mixed:

```text
1. Frame-wise joint-position frame:
   Kimodo does not express every frame's joint positions in a root-heading-local
   frame. It keeps the position channels in the smooth-root/world mixed frame,
   avoiding discontinuities when heading flips quickly.

2. Sequence-level origin and initial heading:
   Kimodo uses a frame-0 canonical root XZ origin. The official canonicalize()
   helper rotates frame-0 heading to zero and translates frame-0 root XZ to
   zero. The model path can then use randomized or user-specified initial
   heading through first_heading_angle; 0 radians means facing +Z.
```

So the raw archive is a correct HY201 -> Kimodo273 representation conversion.
A strict Kimodo-style training dataset should be made as a separate
canonicalized/augmented variant or canonicalized inside the downstream
dataloader, with `first_heading_angle` handled explicitly.

## Critical 6D Detail

HY201 o6dp 6D layout is not Kimodo cont6d layout.

```text
HY identity:     [1, 0, 0, 1, 0, 0]
Kimodo identity: [1, 0, 0, 0, 1, 0]
```

Conversion path:

```text
HY201 [3:135] -> HY 6D decode -> local rotation matrices
local rotation matrices + HY201 [0:3] root translation
  -> official KimodoMotionRep
  -> raw Kimodo273 [T,273]
```

HY201 `[135:201]` auxiliary positions are not used to construct Kimodo273. They
are audited only as a reference, because they came from the previous 272/HY
conversion skeleton and are not Kimodo's official SMPLX22 rest skeleton.

## Full Conversion

Command:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/convert_motionfix_full.sh
```

Output:

```text
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22/manifest.jsonl
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22/conversion_summary.json
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22/stats/Mean.npy
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22/stats/Std.npy
```

Summary:

```text
files_converted: 13460
frames_converted: 1489149
train files: 10774
val files: 660
test files: 2026
shape per clip: [T,273]
fps: 30
```

Stats:

```text
Mean.npy shape: (273,), float32
Std.npy shape: (273,), float32
Std min/max: 0.0215856 / 0.6593291
```

Loader check:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
PYTHONPATH=. /root/miniconda3/envs/mogo/bin/python -m hy201_to_kimodo273.dataset \
  --root /mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22 \
  --split train \
  --index 0
```

Observed:

```text
clips: 10774
relative_path: train/000000_source.npy
motion_shape: [120, 273]
```

## Smoke Conversion

Command:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/convert_motionfix_smoke.sh
```

Output:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/manifest.jsonl
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/conversion_summary.json
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/stats/Mean.npy
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/stats/Std.npy
```

Converted files:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/val/000283_source.npy
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/val/000283_target.npy
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/val/000476_source.npy
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/val/000476_target.npy
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/val/000571_source.npy
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data/val/000571_target.npy
```

Summary:

```text
files: 6
frames: 721
shape per clip: [T,273]
```

## Full Numeric QA

Command:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/audit_motionfix_full.sh
```

Audit JSON:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_full/audit_summary.json
```

Key results:

```text
files_checked: 13460
frames_checked: 1489149
bad_shape: []
non_finite: []
global_min/max: -12.3158 / 13.9145
abs_max: 13.9145
heading_norm_abs_err_max: 4.5053e-08
global_rot_orthogonality_abs_err_max: 1.1102e-15
foot_contact_min/max: 0.0 / 1.0
foot_contact_mean: 0.8274
foot_contact_non_binary_frames: 0
inverse_root_abs_err_max: 0.0
inverse_source_root_abs_err_max: 3.7253e-09
inverse_local_rot_abs_err_max: 4.7684e-07
inverse_pos_rotation_vs_position_abs_err_max: 7.7486e-07
```

Range checks:

```text
smooth_root_min: [-3.9466, 0.0257, -3.7536]
smooth_root_max: [4.3059, 1.6694, 5.0215]
local_joints_pos_min: [-1.1575, -0.1108, -1.0992]
local_joints_pos_max: [1.1139, 2.4883, 1.1211]
velocity_min: [-10.1824, -12.3158, -10.2905]
velocity_max: [9.2693, 13.9145, 13.3450]
```

Interpretation: the full converted dataset has valid shapes and finite values,
binary contacts, normalized heading vectors, and orthogonal Kimodo 6D rotations.
Official Kimodo inverse recovers the source HY root translations and local
rotations to float precision on the audited inverse subset.

Auxiliary-position comparison:

```text
hy_aux_vs_kimodo_local_pos_l2_mean: 0.1247 m
hy_aux_vs_kimodo_local_pos_l2_p95: 0.4581 m
```

This remains an expected skeleton/rest-pose asset mismatch and is not treated
as a failure.

## Full Semantic QA

Command:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/semantic_audit_motionfix_full.sh
```

Audit JSON:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_full/semantic_audit_summary.json
```

What this verifies:

```text
official slice order == repo slice order
saved full feature == official KimodoMotionRep output
smooth_root y == source root y
heading == official hip-vector heading
local_joints_positions xz + smooth_root xz == global FK positions xz
local_joints_positions y == global FK positions y
global_rot_data == official Kimodo cont6d global rotations
velocities == official global finite-difference velocities
foot_contacts == official height/speed contact detector
```

Key results:

```text
files_checked: 13460
frames_checked: 1489149
bad_shape: []
non_finite: []
missing_source_files: []
slice_matches_official: true
saved_vs_official_full_feature_abs_err_max: 0.0
smooth_root_saved_vs_recomputed_abs_err_max: 0.0
smooth_root_y_vs_source_root_y_abs_err_max: 0.0
heading_saved_vs_hips_recomputed_abs_err_max: 0.0
heading_norm_abs_err_max: 4.5053e-08
local_joint_pos_saved_vs_recomputed_abs_err_max: 0.0
local_joint_pos_xz_plus_smooth_root_vs_global_pos_abs_err_max: 2.3842e-07
local_joint_pos_y_vs_global_y_abs_err_max: 0.0
global_rot6d_saved_vs_official_cont6d_abs_err_max: 0.0
global_rot_saved_vs_fk_global_rot_abs_err_max: 4.7718e-07
velocity_saved_vs_global_finite_diff_abs_err_max: 0.0
foot_contact_saved_vs_official_detector_abs_err_max: 0.0
foot_contact_non_binary_frames: 0
```

Semantic conclusion: every stored channel follows official Kimodo SMPLX22
semantics and coordinate convention. The archive is raw
`to_canonicalize=False`, Y-up, XZ-ground, +Z-forward reference, meters.

## Smoke Numeric QA

Command:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/audit_motionfix_smoke.sh
```

Audit JSON:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/audit_summary.json
```

Key results:

```text
bad_shape: []
non_finite: []
heading_norm_abs_err_max: 4.0233e-08
global_rot_orthogonality_abs_err_max: 8.8818e-16
foot_contact_min/max: 0.0 / 1.0
foot_contact_non_binary_frames: 0
inverse_root_abs_err_max: 0.0
inverse_source_root_abs_err_max: 3.7253e-09
inverse_local_rot_abs_err_max: 3.5763e-07
inverse_pos_rotation_vs_position_abs_err_max: 7.1526e-07
```

Interpretation: the current HY201 -> Kimodo273 conversion is internally
self-consistent and preserves source HY root translations and local rotations
to float precision.

Auxiliary-position comparison:

```text
hy_aux_vs_kimodo_local_pos_l2_mean: 0.1379 m
hy_aux_vs_kimodo_local_pos_l2_p95: 0.6206 m
```

This is not treated as a conversion failure. It reflects the skeleton/rest-pose
asset mismatch between the old HY auxiliary position source and official
Kimodo SMPLX22 FK.

## Visual QA

Rest-pose image:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/visual/rest_pose_hy201_vs_kimodo273.png
```

Motion comparison GIFs:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/visual/000283_source_hy_vs_kimodo.gif
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/visual/000476_source_hy_vs_kimodo.gif
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/visual/000571_target_hy_vs_kimodo.gif
```

Static contact sheet:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/visual/hy_vs_kimodo_contact_sheet.jpg
```

Visual rule:

```text
left panel:  HY201 [0:135] through HY WoodenMesh FK
right panel: Kimodo273 through official Kimodo inverse + SMPLX22 FK
ground: XZ plane at Y=0
red axis: +X
green axis: +Z
blue axis: +Y
black arrow: HY root +Z or Kimodo hips-based heading
```

First visual check shows upright motions, consistent Y-up/XZ-ground display,
and no obvious global axis flip. Kimodo skeleton proportions differ from HY
WoodenMesh as expected.

## Handoff Verdict

The current MotionFix HY201 -> Kimodo273 pipeline is ready for downstream use
or for packaging into a remote HY201-to-Kimodo repository. The MotionFix data
path to use is:

```text
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22
```

## HumanML3D Full Conversion

HumanML3D HY201 was converted with the same raw Kimodo273 contract and the same
code path:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/convert_humanml3d_full.sh
```

Input and output:

```text
source: /mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272
target: /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22
manifest: /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/manifest.jsonl
summary: /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/conversion_summary.json
stats: /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/stats/Mean.npy
stats: /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/stats/Std.npy
```

Conversion summary:

```text
files_converted: 26846
frames_converted: 5945004
split train/val/test: 21466 / 1338 / 4042
shape per clip: [T,273]
fps: 30
disk size: 6.2G
stats Mean.npy shape: (273,), float32
stats Std.npy shape: (273,), float32
Std min/max: 0.0277491 / 0.8954669
```

Four HumanML3D clips trigger a smooth-root fallback:

```text
motion_data/000990.npy
motion_data/005836.npy
motion_data/M000990.npy
motion_data/M005836.npy
```

These clips are 3-frame edge cases where Kimodo's official ADMM smooth-root
sparse solver raises `Factor is exactly singular`. For those clips only, root
XZ is left unsmoothed while root Y is preserved exactly. The fallback is
recorded per file in `manifest.jsonl` and summarized in
`conversion_summary.json`. The rest of the 273D feature construction still uses
official Kimodo FK, hip-vector heading, velocity, contact, and cont6d logic.

## HumanML3D QA

Dataset loader checks:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
PYTHONPATH=. /root/miniconda3/envs/mogo/bin/python -m hy201_to_kimodo273.dataset \
  --root /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22 \
  --split train --index 0
```

Observed split sizes:

```text
train: 21466
val: 1338
test: 4042
```

Full numeric audit:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/audit_humanml3d_full.sh
```

Audit output:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_humanml3d_full/audit_summary.json
```

Key numeric results:

```text
files_checked: 26846
frames_checked: 5945004
bad_shape: []
non_finite: []
global_min/max: -15.0565 / 20.0993
abs_max: 20.0993
heading_norm_abs_err_max: 4.5068e-08
global_rot_orthogonality_abs_err_max: 1.1102e-15
foot_contact_min/max: 0.0 / 1.0
foot_contact_non_binary_frames: 0
inverse_root_abs_err_max: 0.0
inverse_source_root_abs_err_max: 3.7253e-09
inverse_local_rot_abs_err_max: 4.7684e-07
inverse_pos_rotation_vs_position_abs_err_max: 9.5367e-07
```

Full semantic audit:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/semantic_audit_humanml3d_full.sh
```

Audit output:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_humanml3d_full/semantic_audit_summary.json
```

Key semantic results:

```text
files_checked: 26846
frames_checked: 5945004
bad_shape: []
non_finite: []
missing_source_files: []
slice_matches_official: true
saved_vs_recomputed_full_feature_abs_err_max: 0.0
smooth_root_saved_vs_recomputed_abs_err_max: 0.0
smooth_root_y_vs_source_root_y_abs_err_max: 0.0
heading_saved_vs_hips_recomputed_abs_err_max: 0.0
heading_norm_abs_err_max: 4.5068e-08
local_joint_pos_saved_vs_recomputed_abs_err_max: 0.0
local_joint_pos_xz_plus_smooth_root_vs_global_pos_abs_err_max: 4.7684e-07
local_joint_pos_y_vs_global_y_abs_err_max: 0.0
global_rot6d_saved_vs_official_cont6d_abs_err_max: 0.0
global_rot_saved_vs_fk_global_rot_abs_err_max: 5.0097e-07
velocity_saved_vs_global_finite_diff_abs_err_max: 0.0
foot_contact_saved_vs_official_detector_abs_err_max: 0.0
foot_contact_non_binary_frames: 0
smooth_root_fallback_files: 4
```

HumanML3D semantic conclusion: the converted archive follows the same raw
Kimodo273 representation as MotionFix: Y-up, XZ-ground, +Z-forward reference,
official SMPLX22 first-22 joints, hip-vector heading, and
`to_canonicalize=False`.
