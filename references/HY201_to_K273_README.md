# hy201_to_kimodo273

Convert HY-Motion 201D `o6dp` clips to Kimodo SMPLX22 273D features.

This repo is the working conversion/QA repo for:

```text
local repo:
/mnt/afs/UMO_debug/hy201_to_kimodo273

remote repo:
https://github.com/CHDTevior/HY201_to_K273.git

source HY201 data:
/mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272
/mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272

reference Kimodo repo:
/mnt/afs/UMO_debug/outside_material/kimodo
https://github.com/nv-tlabs/kimodo

target smoke output:
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data

target full MotionFix output:
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22

target full HumanML3D output:
/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22
```

## Representation Contract

HY201 input:

```text
[0:3]     root translation, world XYZ, Y-up
[3:9]     root local rotation, HY interleaved-column 6D
[9:135]   21 body local rotations, HY interleaved-column 6D
[135:201] auxiliary 22x3 pelvis-relative joint positions
```

Kimodo273 output, official `KimodoMotionRep(SMPLXSkeleton22)` order:

```text
[0:3]     smooth_root_pos
[3:5]     global_root_heading = [cos(theta), sin(theta)], hips-based
[5:71]    local_joints_positions, 22x3
[71:203]  global_rot_data, 22x6 Kimodo cont6d
[203:269] velocities, 22x3, fps-scaled
[269:273] foot_contacts, [left_ankle,left_foot,right_ankle,right_foot]
```

Important: HY201 and Kimodo both use Y-up / XZ-ground / +Z-forward semantics,
and their first 22 joint names and parent topology match. Kimodo's official
docs/code explicitly describe generated motions as Y-up and +Z forward;
`first_heading_angle=0` means facing +Z. The converter follows official Kimodo
semantics for heading too: `global_root_heading` is computed from the hip vector
by `KimodoMotionRep`, not copied from HY root yaw.

The rest skeleton assets are not byte-identical: HY visualization uses
WoodenMesh, Kimodo uses its official SMPLX22 neutral joints. Therefore visual
QA should compare motion/axis/facing behavior, not require every limb endpoint
to overlay exactly.

## 6D Rule

Do not copy HY201 rotation channels into Kimodo channels.

HY identity 6D:

```text
[1, 0, 0, 1, 0, 0]
```

Kimodo identity cont6d:

```text
[1, 0, 0, 0, 1, 0]
```

The converter decodes HY201 local rotations to matrices and calls official
Kimodo code to produce the 273D feature tensor.

## Raw vs Canonical Features

This repo currently writes raw Kimodo273 features:

```text
KimodoMotionRep(..., to_canonicalize=False)
```

That preserves the HY201 source root translation and source initial heading.
It is the right target for auditing the representation conversion itself.

Do not conflate two different "canonicalization" choices:

```text
frame-wise heading-local joint positions:
  Kimodo does not convert every frame's joints into a root-heading-local
  coordinate frame. Joint positions stay in Kimodo's smooth-root/world mixed
  frame. This avoids discontinuities when the body heading flips rapidly.

sequence-level origin/initial-heading handling:
  Kimodo uses a frame-0 canonical origin for root XZ. The official
  canonicalize() helper rotates frame-0 heading to zero and translates frame-0
  root XZ to zero. Training/inference can then use randomized or user-specified
  initial heading through first_heading_angle, where 0 radians means +Z.
```

If the next step is to train a Kimodo-style generator, produce a canonicalized
variant or canonicalize in the downstream dataloader, and explicitly decide
whether to randomize frame-0 heading and pass `first_heading_angle` as a model
condition. Do not assume this raw archive already has Kimodo's training-time
origin/heading preprocessing.

## Smoke Conversion

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/convert_motionfix_smoke.sh
```

Defaults:

```text
input:  /mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272/val/*.npy
output: /mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_smoke/data
files:  first 6 validation clips
```

## Numeric QA

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/audit_motionfix_smoke.sh
```

This checks shape, finite values, heading norm, Kimodo cont6d orthogonality,
binary contacts, official inverse consistency, and optional source-HY rotation
roundtrip.

## Semantic QA

For channel-by-channel coordinate/semantic verification:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/semantic_audit_motionfix_smoke.sh
```

Full dataset:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/semantic_audit_motionfix_full.sh
```

Full semantic audit output:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_full/semantic_audit_summary.json
```

This audit re-computes every channel from source HY201 using official Kimodo
code and checks:

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

Current full semantic audit passed:

```text
files_checked: 13460
frames_checked: 1489149
slice_matches_official: true
saved_vs_official_full_feature_abs_err_max: 0.0
smooth_root_y_vs_source_root_y_abs_err_max: 0.0
heading_saved_vs_hips_recomputed_abs_err_max: 0.0
local_joint_pos_xz_plus_smooth_root_vs_global_pos_abs_err_max: 2.3842e-07
local_joint_pos_y_vs_global_y_abs_err_max: 0.0
global_rot6d_saved_vs_official_cont6d_abs_err_max: 0.0
velocity_saved_vs_global_finite_diff_abs_err_max: 0.0
foot_contact_saved_vs_official_detector_abs_err_max: 0.0
```

## Visual QA

Rest pose:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/render_rest_pose.sh
```

Motion compare GIF:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/render_motionfix_sample.sh
```

The GIF uses each side's own rules:

```text
left:  HY201 [0:135] -> HY WoodenMesh FK
right: Kimodo273 -> official Kimodo inverse + SMPLX22 FK
ground: XZ plane at Y=0
red axis: +X
green axis: +Z
blue axis: +Y
black arrow: HY root local +Z or Kimodo hips-based heading
```

## Full Conversion

The MotionFix HY201 dataset has already been fully converted here:

```text
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22
```

Reproduce it with:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/convert_motionfix_full.sh
```

The converter writes:

```text
manifest.jsonl
conversion_summary.json
stats/Mean.npy
stats/Std.npy
```

Current full conversion summary:

```text
source: /mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272
target: /mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22
files: 13460
frames: 1489149
splits: train=10774, val=660, test=2026
stats: /mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22/stats/{Mean.npy,Std.npy}
```

HumanML3D HY201 has also been fully converted to the same raw Kimodo273
contract:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/convert_humanml3d_full.sh
```

Current HumanML3D conversion summary:

```text
source: /mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272
target: /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22
files: 26846
frames: 5945004
splits: train=21466, val=1338, test=4042
stats: /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/stats/{Mean.npy,Std.npy}
smooth_root_fallback_files: 4
```

The four HumanML3D fallback clips are 3-frame edge cases where Kimodo's
official smooth-root sparse solver returns `Factor is exactly singular`:

```text
motion_data/000990.npy
motion_data/005836.npy
motion_data/M000990.npy
motion_data/M005836.npy
```

For these clips only, the converter keeps root XZ unsmoothed and preserves root
Y exactly. All other Kimodo273 channels still use official Kimodo FK, heading,
velocity, contact, and cont6d logic, and the fallback is recorded in
`manifest.jsonl` and `conversion_summary.json`.

Full numeric audit:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/audit_motionfix_full.sh
bash scripts/audit_humanml3d_full.sh
```

Full audit output:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_motionfix_full/audit_summary.json
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_humanml3d_full/audit_summary.json
```

Key full-QA result:

```text
bad_shape: []
non_finite: []
heading_norm_abs_err_max: 4.5053e-08
global_rot_orthogonality_abs_err_max: 1.1102e-15
foot_contact_min/max: 0.0 / 1.0
foot_contact_non_binary_frames: 0
inverse_source_root_abs_err_max: 3.7253e-09
inverse_local_rot_abs_err_max: 4.7684e-07
inverse_pos_rotation_vs_position_abs_err_max: 7.7486e-07
```

HumanML3D full-QA result:

```text
files_checked: 26846
frames_checked: 5945004
bad_shape: []
non_finite: []
heading_norm_abs_err_max: 4.5068e-08
global_rot_orthogonality_abs_err_max: 1.1102e-15
foot_contact_min/max: 0.0 / 1.0
foot_contact_non_binary_frames: 0
inverse_source_root_abs_err_max: 3.7253e-09
inverse_local_rot_abs_err_max: 4.7684e-07
inverse_pos_rotation_vs_position_abs_err_max: 9.5367e-07
```

HumanML3D full semantic audit:

```bash
cd /mnt/afs/UMO_debug/hy201_to_kimodo273
bash scripts/semantic_audit_humanml3d_full.sh
```

Output:

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_humanml3d_full/semantic_audit_summary.json
```

Key semantic result:

```text
files_checked: 26846
frames_checked: 5945004
slice_matches_official: true
saved_vs_recomputed_full_feature_abs_err_max: 0.0
smooth_root_saved_vs_recomputed_abs_err_max: 0.0
smooth_root_y_vs_source_root_y_abs_err_max: 0.0
heading_saved_vs_hips_recomputed_abs_err_max: 0.0
local_joint_pos_saved_vs_recomputed_abs_err_max: 0.0
local_joint_pos_xz_plus_smooth_root_vs_global_pos_abs_err_max: 4.7684e-07
local_joint_pos_y_vs_global_y_abs_err_max: 0.0
global_rot6d_saved_vs_official_cont6d_abs_err_max: 0.0
velocity_saved_vs_global_finite_diff_abs_err_max: 0.0
foot_contact_saved_vs_official_detector_abs_err_max: 0.0
smooth_root_fallback_files: 4
```

## Dataset Loader

```python
from hy201_to_kimodo273 import Kimodo273MotionDataset

dataset = Kimodo273MotionDataset(
    "/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22",
    split="train",
    normalize=False,
)
item = dataset[0]
motion = item["motion"]  # [T, 273], float32
```

Use the same loader for HumanML3D:

```python
dataset = Kimodo273MotionDataset(
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22",
    split="train",
    normalize=False,
)
```

For PyTorch:

```python
from torch.utils.data import DataLoader
from hy201_to_kimodo273 import Kimodo273TorchDataset, collate_kimodo273_batch

dataset = Kimodo273TorchDataset(
    "/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22",
    split="train",
)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_kimodo273_batch)
```

If `normalize=True`, the loader keeps `foot_contacts` as raw 0/1 by default,
matching the training recommendation in the guide. Set
`normalize_contacts=True` only if your downstream model explicitly expects all
273 channels z-scored.
