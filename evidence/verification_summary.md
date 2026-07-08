# Verification Summary

Date: 2026-07-08, Asia/Shanghai.

## Primary Converted Data: HumanML3D K273

```text
converted root:
  /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22

source HY201 root:
  /mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272

captions:
  /mnt/afs/mogo_base/datasets/HumanML3D/texts

conversion repo:
  /mnt/afs/UMO_debug/hy201_to_kimodo273
  https://github.com/CHDTevior/HY201_to_K273.git
  commit ea668b7
```

HumanML3D semantic audit summary:

```text
files_checked: 26846
frames_checked: 5945004
slice_matches_official: true
saved_vs_recomputed_full_feature_abs_err_max: 0.0
saved_vs_official_full_feature_abs_err_max: 0.0
smooth_root_saved_vs_recomputed_abs_err_max: 0.0
smooth_root_y_vs_source_root_y_abs_err_max: 0.0
heading_saved_vs_hips_recomputed_abs_err_max: 0.0
local_joint_pos_saved_vs_recomputed_abs_err_max: 0.0
local_joint_pos_xz_plus_smooth_root_vs_global_pos_abs_err_max: 4.76837158203125e-07
local_joint_pos_y_vs_global_y_abs_err_max: 0.0
global_rot6d_saved_vs_official_cont6d_abs_err_max: 0.0
velocity_saved_vs_global_finite_diff_abs_err_max: 0.0
foot_contact_saved_vs_official_detector_abs_err_max: 0.0
foot_contact_non_binary_frames: 0
bad_shape: []
non_finite: []
smooth_root_fallback_files: 4
```

Split counts:

```text
converted clip files:
  train: 21466
  val:    1338
  test:   4042
  total: 26846

frames_converted: 5945004
stats_shape: (273,) (273,)
std_minmax: 0.027749110013246536 / 0.8954669237136841
```

Four 3-frame clips use smooth-root XZ fallback because the official Kimodo smooth-root solver raised `Factor is exactly singular`:

```text
motion_data/000990.npy
motion_data/005836.npy
motion_data/M000990.npy
motion_data/M005836.npy
```

Only smooth-root XZ uses fallback for those files. Root Y and the other Kimodo273 channels still use official FK, heading, velocity, contact, and cont6d logic.

## Secondary Converted Data: MotionFix K273

```text
converted root:
  /mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22

converted clip files:
  train: 10774
  val:    660
  test:  2026
  total: 13460

paired records:
  train: 5387 pairs
  val:    330 pairs
  test:  1013 pairs
  total: 6730 pairs

full pair manifests:
  /mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionstreamer272_hml_{train,val,test}.jsonl
  /mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionfix207_{train,val,test}.jsonl
```

## Loader Check

Command used the `mogo` conda environment with:

```text
PYTHONPATH=/mnt/afs/UMO_debug/hy201_to_kimodo273:$PYTHONPATH
/root/miniconda3/envs/mogo/bin/python
```

Observed:

```text
dataset train len 21466 first_shape (124, 273) first_rel motion_data/000002.npy contact_unique [0.0, 1.0]
dataset val len 1338 first_shape (204, 273) first_rel motion_data/000016.npy contact_unique [1.0]
dataset test len 4042 first_shape (175, 273) first_rel motion_data/000000.npy contact_unique [0.0, 1.0]
batch_motion (4, 300, 273)
batch_mask (4, 300)
lengths [124, 136, 300, 154]
```

Caption coverage:

```text
caption_files_total 26846
missing 0
min_lines 1
max_lines 4
mean_lines 2.988378156894882
split train ids 21466 missing 0
split val ids 1338 missing 0
split test ids 4042 missing 0
```

## Environment

```text
use python:
  /root/miniconda3/envs/mogo/bin/python

use torchrun:
  /root/miniconda3/envs/mogo/bin/torchrun

torch:
  2.5.1+cu124

CUDA:
  available
  8 x NVIDIA A100-SXM4-80GB observed
```

Default `/root/miniconda3/bin/python` is Python 3.13 and is not the intended training runtime for this harness.

## Model Interface Check

`FrameMotionTextDiT` plain forward was checked in the `mogo` environment:

```text
motion [2,16,64], text [2,4,64] -> out [2,16,64], finite=True
```

Existing `control_cond` adapter behavior:

```text
control frame length 16 -> error, encoded length 4 != motion length 16
control frame length 64 -> out [2,16,64], finite=True
```

Reason: the existing adapter uses two Conv1d layers with stride 2, so the control frame length is downsampled by 4. The raw-space first version should pass control through `[x_imp, mask]` input projection instead of using that adapter unchanged.

## Known Import Pitfall

Package-level `models.codeflow` import currently pulls evaluation modules. Under NumPy 2.x in the `mogo` environment, this triggers an old `np.float` usage through `common/quaternion.py`.

The raw harness should either avoid package-level initialization for the reused DiT module or fix the compatibility issue directly.
