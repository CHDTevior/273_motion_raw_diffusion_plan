# Verification Summary

Date: 2026-07-08, Asia/Shanghai.

## Converted Data

```text
converted root:
  /mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22

source HY201 root:
  /mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272

conversion repo:
  /mnt/afs/UMO_debug/hy201_to_kimodo273
  https://github.com/CHDTevior/HY201_to_K273.git
  commit b004e82
```

Semantic audit summary:

```text
files_checked: 13460
frames_checked: 1489149
slice_matches_official: true
saved_vs_official_full_feature_abs_err_max: 0.0
smooth_root_y_vs_source_root_y_abs_err_max: 0.0
heading_saved_vs_hips_recomputed_abs_err_max: 0.0
local_joint_pos_xz_plus_smooth_root_vs_global_pos_abs_err_max: 2.384185791015625e-07
local_joint_pos_y_vs_global_y_abs_err_max: 0.0
global_rot6d_saved_vs_official_cont6d_abs_err_max: 0.0
velocity_saved_vs_global_finite_diff_abs_err_max: 0.0
foot_contact_saved_vs_official_detector_abs_err_max: 0.0
```

Split counts:

```text
converted clip files:
  train: 10774
  val:    660
  test:  2026
  total: 13460

MotionFix paired records:
  train: 5387 pairs
  val:    330 pairs
  test:  1013 pairs
  total: 6730 pairs
```

Full pair manifests checked:

```text
/mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionstreamer272_hml_{train,val,test}.jsonl
/mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionfix207_{train,val,test}.jsonl
```

Both manifest families cover all 13460 converted source/target clip paths after stripping the source feature prefix.

The older HY201 subdirectory manifest was checked and is not the full pair index:

```text
/mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272/manifests/motionfix_hy201_{split}.jsonl
total: 6312 pairs
```

## Loader Check

Command used the `mogo` conda environment with:

```text
PYTHONPATH=/mnt/afs/UMO_debug/hy201_to_kimodo273:$PYTHONPATH
/root/miniconda3/envs/mogo/bin/python
```

Observed:

```text
dataset train len 10774 first_shape (120, 273) first_rel train/000000_source.npy contact_unique [0.0, 1.0]
dataset val len 660 first_shape (120, 273) first_rel val/000283_source.npy contact_unique [0.0, 1.0]
dataset test len 2026 first_shape (120, 273) first_rel test/000004_source.npy contact_unique [0.0, 1.0]
batch_motion (3, 120, 273)
batch_mask (3, 120)
lengths [120, 120, 120]
stats_shape (273,) (273,)
std_minmax 0.021585574373602867 0.6593290567398071
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
