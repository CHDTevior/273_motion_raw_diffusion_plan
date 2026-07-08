# Code Inventory

This file records the source files and line references used to build the confirmation plan.

## HY201 to Kimodo273 Conversion Repo

```text
local:
  /mnt/afs/UMO_debug/hy201_to_kimodo273

remote:
  https://github.com/CHDTevior/HY201_to_K273.git

commit:
  ea668b7
```

Key files:

```text
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:22
  DIM_KIMODO273 = 273

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:31
  Kimodo273 slices

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:38
  SMPLX22 joint order

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:102
  HY201 6D decode

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:123
  Kimodo cont6d decode

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:159
  split_kimodo273

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:57
  split_existing/splits split id reader

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:75
  split filtering supports HumanML3D motion_data + split_existing

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:88
  Kimodo273MotionDataset

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:139
  __getitem__ loads [T,273]

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:144
  normalization

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:147
  contacts restored raw 0/1 unless normalize_contacts=True

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:169
  collate_kimodo273_batch

/mnt/afs/UMO_debug/hy201_to_kimodo273/README.md:259
  HumanML3D full conversion summary

/mnt/afs/UMO_debug/hy201_to_kimodo273/README.md:323
  HumanML3D numeric QA result

/mnt/afs/UMO_debug/hy201_to_kimodo273/README.md:339
  HumanML3D semantic audit result

/mnt/afs/UMO_debug/hy201_to_kimodo273/REPORT.md:468
  HumanML3D full conversion section

/mnt/afs/UMO_debug/hy201_to_kimodo273/REPORT.md:519
  HumanML3D QA section
```

## Kimodo

```text
local:
  /mnt/afs/mogeflow-control/external_repos/kimodo

remote:
  https://github.com/nv-tlabs/kimodo.git

commit:
  6bb5848
```

Key files:

```text
external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:34
  official size_dict

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:76
  FK -> global joints/rotations

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:85
  smooth_root_pos

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:88
  local_joints_positions = pelvis-local joints + hips_offset

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:91
  foot contact detector

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:113
  rotate feature sequence

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:147
  translate smooth_root xz only

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:195
  inverse adds smooth_root xz back to joints_pos

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:222
  create_conditions

external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:284
  global joint constraints require smooth_root reference

external_repos/kimodo/kimodo/motion_rep/reps/base.py:165
  rotate_to first heading

external_repos/kimodo/kimodo/motion_rep/reps/base.py:192
  randomize_first_heading

external_repos/kimodo/kimodo/motion_rep/reps/base.py:222
  translate_2d_to_zero

external_repos/kimodo/kimodo/motion_rep/reps/base.py:232
  canonicalize = rotate_to_zero + translate_2d_to_zero

external_repos/kimodo/docs/source/user_guide/constraints.md:30
  training root starts at frame-0 XZ origin

external_repos/kimodo/docs/source/user_guide/constraints.md:31
  initial heading randomly rotated and passed to model

external_repos/kimodo/kimodo/skeleton/definitions.py:342
  right foot = right_ankle, right_foot

external_repos/kimodo/kimodo/skeleton/definitions.py:343
  left foot = left_ankle, left_foot

external_repos/kimodo/kimodo/skeleton/definitions.py:344
  right hand = right_wrist

external_repos/kimodo/kimodo/skeleton/definitions.py:345
  left hand = left_wrist

external_repos/kimodo/kimodo/skeleton/base.py:135
  expand_joint_names

external_repos/kimodo/kimodo/motion_rep/feet.py:35
  contact order starts from left foot joints

external_repos/kimodo/kimodo/motion_rep/feet.py:59
  contacts = [left_ankle, left_foot, right_ankle, right_foot]
```

## Current MoGeFlow-Control Workspace

```text
local:
  /mnt/afs/mogeflow-control
```

Key files:

```text
models/codeflow/dit_blocks.py:607
  FrameMotionTextDiT

models/codeflow/dit_blocks.py:648
  existing control encoder Conv1d stride-2 x2

models/codeflow/dit_blocks.py:680
  _encode_control length check

models/codeflow/dit_blocks.py:708
  DiT forward

train_codeflow.py:2613
  build dataset + DistributedSampler pattern

train_codeflow.py:2689
  model init

train_codeflow.py:2730
  AdamW

train_codeflow.py:2737
  GradScaler

train_codeflow.py:2740
  resume checkpoint

train_codeflow.py:2748
  DDP wrapping

train_codeflow.py:2861
  AMP autocast

train_codeflow.py:2889
  finite-loss guard

train_codeflow.py:2916
  grad clipping

train_codeflow.py:2955
  save latest checkpoint

train_codeflow.py:574
  prepare_flow_training_state

train_codeflow.py:600
  z_t = t*x0 + (1-t)*noise

train_codeflow.py:611
  clean estimate from predicted velocity

models/codeflow/continuous_motion_code_flow.py:60
  z_t = t*x0 + (1-t)*noise

models/codeflow/continuous_motion_code_flow.py:61
  velocity_target = x0 - noise

models/codeflow/continuous_motion_code_flow.py:95
  flow_loss on predicted velocity

models/codeflow/continuous_motion_code_flow.py:98
  clean_pred from velocity

models/codeflow/motion_code_flow.py:707
  ODE sampling grid

models/codeflow/motion_code_flow.py:760
  guided velocity and clean estimate during sampling

models/codeflow/motion_code_flow.py:761
  z = z + dt*v

models/codeflow/motion_code_flow.py:764
  self-conditioning uses clean estimate
```

## Raw-Space Plan Reference

```text
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:15
  HY273 is the raw-space diffusion / flow variable

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:124
  x0 / observed_motion / motion_mask shapes

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:209
  Kimodo-style imputation

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:237
  global joint target requires same-frame smooth_root_ref

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:513
  old conservative baseline said DDPM clean x0 first; current confirmation plan supersedes this with flow-matching first

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:993
  old note placed Flow Matching second; current confirmation plan moves Flow Matching / ODE to first implementation

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1007
  Phase 1 no-control natural prior

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1026
  Phase 2 realistic control sampler patterns

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1197
  old DDPM/DDIM step-wise clamp maps to ODE step-wise clamp in the current plan

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1291
  postprocess is part of the system

outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1420
  implementation traps
```
