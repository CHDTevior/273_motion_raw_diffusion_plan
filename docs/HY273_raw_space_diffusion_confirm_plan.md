# HY273 Raw-Space Flow Harness 确认版计划

本文是实施前确认文档。它吸收了已经完成的 HY201 -> Kimodo273 数据转换、semantic audit 结论，以及“不做错误 canonicalize”的边界。等你确认后，我再按这里的顺序开始写代码。

绝对路径：

```text
/mnt/afs/mogeflow-control/outside_doc/HY273_raw_space_diffusion_confirm_plan.md
```

## 0. 我准备做什么

第一版目标：

```text
raw Kimodo273 / HY273 motion prior
  + same-space observed_motion/motion_mask 控制训练
  + rectified-flow / flow-matching velocity prediction
  + ODE32 clamp-each-step 控制采样
  + root / five-endpoint / fullpose / mixed 控制评估
```

明确不做：

```text
不走 VQ / RVQ / code-space
不训练旧 codeflow terminal logits
不把 HY201 rotation 6D 直接当 Kimodo cont6d
不对 joints_pos 做逐帧 heading canonicalize
不覆盖或改写 raw archive
不把 MotionFix edit instruction 直接误当作 target motion caption
```

模型第一版：

```text
HY273RawFlow = current FrameMotionTextDiT-style flow denoiser

raw model input:  [B,T,546] = concat(imputed noisy HY273, control mask)
raw model output: [B,T,273] = flow velocity v = x0 - noise
clean estimate:   x0_hat = z_t + (1 - t) * v_pred
```

## 1. 已确认数据

转换 repo：

```text
local repo: /mnt/afs/UMO_debug/hy201_to_kimodo273
remote:     https://github.com/CHDTevior/HY201_to_K273.git
commit:     ea668b7
```

第一版训练主数据：

```text
HumanML3D converted Kimodo273:
  /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22

source HY201:
  /mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272

captions:
  /mnt/afs/mogo_base/datasets/HumanML3D/texts

splits:
  /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/split_existing/{train,val,test}.txt
```

HumanML3D numeric audit：

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_humanml3d_full/audit_summary.json
```

HumanML3D semantic audit：

```text
/mnt/afs/UMO_debug/eval_runs/hy201_to_kimodo273_humanml3d_full/semantic_audit_summary.json
```

我已核对的 HumanML3D 关键结果：

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

HumanML3D 数据统计：

```text
converted clip files:
  train: 21466
  val:    1338
  test:   4042
  total: 26846

frames_converted: 5945004
shape per clip: [T,273]
stats:
  stats/Mean.npy
  stats/Std.npy
caption coverage:
  total clips: 26846
  missing caption files: 0
```

HumanML3D 有 4 个 3-frame 极短 clip 触发 Kimodo 官方 smooth-root solver 的 `Factor is exactly singular`。转换器只对这 4 个 clip 的 smooth-root XZ 使用 raw-root fallback，Y 保持原值；其它 channel 仍走官方 Kimodo FK / heading / velocity / contact / cont6d 逻辑，semantic audit 全量通过。

```text
motion_data/000990.npy
motion_data/005836.npy
motion_data/M000990.npy
motion_data/M005836.npy
```

MotionFix K273 仍保留为后续 MotionFix edit/control 数据和协议参考：

```text
MotionFix converted Kimodo273:
  /mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22

MotionFix source HY201:
  /mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272

MotionFix converted clip files:
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

MotionFix 全量 pair manifest 是 MotionFix 根目录下这两套之一：

```text
/mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionstreamer272_hml_{train,val,test}.jsonl
/mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionfix207_{train,val,test}.jsonl
```

这两套都覆盖全部 13460 个 converted Kimodo273 path。HY201 子目录里的
`hymotion201_o6dp_hml272/manifests/motionfix_hy201_{split}.jsonl` 只有
6312 pairs，不覆盖全量 converted data，第一版 pair mode 不用它做全量索引。

## 2. 数据表示契约

HY273 slice：

```text
[0:3]     smooth_root_pos
[3:5]     global_root_heading = [cos(theta), sin(theta)]
[5:71]    local_joints_positions, 22x3
[71:203]  global_rot_data, 22x6
[203:269] velocities, 22x3
[269:273] foot_contacts, 4
```

坐标系：

```text
Y-up
XZ-ground
+Z forward reference
raw archive: to_canonicalize=False
fps: 30
```

SMPLX22 joint order：

```text
0  pelvis
1  left_hip
2  right_hip
3  spine1
4  left_knee
5  right_knee
6  spine2
7  left_ankle
8  right_ankle
9  spine3
10 left_foot
11 right_foot
12 neck
13 left_collar
14 right_collar
15 head
16 left_shoulder
17 right_shoulder
18 left_elbow
19 right_elbow
20 left_wrist
21 right_wrist
```

parents：

```text
[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
```

endpoint preset 先做成配置，不硬编码。

Kimodo EE preset：

```text
LeftFoot   -> left_ankle(7),  left_foot(10)
RightFoot  -> right_ankle(8), right_foot(11)
LeftHand   -> left_wrist(20)
RightHand  -> right_wrist(21)
```

项目 five-point preset 候选：

```text
head:        15
left_wrist:  20
right_wrist: 21
left_foot:   10
right_foot:  11
```

foot contact 通道顺序按 Kimodo detector：

```text
[left_ankle, left_foot, right_ankle, right_foot]
```

## 3. Canonicalize 边界

这里是最容易写错的地方，我会按下面执行。

### 3.1 不做逐帧 heading-local joint position

不会做：

```text
for every frame:
  heading = hips/root heading
  joints_pos = rotate_y(-heading) * joints_pos
```

会保持 Kimodo 的做法：

```text
joints_pos.xz = global_joint_pos.xz - smooth_root_pos.xz
joints_pos.y  = global_joint_pos.y
```

原因：逐帧 heading canonicalize 在空翻、侧手翻、快速翻身时可能让 heading 瞬间翻 180 度，导致 joints_pos 数值不连续。现在转换后的数据已经按 Kimodo 官方表示对齐，不应该再引入这一步。

### 3.2 不把 raw archive 覆盖成 canonical 版本

raw archive 保持：

```text
to_canonicalize=False
```

训练中如果需要增强，只在 dataloader/transform 里临时做，并且输出仍遵守 HY273 slice 语义。

### 3.3 第一版训练 transform

我建议第一版默认做 Kimodo-style sequence transform，但只在 dataloader/训练图里临时做，不改 raw archive：

```text
frame-wise heading-local joints_pos = OFF
root XZ origin shift to frame-0 origin = ON
random first heading augmentation = ON
first_heading_angle/c_dir condition = ON
contacts remain raw 0/1
```

Kimodo 文档里的训练约定是：

```text
frame 0 smooth_root_pos.xz at (0, 0)
initial heading randomly rotated
the chosen initial heading is passed as explicit condition
inference constraints are authored relative to that canonical origin
```

我这里会实现成一个可逆 transform：

```text
1. root_origin_shift:
   delta = -smooth_root_pos[0].xz
   smooth_root_pos.xz += delta
   joints_pos unchanged
   velocities unchanged
   rotations unchanged
   contacts unchanged

2. random_first_heading:
   target_angle ~ Uniform(-pi, pi)
   current_angle = atan2(global_root_heading[0,1], global_root_heading[0,0])
   delta_angle = target_angle - current_angle
   rotate smooth_root_pos, global_root_heading, joints_pos, global_rot_data, velocities
   foot_contacts unchanged

3. condition:
   first_heading_angle = target_angle
   c_dir = [cos(target_angle), sin(target_angle)]
```

注意这里的 rotate 是整段 motion 的刚体 yaw 旋转，不是逐帧把 joints_pos 变成 root-heading-local。Kimodo `rotate()` 对 smooth_root、heading、joints_pos、rot6d、velocity 同时旋转；`translate_2d()` 只平移 smooth_root xz。

如果我们想做一个绝对世界坐标 baseline，也可以通过配置关掉：

```text
--root_origin_shift false
--random_first_heading false
```

但我不建议第一轮默认这么做，因为 Kimodo 的控制约束和评估协议是按 frame-0 canonical origin 组织的。

## 4. 数据读取计划

直接使用转换 repo 的 loader：

```python
from hy201_to_kimodo273 import Kimodo273TorchDataset, collate_kimodo273_batch

dataset = Kimodo273TorchDataset(
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22",
    split="train",
    normalize=False,
)
```

代码参考：

```text
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:57
  split_existing/splits split id reader
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:75
  split filtering: directory prefix first, then split_existing ids
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:88
  Kimodo273MotionDataset
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:139
  __getitem__ loads [T,273]
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:169
  collate_kimodo273_batch
```

normalize=True 时 contacts 默认仍保持 0/1：

```text
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:144
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:147
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/dataset.py:149
```

第一版我会在本项目里写一个薄 wrapper：

```text
data/kimodo273_datasets.py
```

支持两个 mode：

```text
hml3d mode:
  root:
    /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22
  split:
    split_existing/{train,val,test}.txt
  text:
    /mnt/afs/mogo_base/datasets/HumanML3D/texts/{motion_id}.txt
  用于第一版 HY273 raw prior + HumanML3D text-conditioned training + synthetic control training

motionfix_pair mode:
  join source, target, instruction from:
    /mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionstreamer272_hml_{split}.jsonl
  或:
    /mnt/afs/mogo_base/datasets/MotionFix/manifests/motionfix_motionfix207_{split}.jsonl
  映射规则:
    strip "motionstreamer272_hml_joint_vecs/" or "motionfix207_joint_vecs/"
    -> train/000000_source.npy style converted Kimodo273 relative path
  用于之后 MotionFix edit/instruction 条件，不作为第一版 base prior 的主目标
```

第一版 base/control 训练用 HumanML3D mode。HumanML3D text 是 target motion caption，可以作为第一版 `condition_mode=hml_text`。MotionFix instruction 是“如何从 source 改到 target”的编辑指令，不是 target 的普通动作 caption，所以 MotionFix text 仍不直接喂给 base prior。

运行环境也已确认：

```text
use python:   /root/miniconda3/envs/mogo/bin/python
use torchrun: /root/miniconda3/envs/mogo/bin/torchrun
torch:        2.5.1+cu124
cuda visible: 8 x A100-SXM4-80GB
extra path:   PYTHONPATH=/mnt/afs/UMO_debug/hy201_to_kimodo273:$PYTHONPATH
```

默认 `/root/miniconda3/bin/python` 是 Python 3.13，缺少当前需要的部分包，不作为训练入口。

## 5. 模型架构 ASCII

第一版 raw denoiser：

```text
                      text mode
              HumanML3D caption or dropped null
                            |
                            v
                      TextCondition
                            |
                            v
                      text tokens
                            |
                            |
Flow t ----> TimestepEmbed ----+
                               |
c_dir ----> DirectionEmbed ----+----> global cond [B,H]
                               |
                               v
observed_motion [B,T,273]   motion_mask [B,T,273]
          |                         |
          +-----------+-------------+
                      |
noisy x_t [B,T,273]  |
          |           |
          v           v
 x_imp = x_t*(1-mask) + obs*mask
          |
          v
 concat(x_imp, mask.float)  [B,T,546]
          |
          v
 Linear 546 -> H
          |
          v
 frame motion tokens [B,T,H] + frame pos ids
          |
          v
 +-------------------------------------------+
 | FrameMotionTextDiT                         |
 |   double-stream blocks: motion <-> text    |
 |   single-stream blocks: concat tokens      |
 |   AdaLN from timestep + c_dir + text pool  |
 +-------------------------------------------+
          |
          v
 Linear H -> 273
          |
          v
pred [B,T,273]
   0:269     normalized flow velocity for continuous HY273 channels
   269:273   clean contact logits
```

### 5.1 raw-space 和内部 hidden dim 的边界

`raw-space` 只表示 flow/diffusion 的随机变量、训练 target、控制 overwrite、评估输出都在
HY273/Kimodo273 源数据域里，不表示 Transformer 内部必须一直保持 273 维。模型仍然需要正常做
input projection 和 output projection：

```text
x_t / z_t source state        [B,T,273]
observed_motion source state  [B,T,273]
motion_mask                   [B,T,273]
        |
        v
concat(x_imp, mask.float)     [B,T,546]
        |
        v
Linear 546 -> H
        |
        v
DiT / Transformer hidden      [B,T,H]
        |
        v
Linear H -> 273
        |
        v
flow velocity / contact head  [B,T,273]
```

Kimodo 官方 SMPLX22 表示维度按代码实例化核对过：

```text
motion_rep_dim:       273
global_root_dim:        5
body_dim:             268
local_root_dim:         4

[0:3]     smooth_root_pos        3
[3:5]     global_root_heading    2
[5:71]    local_joints_positions 66
[71:203]  global_rot_data        132
[203:269] velocities             66
[269:273] foot_contacts          4
```

Kimodo 官方 two-stage denoiser 也不是在 Transformer 内部固定 273 维，而是投到配置里的
`latent_dim`。你提供的 `nvidia/Kimodo-SMPLX-RP-v1` 配置里：

```text
latent_dim: 1024
ff_size: 2048
num_layers: 16
num_heads: 8
llm_shape: [1,4096]
num_text_tokens_override: 50
input_first_heading_angle: true
motion_mask_mode: concat
```

所以 Kimodo-SMPLX-RP 的实际内部主 hidden dim 是 1024：

```text
root stage:
  input_dim with mask concat = 273 * 2 = 546
  input_linear: 546 -> 1024
  transformer hidden: 1024, 16 layers, 8 heads, FFN 2048
  output_linear: 1024 -> 5

body stage:
  local_motion_rep_dim = 273 - 5 + 4 = 272
  input_dim with mask concat = 272 + 273 = 545
  input_linear: 545 -> 1024
  transformer hidden: 1024, 16 layers, 8 heads, FFN 2048
  output_linear: 1024 -> 268

final output:
  root 5 + body 268 = 273
```

这里还有一个容易漏掉的 prefix 维度：Kimodo 把 LLM text feature 从 4096 投到 1024，
并强制 text token 数到 50；再加 timestep token 和 first-heading token，所以 motion token
前面最多有 `50 + 1 + 1 = 52` 个 1024-d prefix token。

可核对代码：

```text
external_repos/kimodo/kimodo/motion_rep/reps/base.py:65
  motion_rep_dim = sum(size_dict)
external_repos/kimodo/kimodo/motion_rep/reps/base.py:67
  body_slice / body_dim
external_repos/kimodo/kimodo/motion_rep/reps/base.py:69
  global_root_dim / local_root_dim
external_repos/kimodo/kimodo/model/twostage_denoiser.py:33
  input_dim = motion_rep.motion_rep_dim
external_repos/kimodo/kimodo/model/twostage_denoiser.py:37
  root_input_dim = input_dim * 2 when mask concat
external_repos/kimodo/kimodo/model/twostage_denoiser.py:48
  local_motion_rep_dim = input_dim - global_root_dim + local_root_dim
external_repos/kimodo/kimodo/model/twostage_denoiser.py:51
  body_input_dim = local_motion_rep_dim + input_dim when mask concat
external_repos/kimodo/kimodo/model/backbone.py:117
  input_linear = Linear(input_dim, latent_dim)
external_repos/kimodo/kimodo/model/backbone.py:118
  output_linear = Linear(latent_dim, output_dim)
```

实施上第一版会把 `H` 做成显式配置。现在有了 Kimodo-SMPLX-RP 配置后，默认建议用
`hidden_dim=1024` 作为 raw-flow 主尺度；如果显存/吞吐不合适，再只调这个配置项，不改变
273 维 source-domain flow/overwrite 契约。

当前项目参考：

```text
models/codeflow/dit_blocks.py:607  FrameMotionTextDiT
models/codeflow/dit_blocks.py:633  double-stream blocks
models/codeflow/dit_blocks.py:637  single-stream blocks
models/codeflow/dit_blocks.py:708  forward
models/codeflow/motion_code_flow.py:194 timestep embed
models/codeflow/motion_code_flow.py:362 text condition helper
```

已单独跑过形状检查：

```text
FrameMotionTextDiT plain forward:
  motion [2,16,64], text [2,4,64] -> out [2,16,64], finite=True

existing control_cond adapter:
  control frame length 16 -> error, encoded length 4 != motion length 16
  control frame length 64 -> out [2,16,64], finite=True
```

原因是现有 control adapter 是两层 Conv1d stride=2，会把 control frame length 降到 1/4。所以 raw 第一版不走 `control_cond` KV adapter；控制信息直接通过 `[x_imp, mask]` 进入 input projection。以后如果要重新启用 KV adapter，需要明确让 control 条件帧长等于 motion token length 的 4 倍，或者重写 adapter 下采样规则。

另一个已确认的工程坑：

```text
models/codeflow/__init__.py imports eval modules at package import time
common/quaternion.py still uses np.float
NumPy 2.x under mogo env removes np.float
```

因此 raw harness 复用 DiT 时会避免触发包级 `models.codeflow` 初始化，或先做兼容性修复；不在训练入口里临时 monkey patch。

为什么第一版改成 Flow/ODE，而不是 DDPM/DDIM：

```text
train_codeflow.py:574
  prepare_flow_training_state
train_codeflow.py:587
  target_model = model.raw_to_model_latent(target_embeddings)
train_codeflow.py:588
  noise = randn_like(target_model)
train_codeflow.py:599
  t_view = timesteps[:, None, None, None]
train_codeflow.py:600
  z_t = t*x0 + (1-t)*noise
train_codeflow.py:611
  clean_init = predict_clean_from_velocity(z_t, t, v_init)

models/codeflow/continuous_motion_code_flow.py:60
  z_t = t*x0 + (1-t)*noise
models/codeflow/continuous_motion_code_flow.py:61
  velocity_target = x0 - noise
models/codeflow/continuous_motion_code_flow.py:95
  flow_loss = MSE(v_pred, velocity_target)
models/codeflow/continuous_motion_code_flow.py:98
  clean_pred = predict_clean_from_velocity(z_t, t, v_pred)

models/codeflow/motion_code_flow.py:707
  sampling grid
models/codeflow/motion_code_flow.py:760
  v, clean = forward_guided(...)
models/codeflow/motion_code_flow.py:761
  z = z + dt*v
models/codeflow/motion_code_flow.py:764
  x_self_cond = clean.detach()
```

所以 raw HY273 第一版应继承这个连续 flow 训练/采样能力，只把 latent 从 VQ code embedding 换成 normalized HY273 raw feature。DDPM/DDIM 只作为以后对照实验，不作为主线。

## 6. 训练 Tensor Information Flow

```text
Dataset item
  motion_un: [T,273] raw Kimodo273
  length:    scalar
  path:      source/target npy path
      |
      v
Collate
  x0_un:       [B,Tmax,273]
  length_mask: [B,Tmax]
      |
      v
Training transform
  root_origin_shift=true
  random_first_heading=true
  c_dir condition
      |
      v
x0_un_aug [B,T,273]
      |
      +-------------------------------+
      |                               |
      v                               v
HY273Normalizer                  ControlSampler
  continuous 0:269 z-score          sample protocol:
  contacts 269:273 unchanged          none/root/endpoints/fullpose/mixed
      |                             build obs_un/mask from x0_un_aug
      v                               |
x0 [B,T,273]                          v
                                obs_un [B,T,273], mask [B,T,273]
                                      |
                                      v
                                HY273Normalizer
                                      |
                                      v
                                obs [B,T,273]
      |
      v
Rectified flow interpolation
  t:     [B], sampled by logit-normal or uniform schedule
  noise: [B,T,273]
  z_t = t*x0 + (1-t)*noise
  v_target = x0 - noise
      |
      v
Kimodo-style imputation
  z_imp = z_t*(1-mask) + obs*mask
  model_in = concat(z_imp, mask.float)
      |
      v
HY273RawFlow(model_in, t, c_dir, hml_text/null_dropout, length_mask)
      |
      v
pred [B,T,273]
      |
      +--> v_pred_cont  = pred[...,0:269]
      +--> contact_logit= pred[...,269:273]
      +--> contact_prob = sigmoid(contact_logit)
      |
      v
x0_hat_pred
  x0_hat_cont = z_t[...,0:269] + (1-t)*v_pred_cont
  x0_hat = concat(x0_hat_cont, contact_prob)
      |
      +--> training losses use this pre-clamp prediction
      |
      v
x0_hat_for_sampler_or_eval_clamped
  only for ODE sampling / exact-control eval path:
  clamp controlled dims:
    x0_hat_clamped = x0_hat_pred*(1-mask) + obs*mask
      |
      v
Loss
  L_flow_cont_unmasked
  L_clean_cont_optional
  L_contact_bce
  L_control_masked
  L_global_control
  optional-small L_fk / L_foot / L_ground / L_vel_smooth
```

第一版 loss：

```text
L_flow_cont_unmasked:
  MSE(v_pred_cont, v_target[...,0:269]) on valid frames and unmasked dims

L_contact:
  BCEWithLogits(pred[...,269:273], x0[...,269:273]) on valid frames

L_clean_cont_optional:
  SmoothL1(x0_hat_pred[...,0:269], x0[...,0:269]) on valid frames, small weight

L_control:
  SmoothL1(x0_hat_pred, obs) only on motion_mask and valid frames

L_global_control:
  SmoothL1 in reconstructed global joint/root space for selected constraints

L_vel_smooth:
  small temporal smoothness regularizer on generated/unmasked regions
```

FK / foot / ground loss 的接口第一版就实现，权重按 smoke 结果从 0 或小值打开。这样符合原计划里“脚滑和 FK consistency 不是后处理才关心”的边界，同时避免第一轮 shape/smoke 被复杂 kinematics 问题卡住。

## 7. 采样 Tensor Information Flow

```text
User/control protocol
  root / endpoint subset / fullpose / mixed
      |
      v
ConstraintCompiler
  observed_motion_un [B,T,273]
  motion_mask        [B,T,273]
      |
      v
Normalize obs continuous dims, contacts stay 0/1
      |
      v
z_0 ~ N(0,I) [B,T,273]
      |
      v
for ODE step k = 0 ... K-1:
    t_k -> t_{k+1}
    z_t = z_t*(1-mask) + obs*mask
    model_in = concat(z_t, mask.float)
    pred = HY273RawFlow(model_in, t_k, c_dir, hml_text/null_dropout)
    v_cont = pred[...,0:269]
    contact_prob = sigmoid(pred[...,269:273])
    x0_hat_cont = z_t[...,0:269] + (1-t_k)*v_cont
    x0_hat = concat(x0_hat_cont, contact_prob)
    x0_hat = x0_hat*(1-mask) + obs*mask
    z_{t+dt}[...,0:269] = z_t[...,0:269] + dt*v_cont
    z_{t+dt}[...,269:273] = contact_prob
    z_{t+dt} = z_{t+dt}*(1-mask) + obs*mask
      |
      v
Unnormalize
      |
      v
HY273 output
      |
      +--> global joints from smooth_root + joints_pos
      +--> rotations from global_rot6d
      +--> contacts from sigmoid/threshold
      |
      v
Metrics / optional postprocess
```

默认采样：

```text
ODE steps: 32 first, 64 for quality checks if needed
clamp observed dims: every step
gradient guidance on z_t: OFF in first baseline
postprocess: OFF for baseline metrics, optional contact_ik later
```

## 8. 控制协议

训练和评估只覆盖我们真实会用的控制：

```text
root:
  smooth_root xz, optional y, optional heading

five endpoints:
  endpoint subset preset is configurable:
    kimodo_ee: left_ankle, left_foot, right_ankle, right_foot, left_wrist, right_wrist
    five_point: head, left_wrist, right_wrist, left_foot, right_foot
  positions are global targets, compiled into HY273 joints_pos with same-frame smooth_root reference

fullpose:
  all joints positions, optional all global rot6d

mixed:
  root + endpoints
  root + fullpose keyframes
  endpoints + contact
```

Kimodo 约束规则要保留：

```text
global joint position target -> HY273 joints_pos
requires same-frame smooth_root_ref
```

原因是 HY273 joints_pos 的 xz 是 smooth-root-relative。没有 root reference 时不能唯一转成 raw feature。实现上我会让 compiler 要么自动从同一帧 root constraint 取 smooth_root，要么明确报错；不会静默乱填。

## 9. 评估协议

Smoke eval：

```text
fixed 16 samples
protocols:
  root
  endpoint subset
  fullpose keyframes
  mixed
ODE32
save generated npy + metrics json
```

Full eval：

```text
control_root_err
control_endpoint_err
control_fullpose_keyframe_err
uncontrolled_transition_smoothness
foot_skate_from_height
foot_skate_from_pred_contacts
foot_skate_ratio
contact_consistency
```

脚滑是硬指标。只看控制误差不算通过。

foot/contact 指标按 Kimodo SMPLX22 定义：

```text
left foot joints:  left_ankle(7),  left_foot(10)
right foot joints: right_ankle(8), right_foot(11)
contact channels:  [left_ankle, left_foot, right_ankle, right_foot]
```

Kimodo metric 参考：

```text
external_repos/kimodo/kimodo/metrics/constraints.py
external_repos/kimodo/kimodo/metrics/foot_skate.py
```

## 10. 代码实施顺序

确认后按这个顺序做：

```text
Step 1. 数据 wrapper 和 representation 常量
  data/kimodo273_datasets.py
  models/raw_motion/hy273_slices.py
  tests: slice, HumanML3D split loader, captions, contacts 0/1

Step 2. normalizer + transform
  models/raw_motion/hy273_normalizer.py
  root_origin_shift default-on
  random_first_heading default-on
  tests: contacts not normalized, root shift/yaw rotation shape/numeric sanity

Step 3. constraint compiler
  models/raw_motion/hy273_constraints.py
  root/endpoints/fullpose/contact -> obs/mask
  constraints sampled from transformed x0_un_aug, not original x0_un
  tests: global joint target requires smooth_root_ref

Step 4. flow schedule
  models/raw_motion/flow_schedule.py
  sample_t, interpolate z_t, ODE step
  tests: shape, clamp observed dims each step

Step 5. raw flow DiT model
  models/raw_motion/raw_flow_dit.py
  input 546 -> hidden
  FrameMotionTextDiT backbone
  direction condition c_dir
  HumanML3D text condition + null dropout first
  tests: forward [B,T,546] -> velocity/contact head [B,T,273]

Step 6. training harness
  train_hy273_raw_flow.py
  DDP/AMP/resume/checkpoint latest
  phase1 none-control prior mode
  phase2 patterned control sampler curriculum
  one-batch overfit sanity
  4-card smoke

Step 7. sample/eval harness
  sample_hy273_raw.py
  eval_hy273_raw_control.py
  fixed-16 protocols
  ODE32/ODE64 protocols
  metrics json

Step 8. quality extensions
  turn on FK/foot/ground losses after smoke weight check
  separated CFG if text/control branches need it
  contact-aware postprocess
  pair-mode MotionFix edit conditioning after HumanML3D base/control is stable
```

## 11. 第一轮启动命令形态

先 shape/smoke：

```text
PYTHONPATH=/mnt/afs/UMO_debug/hy201_to_kimodo273:$PYTHONPATH \
/root/miniconda3/envs/mogo/bin/python train_hy273_raw_flow.py \
  --data_root /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22 \
  --text_root /mnt/afs/mogo_base/datasets/HumanML3D/texts \
  --split train \
  --condition_mode hml_text \
  --batch_size 4 \
  --max_steps 20 \
  --flow_steps_eval 32 \
  --num_workers 2
```

再 4 卡 DDP：

```text
PYTHONPATH=/mnt/afs/UMO_debug/hy201_to_kimodo273:$PYTHONPATH \
/root/miniconda3/envs/mogo/bin/torchrun --nproc_per_node=4 train_hy273_raw_flow.py \
  --data_root /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22 \
  --text_root /mnt/afs/mogo_base/datasets/HumanML3D/texts \
  --split train \
  --condition_mode hml_text \
  --text_dropout_prob <待定, e.g. 0.1> \
  --batch_size_per_gpu <待测吞吐后定> \
  --epochs <确认后定> \
  --time_schedule logit_normal \
  --sampling_method ode \
  --eval_ode_steps 32 \
  --save_latest_every <N>
```

我会先根据空卡显存试 batch size，不降采样协议，不把 ODE32 改成更低，除非你明确允许。

## 12. 本轮复核证据

数据转换和语义：

```text
/mnt/afs/UMO_debug/hy201_to_kimodo273
  git commit: ea668b7

/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:22
  DIM_KIMODO273 = 273
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:31
  HY273 slices
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:38
  SMPLX22 joint order
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:102
  HY201 6D decode
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:123
  Kimodo cont6d decode
/mnt/afs/UMO_debug/hy201_to_kimodo273/hy201_to_kimodo273/geometry.py:159
  split_kimodo273
```

loader：

```text
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
```

HumanML3D loader/caption smoke：

```text
dataset train len 21466 first_shape (124, 273) first_rel motion_data/000002.npy contact_unique [0.0, 1.0]
dataset val len 1338 first_shape (204, 273) first_rel motion_data/000016.npy contact_unique [1.0]
dataset test len 4042 first_shape (175, 273) first_rel motion_data/000000.npy contact_unique [0.0, 1.0]
batch_motion (4, 300, 273)
batch_mask (4, 300)
caption_files_total 26846
missing caption files: 0
caption line count min/max/mean: 1 / 4 / 2.9884
```

Kimodo representation/control：

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
```

endpoints/contact：

```text
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

当前项目可复用/需避开的部分：

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
```

和你的原始 raw-space 实施文档对齐：

```text
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:15
  HY273 is the raw-space diffusion / flow variable
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:124
  x0 / observed_motion / motion_mask shapes
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:209
  Kimodo-style imputation: x_in = observed*mask + noisy*(1-mask)
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:237
  global joint target requires same-frame smooth_root_ref
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:513
  old conservative baseline said DDPM clean x0 first;
  this confirmation plan supersedes it with flow-matching first to match current backbone
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:993
  old note placed Flow Matching second;
  current backbone review moves Flow Matching / ODE to first implementation
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1007
  Phase 1 no-control natural prior
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1026
  Phase 2 realistic control sampler patterns
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1197
  DDPM/DDIM sampling with step-wise clamp in the old note maps here to ODE sampling with step-wise clamp
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1291
  postprocess is part of the system, not an afterthought
outside_doc/HY273_raw_space_diffusion_training_control_impl.md:1420
  key traps: no contact z-score, no global target without smooth_root_ref,
  no raw-only metric, no uniform random mask, control + foot skate metrics required
```

## 13. 仍需确认的小点

实施前我只剩这几个边界需要你确认；我的默认推荐写在每条后面。

```text
1. 评估默认 endpoint preset:
   推荐默认 kimodo_ee:
     left_ankle, left_foot, right_ankle, right_foot, left_wrist, right_wrist
   同时保留 five_point:
     head, left_wrist, right_wrist, left_foot, right_foot

2. HumanML3D text condition 的 dropout 初值:
   现在默认 condition_mode=hml_text。
   我建议 text_dropout_prob 从 0.1 起，用于 CFG/null 分支鲁棒性。
   MotionFix instruction 仍不用于 base prior，因为它是 source->target edit 指令，不是 target caption。

3. 第一版训练 transform 是否按 Kimodo protocol 默认打开：
   root_origin_shift=true
   random_first_heading=true
   first_heading_angle/c_dir condition=true
```

除此之外，我可以按本文直接开始实施。
