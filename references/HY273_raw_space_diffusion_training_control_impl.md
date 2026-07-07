# HY273 Raw Data Space Diffusion / Flow Training 与控制实现说明

> 交给执行 agent 的实施文档。  
> **前置表示文档**：`HY201_to_kimodo273_guide.md`。本文默认已经完成或即将完成 HY201 → HY273 数据转换。  
> **范围**：只写 raw data space diffusion / flow matching 的训练与控制实现。  
> **不写**：具体模型架构。模型架构部分在第 11 节留空，之后由你们补充。

---

## 0. 本文默认决策

本文默认采用前置文档中的 **A1 路线**：

```text
HY273 = raw-space diffusion / flow matching 的生成变量本身
控制 = 同空间 observed_motion + binary motion_mask
模型输出 = clean HY273 或 flow velocity
```

也就是说，本文不是 MoGeFlow/RVQ code-space 方案，而是 Kimodo-like 的 raw data space 方案。若之后改回 A2/A3，本文件中“控制构造、loss、metric、postprocess”仍可复用，但训练主循环需要改。

---

## 1. 总体思维图

```text
                       ┌──────────────────────────────┐
                       │ HY201 / 272 原始 motion data │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────┐
                  │ HY201 -> HY273 转换与验证           │
                  │ smooth root / heading / FK / vel    │
                  │ foot contact / stats / round-trip   │
                  └─────────────────┬──────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ Dataset item                                           │
        │ x0_hy273: [T, 273]                                     │
        │ length_mask: [T]                                       │
        │ optional text/action/style                             │
        │ derived control candidates: root, endpoints, full pose  │
        └─────────────────────┬─────────────────────────────────┘
                              │
          ┌───────────────────┴────────────────────┐
          │                                        │
          ▼                                        ▼
┌───────────────────────┐              ┌─────────────────────────┐
│ Phase 1: no-control   │              │ Phase 2: control sampler│
│ learn natural prior   │              │ root / EE / full pose   │
└──────────┬────────────┘              │ contact / mixed anchors │
           │                           └───────────┬─────────────┘
           │                                       │
           └───────────────────┬───────────────────┘
                               ▼
                   ┌──────────────────────┐
                   │ noising / interpolation │
                   │ DDPM or Flow Matching   │
                   └───────────┬──────────┘
                               │
                               ▼
          ┌──────────────────────────────────────────┐
          │ Kimodo-style control imputation           │
          │ x_in = m * observed + (1-m) * noisy_state │
          │ model_input = concat(x_in, m)             │
          └────────────────────┬─────────────────────┘
                               │
                               ▼
          ┌──────────────────────────────────────────┐
          │ Model architecture: TBD by user           │
          │ output clean x0_hat or flow velocity      │
          └────────────────────┬─────────────────────┘
                               │
                               ▼
          ┌──────────────────────────────────────────┐
          │ Loss / metrics                            │
          │ raw component loss                         │
          │ control loss                               │
          │ FK consistency                             │
          │ foot lock / contact / ground               │
          └────────────────────┬─────────────────────┘
                               │
                               ▼
          ┌──────────────────────────────────────────┐
          │ Sampling                                  │
          │ step-wise imputation / optional CFG        │
          │ final contact-aware IK / projection        │
          └──────────────────────────────────────────┘
```

---

## 2. HY273 数据契约

前置文档已经固定 HY273 的 J=22 布局。执行 agent 必须以这个布局写统一 slice，不要在各处手写 magic numbers。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class HY273Slices:
    D: int = 273
    J: int = 22

    smooth_root: slice = slice(0, 3)          # [3]
    root_heading: slice = slice(3, 5)         # [2] cos/sin
    joints_pos: slice = slice(5, 71)          # [22*3]
    joints_global_rot6d: slice = slice(71, 203)  # [22*6]
    joints_vel: slice = slice(203, 269)       # [22*3]
    foot_contact: slice = slice(269, 273)     # [4]

    @property
    def non_contact_indices(self):
        return list(range(0, 269))

S = HY273Slices()
```

形状约定：

```text
x0:              [B, T, 273]
observed_motion: [B, T, 273]
motion_mask:     [B, T, 273], bool or 0/1 float
length_mask:      [B, T], True = valid frame
```

归一化约定：

```text
continuous features: z-score
foot_contact: 保持 0/1，不 z-score
```

推荐实现：

```python
import torch

class HY273Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8):
        """
        mean/std: [273]. contact dims can be arbitrary; they are ignored.
        """
        self.mean = mean
        self.std = std.clamp_min(eps)
        self.cont_idx = torch.tensor(S.non_contact_indices, dtype=torch.long)
        self.contact = S.foot_contact

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.cont_idx = self.cont_idx.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y[..., self.cont_idx] = (x[..., self.cont_idx] - self.mean[self.cont_idx]) / self.std[self.cont_idx]
        # contacts stay 0/1
        y[..., self.contact] = x[..., self.contact]
        return y

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y[..., self.cont_idx] = x[..., self.cont_idx] * self.std[self.cont_idx] + self.mean[self.cont_idx]
        y[..., self.contact] = x[..., self.contact]
        return y
```

注意：如果模型输出 contact logits，`unnormalize()` 之前应该先把 contact logits 转成 contact probability，见第 6 节。

---

## 3. Kimodo 的关键参考点

这里不是要照搬 Kimodo 架构，而是要复用它的训练和控制范式。

### 3.1 表示参考

Kimodo 的 motion representation 在代码里是这些字段：

```python
# kimodo/motion_rep/reps/kimodo_motionrep.py
self.size_dict = {
    "smooth_root_pos": torch.Size([3]),
    "global_root_heading": torch.Size([2]),
    "local_joints_positions": torch.Size([nbjoints, 3]),
    "global_rot_data": torch.Size([nbjoints, 6]),
    "velocities": torch.Size([nbjoints, 3]),
    "foot_contacts": torch.Size([4]),
}
```

对应本文 HY273：

```text
smooth_root_pos        -> HY273[0:3]
global_root_heading    -> HY273[3:5]
local_joints_positions -> HY273[5:71]，但这里的 “local” 是 smooth-root-relative xz + global y
global_rot_data        -> HY273[71:203]
velocities             -> HY273[203:269]
foot_contacts          -> HY273[269:273]
```

### 3.2 控制 imputation 参考

Kimodo 的 denoiser 在 `motion_mask_mode == "concat"` 时做：

```python
# kimodo/model/twostage_denoiser.py
x = x * (1 - motion_mask) + observed_motion * motion_mask
x_extended = torch.cat([x, motion_mask], axis=-1)
```

本文统一采用这个范式：

```text
x_in = m * observed_motion + (1-m) * noisy_state
model_input = concat(x_in, m)
```

这就是 raw-space 控制的核心。控制目标和 motion 变量同空间，模型每一步都能看到 clean anchor，mask 告诉模型哪些值是用户确认的。

### 3.3 条件构造参考

Kimodo 的 `create_conditions()` 先创建：

```python
observed_motion = torch.zeros(length, motion_rep_dim)
motion_mask = torch.zeros(length, motion_rep_dim, dtype=torch.bool)
```

然后不同 constraint 只填自己负责的 feature slice。

另外，Kimodo 在 global joint position 约束里要求同帧 smooth root 也被约束，因为 global joint target 需要转换成 smooth-root-relative joint position。如果没有 root reference，转换会不确定。本文也采用这个约束：

```text
global joint target -> HY273 joints_pos 时，必须有 same-frame smooth_root_ref
```

如果用户只给手/脚点但没给 root，UI/backend 需要内部估计 `smooth_root_ref`，或者把这个末端约束只放进 FK/global control loss，而不是直接塞入 HY273 raw feature。

### 3.4 root/body 分解参考

Kimodo 的 two-stage denoiser 做：

```text
stage 1: full noisy motion + constraints -> global root clean prediction
stage 2: predicted global root -> local root representation, then body denoising
```

本文不规定你的模型架构，但建议保留信息流：

```text
root / heading / maybe contact 先形成 clean estimate
body / limbs 在 root-local condition 下生成
```

### 3.5 postprocess 参考

Kimodo 的 `post_process_motion()` 接收：

```python
local_rot_mats, root_positions, contacts, skeleton, constraint_lst
```

并调用 motion correction 做 foot-skate cleanup 和 constraint correction。本文建议第一版就把 postprocess 接口留好，不要等模型训完再补。

---

## 4. 控制数据结构

所有 UI/脚本约束最终都要编译成：

```python
observed_motion: torch.Tensor  # [B, T, 273], same normalized/un-normalized convention as x0
motion_mask: torch.Tensor      # [B, T, 273], bool or float
```

推荐先在 **unnormalized HY273** 空间构造，再 normalize continuous dims。

```python
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class RootConstraint:
    frames: torch.LongTensor              # [K]
    smooth_root_xz: torch.Tensor          # [K, 2]
    root_y: Optional[torch.Tensor] = None # [K]
    heading: Optional[torch.Tensor] = None # [K, 2], cos/sin

@dataclass
class JointPositionConstraint:
    frames: torch.LongTensor              # [K]
    joint_ids: torch.LongTensor           # [M] or [K, M]
    global_positions: torch.Tensor        # [K, M, 3]
    smooth_root_xz_ref: torch.Tensor      # [K, 2], required for raw HY273 imputation

@dataclass
class JointRotationConstraint:
    frames: torch.LongTensor              # [K]
    joint_ids: torch.LongTensor           # [M] or [K, M]
    global_rot6d: torch.Tensor            # [K, M, 6]

@dataclass
class FullPoseConstraint:
    frames: torch.LongTensor              # [K]
    global_joint_positions: torch.Tensor  # [K, 22, 3]
    global_joint_rot6d: Optional[torch.Tensor] # [K, 22, 6]
    smooth_root_xz_ref: torch.Tensor      # [K, 2]
    root_y: Optional[torch.Tensor] = None # [K]
    heading: Optional[torch.Tensor] = None # [K, 2]

@dataclass
class FootContactConstraint:
    frames: torch.LongTensor              # [K]
    contact4: torch.Tensor                # [K, 4], 0/1
```

### 4.1 构造工具

```python
def make_empty_conditions(batch_size: int, num_frames: int, device, dtype=torch.float32):
    obs = torch.zeros(batch_size, num_frames, S.D, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, num_frames, S.D, device=device, dtype=torch.bool)
    return obs, mask


def set_values(obs, mask, b: int, frames: torch.LongTensor, feature_idx, values: torch.Tensor):
    """
    feature_idx can be slice, list[int], or torch.LongTensor.
    values shape should broadcast to obs[b, frames, feature_idx].
    """
    obs[b, frames, feature_idx] = values.to(obs.device, obs.dtype)
    mask[b, frames, feature_idx] = True


def global_pos_to_hy273_jpos(global_positions: torch.Tensor, smooth_root_xz_ref: torch.Tensor) -> torch.Tensor:
    """
    global_positions: [K, M, 3]
    smooth_root_xz_ref: [K, 2]
    return joints_pos feature values: [K, M, 3]
    x,z subtract smooth root; y remains global.
    """
    out = global_positions.clone()
    out[..., 0] = out[..., 0] - smooth_root_xz_ref[:, None, 0]
    out[..., 2] = out[..., 2] - smooth_root_xz_ref[:, None, 1]
    return out


def joint_pos_feature_indices(joint_ids: torch.LongTensor) -> torch.LongTensor:
    """Return flattened HY273 feature indices for joints_pos[joint_ids, xyz]."""
    base = S.joints_pos.start
    ids = joint_ids.to(torch.long)
    idx = base + ids[:, None] * 3 + torch.arange(3, device=ids.device)[None, :]
    return idx.reshape(-1)


def joint_rot_feature_indices(joint_ids: torch.LongTensor) -> torch.LongTensor:
    """Return flattened HY273 feature indices for global_rot6d[joint_ids, 6]."""
    base = S.joints_global_rot6d.start
    ids = joint_ids.to(torch.long)
    idx = base + ids[:, None] * 6 + torch.arange(6, device=ids.device)[None, :]
    return idx.reshape(-1)
```

### 4.2 Root constraint 写入

```python
def apply_root_constraint(obs, mask, b: int, c: RootConstraint):
    frames = c.frames.to(obs.device)

    # smooth_root x,z only
    obs[b, frames, S.smooth_root.start + 0] = c.smooth_root_xz[:, 0].to(obs.device, obs.dtype)
    obs[b, frames, S.smooth_root.start + 2] = c.smooth_root_xz[:, 1].to(obs.device, obs.dtype)
    mask[b, frames, S.smooth_root.start + 0] = True
    mask[b, frames, S.smooth_root.start + 2] = True

    if c.root_y is not None:
        obs[b, frames, S.smooth_root.start + 1] = c.root_y.to(obs.device, obs.dtype)
        mask[b, frames, S.smooth_root.start + 1] = True

    if c.heading is not None:
        obs[b, frames, S.root_heading] = c.heading.to(obs.device, obs.dtype)
        mask[b, frames, S.root_heading] = True
```

### 4.3 末端点 / 任意关节位置写入

```python
def apply_joint_position_constraint(obs, mask, b: int, c: JointPositionConstraint):
    frames = c.frames.to(obs.device)
    joint_ids = c.joint_ids.to(obs.device)

    # Only support fixed joint_ids [M] in this helper.
    assert joint_ids.ndim == 1, "For per-frame joint ids, write a second helper."

    local_like = global_pos_to_hy273_jpos(
        c.global_positions.to(obs.device, obs.dtype),
        c.smooth_root_xz_ref.to(obs.device, obs.dtype),
    )  # [K, M, 3]

    # flattened feature indices [M*3]
    fidx = joint_pos_feature_indices(joint_ids).to(obs.device)
    obs[b, frames[:, None], fidx[None, :]] = local_like.reshape(len(frames), -1)
    mask[b, frames[:, None], fidx[None, :]] = True
```

### 4.4 full-pose keyframe 写入

```python
def apply_full_pose_constraint(obs, mask, b: int, c: FullPoseConstraint):
    frames = c.frames.to(obs.device)

    # root xz from smooth_root_ref
    obs[b, frames, S.smooth_root.start + 0] = c.smooth_root_xz_ref[:, 0].to(obs.device, obs.dtype)
    obs[b, frames, S.smooth_root.start + 2] = c.smooth_root_xz_ref[:, 1].to(obs.device, obs.dtype)
    mask[b, frames, S.smooth_root.start + 0] = True
    mask[b, frames, S.smooth_root.start + 2] = True

    if c.root_y is not None:
        obs[b, frames, S.smooth_root.start + 1] = c.root_y.to(obs.device, obs.dtype)
        mask[b, frames, S.smooth_root.start + 1] = True

    if c.heading is not None:
        obs[b, frames, S.root_heading] = c.heading.to(obs.device, obs.dtype)
        mask[b, frames, S.root_heading] = True

    # all joint positions
    local_like_pos = global_pos_to_hy273_jpos(
        c.global_joint_positions.to(obs.device, obs.dtype),
        c.smooth_root_xz_ref.to(obs.device, obs.dtype),
    )
    obs[b, frames, S.joints_pos] = local_like_pos.reshape(len(frames), -1)
    mask[b, frames, S.joints_pos] = True

    if c.global_joint_rot6d is not None:
        obs[b, frames, S.joints_global_rot6d] = c.global_joint_rot6d.to(obs.device, obs.dtype).reshape(len(frames), -1)
        mask[b, frames, S.joints_global_rot6d] = True
```

### 4.5 foot contact / foot lock 写入

```python
def apply_foot_contact_constraint(obs, mask, b: int, c: FootContactConstraint):
    frames = c.frames.to(obs.device)
    obs[b, frames, S.foot_contact] = c.contact4.to(obs.device, obs.dtype)
    mask[b, frames, S.foot_contact] = True
```

---

## 5. Kimodo-style imputation

### 5.1 通用函数

```python
def apply_control_imputation(noisy_state: torch.Tensor,
                             observed_motion: torch.Tensor | None,
                             motion_mask: torch.Tensor | None):
    """
    noisy_state: [B, T, D]
    observed_motion: [B, T, D], same normalized convention as noisy_state
    motion_mask: [B, T, D], bool or 0/1
    return:
      x_in: [B, T, D]
      mask_float: [B, T, D]
    """
    if observed_motion is None or motion_mask is None:
        mask_float = torch.zeros_like(noisy_state)
        return noisy_state, mask_float

    mask_float = motion_mask.to(dtype=noisy_state.dtype)
    x_in = noisy_state * (1.0 - mask_float) + observed_motion * mask_float
    return x_in, mask_float


def build_model_input(noisy_state, observed_motion=None, motion_mask=None):
    x_in, mask_float = apply_control_imputation(noisy_state, observed_motion, motion_mask)
    return torch.cat([x_in, mask_float], dim=-1)
```

### 5.2 训练时 normalize 顺序

推荐顺序：

```python
# x0_un: [B,T,273], unnormalized HY273
# obs_un: [B,T,273], unnormalized observed motion
# mask: bool [B,T,273]

x0 = normalizer.normalize(x0_un)
obs = normalizer.normalize(obs_un)

# sample noisy state from normalized x0
xt = q_sample(x0, t, eps)

# impute normalized clean observations into normalized noisy state
model_in = build_model_input(xt, obs, mask)
```

---

## 6. 模型输出解释

模型架构未定，但训练代码需要约定输出语义。

### 6.1 DDPM / clean x0 prediction

推荐 Kimodo-like：模型直接输出 clean HY273：

```text
pred = model(model_in, t, condition)
pred[..., 0:269] = continuous clean prediction
pred[..., 269:273] = contact logits
```

为了 sampler 和 continuous loss，需要把 contact logits 转成 probability：

```python
def split_clean_prediction(pred: torch.Tensor):
    """
    pred: [B,T,273]
    return:
      x0_hat_for_sampler: contact dims are probabilities in [0,1]
      contact_logits
      contact_prob
    """
    x0_hat = pred.clone()
    contact_logits = pred[..., S.foot_contact]
    contact_prob = torch.sigmoid(contact_logits)
    x0_hat[..., S.foot_contact] = contact_prob
    return x0_hat, contact_logits, contact_prob
```

如果你们不想用 BCE，也可以让 contact 作为普通连续变量做 smooth L1；但为了 foot-lock 和 postprocess，推荐 logits + BCE。

### 6.2 Flow Matching / velocity prediction

如果用 flow matching：

```text
z ~ N(0, I)
t ~ U(0,1)
x_t = (1-t) * z + t * x0
v_target = x0 - z
pred_v = model(concat(impute(x_t), mask), t, condition)
x0_hat = x_t + (1-t) * pred_v
```

注意：**clean control imputation 会破坏 controlled dims 上的标准 flow target**。因此如果采用 Kimodo-style clean imputation，建议：

```text
1. velocity loss 只在 unmasked dims 上算，或 masked dims 权重极低；
2. controlled dims 在 x0_hat 中直接 clamp 为 observed_motion；
3. control / FK / foot losses 在 clamped clean x0_hat 上算。
```

参考实现：

```python
def fm_make_xt(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    """x_t = (1-t) z + t x0. t shape [B] or [B,1,1]."""
    while t.ndim < x0.ndim:
        t = t[..., None]
    return (1.0 - t) * noise + t * x0


def fm_clean_from_velocity(x_in_for_flow: torch.Tensor,
                           pred_v: torch.Tensor,
                           t: torch.Tensor,
                           observed_motion=None,
                           motion_mask=None):
    while t.ndim < pred_v.ndim:
        t = t[..., None]
    x0_hat = x_in_for_flow + (1.0 - t) * pred_v
    if observed_motion is not None and motion_mask is not None:
        m = motion_mask.to(x0_hat.dtype)
        x0_hat = x0_hat * (1 - m) + observed_motion * m
    return x0_hat
```

---

## 7. Loss 设计

Loss 分两类：

```text
normalized-space loss:
  diffusion/FM 主损失
  raw feature reconstruction
  raw control mask loss

unnormalized/FK-space loss:
  FK consistency
  global joint control error
  foot lock
  ground/contact consistency
  metrics
```

### 7.1 有效帧 mask

```python
def expand_length_mask(length_mask: torch.Tensor, D: int):
    """length_mask [B,T] -> [B,T,D] float"""
    return length_mask[..., None].to(torch.float32).expand(-1, -1, D)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(eps)
```

### 7.2 component-wise Huber loss

```python
import torch.nn.functional as F

LOSS_WEIGHTS_DEFAULT = {
    "root_pos": 10.0,
    "heading": 2.0,
    "joint_pos": 10.0,
    "joint_rot": 10.0,
    "joint_vel": 3.0,
    "contact_bce": 4.0,
    "fk": 5.0,
    "control_raw": 20.0,
    "control_global": 20.0,
    "foot_lock": 10.0,
    "ground": 5.0,
}


def smooth_l1_masked(pred, target, mask, beta=1.0):
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return masked_mean(loss, mask)


def component_losses(x0_hat_norm, x0_norm, contact_logits, length_mask, weights=None):
    weights = weights or LOSS_WEIGHTS_DEFAULT
    lm = length_mask[..., None].to(x0_norm.dtype)
    losses = {}

    losses["root_pos"] = smooth_l1_masked(
        x0_hat_norm[..., S.smooth_root], x0_norm[..., S.smooth_root], lm.expand_as(x0_norm[..., S.smooth_root])
    )
    losses["heading"] = smooth_l1_masked(
        x0_hat_norm[..., S.root_heading], x0_norm[..., S.root_heading], lm.expand_as(x0_norm[..., S.root_heading])
    )
    losses["joint_pos"] = smooth_l1_masked(
        x0_hat_norm[..., S.joints_pos], x0_norm[..., S.joints_pos], lm.expand_as(x0_norm[..., S.joints_pos])
    )
    losses["joint_rot"] = smooth_l1_masked(
        x0_hat_norm[..., S.joints_global_rot6d], x0_norm[..., S.joints_global_rot6d],
        lm.expand_as(x0_norm[..., S.joints_global_rot6d])
    )
    losses["joint_vel"] = smooth_l1_masked(
        x0_hat_norm[..., S.joints_vel], x0_norm[..., S.joints_vel], lm.expand_as(x0_norm[..., S.joints_vel])
    )

    contact_target = x0_norm[..., S.foot_contact]
    bce = F.binary_cross_entropy_with_logits(contact_logits, contact_target, reduction="none")
    losses["contact_bce"] = masked_mean(bce, lm.expand_as(contact_target))

    total = sum(weights[k] * v for k, v in losses.items() if k in weights)
    return total, losses
```

### 7.3 raw control loss

这个 loss 只在受控 feature 上算。它保证模型在训练时意识到 mask 位置是高优先级。

```python
def raw_control_loss(x0_hat_norm, observed_norm, motion_mask, length_mask, weight=20.0):
    if observed_norm is None or motion_mask is None:
        return x0_hat_norm.new_tensor(0.0)
    m = motion_mask.to(x0_hat_norm.dtype) * length_mask[..., None].to(x0_hat_norm.dtype)
    if m.sum() < 1:
        return x0_hat_norm.new_tensor(0.0)
    return weight * smooth_l1_masked(x0_hat_norm, observed_norm, m)
```

在 Flow Matching clean-imputation + clamp 设置下，controlled dims 的 `x0_hat` 可能已经被直接替换成 observed，`raw_control_loss` 会接近 0。这是正常的；真正的作用是 sampling exact clamp 和 unmasked transition 学习。

### 7.4 FK consistency loss

HY273 同时包含 global rotations 和 joint positions，必须绑定一致性。这里需要项目已有的 skeleton/FK 实现。

接口建议：

```python
class HY273Kinematics:
    def hy273_to_global_positions_from_pos(self, x_un: torch.Tensor) -> torch.Tensor:
        """
        x_un: [B,T,273] unnormalized
        return global joint positions reconstructed from HY273 joints_pos + smooth_root: [B,T,22,3]
        """
        smooth_root = x_un[..., S.smooth_root]             # [B,T,3]
        jpos = x_un[..., S.joints_pos].reshape(*x_un.shape[:2], S.J, 3).clone()
        jpos[..., 0] += smooth_root[..., None, 0]
        jpos[..., 2] += smooth_root[..., None, 2]
        # y already global
        return jpos

    def hy273_to_global_rot_mats(self, x_un: torch.Tensor) -> torch.Tensor:
        """global rot6d -> rotation matrices, [B,T,22,3,3]."""
        raise NotImplementedError

    def global_to_local_rot_mats(self, global_rot_mats: torch.Tensor) -> torch.Tensor:
        """global rotations -> local rotations along SMPL-H parent chain."""
        raise NotImplementedError

    def fk_from_rotations(self, x_un: torch.Tensor) -> torch.Tensor:
        """
        Use predicted rotations + recovered root translation to FK global joint positions.
        return [B,T,22,3]
        """
        raise NotImplementedError
```

Loss：

```python
def fk_consistency_loss(x0_hat_un, kin: HY273Kinematics, length_mask, weight=5.0):
    j_from_pos = kin.hy273_to_global_positions_from_pos(x0_hat_un)
    j_from_fk = kin.fk_from_rotations(x0_hat_un)
    m = length_mask[..., None, None].to(x0_hat_un.dtype).expand_as(j_from_pos)
    return weight * smooth_l1_masked(j_from_fk, j_from_pos, m)
```

### 7.5 foot lock / ground loss

```python
def finite_diff_positions(pos: torch.Tensor, fps: float):
    """pos [B,T,J,3] -> velocity [B,T,J,3]. first frame copies second or zero."""
    vel = torch.zeros_like(pos)
    vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * fps
    vel[:, 0] = vel[:, 1] if pos.shape[1] > 1 else 0
    return vel


def foot_lock_loss(x0_hat_un, x0_target_un, kin: HY273Kinematics, length_mask,
                   foot_contact_to_joint_ids: list[int], fps: float, weight=10.0):
    """
    foot_contact_to_joint_ids length 4: [left_heel, left_toe, right_heel, right_toe] joint indices.
    Use target contact as supervision. If toe/heel virtual joints are not in 22 joints,
    kin should expose them via FK/markers; otherwise map to closest ankle/toe joints.
    """
    pred_pos = kin.hy273_to_global_positions_from_pos(x0_hat_un)
    pred_vel = finite_diff_positions(pred_pos, fps)

    contact = x0_target_un[..., S.foot_contact]  # [B,T,4], 0/1
    losses = []
    for ci, jid in enumerate(foot_contact_to_joint_ids):
        v = pred_vel[..., jid, :]  # [B,T,3]
        c = contact[..., ci:ci+1]
        m = c * length_mask[..., None].to(x0_hat_un.dtype)
        losses.append(masked_mean(v.pow(2), m.expand_as(v)))
    return weight * sum(losses) / max(len(losses), 1)


def ground_loss(x0_hat_un, x0_target_un, kin: HY273Kinematics, length_mask,
                foot_contact_to_joint_ids: list[int], ground_y: float = 0.0, weight=5.0):
    pred_pos = kin.hy273_to_global_positions_from_pos(x0_hat_un)
    contact = x0_target_un[..., S.foot_contact]
    losses = []
    for ci, jid in enumerate(foot_contact_to_joint_ids):
        height_err = (pred_pos[..., jid, 1] - ground_y).abs()  # [B,T]
        m = contact[..., ci] * length_mask.to(x0_hat_un.dtype)
        losses.append(masked_mean(height_err, m))
    return weight * sum(losses) / max(len(losses), 1)
```

### 7.6 global control loss

Raw mask loss 在 HY273 feature 空间工作；但用户真正关心的是 global root / end-effector / full pose error。因此训练和验证都应该有 global-space control loss。

推荐做法：control builder 同时返回 `ConstraintEvalTargets`。

```python
@dataclass
class GlobalPositionEvalTarget:
    frames: torch.LongTensor       # [K]
    joint_ids: torch.LongTensor    # [M]
    global_positions: torch.Tensor # [K,M,3]
    weight: float = 1.0

@dataclass
class ConstraintEvalTargets:
    global_joint_pos_targets: list[GlobalPositionEvalTarget]
    # 可继续添加 root path、heading、global rotation targets
```

Loss：

```python
def global_position_control_loss(x0_hat_un, eval_targets: ConstraintEvalTargets,
                                 kin: HY273Kinematics, batch_id: int, weight=20.0):
    if eval_targets is None or not eval_targets.global_joint_pos_targets:
        return x0_hat_un.new_tensor(0.0)
    pred_pos = kin.hy273_to_global_positions_from_pos(x0_hat_un)  # [B,T,J,3]
    total = x0_hat_un.new_tensor(0.0)
    denom = 0.0
    for tgt in eval_targets.global_joint_pos_targets:
        frames = tgt.frames.to(x0_hat_un.device)
        jids = tgt.joint_ids.to(x0_hat_un.device)
        target = tgt.global_positions.to(x0_hat_un.device, x0_hat_un.dtype)
        pred = pred_pos[batch_id, frames[:, None], jids[None, :], :]
        total = total + tgt.weight * F.smooth_l1_loss(pred, target, reduction="mean")
        denom += tgt.weight
    return weight * total / max(denom, 1e-8)
```

---

## 8. DDPM 训练主循环参考

Kimodo 用的是 clean x0 prediction，建议第一版先做这个。Flow Matching 放到第 9 节。

### 8.1 diffusion utilities

```python
class DiffusionSchedule:
    def __init__(self, num_steps: int = 1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor):
        """x_t = sqrt(ab) x0 + sqrt(1-ab) noise"""
        ab = self.alpha_bars[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
```

### 8.2 training step

```python
def train_step_ddpm(batch, model, normalizer: HY273Normalizer, schedule: DiffusionSchedule,
                    kin: HY273Kinematics, optimizer, control_sampler, cfg, device):
    """
    batch fields expected:
      batch["x0_hy273"]     [B,T,273] unnormalized
      batch["length_mask"]  [B,T] bool
      optional batch["text"], batch["action"], etc.
    control_sampler returns:
      obs_un, motion_mask, eval_targets
    model architecture is intentionally unspecified.
    """
    x0_un = batch["x0_hy273"].to(device)
    length_mask = batch["length_mask"].to(device).bool()
    B, T, D = x0_un.shape

    # 1) sample constraints from ground-truth x0_un
    obs_un, motion_mask, eval_targets = control_sampler.sample(batch, x0_un, length_mask)
    obs_un = obs_un.to(device)
    motion_mask = motion_mask.to(device).bool()

    # 2) normalize
    x0 = normalizer.normalize(x0_un)
    obs = normalizer.normalize(obs_un)

    # 3) diffusion noising
    t = torch.randint(0, schedule.num_steps, (B,), device=device)
    eps = torch.randn_like(x0)
    xt = schedule.q_sample(x0, t, eps)

    # 4) Kimodo-style imputation
    model_in = build_model_input(xt, obs, motion_mask)  # [B,T,546]

    # 5) optional condition dropout for CFG training
    cond = build_condition(batch, cfg, device)  # implement in your project
    cond = maybe_drop_condition(cond, cfg)      # text/control dropout etc.

    # 6) forward: clean x0 prediction
    pred = model(model_in, t, cond, length_mask=length_mask)
    x0_hat, contact_logits, contact_prob = split_clean_prediction(pred)

    # 7) normalized-space losses
    comp_total, comp_dict = component_losses(x0_hat, x0, contact_logits, length_mask)
    l_ctrl_raw = raw_control_loss(x0_hat, obs, motion_mask, length_mask, weight=cfg.w_control_raw)

    # 8) unnormalized-space losses
    x0_hat_un = normalizer.unnormalize(x0_hat)
    l_fk = fk_consistency_loss(x0_hat_un, kin, length_mask, weight=cfg.w_fk)
    l_foot = foot_lock_loss(x0_hat_un, x0_un, kin, length_mask, cfg.foot_contact_to_joint_ids, cfg.fps, cfg.w_foot_lock)
    l_ground = ground_loss(x0_hat_un, x0_un, kin, length_mask, cfg.foot_contact_to_joint_ids, cfg.ground_y, cfg.w_ground)

    # Optional: global-space control loss per batch item.
    l_global = x0_hat.new_tensor(0.0)
    for b in range(B):
        l_global = l_global + global_position_control_loss(x0_hat_un, eval_targets[b], kin, b, weight=cfg.w_control_global)
    l_global = l_global / B

    loss = comp_total + l_ctrl_raw + l_fk + l_foot + l_ground + l_global

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    logs = {"loss": loss.detach(), "control_raw": l_ctrl_raw.detach(), "fk": l_fk.detach(),
            "foot_lock": l_foot.detach(), "ground": l_ground.detach(), "global_ctrl": l_global.detach()}
    logs.update({f"comp/{k}": v.detach() for k, v in comp_dict.items()})
    return logs
```

`build_condition()`、`maybe_drop_condition()`、`model()` 的细节留给架构实现。

---

## 9. Flow Matching 训练主循环参考

如果你们决定用 flow matching，保持控制接口不变，但注意 masked dims 的 velocity loss。

```python
def train_step_fm(batch, model, normalizer: HY273Normalizer, kin: HY273Kinematics,
                  optimizer, control_sampler, cfg, device):
    x0_un = batch["x0_hy273"].to(device)
    length_mask = batch["length_mask"].to(device).bool()
    B, T, D = x0_un.shape

    obs_un, motion_mask, eval_targets = control_sampler.sample(batch, x0_un, length_mask)
    obs_un = obs_un.to(device)
    motion_mask = motion_mask.to(device).bool()

    x0 = normalizer.normalize(x0_un)
    obs = normalizer.normalize(obs_un)

    z = torch.randn_like(x0)
    t = torch.rand(B, device=device).clamp(cfg.fm_t_min, cfg.fm_t_max)
    xt = fm_make_xt(x0, t, z)
    v_target = x0 - z

    # clean observation imputation
    x_in, mask_float = apply_control_imputation(xt, obs, motion_mask)
    model_in = torch.cat([x_in, mask_float], dim=-1)

    cond = build_condition(batch, cfg, device)
    cond = maybe_drop_condition(cond, cfg)

    pred_v = model(model_in, t, cond, length_mask=length_mask)

    # Because x_in has clean values at masked dims, do FM velocity loss mainly on unmasked dims.
    valid = length_mask[..., None].to(x0.dtype)
    unmasked = (1.0 - mask_float) * valid
    fm_loss = masked_mean((pred_v - v_target).pow(2), unmasked.expand_as(pred_v)) * cfg.w_fm

    x0_hat = fm_clean_from_velocity(x_in, pred_v, t, observed_motion=obs, motion_mask=motion_mask)

    # contact interpretation: if model output has no contact logits head, use x0_hat contact as probability.
    # Better architecture: add separate contact logits head and use BCE.
    x0_hat[..., S.foot_contact] = x0_hat[..., S.foot_contact].clamp(0.0, 1.0)

    l_ctrl_raw = raw_control_loss(x0_hat, obs, motion_mask, length_mask, weight=cfg.w_control_raw)
    x0_hat_un = normalizer.unnormalize(x0_hat)

    l_fk = fk_consistency_loss(x0_hat_un, kin, length_mask, weight=cfg.w_fk)
    l_foot = foot_lock_loss(x0_hat_un, x0_un, kin, length_mask, cfg.foot_contact_to_joint_ids, cfg.fps, cfg.w_foot_lock)
    l_ground = ground_loss(x0_hat_un, x0_un, kin, length_mask, cfg.foot_contact_to_joint_ids, cfg.ground_y, cfg.w_ground)

    l_global = x0_hat.new_tensor(0.0)
    for b in range(B):
        l_global = l_global + global_position_control_loss(x0_hat_un, eval_targets[b], kin, b, weight=cfg.w_control_global)
    l_global = l_global / B

    loss = fm_loss + l_ctrl_raw + l_fk + l_foot + l_ground + l_global

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    return {
        "loss": loss.detach(),
        "fm": fm_loss.detach(),
        "control_raw": l_ctrl_raw.detach(),
        "fk": l_fk.detach(),
        "foot_lock": l_foot.detach(),
        "ground": l_ground.detach(),
        "global_ctrl": l_global.detach(),
    }
```

建议：如果第一版目标是稳，先用 DDPM clean x0 prediction；FM 版本可以第二阶段再做。

---

## 10. Control sampler / curriculum

不要 uniform random feature mask。按真实使用方式采样。

### 10.1 Phase 1：无控制自然先验

```text
observed_motion = 0
motion_mask = 0
text/action/style 可以保留
目标：自然 motion prior、contact prior、FK consistency
```

建议训练到：

```text
unconditional / text-conditioned motion 看起来自然
foot contact BCE 稳定下降
foot skate metric 可接受
FK consistency 不爆
```

### 10.2 Phase 2：控制训练

推荐 pattern 概率起点：

```python
CONTROL_PATTERN_PROBS = {
    "none": 0.10,
    "root_sparse": 0.10,
    "root_dense_path": 0.10,
    "root_heading": 0.05,
    "single_end_effector": 0.15,
    "two_hands": 0.10,
    "hands_feet_subset": 0.15,
    "feet_or_contact": 0.10,
    "full_pose_sparse": 0.10,
    "mixed_root_ee_or_fullpose": 0.05,
}
```

Keyframe 数量 curriculum：

```text
0% - 20% Phase 2: 1-3 anchors
20% - 60%:         1-10 anchors
60% - 100%:        1-20 anchors, mixed constraints 增加
```

真实模式优先：

```text
root only
root + heading
root dense path
root + one hand
root + two hands
root + hands + feet
五末端任意子集
full-pose start/end
full-pose sparse keyframes
contact / foot-lock constraints
```

### 10.3 从 x0 采样约束

训练时约束来自 ground truth x0_un，因此可以精确构造 observed/mask 和 global eval targets。

```python
class HY273ControlSampler:
    def __init__(self, probs, endpoint_joint_ids, rng=None):
        self.probs = probs
        self.endpoint_joint_ids = endpoint_joint_ids
        self.rng = rng

    def sample(self, batch, x0_un, length_mask):
        B, T, D = x0_un.shape
        obs, mask = make_empty_conditions(B, T, x0_un.device, x0_un.dtype)
        eval_targets = []
        for b in range(B):
            valid_T = int(length_mask[b].sum().item())
            pattern = self.sample_pattern()
            targets_b = ConstraintEvalTargets(global_joint_pos_targets=[])

            if pattern == "none":
                pass
            elif pattern == "root_sparse":
                self._sample_root_sparse(x0_un, obs, mask, targets_b, b, valid_T)
            elif pattern == "single_end_effector":
                self._sample_single_ee(x0_un, obs, mask, targets_b, b, valid_T)
            # add other patterns
            else:
                self._sample_mixed(x0_un, obs, mask, targets_b, b, valid_T)

            eval_targets.append(targets_b)
        return obs, mask, eval_targets

    def sample_pattern(self):
        raise NotImplementedError
```

核心工具：从 HY273 reconstruct global joint positions。

```python
def hy273_global_joints_from_feature(x_un: torch.Tensor):
    """x_un: [B,T,273] -> [B,T,22,3] global joints from joints_pos + smooth_root."""
    smooth = x_un[..., S.smooth_root]
    jp = x_un[..., S.joints_pos].reshape(*x_un.shape[:2], S.J, 3).clone()
    jp[..., 0] += smooth[..., None, 0]
    jp[..., 2] += smooth[..., None, 2]
    return jp
```

Example: single end-effector constraint from GT。

```python
def sample_frames(valid_T, num_keyframes, device):
    if num_keyframes >= valid_T:
        return torch.arange(valid_T, device=device)
    # include sparse random frames; optionally force endpoints
    return torch.randperm(valid_T, device=device)[:num_keyframes].sort().values


def apply_gt_end_effector_constraint(x0_un, obs, mask, targets_b, b, frames, joint_ids):
    """
    x0_un: [B,T,273] unnormalized
    frames: [K]
    joint_ids: [M]
    """
    global_pos = hy273_global_joints_from_feature(x0_un)[b, frames[:, None], joint_ids[None, :], :]  # [K,M,3]
    smooth_root_xz = x0_un[b, frames, S.smooth_root][:, [0, 2]]

    c = JointPositionConstraint(
        frames=frames,
        joint_ids=joint_ids,
        global_positions=global_pos,
        smooth_root_xz_ref=smooth_root_xz,
    )
    apply_joint_position_constraint(obs, mask, b, c)

    targets_b.global_joint_pos_targets.append(
        GlobalPositionEvalTarget(frames=frames, joint_ids=joint_ids, global_positions=global_pos, weight=1.0)
    )
```

---

## 11. 模型架构占位

这一节由你们之后补。训练代码只要求模型满足下面接口之一。

### 11.1 DDPM clean x0 prediction 接口

```python
class UserMotionDenoiser(torch.nn.Module):
    def forward(self, model_input, timesteps, condition, length_mask=None):
        """
        model_input: [B,T,546] = concat(noisy_or_imputed_hy273, motion_mask)
        timesteps: [B] long
        condition: arbitrary dict/object
        length_mask: [B,T]
        return:
          pred_clean: [B,T,273]
          contact dims are logits
        """
        raise NotImplementedError
```

### 11.2 Flow Matching velocity prediction 接口

```python
class UserMotionFlow(torch.nn.Module):
    def forward(self, model_input, t_continuous, condition, length_mask=None):
        """
        model_input: [B,T,546]
        t_continuous: [B] float in [0,1]
        return:
          pred_velocity: [B,T,273]
        """
        raise NotImplementedError
```

### 11.3 架构待补

```text
TODO(user):
- backbone 类型：Transformer / DiT / U-Net / Mamba / hybrid
- root/body 是否分支
- text/action/style condition 形式
- timestep embedding 形式
- contact 是否单独 head
- 是否支持 separated CFG
- 是否做 root-local body conditioning
- 是否做 extra/register tokens
```

---

## 12. Sampling / inference

### 12.1 DDPM/DDIM sampling 伪代码

```python
@torch.no_grad()
def sample_ddpm(model, normalizer, schedule, constraints, num_frames, cfg, device, condition=None):
    B = constraints.batch_size
    obs_un, mask, eval_targets = constraints.to_observed_motion(num_frames, device)
    obs = normalizer.normalize(obs_un)

    x = torch.randn(B, num_frames, S.D, device=device)
    length_mask = torch.ones(B, num_frames, dtype=torch.bool, device=device)

    for step in reversed(range(schedule.num_steps)):
        t = torch.full((B,), step, dtype=torch.long, device=device)
        model_in = build_model_input(x, obs, mask)

        pred = model(model_in, t, condition, length_mask=length_mask)
        x0_hat, _, _ = split_clean_prediction(pred)

        # optional: force exact clean values at controlled dims in x0 prediction
        m = mask.to(x0_hat.dtype)
        x0_hat = x0_hat * (1 - m) + obs * m

        x = ddim_or_ddpm_step(x, x0_hat, t, schedule, cfg)  # implement sampler

        # optional: clamp current state too, keeps controls stable throughout loop
        x = x * (1 - m) + obs * m

    x_un = normalizer.unnormalize(x)
    x_un = postprocess_motion_if_enabled(x_un, constraints, cfg)
    return x_un
```

### 12.2 Flow Matching sampling 伪代码

```python
@torch.no_grad()
def sample_fm(model, normalizer, constraints, num_frames, cfg, device, condition=None):
    B = constraints.batch_size
    obs_un, mask, eval_targets = constraints.to_observed_motion(num_frames, device)
    obs = normalizer.normalize(obs_un)

    x = torch.randn(B, num_frames, S.D, device=device)
    length_mask = torch.ones(B, num_frames, dtype=torch.bool, device=device)
    m = mask.to(x.dtype)

    # integrate t: 0 -> 1
    for i in range(cfg.num_fm_steps):
        t0 = i / cfg.num_fm_steps
        t = torch.full((B,), t0, device=device)
        dt = 1.0 / cfg.num_fm_steps

        x_in, mask_float = apply_control_imputation(x, obs, mask)
        model_in = torch.cat([x_in, mask_float], dim=-1)
        v = model(model_in, t, condition, length_mask=length_mask)

        # Euler; replace by Heun/RK if needed
        x = x + dt * v
        x = x * (1 - m) + obs * m

    x_un = normalizer.unnormalize(x)
    x_un = postprocess_motion_if_enabled(x_un, constraints, cfg)
    return x_un
```

### 12.3 Separated CFG

如果你们有 text/style/action + control 两类条件，建议保留 Kimodo 的 separated CFG 思路。

```text
pred = pred_uncond
     + w_text    * (pred_text_only    - pred_uncond)
     + w_control * (pred_control_only - pred_uncond)
```

实现注意：

```text
text_only:    text condition on, motion_mask zero
control_only: text dropped, motion_mask on
uncond:       text dropped, motion_mask zero
```

这能让 inference 时单独调：

```text
w_control ↑ : 更听 root/末端/full pose anchors
w_text ↑    : 更听文本/动作语义
```

---

## 13. Postprocess 接口

神经网络不要承担 100% exact hit 和 0 foot skate。第一版就要留 postprocess。

```python
def postprocess_motion_if_enabled(x_un: torch.Tensor, constraints, cfg):
    if not cfg.enable_postprocess:
        return x_un

    # TODO:
    # 1. HY273 -> global positions / global rotations / local rotations / root translation
    # 2. detect or use predicted foot contacts
    # 3. foot locking: stance foot target averaged over contact segment
    # 4. IK or constrained optimization:
    #    - keep full-pose anchors exact
    #    - keep end-effector anchors exact or near-exact
    #    - keep root path within root_margin
    #    - reduce stance foot velocity
    # 5. convert corrected motion back to HY273
    raise NotImplementedError
```

最小可用 postprocess：

```text
1. contact segment 检测：contact_prob > threshold 且连续 >= N 帧
2. 每段 stance foot target = segment 内 foot global position median/mean
3. 对 root translation 做小幅 correction，使 stance foot 接近 target
4. 对腿部做 IK，恢复膝/踝姿态
5. 对受控手/脚/full pose anchor 重新 project
```

---

## 14. 验收指标

每次训练/采样必须记录：

```text
1. root path error
   mean/max L2 in xz at root-controlled frames

2. end-effector error
   mean/max L2 for head/hand/foot target frames

3. full-pose keyframe error
   MPJPE at full-pose controlled frames
   rotation geodesic error if rotation controlled

4. foot skate
   contact=1 时 foot global speed mean/max
   foot_skate_ratio = speed > threshold 的 contact frames 比例

5. contact consistency
   contact=1 时 foot height near ground
   contact=0 时不强行压地

6. FK consistency
   FK(rotations, root) vs HY273 joints_pos reconstructed global positions

7. transition smoothness
   anchor 附近 velocity/acceleration/jerk

8. diversity / naturalness
   同一约束多 sample 的合理变化
```

建议 hard gates：

```text
controlled full-pose MPJPE: 目标 < 2-3 cm，postprocess 后更低
end-effector position error: 目标 < 3 cm，postprocess 后更低
root xz error: 目标 < 3 cm
stance foot speed: 目标尽量 < 2-5 cm/s
```

阈值需要按骨架、单位、采样率实测修正。

---

## 15. Agent 实施清单

建议文件结构：

```text
motion/
  hy273_slices.py
  hy273_normalizer.py
  hy273_kinematics.py
  hy273_constraints.py
  hy273_control_sampler.py
  hy273_losses.py
  diffusion_schedule.py
  train_ddpm_raw.py
  train_fm_raw.py
  sample_raw.py
  postprocess_contact_ik.py
  metrics_control_foot.py

tests/
  test_hy273_slices.py
  test_normalizer_contact_not_zscore.py
  test_constraint_builder_root.py
  test_constraint_builder_end_effector.py
  test_imputation.py
  test_fk_consistency_shapes.py
  test_foot_lock_loss.py
```

实施顺序：

```text
Step 1. 写 HY273Slices + normalizer
Step 2. 写 constraint builder，先支持 root / full-pose / one EE
Step 3. 写 imputation 单测
Step 4. 写 DDPM clean x0 training step，不接复杂模型，先 dummy model 跑通 shape
Step 5. 接真实模型架构
Step 6. 加 component losses + contact BCE
Step 7. 加 FK consistency + foot lock
Step 8. 加 Phase 2 control sampler curriculum
Step 9. 写 DDPM sampling + step-wise clamp
Step 10. 写 metrics
Step 11. 写 postprocess
Step 12. 再考虑 Flow Matching 版本
```

---

## 16. 关键陷阱

```text
1. 不要把 foot_contact z-score。
2. 不要把 global 手/脚 target 直接塞进 joints_pos；必须减 smooth_root xz。
3. 如果没有 smooth_root_ref，不要做 raw HY273 joint-position imputation。
4. 不要只看 raw feature loss；必须看 global FK/control metric。
5. 不要期待网络输出 exact hit；sampling clamp + postprocess 是系统的一部分。
6. Flow Matching 下 clean imputation 会破坏 masked dims 的 velocity target；velocity loss 应主要算 unmasked dims。
7. position 和 rotation 同时生成会不一致；必须加 FK consistency。
8. 脚滑不是 smoothness 问题；必须用 contact + foot velocity + ground height。
9. 训练 mask pattern 要贴近真实 UI，不要均匀随机 mask。
10. 多段/多 prompt/长序列过渡，需要把上一段末尾 full-pose/EE 作为下一段开头 overlap constraints。
```

---

## 17. 和前置 HY273 转换文档的接口

本文依赖前置文档交付：

```text
convert_hy201_to_hy273.py
convert_hy273_to_hy201.py
HY273 Mean/Std, contact dims excluded
round-trip validation
contact sanity validation
```

训练侧只接受：

```text
x0_hy273_un: [T,273]
length
optional text/action/style
optional cached global joints/rotations for faster control sampling
```

如果 dataset 里缓存：

```text
global_joint_pos: [T,22,3]
global_joint_rot6d: [T,22,6]
```

control sampler 会简单很多，且能避免重复 FK。
