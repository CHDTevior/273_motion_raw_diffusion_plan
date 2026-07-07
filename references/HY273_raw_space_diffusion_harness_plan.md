# HY273 Raw-Space Diffusion Harness Plan

本文是实施前方案，不是代码实现。目标是在确认后把项目从 VQ/code-space 训练路径切到 HY273 raw data space diffusion/control harness。

## 0. 结论

采用 A1 路线：

```text
生成变量: normalized HY273 raw motion, shape [B, T, 273]
控制协议: same-space observed_motion + binary motion_mask, shape [B, T, 273]
第一版模型目标: DDPM clean x0 prediction
第一版架构: 复用当前 FrameMotionTextDiT 的 text/time/frame-token denoiser 设计
第一版不做: VQ tokenizer, RVQ code target, terminal logits, latent nearest decode
```

第一版先做 Kimodo-like raw-space imputation control，不把旧 KV adapter 作为必须路径。原因是 raw-space 控制已经能把确认帧/确认维度作为 clean anchor 注入每个 denoise step；旧 KV adapter 可以作为后续可选增强，而不是第一版跑通训练的依赖。

## 1. 已调研代码

外部仓库：

```text
Kimodo:
  local: /mnt/afs/mogeflow-control/external_repos/kimodo
  remote: https://github.com/nv-tlabs/kimodo
  commit: 6bb5848

KV-Control:
  local: /mnt/afs/mogeflow-control/external_repos/KV-Control
  remote: https://github.com/CHDTevior/KV-Control
  commit: 0b94908

sigraph-asia-kvControl harness:
  local: /mnt/afs/mogeflow-control/external_repos/sigraph-asia-kvControl
  remote: https://github.com/CHDTevior/sigraph-asia-kvControl
  commit: 7f7dd0e
```

关键参考点：

```text
HY273 方案文档:
  outside_doc/HY273_raw_space_diffusion_training_control_impl.md:12
  outside_doc/HY273_raw_space_diffusion_training_control_impl.md:95
  outside_doc/HY273_raw_space_diffusion_training_control_impl.md:456
  outside_doc/HY273_raw_space_diffusion_training_control_impl.md:507

Kimodo representation:
  external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:34
  external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:76
  external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:85
  external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:90

Kimodo control imputation:
  external_repos/kimodo/kimodo/model/twostage_denoiser.py:98
  external_repos/kimodo/kimodo/model/twostage_denoiser.py:102
  external_repos/kimodo/kimodo/model/twostage_denoiser.py:103

Kimodo global joint condition rule:
  external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:284
  external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:289
  external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:291
  external_repos/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py:295

Kimodo DDPM/DDIM:
  external_repos/kimodo/kimodo/model/diffusion.py:12
  external_repos/kimodo/kimodo/model/diffusion.py:96
  external_repos/kimodo/kimodo/model/diffusion.py:113

Kimodo separated CFG:
  external_repos/kimodo/kimodo/model/cfg.py:94
  external_repos/kimodo/kimodo/model/cfg.py:126
  external_repos/kimodo/kimodo/model/cfg.py:127

Kimodo metrics:
  external_repos/kimodo/kimodo/metrics/constraints.py:58
  external_repos/kimodo/kimodo/metrics/constraints.py:67
  external_repos/kimodo/kimodo/metrics/constraints.py:73
  external_repos/kimodo/kimodo/metrics/foot_skate.py:31
  external_repos/kimodo/kimodo/metrics/foot_skate.py:80
  external_repos/kimodo/kimodo/metrics/foot_skate.py:136

当前项目 denoiser:
  models/codeflow/dit_blocks.py:607
  models/codeflow/dit_blocks.py:633
  models/codeflow/dit_blocks.py:637
  models/codeflow/dit_blocks.py:641
  models/codeflow/dit_blocks.py:708
  models/codeflow/motion_code_flow.py:176
  models/codeflow/motion_code_flow.py:194
  models/codeflow/motion_code_flow.py:362
  models/codeflow/motion_code_flow.py:389
```

## 2. Tensor Information Flow

训练时：

```text
raw sample
  x0_un: [B,T,273]
  text:  list[str] or optional empty text
  lengths: [B]
        |
        v
HY273Normalizer
  continuous dims 0:269 z-score
  contact dims 269:273 keep 0/1
        |
        v
x0: [B,T,273]
        |
        +---------------------------+
        |                           |
        v                           v
DDPM q_sample                  ControlSampler
  eps ~ N(0,I)                   obs_un, mask
  t ~ randint[0,N)               root / endpoints / fullpose / mixed
  xt = sqrt_ab*x0 + sqrt_1m_ab*eps
        |                           |
        |                           v
        |                       obs = normalize(obs_un)
        |                       mask: [B,T,273] bool
        +-------------+-------------+
                      |
                      v
Kimodo-style imputation
  x_in = xt * (1-mask) + obs * mask
  model_in = concat(x_in, mask.float)
  model_in: [B,T,546]
                      |
                      v
HY273RawDenoiser
  input_proj 546 -> hidden
  FrameMotionTextDiT(text tokens, timestep embedding, length mask)
  output_proj hidden -> 273
                      |
                      v
pred
  pred[...,0:269] = normalized clean continuous x0
  pred[...,269:273] = contact logits
                      |
                      v
loss
  raw clean loss on 0:269
  contact BCE on 269:273
  masked control loss
  optional unnormalized FK / foot / ground losses
```

采样时：

```text
x_T ~ N(0,I), shape [B,T,273]
obs, mask from user constraints
for ddim step k = K-1 ... 0:
    x_t = x_t * (1-mask) + obs * mask
    model_in = concat(x_t, mask)
    pred = denoiser(model_in, t, text)
    x0_hat = continuous pred + sigmoid(contact logits)
    x0_hat = x0_hat * (1-mask) + obs * mask
    x_{t-1} = DDIM(x_t, x0_hat, t)
    x_{t-1} = x_{t-1} * (1-mask) + obs * mask
unnormalize -> HY273
decode HY273 -> joints / rotations / contacts
optional contact-aware postprocess
```

这里的关键是：控制不再是最后对结果修一下，而是在每个 denoise step 里参与预测输入和 clean estimate。

## 3. HY273 Contract

统一 slice 必须集中定义，不允许散落 magic number：

```text
smooth_root:          0:3       [3]
root_heading:         3:5       [2]
joints_pos:           5:71      [22*3]
joints_global_rot6d:  71:203    [22*6]
joints_vel:           203:269   [22*3]
foot_contact:         269:273   [4]
```

shape:

```text
x0_un:           [B,T,273] unnormalized HY273
x0:              [B,T,273] normalized HY273
observed_motion: [B,T,273] same normalized convention as current x_t
motion_mask:     [B,T,273] bool or 0/1
length_mask:     [B,T] True = valid frame
```

normalization:

```text
continuous: 0:269 z-score
contact:    269:273 keep binary 0/1
```

## 4. Proposed Module Tree

建议新增路径，不污染旧 VQ/codeflow 主路径：

```text
models/raw_motion/
  __init__.py
  hy273_slices.py          # HY273Slices and feature index helpers
  hy273_normalizer.py      # continuous z-score, contacts unchanged
  hy273_constraints.py     # root / endpoints / fullpose / contact -> obs+mask
  hy273_control_sampler.py # training-time random control patterns
  hy273_kinematics.py      # HY273 -> global joints / FK consistency utilities
  hy273_losses.py          # raw, contact, control, FK, foot, ground losses
  diffusion_schedule.py    # cosine DDPM + DDIM sampler
  raw_dit.py               # HY273RawDenoiser based on FrameMotionTextDiT
  raw_diffusion.py         # training/sampling wrapper
  metrics_control_foot.py  # control + foot-skate + transition metrics

data/
  hy273_dataset.py         # text + raw HY273 sequence loading

scripts/launch/
  train_hy273_raw_ddpm_ddp4.sh
  eval_hy273_raw_control_ddp4.sh

train_hy273_raw_ddpm.py
sample_hy273_raw.py
eval_hy273_raw_control.py

tests/raw_motion/
  test_hy273_slices.py
  test_hy273_normalizer.py
  test_hy273_constraints.py
  test_hy273_imputation.py
  test_raw_dit_forward.py
  test_ddim_control_clamp.py
```

若你希望把所有新增代码放在 `motion/` 而不是 `models/raw_motion/`，实施前可以改名；逻辑不受影响。

## 5. Data Harness

第一版 dataset contract：

```python
{
    "motion": FloatTensor[T,273],      # unnormalized HY273
    "length": int,
    "text": str,
    "name": str,
}
```

collate 后：

```text
x0_un:      [B,Tmax,273]
lengths:    [B]
length_mask:[B,Tmax]
texts:      list[str]
names:      list[str]
```

数据 loader 先支持两种来源：

```text
1. hy273_npz_dir:
   motions/<id>.npy or .npz
   texts/<id>.txt
   splits/train.txt, val.txt, test.txt

2. manifest json/jsonl:
   {"id": "...", "motion_path": "...", "text": "...", "length": ...}
```

真实数据没到之前，先写 synthetic/small smoke loader，只用于 shape、loss、DDP sanity，不把它当实验结果。

## 6. Model Harness

当前项目最适合复用的是 `FrameMotionTextDiT`：

```text
models/codeflow/dit_blocks.py:607
  Text-conditioned DiT over one structured motion token per frame.

models/codeflow/dit_blocks.py:633-639
  double-stream blocks + single-stream blocks.

models/codeflow/dit_blocks.py:708-765
  forward(motion tokens, text tokens, timestep cond, length mask, optional control_cond)
```

raw-space denoiser 第一版：

```text
HY273RawDenoiser
  text_encoder: reuse FrozenCLIPTextEncoder path from MotionCodeFlow
  timestep_embed: reuse TimestepEmbedder
  input_proj: Linear(546, hidden_size)
  backbone: FrameMotionTextDiT
  output_proj: Linear(hidden_size, 273)
```

forward:

```python
def forward(model_in, timesteps, texts, lengths, text_drop_prob=0.0):
    # model_in: [B,T,546] = concat(imputed x_t, mask)
    motion_tokens = input_proj(model_in)
    text_cond = encode_text(texts, drop_prob=text_drop_prob)
    cond = timestep_embed(timesteps) + text_pooled_proj(text_cond.pooled)
    hidden = frame_dit(
        motion=motion_tokens,
        text=text_token_proj(text_cond.tokens),
        cond=cond,
        motion_valid=length_mask,
        text_padding_mask=text_cond.padding_mask,
        motion_pos_ids=frame_pos_ids,
    )
    pred = output_proj(hidden)
    return pred * length_mask[...,None]
```

v1 不启用旧 `control_cond` KV adapter。v1.1 可以加一个实验开关：

```text
--raw_use_kv_adapter
  control_cond = concat(observed_motion, motion_mask) or compact endpoint/root feature
```

但这个不是第一版必需项。第一版先验证 raw imputation control 本身能否训起来和受控。

## 7. Training Harness

第一版采用 clean x0 DDPM：

```text
N_train_steps: 1000 diffusion steps
sampler: DDIM, eval default 32 steps
prediction: clean x0
continuous output: pred[...,0:269]
contact output: logits pred[...,269:273]
```

训练 step：

```text
1. x0_un, lengths, texts = batch
2. x0 = normalize(x0_un)
3. obs_un, mask = control_sampler(x0_un, lengths)
4. obs = normalize(obs_un)
5. t ~ randint(0, N)
6. xt = q_sample(x0, t, noise)
7. x_in = xt*(1-mask) + obs*mask
8. model_in = concat(x_in, mask)
9. pred = model(model_in, t, texts, lengths, text_drop_prob)
10. x0_hat, contact_logits, contact_prob = split_clean_prediction(pred)
11. x0_hat_clamped = x0_hat*(1-mask) + obs*mask
12. loss = weighted losses
```

loss first pass:

```text
L_cont_clean:
  SmoothL1(pred_cont, x0_cont) on valid frames, dims 0:269

L_contact:
  BCEWithLogits(pred_contact_logits, x0_contact) on valid frames

L_control:
  SmoothL1(x0_hat_clamped, obs) only on mask, valid frames
  This is mostly a diagnostic/weighting term because clamp guarantees exact sampler anchors.

L_vel_smooth:
  optional temporal velocity/acceleration smoothness on unmasked regions

L_fk / L_foot / L_ground:
  optional after HY273 kinematics is verified
```

推荐初始权重：

```text
cont_clean: 1.0
contact_bce: 0.2
control: 1.0
vel_smooth: 0.02
fk: 0.0 for first smoke, then 0.1
foot_lock: 0.0 for first smoke, then 0.05
ground: 0.0 for first smoke, then 0.02
```

训练阶段：

```text
Phase A: prior warmup
  control probability = 0.0
  purpose = learn natural HY273 distribution

Phase B: mixed control
  root trajectory, endpoints subset, fullpose keyframes, contact, mixed
  purpose = learn in-between natural transition under confirmed frames

Phase C: hard control
  sparse anchors + long gaps + mixed endpoint/root/fullpose
  purpose = evaluate real control use cases
```

DDP:

```text
torchrun --nproc_per_node=4 train_hy273_raw_ddpm.py ...
AMP enabled
gradient clip enabled
save latest every N steps
save full state: model, optimizer, scaler, scheduler, epoch, global_step, config, normalizer stats
resume from latest must be default-supported
```

## 8. Control Sampler

训练时只采样真实使用边界：

```text
root:
  smooth_root xz, optional y, optional heading

endpoints:
  five limb endpoints subset
  likely pelvis/root + L/R hands + L/R feet, exact joint ids from skeleton contract

fullpose:
  all joints positions, optional all global rot6d

mixed:
  root + endpoints
  root + fullpose keyframes
  endpoints + contact
```

采样模式概率第一版：

```text
none/prior:      0.20
root:            0.20
endpoints:       0.25
fullpose:        0.20
mixed/contact:   0.15
```

关键规则：

```text
global joint position -> HY273 joints_pos 时，必须有同帧 smooth_root_ref。
没有 root reference 时，不直接写入 HY273 raw feature；只能放到 global/FK loss 或先估计 root。
```

这点来自 Kimodo `create_conditions()` 的 global joint condition 约束。

## 9. Sampling And Evaluation

sampling options:

```text
--num_ddim_steps 32
--cfg_type none|regular|separated
--cfg_text_weight
--cfg_control_weight
--clamp_observed_each_step true
--postprocess none|contact_ik
```

第一版评估协议：

```text
Smoke-16:
  固定 16 个样本
  每类控制都跑：root / endpoints / fullpose / mixed
  输出可视化 npy + json metrics

Full control eval:
  root error
  endpoint error
  fullpose keyframe error
  uncontrolled transition smoothness
  foot skate from height
  foot skate from predicted contact
  foot skate ratio
  contact consistency
```

metric 实现参考 Kimodo：

```text
root/end-effector/fullbody errors:
  external_repos/kimodo/kimodo/metrics/constraints.py:58
  external_repos/kimodo/kimodo/metrics/constraints.py:67
  external_repos/kimodo/kimodo/metrics/constraints.py:73

foot skate:
  external_repos/kimodo/kimodo/metrics/foot_skate.py:31
  external_repos/kimodo/kimodo/metrics/foot_skate.py:80
  external_repos/kimodo/kimodo/metrics/foot_skate.py:136
```

用户之前强调脚和地面接触滑动最不能接受，所以 full eval 必须默认打印 foot skate，不只打印控制误差。

## 10. Implementation Order

确认后按这个顺序实施：

```text
Step 1. 数据契约和纯函数
  hy273_slices
  normalizer
  imputation
  constraint -> obs/mask
  unit tests

Step 2. DDPM schedule + sampler
  cosine beta
  q_sample
  DDIM clean-x0 step
  clamp-observed-each-step test

Step 3. raw DiT denoiser
  input/output projection
  reuse CLIP text encoder + FrameMotionTextDiT
  forward shape test on CPU/GPU

Step 4. train_hy273_raw_ddpm.py
  dataset interface
  synthetic smoke data
  DDP/AMP/resume
  one-batch overfit sanity

Step 5. control sampler + losses
  root/endpoints/fullpose/contact/mixed
  raw clean/contact/control losses
  first short GPU run

Step 6. sample/eval harness
  DDIM32
  fixed 16 sample smoke
  full control eval
  save metrics json and generated motions

Step 7. optional quality additions
  FK/foot/ground losses
  contact-aware postprocess interface
  separated CFG
  optional root-body/two-stage branch
  optional existing KV adapter branch
```

## 11. Current Blockers / Need Confirmation

这些不是阻塞写 harness skeleton 的问题，但会影响真实训练是否能开始：

```text
1. outside_doc 里目前没有 HY201_to_kimodo273_guide.md。
   原方案文档第 4 行引用了它，需要你之后补齐或确认 HY273 converter 的真实来源。

2. 需要真实 HY273 数据落盘格式。
   我可以先支持 npy/npz/manifest 三种格式，但真实数据字段需要最终确认。

3. 需要 HY273 skeleton contract。
   包括 22 joints 顺序、parent tree、foot joint ids、five endpoints ids、fps、坐标系。

4. 需要确认 text conditioning。
   默认复用当前 CLIP text encoder；如果新数据没有文本，可以支持 empty text / action label。

5. 需要确认第一版是否强制 two-stage root/body。
   我的建议是第一版不强制 Kimodo two-stage，先用我们当前 FrameMotionTextDiT 单阶段 raw denoiser。
   如果控制 root/body 耦合不好，再加 Kimodo-like root-first branch。
```

## 12. Recommendation

我建议第一版按下面最小闭环做：

```text
HY273 npy/manifest dataset
  -> normalizer
  -> DDPM clean x0
  -> FrameMotionTextDiT raw denoiser
  -> Kimodo-style obs/mask imputation
  -> root/endpoints/fullpose/mixed control sampler
  -> DDIM32 clamp-each-step sampling
  -> control + foot-skate evaluation
```

这个闭环最贴近你的新边界：不用 VQ，不依赖旧 code-space backbone checkpoint，但保留我们当前 denoising model 的设计资产。
