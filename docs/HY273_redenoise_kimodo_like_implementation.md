# HY273 `redenoise_kimodo_like` 实现说明

## 1. 目标与边界

这一版在 HY273 raw data space 中训练 rectified flow，但把去噪器拆成 Kimodo 风格的两个连续阶段：

1. root denoiser 先预测完整 clean global root；
2. 把该 root 预测转成 local root velocity/height；
3. body denoiser在同一次 forward 内预测 clean body；
4. Stage 1 学文本到自然动作，Stage 2 从同一 checkpoint 严格续训控制。

v1 正式控制协议只包括：

- root sparse waypoint / dense path；
- sparse full-pose position keyframe；
- 头、双手、双脚五末端点的非空子集；
- 上述任意两个 pattern 的组合。

v1 不声明 contact channel 控制，也不声明 endpoint rotation 控制。Contact 仍作为生成输出接受 BCE、foot-lock 等监督。

## 2. 数据契约

输入 motion 为 `[B,T,273]`：

```text
[0:3]      smooth global root xyz
[3:5]      global root heading [cos(yaw), sin(yaw)]
[5:71]     22 joint positions; xz relative to smooth root, y global
[71:203]   22 global rotations in cont6d
[203:269]  22 global joint velocities
[269:273]  four binary foot contacts (never z-score normalized)
```

训练 transform：整段先做首帧 root XZ origin shift，再做随机整段 yaw；joint positions 不做逐帧 heading canonicalization。统计只来自 train split，并用 `0/90/180/270` 度四点 yaw quadrature 对齐随机 yaw 分布。

实现：

- transform: `models/raw_motion/hy273_normalizer.py`
- stats builder: `tools/build_hy273_redenoise_stats.py`
- global-to-local root: `models/raw_motion/hy273_root_conditioning.py:22-103`

严格资产清单：

```text
files:       42,994
bytes:       58,953,101,623
manifest:    .../derived_stats/redenoise_kimodo_like_v1/training_assets.json
sha256:      ff8da22b41f440931c35a9c1a86291c07c134a25e55a677fc9e95534f3113e84
initial model sha256: efbdf65ce2adb26a71e87bcae9c9cc55e4260c5523d8495508a0bb669116302a
```

它覆盖 train motions、对应 text、split/source manifest、full/local-root stats、HYText index/manifest、全部 HYText shards，以及 FK loss 使用的 SMPL-X22 neutral skeleton。构建与验证逻辑在 `models/raw_motion/asset_integrity.py` 和 `tools/build_hy273_training_asset_manifest.py`。

## 3. 模型与 tensor information flow

核心模型：`models/raw_motion/kimodo_like_flow_dit.py:30-253`。

```text
clean HY273 x0 [B,T,273]
        |
        | random yaw + root-origin shift + normalize
        v
continuous x0[:269] ---- noise eps [B,T,269]
        |                    |
        +--- z_t = t*x0 + (1-t)*eps
                             |
observed obs [B,T,273] ------+---- mask m [B,T,273]
        |                          |
        +--> z_imp = m*obs + (1-m)*z_t/contact_aux
                                      |
                                      v
root input = concat(z_imp273, mask273) [B,T,546]
                                      |
                             root input projection
                                      |
                         Root FrameMotionTextDiT
                         + time + c_dir + HYText
                                      |
                                      v
                      clean global root_hat [B,T,5]
                                      |
                         detach during training only
                                      |
                         FP32 global -> local
                  [yaw_vel, world_vx, world_vz, root_y]
                              [B,T,4]
                                      |
                                      +--------------------+
                                                           |
body input = concat(local_root4, z_imp_body268, mask273) [B,T,545]
                                                           |
                                                  body input projection
                                                           |
                                                Body FrameMotionTextDiT
                                                + same time/c_dir/HYText
                                                           |
                                                           v
                                             clean body_hat [B,T,268]
                                                           |
                clean HY273 prediction = concat(root_hat5, body_hat268)
                                      [B,T,273]
```

关键约束：body bridge 使用完整 `root_prediction_raw`，绝不把 sparse root observation 拼进预测轨迹后再做 finite difference。对应代码为 `kimodo_like_flow_dit.py:223-232`。这样 body 与最终输出 root 描述同一条轨迹，避免 waypoint 两侧速度尖峰。

生产规模：

```text
hidden dim:                 1024
root DiT:                   3 double + 6 single blocks
body DiT:                   3 double + 6 single blocks
trainable parameters:       387,632,913
root/body backbone params:  188,918,016 each
text:                       cached Qwen3-8B tokens + CLIP ViT-L/14 pooled
```

## 4. 训练目标

网络输出语义固定为 clean x0 prediction；`redenoise_kimodo_like` 会拒绝 `prediction_type=velocity`。主 representation loss 可以在 velocity space 计算：

```text
x0_hat = network(...)
denom  = clamp(1-t, min=0.05)
v_hat  = (x0_hat - z_imp) / denom
v_tgt  = (x0 - z_imp) / denom
L_repr = semantic_weighted_MSE(v_hat, v_gt)
```

因此这是“x0 head + v-space training loss”，不是 velocity head。实现位于 `train_hy273_raw_flow.py:1281-1349`。

其他监督：

```text
0.10 * BCE(contact logits, GT contact)
0.01 * clean root velocity SmoothL1
0.01 * clean joint velocity SmoothL1
0.01 * foot-lock velocity loss on GT-contact frames
0.07 * FK position consistency, first 5K steps linear warmup
```

Stage 2 额外：

```text
0.25 * SmoothL1(x0_hat, observed target) on controlled continuous entries
0.00 * controlled-contact BCE  # v1 不提供 contact control
```

受控 entries 仍从主 representation loss 中排除，避免同一 entry 同时接受 flow target 和 hard-control target。

## 5. 两阶段训练

```text
Stage 1: text-only backbone pretraining
  global steps: 0 -> 200,000
  controls: none
  checkpoint: every 50,000 + latest
  optimizer/EMA: AdamW + EMA

Stage 2: control curriculum continuation
  strict resume: exactly step 200,000
  global steps: 200,000 -> 400,000
  preserves: model, optimizer, EMA, dataloader cursor, architecture, assets
  changes only: phase, control modes, controlled-entry loss
```

Curriculum 在 `train_hy273_raw_flow.py:1133-1158` 和 `hy273_constraints.py:143-236`：

```text
10%  no control
65%  one pattern
25%  union of two distinct patterns

sparse Kmax: 1 -> 20 over the 200K Stage-2 updates
K sampling: low-biased squared-uniform
root path: root XZ + heading, not root Y
endpoint: random non-empty subset of head/hands/feet + hidden root reference
```

最后一个实际 update 使用 progress=1.0，因此 Kmax=20 在训练内可达。

## 6. 推理与 separated CFG

ODE state 始终是一份未永久 clamp 的共享 state。每一步构造四个 batch branch：

```text
joint:    text + control overwrite
text:     text + no control mask
control:  empty text + control overwrite
empty:    empty text + no control mask
```

连续通道 guidance：

```text
D = D_empty
  + w_text * (D_text - D_empty)
  + w_ctrl * (D_control - D_empty)

defaults: w_text=3.5, w_ctrl=2.0
```

Contact 是 logits 而非与连续通道同构的 raw feature，默认取 joint branch，不放大 logits；可显式启用 contact CFG。实现和 oracle algebra tests 分别在 `sample_hy273_raw.py:82-270`、`tests/test_raw_flow_sampling.py`。

每一步仅 branch-local input overwrite，ODE update 本身不持久 clamp：

```text
state_k
  -> branch-local overwrite
  -> x0 predictions / separated CFG
  -> v = (x0_hat - state_k)/(1-t)
  -> state_{k+1} = state_k + dt*v
```

输出分开保存：

- `samples.npy`: raw pre-clamp，评估模型实际控制响应；
- `samples_exact_clamped.npy`: 最终系统 exact output；
- `final_clean_prediction.npy` 与各 CFG branch prediction 用于诊断。

## 7. 评估协议

`eval_hy273_raw_control.py` 默认：

```text
weight source:       EMA, strict load
seed:                3407
ODE steps:           32
text CFG:            3.5
control CFG:         2.0
control sampler:     Stage-2 distribution conditioned on having control
normalizer/assets:   checkpoint contract + content hash verification
```

同时报告 raw/exact 两套 endpoint feature error、endpoint FK error、root XZ error、FK consistency 和 foot skate。模型控制能力结论必须看 `raw_*`；`exact_*` 代表带最终 overwrite 的系统能力。

## 8. 续训与故障恢复

每个 checkpoint 包含：

```text
model
optimizer
EMA
resolved args contract
global step
exact next epoch/batch cursor
full normalizer mean/std/variance_eps
```

Stage 2 还要求显式提供 Stage-1 checkpoint SHA256。资产 manifest、模型结构、统计、batch/world size 和 cursor 不匹配时直接失败，不静默降级。

生产入口：

```text
scripts/launch/train_redenoise_kimodo_like_stage1_ddp8.sh
scripts/launch/train_redenoise_kimodo_like_stage2_control_ddp8.sh
```

launcher 会检查配置、资产、GPU 列表数量、每张卡是否空闲以及 run directory 是否已存在。正式保存周期保持 50K。

生产 launcher 不接受隐式 `MAX_STEPS/MAX_EPOCHS/SAVE_EVERY` 覆盖。短门禁必须显式使用：

```text
ALLOW_SHORT_PILOT=1 PILOT_MAX_STEPS=<N>
```

训练进程会在 YAML/CLI merge 后再次校验 `stage1_production/stage1_pilot` 或 `stage2_production/stage2_pilot`，Shell preflight 不能替代该最终合约。

## 9. 当前验证

```text
unit/integration tests:       66 passed
official local-root parity:   max abs error 7.39e-6
real HYText forward/backward: passed
DDP8 Stage-1 smoke:           passed, finite loss/backward/EMA/checkpoint
DDP8 Stage-2 transition:      passed, strict SHA/optimizer/EMA/cursor/normalizer resume
separated CFG ODE smoke:      passed, raw finite, exact clamp error ~1.49e-8
asset pin:                    42,994 files / 58.95 GB content-addressed
```

正式 200K Stage 1 只会在二次 `gpt-5.6-sol/max` 对抗审核通过后启动。
