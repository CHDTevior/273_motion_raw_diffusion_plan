# HY273 Loss and Root-Body Experiment Plan

**Problem**: 当前 one-stage HY273 raw-flow 的训练 loss 正常下降，但 200K 样本的 root/joint 运动速度只有同批 GT 的 46.5%/53.2%，root path 只有 56.3%，存在明显欠运动。当前模型也没有复现 Kimodo 的 interleaved root-to-body denoiser。

**Method thesis**: 保留 rectified flow 所需的 clean-x0 MSE 主目标，只改变各语义块的优化预算；再用一个独立的小权重 FK-position consistency 辅助项约束同一预测中的 position/rotation。loss 结论经从头训练确认后，才比较 one-stage 与 interleaved two-stage。

**Date**: 2026-07-10

## Decision Summary

1. 300K checkpoint 已保存为 `checkpoints/t2m/hy273_raw_flow_hml3d_stage1_x0_hytext_ddp8_from5k_20260709_095555/model/step_00300000.pt`。原 L0 主线已按用户要求在约 308K 停止并归档；未保存的 300K-308K 状态不参与实验。
2. 当前执行范围由用户收缩为 L2 和 L3，不再训练 L0/L1。300K EMA 可视化作为历史 baseline，而不是 matched continuation run。
3. L2/L3 均从随机初始化开始，使用相同 seed、相同 rank0 初始权重 SHA 和相同训练 trace；各使用 4 卡 DDP，从 global step 0 训练到 200K。
4. 保持等效 global batch 128、LR、optimizer、EMA、数据与随机 trace 一致；每 50K 保存并原子维护 `latest.pt`。
5. 由于省略 L1，本轮不能把收益单独归因于“per-block reduction”或“Kimodo semantic ratios”；只检验完整 L2 recipe 以及 L3 consistency 的增量价值。
6. scratch loss 校准、前 100 step 配对 trace、初始权重 SHA、真实 checkpoint/restart smoke 和冻结 launcher 未通过前，不启动两个长训。物理/文本评估器必须在第一个 50K checkpoint 用于模型判断前通过。
7. 不用 Smooth-L1/Huber 替换 rectified-flow 的 MSE 主目标。Huber 只允许作为小权重几何辅助项。
8. two-stage 和后续 control curriculum 保留为下一决策阶段，本轮先完成 L2/L3。

## What Kimodo Optimizes

Kimodo 是 clean-x0 DDPM。技术报告对各语义块分别做 Smooth-L1，并给出近似如下的权重：

```text
10 * root_position
 2 * root_heading
10 * joint_position
 3 * joint_velocity
10 * global_rotation_6d
 4 * foot_contact
 5 * FK(predicted rotation/root -> GT joint position)
```

这样设计有四个主要原因：

1. 每个块先独立 reduction，避免 132 维 rotation 仅因维数多就支配 5 维 root。
2. root path、直接 joint position 和可驱动骨架的 rotation 都是主任务，所以获得高权重；heading、velocity、contact 是独立语义，不与大块维数混算。
3. Smooth-L1 对 DDPM clean denoising 的大误差更稳健，但它估计的是 robust conditional location，不是条件均值。
4. FK-to-GT 从 rotation 分支直接监督实际骨架位置，减少“rotation 数值接近但 FK 位置仍偏”的情况。

Kimodo 的 two-stage 不是“先完整生成 root，再完整生成 body”两次采样，而是在每个 denoise step 内先预测 global root，再把它转成 local-root velocity 供 body 分支使用。

参考：

- Kimodo technical report, loss/training: https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf#page=10
- Kimodo technical report, one-stage ablation: https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf#page=14
- local source: `external_repos/kimodo/kimodo/model/twostage_denoiser.py:107-152`

## Why We Should Not Copy It Literally

本项目是 rectified flow，路径为：

```text
z_t = t * x0 + (1 - t) * noise
x0_hat = model(z_t, t, text, control)
v_hat = (x0_hat - z_t) / (1 - t)
```

MSE 的最优解是 `E[x0 | z_t, condition]`，因此上式对应标准 conditional flow field。若把主目标替换为 Huber，最优解变成 robust conditional location，不能再宣称它与标准 rectified-flow 条件均值等价。`2 * SmoothL1` 只能匹配零附近曲率，不能修复这个语义差异。

另有两点不能照抄：

1. 我们的 contact head 输出 logits，contact 不进入 continuous ODE state，所以继续用 BCEWithLogits，不改成 Kimodo 的 clean contact regression。
2. 我们更需要先约束同一份生成结果中的 `FK(rotation)` 与 `joint_position` 一致。Kimodo 的 FK-to-GT 可作为诊断，但下一轮不把它和 consistency 混成一个 loss。

## Current Loss Imbalance

当前实现对 269 个 continuous entries 统一求 mean MSE。若各维误差同量级，其隐式预算是：

```text
root position + heading    1.86%
joint position            24.54%
global rotation 6D        49.07%
joint velocity            24.54%
```

近期总 loss 中 continuous x0 MSE 约占 90.4%，contact 约 8.6%，三个动态项合计约 1%。因此 root 欠运动更可能来自主损失预算和模型结构，而不是“训练 loss 没有下降”。

## Proposed Loss Family

### Common contract

- 每个 block 在 valid frames 上独立 mean，包括该 block 的 feature 维。
- continuous 主目标始终是 MSE，保持 flow 条件均值语义。
- Stage 1 监督所有 continuous dims。
- Stage 2 主生成 MSE 监督 unmasked continuous dims；同时固定保留当前训练中的 `0.25 * SmoothL1(clean continuous prediction, observed)` 和 `0.05 * BCE(contact logits, observed contact)`，只在对应 masked dims 上计算。所有架构使用完全相同的 control loss。
- Contact 保持 `0.1 * BCEWithLogits`；现有三个 `0.01` dynamic 项保持不变，以隔离主变量。
- 对每个变体增加一个固定标量 `alpha`，在同一随机初始化、同一 calibration trace 上匹配 L0 的 output-gradient RMS，且必须在第一次 optimizer step 前完成。`alpha` 只匹配主表示项整体尺度，不动态更新。

定义五个 block MSE：

```text
m_root, m_heading, m_jpos, m_rot6d, m_vel
```

### L0: Current per-entry MSE

```text
L_repr = MSE over all 269 continuous entries
```

### L1: Equal-block MSE

```text
L_repr = alpha_equal * (
    m_root + m_heading + m_jpos + m_rot6d + m_vel
) / 5
```

L1 只回答“从 per-entry reduction 改为 per-semantic-block reduction 是否有效”。

### L2: Semantic-weighted MSE

```text
L_repr = alpha_semantic * (
    10 * m_root
  +  2 * m_heading
  + 10 * m_jpos
  + 10 * m_rot6d
  +  3 * m_vel
) / 35
```

L2 相对 L1 只增加 Kimodo 启发的语义权重，仍保持 MSE flow 目标。

### L3: L2 plus internal FK consistency

先在 clean prediction 上构造同一结果的两套世界坐标关节。实现必须走已审计的 helper，不能自行猜 pelvis/root：

```text
x0_hat_norm = concat(x0_cont_hat, sigmoid(contact_logits))
x0_hat_clamped_norm = clamp_clean_anchors(x0_hat_norm, obs_norm, mask)
x0_hat_clamped_un = normalizer.denormalize(x0_hat_clamped_norm)

J_pos = reconstruct_global_joints_from_features(x0_hat_clamped_un)
J_fk  = fk_positions_from_global_rot6d(x0_hat_clamped_un)

L_cons_raw = valid_frame_mean SmoothL1(
    (J_fk - J_pos) / 0.05m,
    beta=1.0,
)

L_total = L2 + lambda_cons * L_cons_raw
```

- Stage 1 的 `x0_hat_clamped == x0_hat`。
- Stage 2 必须先把 clean anchors clamp 到 `x0_hat`，再算 consistency。
- `fk_positions_from_global_rot6d` 从预测 position representation 恢复真实 pelvis translation；`[0:3] smooth_root` 不能直接当 pelvis。
- loss 在 denormalized meter space 计算，并严格排除 padded frames。
- 两侧都保留梯度；MSE 主项负责把两侧锚定到 GT，consistency 负责让它们落在同一骨架流形上。
- `FK-to-GT`、`J_pos-to-GT` 单独记录为 metric，不在本轮再加第二个 FK loss。
- `lambda_cons` 使固定 trace 上该项相对完整 L2 共同训练目标（semantic MSE + contact + 三个 dynamic auxiliaries）的 aggregate output-gradient RMS 为 5%-10%，且任何 timestep bin 的 aggregate RMS 比不超过 15%。选择过程只读取梯度，不得查看生成质量；无合法点则 L3 不启动。
- scratch seed 3407 上固定 `lambda_cons=0.07`，并在前 5K optimizer steps 从 0 线性 warm up 到该值。

## Claim Map

| Claim | Primary evidence | Falsification |
|---|---|---|
| C1: semantic block budgeting 缓解欠运动 | root/joint speed 和 root path 的 GT distribution distance 显著下降 | 只增加高频抖动，或文本/物理指标退化 |
| C2: internal FK consistency 改善双表示一致性 | FK-vs-position p95 显著下降，FK-to-GT 和 skate 不退化 | 两个预测一起偏离 GT，或 contact/skate 变差 |
| C3: interleaved root-body 本身有收益 | A2 优于参数匹配及计算匹配 one-stage | 收益可由参数量、FLOPs或训练时长解释 |

## B0: Preflight

训练启动前必须完成：scratch 梯度校准、相同初始权重 SHA、L2/L3 前 100 optimizer steps 的逐 rank/逐 microbatch trace hash、一轮真实 checkpoint 保存后恢复、冻结的 GPU/端口/步数/保存协议，以及 NaN/OOM/grad-clipping 检查。

以下评估门禁可与训练并行实现，但必须在 50K checkpoint 首次参与模型判断前完成；未通过时允许继续优化，不允许做模型优劣结论：

1. **EMA/raw loading**: 评估器可明确选择 raw 或 EMA，默认模型选择使用 EMA；短 fork 同时报告 raw。
2. **Physical metrics**:
   - joints 必须由 predicted rotations + predicted root 做 FK；
   - frame displacement 乘 `30 FPS * 100`，单位为 cm/s；
   - 报 mean/p95/max、skate ratio、contact occupancy、heuristic-contact skate、ground height 和 contact consistency；
   - 禁止“预测无 contact 时 skate=0”成为好结果。
3. **Representation metrics**: FK-vs-position、FK-to-GT、position-to-GT 分开统计，包含 feet/hands/all 的 mean/p95/max。
4. **Motion metrics**: root speed、joint speed、root path、multi-scale velocity、acceleration、jerk 和 high-frequency spectral energy。
5. **Text evaluator bridge**: K273 -> HumanML3D evaluator 的 GT parity/audit 必须通过，才能用 FID、R precision 和 matching score选模型。门槛固定为 joint reconstruction max error `<1e-4 m`、同一 GT 经两条 evaluator 输入路径的 motion-embedding cosine `>0.999`，且 R@1/2/3 与 reference GT pipeline 的绝对差均 `<0.01`；缺少 reference pipeline 即保持 blocked。
6. **Control metrics**: pre-clamp clean prediction 与 post-clamp system output 分开；按 root/endpoints/fullpose/mixed 分层，并统计 anchor 邻域和未控制帧过渡。
7. **Deterministic stateless trace**: RNG 由 seed/rank/global-step/micro-step/stream 唯一决定；dataset crop/caption 由 seed/epoch/index/stream 唯一决定。所有分支记录前 100 steps tensor SHA256 并逐字节比较。self-conditioning 当前关闭；未来开启时还要覆盖启用 bit 和 first-pass stochastic state。
8. **Gradient audit**: output-gradient RMS、parameter-gradient RMS、分量间 cosine、五个 timestep bins、grad clipping hit rate。
9. **Regression assertions**: contact auxiliary 在 train/sample 均保持 `[0,1]` 且使用同一 blend 语义；train/sample/eval 的 root-origin/c_dir transform 一致。

已经确认的事实：当前 FK helper 在前 256 条 test GT 上复现 K273 position，all-joint mean error 为 `6.2e-8 m`，max 为 `1.93e-6 m`。这只证明骨架/通道语义正确，不代表现有 skate evaluator 已正确。

## B1: User-Selected L2/L3 Scratch Runs

- 两个分支都从 step 0 随机初始化；rank0 初始模型 SHA 必须与 calibration 记录一致，且 L2/L3 彼此一致。
- 所有分支使用 B0 的 deterministic trace。
- 每个分支绑定 4 张空闲 A100；使用 microbatch 16、gradient accumulation 2，保持 global batch `4*16*2=128`。
- 模型、HYText、AdamW、LR、t distribution、transforms、EMA 和 ODE32 全部不变。
- 两个 run 各训练 200K optimizer steps：从 global step 0 到 200K。
- 保存 global steps 50K/100K/150K/200K，并原子更新 `latest.pt`；checkpoint 保存精确的下一数据游标，支持任一分支单独断线后恢复配对 trace。

Runs:

```text
R-L2  semantic-weighted MSE                 200K, GPU 0-3
R-L3  semantic-weighted MSE + consistency   200K, GPU 4-7
```

Evaluation:

- HumanML3D val 中固定 1024 clips 作为 development set；其中前 256 及一个 paired generation seed只做快速淘汰。
- L2 和 L3 在完整 1024 development clips、3 个 paired generation seeds 上做 matched comparison。300K archive 只作历史参照，因训练预算与初始化不同，不参与 L2/L3 因果归因。
- raw 与 EMA 都报告；ODE64 只做 sampler sensitivity，不参与主选择。

Calibration record before launch:

- scratch seed 3407、16 个固定 train clips、五个 timestep bins 的最终报告位于 `run_logs/hy273_l2_l3_calibration_scratch_seed3407_n16_final.json`。
- 完整 scratch launch preflight 报告位于 `run_logs/hy273_l2_l3_scratch_preflight_report.json`，包含初始 SHA、四个 rank 的 100-step paired trace、NaN/OOM/grad clipping、checkpoint 游标和 exact restart 结果。
- 启动字节契约位于 `run_logs/hy273_l2_l3_scratch_source_manifest.sha256`；生产 launcher 在 GPU 检查后、训练进程创建前校验全部训练代码、配置、数据元信息、HYText cache 元信息和 calibration artifact，并要求 preflight report 绑定同一 manifest SHA。
- 初始模型 SHA256：`808a1639f7ec134887ce07b3d2634849dccfe68e6441bac0addba4f4cc579e59`。
- 按全部梯度元素平方和汇总：`alpha_semantic=0.3247346107`。
- `lambda_cons=0.07`；相对完整 L2 共同目标的全局 aggregate 梯度比为 `7.78%`，最高 timestep-bin 为 `11.27%`。

预注册主指标：

```text
For k in {root_speed, joint_speed, root_path}:
  D_k = W1(log(metric_gen_k + 1e-6), log(metric_gt_k + 1e-6))

D_motion = mean(D_root_speed, D_joint_speed, D_root_path)
```

同时报告每项的 p10/p25/p50/p75/p90 ratio；median ratio 只作解释，不能替代 `D_motion`。这样 variance、tails 或 mode collapse 变差时不能靠一个 median 过门。

Promotion gate:

1. `D_motion` 相对归档 300K baseline 至少降低 10%，paired block-bootstrap 95% CI 不跨 0，且三个 `D_k` 中任一项不得退化超过 5%。
2. p10-p90 quantile ratios整体朝 1.0 移动；不得靠 acceleration/jerk/high-frequency energy 超过 GT 20% 来提高速度。
3. 文本 non-inferiority 使用三个同时必须通过的 one-sided 95% CI：`upper(FID_candidate/FID_baseline - 1) < 0.05`、`upper(Matching_candidate/Matching_baseline - 1) < 0.05`、`lower(R@3_candidate - R@3_baseline) > -0.02`。这是 intersection-union gate，不从中挑最好看的一个。
4. heuristic-contact FK skate、contact coverage/consistency 无显著退化。
5. L3 还必须让 FK-vs-position p95 至少降低 20%，且 FK-to-GT 不退化。
6. 无 NaN；grad clipping hit rate低于 5%。

L3 的增量结论必须直接对 L2 做 paired comparison；若 consistency 指标改善但 motion/text/skate 退化，则选择 L2。

## B1.5: User-Directed L3 Clean-Head JiT V-Loss Comparator

After the 100K visual review, the user selected L3 over L2 and requested a second
independent L3 run using the clean-prediction/velocity-loss recipe from JiT. L2
was stopped at approximately 137K; its durable 50K and 100K checkpoints remain
archived. The original L3 run continues unchanged on GPU 4-7.

The new comparator starts from random initialization on GPU 0-3 and keeps L3's
data, architecture, HYText conditioning, semantic block weights, FK consistency,
optimizer, LR, global batch 128, EMA, 200K budget, and 50K checkpoint cadence.
Its changed denoising contract is:

```text
z_t      = t * x0 + (1 - t) * noise
x0_hat   = model(z_t, t, text)
v_target = (x0 - z_t)     / max(1 - t, 0.05)
v_pred   = (x0_hat - z_t) / max(1 - t, 0.05)
L_repr   = semantic_weighted_MSE(v_pred, v_target)

t = sigmoid(N(P_mean=-0.8, P_std=0.8))
```

The timestep distribution and `t_eps=0.05` follow the public JiT implementation.
This is therefore a JiT-recipe comparator, not a strict loss-only ablation against
the original L3, which uses `logit-N(0,1)` and x0-space MSE.

Pre-optimizer calibration on 4096 real HumanML3D samples fixed:

```text
representation_scale = 0.09397019716051493
FK lambda             = 0.07
aggregate FK gradient ratio = 7.044%
worst fixed-timestep ratio  = 8.006%
initial model SHA256         = 808a1639f7ec134887ce07b3d2634849dccfe68e6441bac0addba4f4cc579e59
```

A real-cache DDP4 500-step pilot is a mandatory launch gate. It completed with
500 finite logged steps, zero gradient-clipping hits, gradient-norm p95/p99/max
of 0.495/0.704/0.956, and flow loss decreasing from 0.391 at step 1 to 0.0207 at
step 500. The machine-readable gate is
`run_logs/hy273_l3_vloss_jit_preflight_report.json`.

## Deferred: Multi-Seed Loss Confirmation

- 当前 L2/L3 已经是 seed 3407 的独立 scratch runs。本节仅保留未来的多训练 seed 复核，不再把当前实验描述成 continuation。
- 固定比较 L0 与 provisional winner；若 L2/L3 机制结果相反且都通过，可保留两者。
- 每个 loss 使用 training seeds `3407, 3408, 3409`，从随机初始化训练 50K。每个 seed 单独估计 sample-level CI，训练 seed 不与生成样本混池；最终报告三 seed mean/std、range，并要求三个 seed 的主指标改善方向一致。
- 在 10K/25K/50K 使用同一 validation protocol；模型选择只看预注册的 50K endpoint。
- 每个 scratch seed 都在第一次 optimizer step 前，用同一 calibration trace 重新校准 `alpha`；consistency 的 `lambda_cons` 也在随机初始化重新校准并做 5K warm up。持续记录各 timestep bin 梯度比例，超过 15% 立即判该 run 无效而不是临时调参。
- 只有 3-seed 50K 结果通过 B1 的 promotion/non-inferiority gates，候选 loss 才成为 architecture loss。
- 若结果不清楚或不同 seed 方向不一致，保留简单的 L0。

## B2: Architecture Isolation

### Mandatory implementation preflight

1. 从 train split 生成独立的 `global_root/body/local_root` stats。
2. local root 严格按 Kimodo 源码计算：heading angular velocity、global root XZ velocity、global root Y，单位均按 30 FPS；与 Kimodo reference tensor 做数值对齐测试。
3. Two-stage tensor contract：

```text
root input:  [z_imp(273), mask(273)] -> 546 -> root continuous x0[5]
root x0[5] --unnormalize/derive/normalize--> local root[4]
body input:  [local root(4), noisy body(268), global mask(273)] -> 545
body output: body continuous x0[264] + contact logits[4]
model output: root x0[5] + body continuous x0[264] + contact logits[4]
ODE update:   only first 269 continuous entries; contacts use sigmoid/BCE/blend
repeat at every ODE step
```

训练时 local-root conversion detach；推理时保留可微接口供未来 guidance 使用。两个 denoiser共享 HYText frontend，但各自有独立 backbone。

### Compute controls

```text
A1 current one-stage
A2 interleaved two-stage
A3 parameter-matched one-stage: trainable params within +/-3% of A2
A4 compute-matched one-stage: forward+backward FLOPs within +/-5% of A2
```

启动前冻结 A3/A4 的 depth/width 配置，并报告 trainable params、FLOPs、optimizer memory、peak VRAM、samples/s 和 ODE32 latency。FLOPs 使用固定 workload `B=1,T=300,text_tokens=128,bf16,self_cond=off,CFG=1`，统计一次 train forward+backward；latency 使用同一 shape 的 ODE32/contact-blend。另在固定真实 length trace 上报告吞吐。若一个模型不能同时匹配 params 与 FLOPs，A3/A4 都保留。

### Run protocol

- 所有模型使用 B1.5 winner，从头训练。
- seed 3407 先到 50K 只作故障筛查，不得据此删除 required comparator。
- A2、A3、A4 都固定使用 training seeds `3407,3408,3409` 训练到 100K。A1 作为当前容量参考至少完成 seed 3407。
- 不在 B2 单独宣布 two-stage 成功，最终架构判断要等 B3 control curriculum。

## B3: Short Control Curriculum

- 从 A2/A3/A4 的三 seed 100K checkpoints 各训练 50K mixed control；control loss 固定为 `0.25 * clean continuous SmoothL1 on masked dims + 0.05 * contact BCE on masked contact dims`，其余 loss、control trace 与 mode 概率完全一致。
- Control modes 固定为 `none,root,endpoints,fullpose,mixed`。
- Endpoint 固定 `kimodo_ee/random_nonempty/kimodo_hidden_root`，首协议为 position-only endpoints。
- 每个 mode 分开报告；不能用 mode 混合均值掩盖失败。
- 模型能力使用 pre-clamp FK/world errors；post-clamp exact hit 只记作系统保证。
- 必须报告未控制帧、anchor 前后窗口的 velocity/acceleration/jerk，以及控制激活时的文本匹配。

Architecture gate:

1. A2 必须分别对 A3 和 A4 的 endpoint/fullpose pre-clamp FK p95 改善至少 10%；每个 training seed 方向一致，且每个 seed 的 paired sample CI 报告完整。
2. root/mixed pre-clamp FK/world p95 相对 A3/A4 均满足 5% non-inferiority，不能只靠 endpoints/fullpose 过门。
3. heuristic-contact foot skate 和 transition jerk 满足 5% non-inferiority，contact coverage 不下降。
4. uncontrolled T2M 使用前述文本 non-inferiority gate；control 激活时的 matching/R@3 也必须满足相同 gate，不能用 uncontrolled text 代替。
5. 参数、FLOPs、wall time 和 latency 全部随结果报告；若收益只来自更多 compute，不支持 C3。

## B4: Final Training

- 选定 architecture/loss 后从头训练。
- Stage 1: 500K pure T2M。
- Stage 2: additional 500K mixed control，总计 1M。
- 启动前把 launcher 改成显式 `RESUME_STEP=500000 + ADDITIONAL_STEPS=500000 -> TARGET_GLOBAL_STEP=1000000` 并 assert checkpoint/global target。当前 Stage-2 launcher 的 `max_steps=1500000` 必须先修，否则会多跑 500K。
- 每 50K 保存 checkpoint，原子更新 `latest.pt`。
- 每 50K 在 fixed val-256 跑 pilot；每 100K 在 val-1024/3 generation seeds 上确认。训练与架构选择只使用 val；HumanML3D full test 保持锁定，只在选定最终模型后的 500K Stage-1 endpoint 和 1M final endpoint 各运行一次。
- raw/no-postprocess 是主表；postprocess 单列，不能替代模型结果。
- 1M 后不自动延长。若需续训，基于 900K/1M 的预注册 endpoint 新建 extension plan。

## Stop/Go Rules

1. B0 的训练启动门禁未通过，不启动 loss/architecture 训练；评估门禁未通过时不阻止算力推进，但阻止 checkpoint 选择与结论。
2. 若 L1/L2 都不能改善 `D_motion`，不继续调 semantic weights，转向架构假设。
3. 若 L3 只降低 consistency、但 FK-to-GT/skate/文本变差，删除 consistency，不临时改 lambda。
4. 若 late fork 赢但 scratch confirmation 不赢，以 scratch 结果为准。
5. 若 A2 不优于 A3/A4，保留简单 one-stage。
6. 训练 loss、post-clamp feature error、单个 GIF 均不能单独决定模型。

## Final Checklist

- [x] 300K checkpoint saved
- [ ] EMA/raw evaluator semantics fixed
- [ ] FK skate/contact metrics fixed and GT-audited
- [ ] K273-to-HumanML3D evaluator bridge audited
- [x] deterministic first-100-step scratch trace matched
- [x] scratch output-gradient audit implemented
- [x] checkpoint/restart cursor smoke passed
- [ ] L2/L3 scratch runs reached 200K
- [ ] local-root stats and conversion match Kimodo reference
- [ ] A2 compared with both parameter- and compute-matched one-stage
- [ ] control selected by pre-clamp FK/world metrics
- [ ] no-postprocess and postprocess results separated
