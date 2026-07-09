# HY273 Raw-Flow HYText Cache Integration Review

## Scope

This patch connects HY-Motion-style text conditioning to the HY273 raw-space x0 rectified-flow trainer.

The intended first training target is still:

```text
HumanML3D K273 raw source data
  -> same-space observed_motion/mask overwrite
  -> x0-pred raw-flow denoiser
  -> ODE sampler with step-wise clamp
```

Only the text encoder path changes:

```text
old: OpenAI CLIP ViT-B/32 online encoder
new: cached HYText = Qwen3-8B token embeddings + CLIP ViT-L/14 pooled embedding
```

The large text towers are not loaded by DDP training. They are used only by the offline cache builder.

## Main Files

```text
models/raw_motion/text_condition.py
  Shared RawTextCondition dataclass.

models/raw_motion/hytext_cache.py
  Memmap cache reader and CachedHYTextEncoder projection bridge.
  The reader validates cache metadata and groups random batch lookups by shard.

models/raw_motion/raw_flow_dit.py
  Adds text_encoder=hy_cache / hytext_cache / qwen_clip_cache.

tools/cache_hy273_hytext_embeddings.py
  Offline cache builder using local Qwen3-8B and CLIP ViT-L/14 weights.

tools/check_hy273_hytext_cache_coverage.py
  Verifies that train/val/test captions and the empty caption all hit the cache index.

train_hy273_raw_flow.py
  Adds max_text_tokens and HYText cache args, saved into checkpoint args.

sample_hy273_raw.py
  Adds HYText override args for sampling/eval.

configs/raw_flow_hy273_hytext.yaml
  Dedicated HYText-cache config for stage-1 training.

scripts/launch/train_hy273_raw_flow_stage1_x0_hytext_ddp8.sh
  Stage-1 8-GPU launch script for HYText-cache T2M x0 training.

tests/test_raw_flow_model.py
  Adds fake-cache forward and forced text-drop tests.
```

## Tensor Information Flow

```text
Caption string
  |
  | offline cache builder only
  v
Qwen3-8B chat-template encoder
  -> ctxt_raw: [B, 128, 4096]
  -> ctxt_len: [B]

CLIP ViT-L/14 text encoder
  -> vtxt_raw: [B, 1, 768]

Saved cache
  index.json: sha1(normalized_caption) -> shard,row
  shards/shard_xxxxx/ctxt.npy      float16 [N,128,4096]
  shards/shard_xxxxx/vtxt.npy      float16 [N,1,768]
  shards/shard_xxxxx/ctxt_len.npy  int16   [N]

DDP training batch
  captions: List[str]
  |
  v
CachedHYTextEncoder
  ctxt_raw row lookup -> token_proj 4096 -> H
  vtxt_raw row lookup -> pooled_proj 768 -> H
  ctxt_len -> padding_mask [B,128]
  |
  v
RawTextCondition
  tokens:       [B,128,H]
  pooled:       [B,H]
  padding_mask: [B,128]

Raw-flow denoiser
  motion token:
    model_in = concat(z_imp, motion_mask) [B,T,546]
    input_proj -> [B,T,H]

  global cond:
    timestep_embed(t) [B,H]
    + direction_embed(c_dir) [B,H]
    + text pooled [B,H]

  cross-attention text:
    Qwen token projections [B,128,H]

  output:
    pred [B,T,273]
      [0:269] x0 continuous prediction
      [269:273] contact logits
```

## Cache Build Command

```bash
/root/miniconda3/envs/mogo/bin/python tools/cache_hy273_hytext_embeddings.py \
  --data_root /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22 \
  --text_root /mnt/afs/mogo_base/datasets/HumanML3D/texts \
  --splits train,val,test \
  --output_dir /mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128 \
  --qwen_path /mnt/afs/HY-Motion-1.0/ckpts/Qwen3-8B \
  --clip_path /mnt/afs/HY-Motion-1.0/ckpts/clip-vit-large-patch14 \
  --max_length_llm 128 \
  --batch_size 4 \
  --shard_size 4096 \
  --device cuda:0 \
  --storage_dtype fp16
```

Use an actually idle GPU for this command. It loads Qwen3-8B once and writes a memmap cache.

Before long training, run cache coverage:

```bash
/root/miniconda3/envs/mogo/bin/python tools/check_hy273_hytext_cache_coverage.py \
  --data_root /mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22 \
  --text_root /mnt/afs/mogo_base/datasets/HumanML3D/texts \
  --cache_dir /mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128 \
  --splits train,val,test \
  --output_json /mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128/coverage_report.json
```

This must pass with `empty_key_present=true` and no missing captions.

## Training Command

```bash
HYTEXT_CACHE_DIR=/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128 \
bash scripts/launch/train_hy273_raw_flow_stage1_x0_hytext_ddp8.sh
```

This keeps the current stage-1 training policy:

```text
prediction_type=x0
control_modes=none
flow_loss=1.0
contact_loss=0.1
clean_root_vel=0.01
clean_joint_vel=0.01
foot_lock=0.01
EMA=0.995 every 10 steps
```

## Local Smoke Checks

Passed:

```bash
/root/miniconda3/envs/mogo/bin/python -m pytest \
  tests/test_raw_flow_model.py tests/test_raw_flow_sampling.py -q

# result: 9 passed
```

Passed one-step train smoke with fake cache:

```text
[train] epoch=0 step=1 loss=1.865776 ...
```

This smoke uses fake zero text cache with cache-miss fallback. It validates code connectivity only; it does not validate HYText quality.

## Post-Review Fixes

The follow-up patch addresses the high-priority review issues:

```text
P0:
  - Added configs/raw_flow_hy273_hytext.yaml so HYText training does not rely on CLI overrides.
  - train_hy273_raw_flow_stage1_x0_hytext_ddp8.sh now defaults CONFIG to the HYText config.
  - Added tools/check_hy273_hytext_cache_coverage.py.
  - CachedHYTextEncoder validates manifest format, ctxt/vtxt dims, max_length_llm, and first-shard array shape.

P1:
  - cache builder shard_size default changed 128 -> 4096.
  - hytext_max_open_shards default changed to 64 in configs.
  - HYTextMemmapCache.lookup_rows now groups random batch rows by shard and vector-indexes each shard.
  - sample_hy273_raw.py can override hytext_ctxt_dim, hytext_vtxt_dim, hytext_max_open_shards, and hytext_allow_cache_miss.
```

Remaining known issue:

```text
HumanML3D segment-caption span handling is still unchanged.
The current dataset keeps the existing full-caption/random-crop behavior.
Treat this as a quality ablation after the HYText cache pilot starts.
```

## Reviewer Prompt

Please review this HY273 raw-flow HYText-cache integration adversarially.

Check whether the implementation correctly matches the intended semantics:

1. DDP training must not instantiate Qwen3-8B or CLIP ViT-L/14 online.
2. Offline cache should match HY-Motion HYText semantics:
   - Qwen3-8B chat template with the human-motion summarization system prompt.
   - crop_start computed with the `<BOC>` marker.
   - cached `ctxt_raw` is `[B,128,4096]`.
   - cached `vtxt_raw` is CLIP ViT-L/14 pooled `[B,1,768]`.
3. The training model should project:
   - Qwen token embeddings `4096 -> hidden_dim` as cross-attention text tokens.
   - CLIP pooled embedding `768 -> hidden_dim` into the global conditioning vector.
4. Text dropout / CFG force-drop should use the cached empty-caption row, not random zeros unless the empty row is missing.
5. Cache lookup should be efficient for random DDP batches. Specifically, verify that it does not `torch.load` large shard tensors per batch.
6. x0-pred raw-flow behavior should be unchanged:
   - `[0:269]` direct clean continuous x0 prediction.
   - `[269:273]` contact logits with BCE.
   - contact channels remain raw 0/1/probability, not z-scored.
7. Check checkpoint compatibility:
   - new args are saved.
   - sampling can reconstruct the HYText-cache model from checkpoint args.
8. Look for failure modes:
   - missing caption cache keys.
   - duplicate normalized captions.
   - Qwen tokenizer padding/crop length off-by-one.
   - FP16 cache precision loss.
   - memmap read concurrency under multiple dataloader/DDP workers.
   - excessive text projection parameter count or optimizer side effects.

Please report concrete bugs by file/function and suggest minimal patches.
