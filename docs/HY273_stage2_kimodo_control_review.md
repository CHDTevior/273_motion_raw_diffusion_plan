# HY273 Stage-2 Kimodo Control Review

## Scope

This patch prepares the existing HYText Stage-1 checkpoint for Stage-2 control training. It does not alter the Stage-1 model, raw HY273 representation, x0 objective, or the running process.

## Stage-2 Launch Contract

```bash
RESUME=/path/to/stage1_step_00500000.pt \
bash scripts/launch/train_hy273_raw_flow_stage2_x0_control_ddp8.sh
```

The resolved defaults are:

```text
config=configs/raw_flow_hy273_hytext.yaml
text_encoder=hy_cache
max_text_tokens=128
prediction_type=x0
control_modes=none,root,endpoints,fullpose,mixed
endpoint_preset=kimodo_ee
endpoint_subset_mode=random_nonempty
endpoint_root_ref_mode=kimodo_hidden_root
max_steps=1500000                 # total global step, not 1.5M additional steps
max_epochs=10000                  # enough to reach total step 1.5M after Stage 1
save_every=50000
```

Resume is fail-closed:

1. Model state loads with `strict=True`.
2. Shape-preserving semantic fields such as prediction type, HYText token length, transforms, and data roots must match the checkpoint args contract.
3. Optimizer state must load successfully.
4. Requested EMA requires an exact checkpoint EMA key/shape contract; it is not silently reinitialized.

## Kimodo Endpoint Protocol

The logical endpoint groups follow Kimodo `SMPLXSkeleton22.expand_joint_names`:

```text
LeftFoot  -> left_ankle(7), left_foot(10)
RightFoot -> right_ankle(8), right_foot(11)
LeftHand  -> left_wrist(20)
RightHand -> right_wrist(21)
```

For each training motion, endpoint control samples one nonempty subset of these four logical groups and keeps that subset across its sampled keyframes. Foot groups remain atomic: selecting a foot always masks both ankle and foot position channels.

This project's confirmed first protocol constrains endpoint positions only. It does not mask endpoint rotation channels. At every endpoint keyframe it also masks HY273 channels `[0:5]`:

```text
[0:3] smooth_root_pos: x/z reference plus root y
[3:5] global_root_heading
```

The root reference is required because HY273 joint-position x/z channels are relative to `smooth_root_pos`. It is an internal compiler condition even when the user-facing constraint only specifies global endpoint positions.

## Local Verification

```text
python -m pytest -q \
  tests/test_hy273_constraints.py \
  tests/test_raw_flow_model.py \
  tests/test_raw_flow_sampling.py

17 passed
```

The Stage-2 launcher was also dry-run through `/bin/echo`. It resolved to HYText cache, 128 text tokens, Kimodo logical subsets, 10,000 maximum epochs, total step 1.5M, and 50K checkpoint intervals.

The first `gpt-5.6-sol` / `max` adversarial review found three P2 consistency gaps. The follow-up patch:

1. Extends the resume contract to data split, random seed, optimizer/numerical settings, flow time distribution, HYText cache strictness, self-conditioning probability, and unchanged Stage-1 loss semantics.
2. Makes sampling inherit the checkpoint endpoint preset, subset mode, root-reference mode, and keyframe count, with explicit sampling overrides available.
3. Tracks which argparse destinations were explicitly present so command-line values always override YAML, even when equal to parser defaults.

The follow-up review found one P2 edge case: argparse accepted abbreviated option names that the explicit-option detector did not recognize. The training parser now disables option abbreviation, so only exact CLI names are accepted and YAML precedence cannot diverge from parsing. A regression test covers the rejected abbreviation.

## Reviewer Prompt

Review the uncommitted patch adversarially. Findings should lead, ordered by severity and grounded in file/line references.

Check:

1. Whether Stage 2 can silently change any Stage-1 HYText/model/data semantic while still loading the checkpoint.
2. Whether optimizer and EMA continuity are fail-closed.
3. Whether the endpoint masks exactly implement the four Kimodo logical groups and arbitrary nonempty subsets.
4. Whether same-frame root position/y/heading references are present for endpoint controls.
5. Whether DDP rank seeding, frame sampling, padding masks, normalization, or CLI/YAML merging can invalidate the protocol.
6. Whether `max_epochs=10000` allows a Stage-1 500K checkpoint to reach total step 1.5M for the current HumanML3D train split and global batch size.
7. Whether the tests use independent expected joint IDs and catch semantic regressions rather than deriving all expectations from implementation constants.

Do not report endpoint rotations as a blocker: position-only endpoint control is the explicitly selected first protocol.
