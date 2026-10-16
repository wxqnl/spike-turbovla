# Vision attribution runs (D0–D3)

Purpose: split the LIBERO gap between Spike-Driven VLA and ANN VLAs into the cost of a **spiking** vision encoder, of its **pretraining objective**, and of **capacity**.

All four runs keep everything after the vision encoder as in the spike-driven-vla base:
- spike projections and `spike_text_only` fusion (text top-3 WTA, SDSA cross);
- Shiftmax spike action head with eight continuous state tokens;
- light RGB augmentation;
- 4 GPUs × 80 per GPU without accumulation (global 320) for 32k steps: the same 10.24M samples as the base's 256 × 40k;
- learning rates scaled by sqrt(320/256): vision 1.12e-5, other 5.6e-5.

Only the vision slot changes. All runs use 224 px input and 14×14 tokens, and train with the masked action L1 only, without distillation.

| Variant | Vision encoder | Size | Pretraining | Spiking |
| --- | --- | --- | --- | --- |
| `d0_sdtv3_nokd` | SDT-V3 (`V3_19.0M_1x4.pth`) | 19M | ImageNet-1k supervised | yes |
| `d1_deit_s_nokd` | `facebook/deit-small-patch16-224` | 22M | ImageNet-1k supervised | no |
| `d2_dinov3_s_nokd` | `facebook/dinov3-vits16-pretrain-lvd1689m` | 21M | self-supervised (LVD-1689M) | no |
| `d3_dinov3_b_nokd` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 86M | self-supervised (LVD-1689M) | no |

How to read the differences:
- D1 − D0: the cost of spiking at matched size and objective.
- D2 − D1: the pretraining objective.
- D3 − D2: capacity.

The current base (92.05%, with KD from the ANN teacher) is not part of this comparison.

## Setup

Use the same environment, assets and text-projection initialization as the base ([README](README.md), steps 2–3).

For D1–D3, download the Hugging Face backbones on the training machine. Each directory must contain `config.json`, the weights and `preprocessor_config.json`:

```bash
export SPIKE_DEIT_S=$SPIKE_WORK_ROOT/pretrained/hf/deit-small-patch16-224
export SPIKE_DINOV3_S=$SPIKE_WORK_ROOT/pretrained/hf/dinov3-vits16-pretrain-lvd1689m
export SPIKE_DINOV3_B=${SPIKE_DINO_PATH:-$SPIKE_WORK_ROOT/pretrained/TurboVLA/backbones/dinov3-vitb16}
```

The recipe checks each directory's `model_type`, `hidden_size` and `patch_size` before training.

## Training

Preview each run first with `--world-size 4 --dry-run`. The last `--vision_encoder_type`, `--vision_model_path` and `--distill_*_weight` in the preview are the ones that take effect.

```bash
run() {  # $1 = variant, $2 = run dir, $3 = optional HF backbone dir
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --standalone --nproc_per_node=4 --module experiments.libero.train_action_head \
    --variant "$1" --run_dir "$2" \
    --vision_checkpoint "${SPIKE_VISION_CHECKPOINT:-$SPIKE_WORK_ROOT/pretrained/SDT-V3/V3_19.0M_1x4.pth}" \
    --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
    --max_steps 32000 --batch_size 80 --grad_accum_steps 1 \
    ${3:+--vision_backbone "$3"} --resume_mode none --approve_training 2>&1 | tee "$2.train.log"
}
run d0_sdtv3_nokd    "$SPIKE_RUN_ROOT/d0_sdtv3_nokd_b320"
run d1_deit_s_nokd   "$SPIKE_RUN_ROOT/d1_deit_s_nokd_b320"   "$SPIKE_DEIT_S"
run d2_dinov3_s_nokd "$SPIKE_RUN_ROOT/d2_dinov3_s_nokd_b320" "$SPIKE_DINOV3_S"
run d3_dinov3_b_nokd "$SPIKE_RUN_ROOT/d3_dinov3_b_nokd_b320" "$SPIKE_DINOV3_B"
```

`--vision_checkpoint` is required by the shared launcher for every variant, but only D0 reads it.

## Evaluation

Run the EMA refresh (README step 5) on the final (32k) checkpoint, then the four-suite evaluation (seed 7, 4 × 500 episodes), the same as the base. Keep the usual `--dinov3_path` (DINOv3-B) for every variant: training normalized all images with its processor (ImageNet mean/std). Do not pass the DeiT directory there, since its `preprocessor_config.json` says mean/std 0.5. Evaluation rebuilds the D1–D3 backbone from the directory recorded in the checkpoint, so that directory must exist on the evaluation machine. It preprocesses at the trained 224 px.
