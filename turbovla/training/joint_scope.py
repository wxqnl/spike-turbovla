"""Initialization and first-step evidence for joint training, not a new model."""
import json
from pathlib import Path

import torch

PROJECTION_KIND = "common_pretraining_smoothspike_projection_initialization"
GROUPS = {
    "vision_backbone": "vision_encoder.backbone.",
    "vision_projection": "vision_projection.",
    "text_projection": "text_encoder.text_projection.",
    "cross_fusion": "vision_language_interaction.fusion_layers.",
    "text_interaction": "vision_language_interaction.text_layers.",
    "state_projection": "action_head.state_projection.",
    "action_decoder": "action_head.decoder.",
}


def load_calibrated_projection(model, args):
    if model.config.text.encoder_type != "smoothspike_bert":
        raise ValueError("SmoothSpike projection initialization cannot be applied to ANN BERT")
    payload = torch.load(args.text_projection_init, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_kind") != PROJECTION_KIND:
        raise ValueError("Only the common-pretraining projection artifact is allowed")
    metadata = payload["metadata"]
    expected = {
        "bert_path": args.bert_path,
        "smoothspike_path": args.text_model_path,
        "grounding_init": args.pretrained_init_ckpt,
        "sentence_mask": model.config.sentence_mask_version,
        "timesteps": args.text_timesteps,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Projection initialization mismatch for {key}")
    if metadata.get("source_policy_checkpoint") is not None:
        raise ValueError("Projection must not inherit a previously trained VLA policy")
    state = payload["text_projection"]
    if set(state) != {"weight", "bias"} or not all(torch.isfinite(p).all() for p in state.values()):
        raise ValueError("Invalid projection-only initialization")
    projection = model.text_encoder.text_projection
    if model.spike_projectors:
        projection.load_common_projection(state)
    else:
        projection.load_state_dict(state, strict=True)
    if not all(p.requires_grad for p in model.text_encoder.text_projection.parameters()):
        raise RuntimeError("Calibrated text projection must remain trainable")
    return metadata


def check_preload(model, report):
    if model.config.interaction.backbone_type in {"spike_temporal", "spike_wta", "spike_text_only"}:
        from .temporal_fusion import common_pretrained_names
        expected = len(common_pretrained_names(model))
        if report["mapped"] != expected or report["skipped_missing"] or report["skipped_shape"]:
            raise RuntimeError(f"Incomplete common initialization for temporal fusion: {report}")
        return
    expected = sum(
        key.startswith("vision_language_interaction.") and ".spike_nodes." not in key
        for key in model.state_dict()
    )
    if report["mapped"] != expected or report["skipped_missing"] or report["skipped_shape"]:
        raise RuntimeError(f"Incomplete common GroundingDINO initialization: {report}, expected={expected}")


def joint_scope(model, args):
    wta_full_resume = (args.resume_mode == "all"
                       and model.config.interaction.backbone_type in {"spike_wta", "spike_text_only"})
    if (args.resume_mode != "none" and not wta_full_resume) or args.freeze_backbones or not args.freeze_text_encoder:
        raise ValueError("Joint stage requires fresh initialization (or full WTA resume) and only frozen text")
    if model.config.vision.frozen or not model.config.text.frozen:
        raise ValueError("Incorrect vision/text gradient configuration")
    # DINOv3's unused masked-image token is frozen by the existing vision adapter.
    # It is not part of the policy's forward path; all used vision weights train.
    unused = {"vision_encoder.backbone.embeddings.mask_token"} if model.config.vision.encoder_type == "dinov3" else set()
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad and name not in unused]
    if not frozen or any(not name.startswith("text_encoder.bert.") for name in frozen):
        raise RuntimeError("A used module other than the text backbone is frozen")
    if any(p.requires_grad for p in model.text_encoder.bert.parameters()):
        raise RuntimeError("Text backbone must remain frozen")
    groups = {
        group: [(name, p) for name, p in model.named_parameters() if name.startswith(prefix) and name not in unused]
        for group, prefix in GROUPS.items()
    }
    if any(not values or any(not p.requires_grad for _, p in values) for values in groups.values()):
        raise RuntimeError("A required joint-training parameter group is absent or frozen")
    return groups


def gradient_report(groups):
    report = {}
    for group, values in groups.items():
        gradients = [p.grad.detach() for _, p in values if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"Missing or non-finite gradients in {group}")
        norm = float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt())
        if norm == 0:
            raise RuntimeError(f"All gradients are zero in {group}")
        report[group] = {"parameters": sum(p.numel() for _, p in values),
                         "tensors_with_grad": len(gradients), "gradient_l2": norm}
    return report


def representative_snapshots(groups):
    result = {}
    for group, values in groups.items():
        name, parameter = next((n, p) for n, p in values if p.ndim >= 2)
        result[group] = (name, parameter, parameter.detach().clone())
    return result


def record_first_update(model, args, groups, before, gradients, rank):
    updates = {}
    for group, (name, parameter, original) in before.items():
        change = float((parameter.detach() - original).abs().max())
        if change == 0 or not torch.isfinite(parameter).all():
            raise RuntimeError(f"No finite optimizer update for {name}")
        updates[group] = {"representative_parameter": name, "maximum_change": change}
    if any(p.grad is not None for p in model.text_encoder.bert.parameters()):
        raise RuntimeError("Frozen text backbone received gradients")
    if rank == 0:
        result = {
            "training_scope": "joint_except_frozen_text_backbone",
            "step": 1, "resumed_policy": False, "gradients": gradients, "updates": updates,
            "vision_requires_grad": True, "text_projection_requires_grad": True,
            "text_backbone_frozen": True,
            "text_projection_init": args.text_projection_init,
        }
        path = Path(args.checkpoint_dir).parent / "joint_training_audit.json"
        with path.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"event": "joint_first_update_verified", **result}), flush=True)
