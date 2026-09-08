"""Prepare a fresh text interface and verify BC gradients on real training data.

This file never reads a checkpoint from an earlier spike-turbovla training run.
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from turbovla.models.configuration import TextEncoderConfig
from turbovla.models.text_encoder import TurboVLATextEncoder
from turbovla.training import trainer
from turbovla.training.joint_scope import (
    PROJECTION_KIND, load_calibrated_projection, check_preload, joint_scope, gradient_report,
)

from experiments.libero.settings import CODE, BERT, SMOOTH, INIT, DATASETS, SEED


def emit(event, **values):
    print(json.dumps({"event":event, **values}), flush=True)


def metrics(pred, target, selected):
    p, t = pred[selected], target[selected]
    return {"relative_mse":float((p-t).square().mean()/t.square().mean().clamp_min(1e-12)),
            "cosine":float(F.cosine_similarity(p,t,dim=-1).mean()),
            "mae":float((p-t).abs().mean())}


def calibrate(destination):
    layout=json.loads((CODE/"experiments/libero/configs/online_text_layout.json").read_text())
    instructions=list(layout["padding_length_by_instruction"])
    training_tasks=set()
    for dataset in DATASETS:
        for line in (Path(dataset)/"meta/tasks.jsonl").read_text().splitlines():
            training_tasks.add(json.loads(line)["task"])
    if len(instructions)!=40 or set(instructions)!=training_tasks:
        raise RuntimeError("Calibration instructions do not match the four training datasets")
    config=TextEncoderConfig(
        encoder_type="bert",model_name_or_path=BERT,padding_length=layout["output_padding_length"],
        padding_length_by_instruction=layout["padding_length_by_instruction"],
        frozen=True,local_files_only=True,attention_implementation="eager")
    device=torch.device("cuda:0")
    reference=TurboVLATextEncoder(config,256,"legacy").eval().to(device)
    grounding=trainer._extract_state_dict(torch.load(INIT,map_location="cpu",weights_only=False))
    grounding={k[7:] if k.startswith("module.") else k:v for k,v in grounding.items()}
    projection={key:grounding["feat_map."+key] for key in ("weight","bias")}
    reference.text_projection.load_state_dict(projection,strict=True)
    reference.requires_grad_(False)
    del grounding,projection
    spike_config=copy.deepcopy(config)
    spike_config.encoder_type="smoothspike_bert"
    spike_config.model_name_or_path=SMOOTH
    spike_config.timesteps=4
    spike=TurboVLATextEncoder(spike_config,256,"legacy").eval().to(device)
    spike.requires_grad_(False)
    spike.text_projection.load_state_dict(reference.text_projection.state_dict(),strict=True)
    with torch.no_grad():
        target,padding,masks=reference(instructions,device)
        hidden,valid,spike_masks=spike.encode_bert_hidden(instructions,device)
        if not torch.equal(valid,~padding) or not torch.equal(masks,spike_masks):
            raise RuntimeError("ANN/SNN text layout mismatch")
        raw=spike.text_projection(hidden)
        # Include PAD memory rows because the original action head consumes them.
        x=hidden.reshape(-1,hidden.shape[-1]).double()
        y=target.reshape(-1,target.shape[-1]).double()
        mx,my=x.mean(0),y.mean(0)
        xc,yc=x-mx,y-my
        gram=xc.T@xc/len(x)
        ridge=0.01*gram.diag().mean()
        weight=torch.linalg.solve(
            gram+ridge*torch.eye(gram.shape[0],device=device,dtype=gram.dtype),
            xc.T@yc/len(x))
        bias=my-mx@weight
        spike.text_projection.weight.copy_(weight.T)
        spike.text_projection.bias.copy_(bias)
        calibrated=spike.text_projection(hidden)
        if not torch.isfinite(calibrated).all():
            raise RuntimeError("Non-finite calibrated text features")
    metadata={
        "bert_path":BERT,"smoothspike_path":SMOOTH,"grounding_init":INIT,
        "source_policy_checkpoint":None,"sentence_mask":"legacy","head_mask":"corrected",
        "float32_matmul_precision":torch.get_float32_matmul_precision(),
        "timesteps":4,"training_instructions":instructions,"instruction_count":40,
        "training_metadata":[str(Path(p)/"meta/tasks.jsonl") for p in DATASETS],
        "fit_uses_actions":False,"fit_uses_evaluation_trajectories":False,
        "ridge_relative_regularization":0.01,"ridge_strength":float(ridge),
        "fit_memory_rows":"all_including_padding","policy_optimizer_steps":0,
        "deployed_runtime":"online SmoothSpike, no embedding lookup",
        "projection_trainable_in_joint_run":True,
    }
    report={"metadata":metadata,
            "raw_valid":metrics(raw,target,valid),
            "calibrated_valid":metrics(calibrated,target,valid),
            "raw_padding":metrics(raw,target,padding),
            "calibrated_padding":metrics(calibrated,target,padding),
            "scope":"fit on 40 training instruction templates; not unseen-language validation"}
    state={k:v.detach().cpu() for k,v in spike.text_projection.state_dict().items()}
    temporary=destination/"text_projection.pth.tmp"
    torch.save({"checkpoint_kind":PROJECTION_KIND,"text_projection":state,"metadata":metadata},temporary)
    temporary.replace(destination/"text_projection.pth")
    (destination/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    emit("common_pretraining_calibration_complete",output=str(destination),**{k:v for k,v in report.items() if k!="metadata"})


def preflight(destination):
    from experiments.libero.run_cross_joint_stage import training_arguments
    from turbovla.data.libero_lerobot import LiberoMixedLeRobotDataset,vla_collate_fn
    from turbovla.training import train_mixed
    argv=sys.argv
    try:
        sys.argv=[argv[0],*training_arguments(CODE,destination)]
        args=train_mixed.parse_args_and_record()
    finally:
        sys.argv=argv
    trainer.set_seed(args.seed)
    model=trainer.build_model_architecture(args)
    report=trainer.load_interaction_from_init_checkpoint(model,args.pretrained_init_ckpt)
    check_preload(model,report)
    trainer.load_text_projection_from_init_checkpoint(model,args.pretrained_init_ckpt)
    if args.text_projection_init:
        load_calibrated_projection(model,args)
    device=torch.device("cuda:0")
    model.to(device).train()
    groups=joint_scope(model,args)
    dataset=LiberoMixedLeRobotDataset(
        dataset_dir=args.dataset_dir,dataset_dirs=args.dataset_dirs,
        stats_path=args.stats_path,stats_key=args.stats_key,
        normalize_binary_gripper=args.normalize_binary_gripper,
        gripper_convention=args.lerobot_gripper_convention,
        LOCAL_DINOV3_PATH=args.dinov3_path,chunk_size=args.chunk_size,
        seed=args.seed,local_files_only=True,expected_image_size=args.expected_image_size,
        vision_encoder_type=args.vision_encoder_type,vision_image_size=args.vision_image_size,
        enable_dinov3_teacher=False)
    samples,frames,seen=[],[],{}
    for episode in dataset.episodes:
        selected=seen.setdefault(episode.suite_index,set())
        if len(selected)>=2 or episode.fallback_instruction in selected:
            continue
        selected.add(episode.fallback_instruction)
        decoded=dataset._load_episode(episode)
        step=episode.expected_length//2
        samples.append(dataset._build_step_sample(decoded,step))
        frames.append({"suite_index":episode.suite_index,"episode_index":episode.episode_index,
                       "step":step,"instruction":episode.fallback_instruction})
    if len(samples)!=8:
        raise RuntimeError("Expected eight real frames, two instructions from each training suite")
    images,instructions,states,targets,mask=vla_collate_fn(samples)
    predictions=model(instructions,trainer.move_samples_to_device(images,device),states.to(device))
    loss=trainer.masked_l1_loss(predictions,targets.to(device),mask.to(device))
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite pure behavior-cloning loss")
    loss.backward()
    gradients=gradient_report(groups)
    if any(p.grad is not None for p in model.text_encoder.bert.parameters()):
        raise RuntimeError("Frozen text backbone received gradients")
    result={"training_scope":"joint_except_frozen_text_backbone","source_policy_checkpoint":None,
            "float32_matmul_precision":torch.get_float32_matmul_precision(),
            "loss_used":"LIBERO ground-truth action L1 only; no distillation in this gradient check",
            "action_l1":float(loss),"optimizer_steps":0,"frames":frames,"gradients":gradients,
            "text_backbone_frozen":True,"text_projection_trainable":True,
            "mapped_grounding_tensors":report["mapped"],
            "peak_gpu_allocated_gib":torch.cuda.max_memory_allocated()/2**30}
    (destination/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    emit("joint_bc_gradient_preflight_complete",**result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["calibrate","preflight"],required=True)
    parser.add_argument("--output_dir",required=True)
    args=parser.parse_args()
    destination=Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__,destination/"prepare_cross_joint.py")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.mode=="calibrate":
        calibrate(destination)
    else:
        preflight(destination)


if __name__=="__main__":
    main()
