"""Bounded head-only repair using real, frozen full-model LIBERO features.

The cache is produced by the complete original AA and A policies at B=1.
A supplies action-only distillation targets; no latent spaces are mixed.
Episode-disjoint repair validation is not a held-out pretraining benchmark.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from turbovla.data.libero_lerobot import LiberoMixedLeRobotDataset, vla_collate_fn
from turbovla.evaluation.policy import configure_transformers_offline
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import build_turbovla
from turbovla.models.spike_action_head import SpikeTurboVLAActionHead
from turbovla.models.components.wta_operators import running_norms, verify_running_norms
from turbovla.training.trainer import masked_l1_loss, masked_mse_loss, move_samples_to_device

SOURCE = Path(__file__).resolve().parents[2]


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def save_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def load_checkpoint(path):
    cp = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = TurboVLAConfig.from_mapping(cp["model_config"])
    cfg.vision.model_source_path = str(SOURCE / "third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py")
    return cp, cfg


def make_dataset(cp):
    a = SimpleNamespace(**cp["args"])
    stats = SOURCE / "experiments/libero/configs/libero_all4_stats.json"
    return LiberoMixedLeRobotDataset(
        dataset_dir=a.dataset_dir, dataset_dirs=a.dataset_dirs,
        stats_path=stats, stats_key=a.stats_key,
        normalize_binary_gripper=a.normalize_binary_gripper,
        gripper_convention=a.lerobot_gripper_convention,
        LOCAL_DINOV3_PATH=a.dinov3_path, chunk_size=a.chunk_size,
        seed=a.seed, local_files_only=True,
        expected_image_size=a.expected_image_size,
        vision_encoder_type=a.vision_encoder_type, vision_image_size=a.vision_image_size,
        shuffle_buffer=0, shuffle_steps_within_episode=False, step_mix_buffer_size=0,
        enable_dinov3_teacher=False,
    )


def make_model(cp, cfg, device):
    model = build_turbovla(cfg)
    model.load_state_dict(cp["ema_model_state_dict"], strict=True)
    return model.to(device).float().eval().requires_grad_(False)


@torch.inference_mode()
def cache_features(opt, device):
    started = time.monotonic()
    output = Path(opt.output)
    output.mkdir(parents=True, exist_ok=False)
    cp, cfg = load_checkpoint(opt.checkpoint)
    tcp, tcfg = load_checkpoint(opt.teacher)
    model, teacher = make_model(cp, cfg, device), make_model(tcp, tcfg, device)
    dataset = make_dataset(cp)
    fields = ("memory", "padding", "state", "target", "mask", "teacher", "baseline")
    rows = {split: {field: [] for field in fields} for split in ("train", "validation")}
    records = {split: [] for split in rows}
    sequences = []
    nmax = cfg.vision.num_views * cfg.vision.output_grid_size ** 2 + max(cfg.text.padding_length_by_instruction.values())
    emit("cache_models_loaded", shard=opt.shard, device=str(device), gpu=torch.cuda.get_device_name(device), max_memory_length=nmax)
    for suite_index, suite in enumerate(dataset.suites):
        for task_index, instruction in sorted(suite.tasks.items()):
            if (suite_index * 10 + task_index) % opt.shards != opt.shard:
                continue
            rng = np.random.default_rng(opt.seed + 1009 * suite_index + 9176 * task_index)
            episodes = [ep for ep in dataset.episodes if ep.suite_index == suite_index and ep.fallback_instruction == instruction]
            needed = opt.train_episodes + opt.validation_episodes
            if len(episodes) < needed:
                raise ValueError(f"Task {suite_index}/{task_index} has fewer than {needed} real episodes")
            chosen = rng.permutation(len(episodes))[:needed]
            for number, index in enumerate(chosen):
                episode = episodes[int(index)]
                split = "train" if number < opt.train_episodes else "validation"
                data = dataset._load_episode(episode)
                if split == "validation":
                    state_series = (torch.as_tensor(data["states"]) - dataset.proprio_mean) / (dataset.proprio_std + 1e-6)
                    sequences.append(dict(suite=suite_index, task=task_index, episode=episode.episode_index, states=state_series.clone()))
                frame_count = opt.train_frames if split == "train" else opt.validation_frames
                last_start = max(0, episode.expected_length - cfg.action.horizon)
                steps = np.unique(np.rint(np.linspace(0, last_start, frame_count)).astype(int))
                for step in steps:
                    images, instructions, state, target, mask = vla_collate_fn([dataset._build_step_sample(data, int(step))])
                    images = move_samples_to_device(images, device)
                    state_device = state.to(device)
                    condition, features = model.encode_condition(instructions, images, return_distillation_features=True)
                    visual = condition.shape[1] - features["text_valid"].shape[1]
                    padding = torch.cat((torch.zeros((1, visual), device=device, dtype=torch.bool), ~features["text_valid"]), 1)
                    prediction = model.action_head(condition, state_device, padding)
                    teacher_action = teacher(instructions, images, state_device)
                    if not torch.isfinite(prediction).all() or not torch.isfinite(teacher_action).all():
                        raise RuntimeError("Non-finite action on a real observation")
                    values = dict(
                        memory=F.pad(condition, (0, 0, 0, nmax-condition.shape[1])).cpu(),
                        padding=F.pad(padding, (0, nmax-padding.shape[1]), value=True).cpu(),
                        state=state, target=target, mask=mask,
                        teacher=teacher_action.cpu(), baseline=prediction.cpu(),
                    )
                    for field, value in values.items():
                        rows[split][field].append(value)
                    records[split].append(dict(suite_index=suite_index, task_index=task_index, episode_index=episode.episode_index, frame_index=int(step), instruction=instruction))
                emit("cache_episode", shard=opt.shard, suite=suite_index, task=task_index, episode=episode.episode_index, split=split, train_samples=len(records["train"]), validation_samples=len(records["validation"]), elapsed=time.monotonic()-started)
    packed = {split: {field: torch.cat(values) for field, values in split_rows.items()} for split, split_rows in rows.items()}
    payload = dict(data=packed, records=records, validation_state_sequences=sequences, checkpoint=opt.checkpoint, teacher_checkpoint=opt.teacher, seed=opt.seed, precision="FP32 highest; no TF32; complete original models B=1", episode_split="disjoint for head repair; original models may have seen both during pretraining")
    with (output / "features.pt").open("xb") as stream:
        torch.save(payload, stream)
    save_json(output / "manifest.json", {key:value for key,value in payload.items() if key not in ("data", "validation_state_sequences")} | dict(train_samples=len(records["train"]), validation_samples=len(records["validation"]), elapsed=time.monotonic()-started))
    emit("cache_complete", shard=opt.shard, output=str(output), train_samples=len(records["train"]), validation_samples=len(records["validation"]), elapsed=time.monotonic()-started)


@torch.inference_mode()
def measure_predictions(pred, target, mask):
    errors = (pred-target).abs() * mask[..., None]
    count = mask.sum()
    return dict(l1=(errors.sum()/(count*7)).item(), first_l1=(pred[:, 0]-target[:, 0]).abs().mean().item(),
                per_dim_l1=(errors.sum((0,1))/count).tolist(),
                gripper_sign_accuracy=((((pred[..., 6]>=0)==(target[..., 6]>=0))*mask).sum()/count).item())


@torch.inference_mode()
def evaluate(head, data, batch_size=1):
    head.eval()
    predictions = []
    for index in range(0, len(data["state"]), batch_size):
        stop = index + batch_size
        length = int(data["memory_length"][index]) if batch_size == 1 else data["memory"].shape[1]
        predictions.append(head(data["memory"][index:stop, :length], data["state"][index:stop], data["padding"][index:stop, :length]))
    predictions = torch.cat(predictions)
    return measure_predictions(predictions, data["target"], data["mask"]), predictions


@torch.inference_mode()
def refresh_head_statistics(head, training, seed):
    head.eval()
    norms = running_norms(head)
    for _, module in norms:
        module.reset_running_stats()
        module.calibrating = True
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(training["state"]), generator=generator).to(training["state"].device)
    try:
        # Same 128x16 train-only refresh budget as the original AA export.
        for index in range(128):
            positions = order[(torch.arange(16, device=order.device) + index*16) % len(order)]
            pred = head(training["memory"][positions], training["state"][positions], training["padding"][positions])
            if not torch.isfinite(pred).all():
                raise RuntimeError("Non-finite action during train-only head normalization refresh")
    finally:
        for _, module in norms:
            module.calibrating = False
    return verify_running_norms(head)


@torch.inference_mode()
def state_resolution(head, sequences, device):
    current = torch.cat([x["states"] for x in sequences]).to(device)
    left, offset = [], 0
    for item in sequences:
        length = len(item["states"])
        left.extend(range(offset, offset+length-1))
        offset += length
    left = torch.tensor(left, device=device)
    right = left+1
    state = {}
    tokens = torch.stack([head.state_projection.forward_step(current, state, t) for t in range(head.config.spike_timesteps)])
    changed = (current[left]-current[right]).abs().mean(-1) > 1e-6
    same = (tokens[:, left]==tokens[:, right]).all(0).flatten(1).all(1)
    return dict(real_states=len(current), adjacent_pairs=len(left), changed_pairs=changed.sum().item(),
                identical_encoded_pairs=(same & changed).sum().item(),
                collision_fraction=same[changed].float().mean().item())


def train_head(opt, device):
    started = time.monotonic()
    output = Path(opt.output)
    output.mkdir(parents=True, exist_ok=False)
    cp, cfg = load_checkpoint(opt.checkpoint)
    cfg.action.spike_state_population = opt.population
    cfg.action.spike_state_population_range = 4.0
    cfg.action.spike_state_attention_gain = opt.state_gain
    cfg.action.spike_gradient_checkpointing = False
    cfg = TurboVLAConfig.from_mapping(cfg.to_dict())
    head = SpikeTurboVLAActionHead(cfg.action, cfg.interaction.hidden_dim, cfg.interaction.nheads, cfg.interaction.dim_feedforward, cfg.interaction)
    old_head = {key[len("action_head."):]: value for key,value in cp["ema_model_state_dict"].items() if key.startswith("action_head.")}
    missing, unexpected = head.load_state_dict(old_head, strict=False)
    expected_missing = ["state_projection.population_projection.weight"] if opt.population else []
    if sorted(missing) != expected_missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint migration: missing={missing}, unexpected={unexpected}")
    if opt.population:
        head.state_projection.initialize_population_residual(scale=0.25)
    head = head.to(device).float()
    shards = [torch.load(path, map_location="cpu", mmap=True, weights_only=False) for path in opt.cache]
    for shard in shards:
        if shard["checkpoint"] != opt.checkpoint:
            raise ValueError("Cached upstream features do not belong to the requested checkpoint")
    data = {split: {name: torch.cat([shard["data"][split][name] for shard in shards]).to(device) for name in shards[0]["data"][split]} for split in ("train", "validation")}
    visual_length = cfg.vision.num_views * cfg.vision.output_grid_size ** 2
    for split in data:
        lengths = [visual_length + cfg.text.padding_length_by_instruction[record["instruction"]]
                   for shard in shards for record in shard["records"][split]]
        data[split]["memory_length"] = torch.tensor(lengths, device=device)
    sequences = [sequence for shard in shards for sequence in shard["validation_state_sequences"]]
    train_records = {(r["suite_index"],r["episode_index"]) for shard in shards for r in shard["records"]["train"]}
    val_records = {(r["suite_index"],r["episode_index"]) for shard in shards for r in shard["records"]["validation"]}
    if train_records & val_records:
        raise RuntimeError("Repair training and validation episodes overlap")
    baseline = measure_predictions(data["validation"]["baseline"], data["validation"]["target"], data["validation"]["mask"])
    teacher = measure_predictions(data["validation"]["teacher"], data["validation"]["target"], data["validation"]["mask"])
    ema = copy.deepcopy(head).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(head.parameters(), lr=opt.lr, weight_decay=1e-10)
    generator = torch.Generator(device="cpu").manual_seed(opt.seed)
    best, best_state, results = None, None, []
    handles, audited = [], set()
    def event_check(name):
        def hook(module, inputs):
            events = inputs[0]
            if not ((events==0) | (events==1) | (events==-1)).all().item():
                raise RuntimeError(f"Learned projection received non-unit events: {name}")
            audited.add(name)
        return hook
    for name, module in head.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(event_check(name)))
    emit("training_started", population=opt.population, state_gain=opt.state_gain, train_samples=len(data["train"]["state"]), validation_samples=len(data["validation"]["state"]), train_episodes=len(train_records), validation_episodes=len(val_records), baseline=baseline, teacher=teacher, device=str(device), max_steps=opt.steps, trainable_parameters=sum(p.numel() for p in head.parameters()))
    for step in range(opt.steps+1):
        if step in (0, 128, 256, 512, opt.steps):
            candidate = copy.deepcopy(ema).eval()
            layer_stats = refresh_head_statistics(candidate, data["train"], opt.seed)
            values, predictions = evaluate(candidate, data["validation"])
            values.update(step=step, elapsed=time.monotonic()-started)
            results.append(values)
            emit("validation", **values)
            if step > 0 and (best is None or values["l1"] < best["l1"]):
                best = values
                best_state = {key:value.detach().cpu().clone() for key,value in candidate.state_dict().items()}
                best_layers = layer_stats
                torch.save(dict(state_dict=best_state, step=step, validation=values, model_config=cfg.to_dict()), output / f"head_step{step}.pth")
            del candidate
        if step == opt.steps:
            break
        head.train()
        positions = torch.randint(len(data["train"]["state"]), (opt.batch_size,), generator=generator).to(device)
        batch = {name:value[positions] for name,value in data["train"].items()}
        pred = head(batch["memory"], batch["state"], batch["padding"])
        loss_bc = masked_l1_loss(pred, batch["target"], batch["mask"])
        loss_kd = masked_mse_loss(pred, batch["teacher"], batch["mask"])
        loss = loss_bc + 0.1*loss_kd
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite real-data training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for average, parameter in zip(ema.parameters(), head.parameters()):
                average.lerp_(parameter, 1-opt.ema_decay)
            for average, buffer in zip(ema.buffers(), head.buffers()):
                average.copy_(buffer)
        if step == 0:
            for handle in handles:
                handle.remove()
            emit("event_input_audit", learned_linear_modules=len(audited), all_inputs_unit_signed_or_binary=True)
        if (step+1)%32 == 0:
            emit("training_step", step=step+1, bc=loss_bc.item(), kd=loss_kd.item(), grad_norm=float(norm), elapsed=time.monotonic()-started)
    head.load_state_dict(best_state, strict=True)
    head.eval()
    resolution = state_resolution(head, sequences, device)
    final_values, final_predictions = evaluate(head, data["validation"])
    zero_state_data = dict(data["validation"], state=torch.zeros_like(data["validation"]["state"]))
    zero_values, zero_predictions = evaluate(head, zero_state_data)
    sensitivity = dict(
        intervention="Replace normalized proprioception with zero; images and language unchanged",
        zero_state_validation=zero_values,
        l1_change=zero_values["l1"]-final_values["l1"],
        mean_action_change=measure_predictions(zero_predictions, final_predictions, data["validation"]["mask"])["l1"],
    )
    export = {key:copy.deepcopy(cp[key]) for key in ("args","numerics","ema_decay","training_scope") if key in cp}
    export["model_config"] = cfg.to_dict()
    export["ema_model_state_dict"] = {key:value for key,value in cp["ema_model_state_dict"].items() if not key.startswith("action_head.")}
    export["ema_model_state_dict"].update({"action_head."+key:value for key,value in best_state.items()})
    export["global_step"] = int(cp["global_step"]) + int(best["step"])
    export["export_kind"] = "ema_evaluation_only"
    export["training_scope"] = dict(kind="bounded_head_only_repair", frozen="complete original AA vision/text/fusion", source_global_step=int(cp["global_step"]), repair_steps=int(best["step"]), population=opt.population, state_attention_gain=opt.state_gain, action_teacher=shards[0]["teacher_checkpoint"])
    export["args"].update(spike_state_population=opt.population, spike_state_population_range=4.0, spike_state_attention_gain=opt.state_gain, stats_path=str(SOURCE/"experiments/libero/configs/libero_all4_stats.json"), vision_model_source_path=cfg.vision.model_source_path)
    normalization = copy.deepcopy(cp["wta_normalization"])
    layer_stats = normalization["ema"].get("layers", {})
    normalization["ema"]["layers"] = {
        **{name:value for name,value in layer_stats.items() if not name.startswith("action_head.")},
        **{"action_head."+name:value for name,value in best_layers.items()},
    }
    normalization["ema"].update(status="refreshed_with_training_data", source="training_data", repair_head_refresh=dict(batches=128,batch_size=16,cache=opt.cache,optimizer_steps=0), frozen_upstream_statistics="unchanged from original evaluation export")
    export["wta_normalization"] = normalization
    export["ema_decay"] = opt.ema_decay
    export["repair"] = dict(source_checkpoint=opt.checkpoint, cache=opt.cache, seed=opt.seed, best_validation=best, original_baseline=baseline, teacher=teacher, state_resolution=resolution, state_sensitivity=sensitivity, optimizer_steps_run=opt.steps, selected_head_steps=best["step"], head_ema_decay=opt.ema_decay, batch_size=opt.batch_size, learning_rate=opt.lr, kd_weight=0.1, warning="Episode-disjoint only for head repair; no new full-benchmark success claim")
    with (output/"model_ema.pth").open("xb") as stream:
        torch.save(export, stream)
    torch.save(dict(predictions=final_predictions.cpu(), target=data["validation"]["target"].cpu(), mask=data["validation"]["mask"].cpu(), records=[record for shard in shards for record in shard["records"]["validation"]]), output/"validation_predictions.pt")
    save_json(output/"result.json", dict(repair=export["repair"], validation_history=results, final_validation=final_values, event_input_audit=dict(learned_linear_modules=len(audited), unit_events=True), elapsed=time.monotonic()-started))
    emit("training_complete", output=str(output), selected_step=best["step"], final_validation=final_values, state_resolution=resolution, state_sensitivity=sensitivity, elapsed=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cache","train"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--cache", nargs="+")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--train-frames", type=int, default=16)
    parser.add_argument("--validation-frames", type=int, default=8)
    parser.add_argument("--population", type=int, default=0)
    parser.add_argument("--state-gain", type=float, default=1.)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ema-decay", type=float, default=.99)
    opt = parser.parse_args()
    configure_transformers_offline()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    device = torch.device("cuda:0")
    if opt.mode == "cache":
        cache_features(opt, device)
    else:
        train_head(opt, device)


if __name__ == "__main__":
    main()
