"""Fresh common initialization -> BC-gradient check -> joint training -> LIBERO eval."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from experiments.libero.settings import (
    WORK, CODE, ROOT, CONTROL, RUN, CALIBRATION, PREFLIGHT, EVALUATION,
    PYTHON, BERT, DINO, SMOOTH, VISION, TEACHER, INIT, DATASETS,
    SUITES, GPU_IDS, GPUS, LIBERO_ROOT, LIBERO_CONFIG,
    MODULES, VISION_TYPE, TEXT_TYPE, CROSS_TYPE, BATCH_SIZE, GRAD_ACCUM,
    GLOBAL_BATCH, MAX_STEPS, SEED, EVAL_TRIALS, CHECKPOINT_PREFIX,
)


def stamp():
    return datetime.now().astimezone().isoformat()


def status(stage,**values):
    record={"time":stamp(),"stage":stage,**values}
    temporary=CONTROL/"status.json.tmp"
    temporary.write_text(json.dumps(record,indent=2)+"\n")
    temporary.replace(CONTROL/"status.json")
    print(json.dumps(record),flush=True)


def environment(code,gpus):
    env=os.environ.copy()
    for name in ("RANK","WORLD_SIZE","LOCAL_RANK","MASTER_ADDR","MASTER_PORT"):
        env.pop(name,None)
    env.update(CUDA_VISIBLE_DEVICES=gpus,NCCL_NVLS_ENABLE="0",
               PYTHONPATH=f"{code}:{code}/third_party/vla_adapter:{code}/third_party/starvla_runtime",
               PYTHONUNBUFFERED="1",TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",USE_TF="0",USE_FLAX="0",
               HF_HUB_DISABLE_TELEMETRY="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1",TF_NUM_INTEROP_THREADS="1",TF_NUM_INTRAOP_THREADS="1",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    return env


def free_gpus(wait_seconds=0):
    deadline=time.monotonic()+wait_seconds
    while True:
        output=subprocess.check_output(
            ["nvidia-smi","-i",GPUS,"--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],cwd=WORK,text=True)
        usage={int(row.split(",")[0]):int(row.split(",")[1]) for row in output.strip().splitlines()}
        if set(usage)==set(GPU_IDS) and max(usage.values())<=100:
            return usage
        if time.monotonic()>=deadline:
            raise RuntimeError(f"Permitted GPUs occupied; no process displaced: {usage}")
        time.sleep(3)


def run_command(command,log,exit_file,env,timeout):
    if log.exists() or exit_file.exists():
        raise RuntimeError(f"Refusing duplicate command or implicit retry: {log}")
    print(json.dumps({"time":stamp(),"event":"command_start","command":shlex.join(command),
                      "gpu":env["CUDA_VISIBLE_DEVICES"],"log":str(log)}),flush=True)
    with log.open("x") as stream:
        stream.write(stamp()+"\ncommand: "+shlex.join(command)+"\n")
        stream.flush()
        result=subprocess.run(["timeout","--signal=TERM","--kill-after=10s",str(timeout),*command],
                              cwd=WORK,env=env,stdout=stream,stderr=subprocess.STDOUT)
    exit_file.write_text(str(result.returncode)+"\n")
    if result.returncode:
        raise RuntimeError(f"Command failed with {result.returncode}: {log}; no automatic retry")


def training_arguments(code,run):
    arguments = [
        "--dataset_dir",DATASETS[0],"--dataset_dirs",",".join(DATASETS),"--dataset_format","lerobot",
        "--lerobot_gripper_convention","binary_open","--normalize_binary_gripper","false",
        "--stats_path",str(code/"experiments/libero/configs/libero_all4_stats.json"),
        "--stats_key","libero_all4_no_noops",
        "--text_layout_path",str(code/"experiments/libero/configs/online_text_layout.json"),
        "--dinov3_path",DINO,"--bert_path",BERT,"--no_allow_hf_download",
        "--text_encoder_type",TEXT_TYPE,"--text_model_path",SMOOTH if TEXT_TYPE == "smoothspike_bert" else BERT,
        "--text_timesteps","4" if TEXT_TYPE == "smoothspike_bert" else "1",
        "--text_mask_version","legacy","--text_sentence_mask_version","legacy",
        "--text_head_mask_version","corrected","--teacher_text_head_mask_version","corrected",
        "--joint_scope_audit",
        "--vision_encoder_type",VISION_TYPE,"--vision_model_path",VISION if VISION_TYPE == "sdtv3_19m" else DINO,
        "--vision_image_size","224" if VISION_TYPE == "sdtv3_19m" else "256",
        "--vision_output_grid_size","14",
        "--cross_attention_type",CROSS_TYPE,"--cross_timesteps","4","--cross_gradient_checkpointing",
        "--pretrained_init_ckpt",INIT,"--teacher_checkpoint",TEACHER,
        "--distill_feature_weight","0.1","--distill_action_weight","0.1","--distill_text_weight","0",
        "--checkpoint_dir",str(run/"checkpoints"),"--checkpoint_prefix",CHECKPOINT_PREFIX,
        "--batch_size",str(BATCH_SIZE),"--grad_accum_steps",str(GRAD_ACCUM),
        "--precision","fp32","--dinov3_precision","fp32","--vision_precision","fp32",
        "--float32_matmul_precision","highest",
        "--no_freeze_backbones","--freeze_text_encoder",
        "--lr","5e-5","--head_lr","5e-5","--dinov3_lr","1e-5","--vision_encoder_lr","1e-5",
        "--weight_decay","1e-10","--head_weight_decay","1e-10",
        "--dinov3_weight_decay","1e-10","--vision_encoder_weight_decay","1e-10",
        "--max_steps",str(MAX_STEPS),"--lr_schedule_steps",str(MAX_STEPS),"--warmup_steps","3334",
        "--min_lr_ratio","1","--save_steps","5000","--extra_save_steps","20","1000",
        "--num_workers","4","--seed",str(SEED),"--log_freq","20","--resume_mode","none"]
    if VISION_TYPE == "sdtv3_19m":
        arguments += [
            "--vision_pretrained_checkpoint", VISION,
            "--vision_model_source_path", str(code/"third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"),
        ]
    if TEXT_TYPE == "smoothspike_bert":
        arguments += ["--text_projection_init", str(CALIBRATION/"text_projection.pth")]
    return arguments


def evaluate(code,checkpoint):
    EVALUATION.mkdir(exist_ok=False)
    status("joint_evaluating",checkpoint=str(checkpoint),evaluation=str(EVALUATION))
    def suite_job(gpu,suite):
        command=[PYTHON,str(code/"experiments/libero/evaluate.py"),
                 "--ckpt_path",str(checkpoint),"--checkpoint_weights","ema",
                 "--libero_root",LIBERO_ROOT,"--dinov3_path",DINO,"--bert_path",BERT,
                 "--stats_path",str(code/"experiments/libero/configs/libero_all4_stats.json"),
                 "--stats_key","libero_all4_no_noops","--normalize_binary_gripper","false",
                 "--task_suite_name",suite,"--num_trials_per_task",str(EVAL_TRIALS),"--num_steps_wait","10",
                 "--num_open_loop_steps","12","--chunk_size","12","--env_img_res","256",
                 "--seed","7","--control_mode","relative","--precision","fp32",
                 "--float32_matmul_precision","auto",
                 "--mujoco_gl","osmesa","--pyopengl_platform","osmesa","--save_video","false",
                 "--video_out_path",str(EVALUATION/"videos"),
                 "--result_json_path",str(EVALUATION/(suite+".result.json")),
                 "--log_path",str(EVALUATION/(suite+".events.log"))]
        env=environment(code,str(gpu))
        env.update(LIBERO_CONFIG_PATH=LIBERO_CONFIG,
                   MUJOCO_GL="osmesa",PYOPENGL_PLATFORM="osmesa",OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1",LP_NUM_THREADS="4")
        run_command(command,EVALUATION/(suite+".stdout.log"),EVALUATION/(suite+".exit_code"),env,14400 * max(1, (EVAL_TRIALS + 4) // 5))
    def gpu4():
        suite_job(GPU_IDS[0],"libero_spatial")
        suite_job(GPU_IDS[0],"libero_object")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(gpu4),pool.submit(suite_job,GPU_IDS[1],"libero_goal"),pool.submit(suite_job,GPU_IDS[2],"libero_10")]
        for job in jobs:
            job.result()
    summary={"checkpoint":str(checkpoint),"checkpoint_weights":"ema","seed":7,"trials_per_task":EVAL_TRIALS,
             "float32_matmul_precision_requested":"auto",
             "suites":{},"total_successes":0,"total_episodes":0}
    for suite in SUITES:
        result=json.loads((EVALUATION/(suite+".result.json")).read_text())
        if result["total_episodes"]!=10 * EVAL_TRIALS or result["num_trials_per_task"]!=EVAL_TRIALS:
            raise RuntimeError(f"Incomplete evaluation: {suite}")
        if result["numerics"]["float32_matmul_precision"] != "highest":
            raise RuntimeError(f"New-run train/eval matmul precision mismatch: {suite}")
        summary["numerics"] = result["numerics"]
        summary["suites"][suite]={"successes":result["total_successes"],"episodes":10 * EVAL_TRIALS}
        summary["total_successes"]+=result["total_successes"]
        summary["total_episodes"]+=10 * EVAL_TRIALS
    summary["overall_success_rate"]=summary["total_successes"]/summary["total_episodes"]
    summary["finished_at"]=stamp()
    (EVALUATION/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (EVALUATION/"overall.exit_code").write_text("0\n")
    return summary


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock=(CONTROL/"pipeline.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if RUN.exists() or (CONTROL/"pipeline.exit_code").exists():
        raise RuntimeError("Existing run found; no implicit restart or overwrite")
    try:
        if CALIBRATION.exists() or PREFLIGHT.exists():
            raise RuntimeError("Existing preparation output found; choose a new SPIKE_RUN_NAME")
        free_gpus()
        if TEXT_TYPE == "smoothspike_bert":
            status("text_interface_calibration", calibration=str(CALIBRATION), modules=MODULES)
            run_command([PYTHON, str(CODE/"experiments/libero/prepare_cross_joint.py"),
                         "--mode", "calibrate", "--output_dir", str(CALIBRATION)],
                        CONTROL/"calibration.log", CONTROL/"calibration.exit_code",
                        environment(CODE, str(GPU_IDS[0])), 900)
            calibration=json.loads((CALIBRATION/"result.json").read_text())
            if calibration["metadata"]["source_policy_checkpoint"] is not None:
                raise RuntimeError("Calibration inherited a trained VLA checkpoint")
        if shutil.disk_usage(WORK).free<32*2**30:
            raise RuntimeError("Less than 32 GiB disk free; no history removed")
        free_gpus()
        status("joint_gradient_preflight",preflight=str(PREFLIGHT),modules=MODULES)
        run_command([PYTHON,str(CODE/"experiments/libero/prepare_cross_joint.py"),
                     "--mode","preflight","--output_dir",str(PREFLIGHT)],
                    CONTROL/"preflight.log",CONTROL/"preflight.exit_code",environment(CODE,str(GPU_IDS[0])),900)
        check=json.loads((PREFLIGHT/"result.json").read_text())
        if check["optimizer_steps"]!=0 or check["training_scope"]!="joint_except_frozen_text_backbone":
            raise RuntimeError("Incorrect joint gradient preflight scope")
        if any(item["gradient_l2"]<=0 for item in check["gradients"].values()):
            raise RuntimeError("A required joint-training gradient is missing")
        free_gpus(60)
        RUN.mkdir(exist_ok=False)
        (RUN/"checkpoints").mkdir()
        snapshot=RUN/"source_snapshot"
        shutil.copytree(CODE,snapshot,symlinks=True,ignore=shutil.ignore_patterns(".git", ".deps", ".venv", "venv", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", "*.egg-info", ".env", ".env.*", "outputs", "checkpoints", "pretrained", "datasets", "build", "dist", ".bootstrap-*"))
        initialization={
            "training_scope":"joint_except_frozen_text_backbone","source_policy_checkpoint":None,
            "modules":MODULES,
            "vision_initialization":VISION if VISION_TYPE == "sdtv3_19m" else DINO,
            "interaction_initialization":INIT,
            "text_backbone_initialization":SMOOTH if TEXT_TYPE == "smoothspike_bert" else BERT,
            "text_projection_initialization":str(CALIBRATION/"text_projection.pth") if TEXT_TYPE == "smoothspike_bert" else INIT,
            "action_head_initialization":f"original model constructor, seed {SEED}",
            "teacher":TEACHER,"optimizer_initial_state":"new; no resume",
            "batch_per_gpu":BATCH_SIZE,"gradient_accumulation":GRAD_ACCUM,
            "global_batch":GLOBAL_BATCH,"optimizer_steps":MAX_STEPS,
            "float32_matmul_precision":"highest",
            "loss":"LIBERO action L1 + 0.1 pre-fusion visual MSE KD + 0.1 action MSE KD",
            "calibration_note":"initialization only; projection participates in joint optimization" if TEXT_TYPE == "smoothspike_bert" else "not used for ANN BERT",
            "mask_note":"legacy sentence protocol; corrected batch/head indexing for both student and teacher",
        }
        (RUN/"initialization.json").write_text(json.dumps(initialization,indent=2)+"\n")
        command=[PYTHON,"-m","torch.distributed.run","--standalone","--nproc_per_node=3",
                 str(snapshot/"experiments/libero/train.py"),*training_arguments(snapshot,RUN)]
        status("joint_training",run=str(RUN),**initialization)
        run_command(command,RUN/"train.log",RUN/"train.exit_code",environment(snapshot,GPUS),604800)
        import torch
        checkpoint=RUN/"checkpoints"/f"{CHECKPOINT_PREFIX}_{MAX_STEPS}.pth"
        saved=torch.load(checkpoint,map_location="cpu",mmap=True,weights_only=False)
        if (saved["global_step"]!=MAX_STEPS or not saved.get("ema_model_state_dict") or
            saved.get("numerics",{}).get("float32_matmul_precision")!="highest" or
            saved.get("training_scope")!="joint_except_frozen_text_backbone"):
            raise RuntimeError("Final joint checkpoint is incomplete")
        del saved
        free_gpus(60)
        result=evaluate(snapshot,checkpoint)
        status("complete",run=str(RUN),evaluation=str(EVALUATION),result=result)
        (CONTROL/"pipeline.exit_code").write_text("0\n")
    except BaseException as error:
        status("failed",error=repr(error),no_automatic_retry=True)
        (CONTROL/"pipeline.exit_code").write_text("1\n")
        raise


if __name__=="__main__":
    main()
