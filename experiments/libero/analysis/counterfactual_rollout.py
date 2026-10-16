"""Does the policy read where the target is? Plate-shift counterfactuals on LIBERO-Spatial.

The official rollout runs unchanged (vla_adapter.rollout, as in failure_rollout.py). After each episode, still in the
same scene model and before the next reset, we restore simulator states of that episode, move only the plate
(its free joint, in xy) and let the same policy act:

  closed-loop branches  from the first policy frame ("t0") and from the start of transport ("transport": first policy
                        query once the target bowl is grasped and lifted >= 5 cm), plate offsets of +-1.5 cm in x and in y plus
                        an unshifted re-run; each runs to success or the episode's remaining step budget. Causal
                        tracking gain = landing shift / actual plate shift, ideal 1.
  open-loop probes      at the same two frames, 8 random plate offsets in [-2, 2]^2 cm plus the unshifted render;
                        records the predicted action chunk and features at each stage of the policy (vision encoder
                        output, projected visual tokens, fused visual tokens, action-decoder output), each reduced by
                        a fixed seeded Gaussian projection to 2048 dims, to locate where the plate's position is lost.

Branches do not reset the environment and use no environment randomness, so the official episode sequence (episode
order is part of the protocol) is unchanged; the unshifted re-runs (both start at policy queries) double as determinism checks.
Run like failure_rollout.py, plus --record_path and --num_trials_per_task N (first N initial states, in order).
"""
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_rollout import RecordingEnv  # noqa: E402
from vla_adapter import rollout  # noqa: E402

LIFT_TRANSPORT = 0.05
SHIFTS = [(0.0, 0.0), (0.015, 0.0), (-0.015, 0.0), (0.0, 0.015), (0.0, -0.015)]
PROBE_OFFSETS, PROBE_RANGE, PROJ_DIM = 8, 0.02, 2048
# "approach" (head_probe.py): first policy query after transport with the held bowl within APPROACH_XY of the plate
FRAMES, APPROACH_XY = ("t0", "transport"), 0.05


class StateRecordingEnv(RecordingEnv):
    """Also keeps, after every step, what a restore needs beyond the MuJoCo state:
    - the gripper's integrated command: the Panda gripper integrates its +-1 command by 0.01 per step in
      `gripper.current_action`; it starts at 0 in every official episode because each hard reset rebuilds the model;
    - the OSC controller goal: the orientation goal is only updated by a nonzero rotation delta, and env.reset() sets
      it at the robot's default joint pose, before set_init_state moves the arm, so the 10 settle steps turn the arm
      toward that default orientation."""

    def _snapshot(self, obs):
        super()._snapshot(obs)
        self._states.append(self._inner.sim.get_state().flatten().copy())
        controller = self._inner.robots[0].controller
        self._ctrl_goal.append((np.array(controller.goal_pos, np.float64), np.array(controller.goal_ori, np.float64)))
        command = np.asarray(self._inner.robots[0].gripper.current_action, np.float64)
        # starts as zeros(1) and becomes [-c, c] after the first step; identical behaviour, keep the 2-vector form
        self._grip.append(np.array([-command[0], command[0]]) if command.size == 1 else command.copy())

    def set_init_state(self, state):
        self._states, self._grip, self._ctrl_goal = [], [], []
        return super().set_init_state(state)

    def episode(self):
        out = super().episode()
        out["sim_states"], out["gripper_command"] = np.stack(self._states), np.stack(self._grip)
        out["controller_goal_pos"] = np.stack([g[0] for g in self._ctrl_goal])
        out["controller_goal_ori"] = np.stack([g[1] for g in self._ctrl_goal])
        return out


def project(x, name):
    """[N, ...] -> [N, PROJ_DIM]: x @ R with R ~ N(0, 1/PROJ_DIM), generated in fixed seeded row blocks."""
    x = x.reshape(len(x), -1).float()
    out = torch.zeros(len(x), PROJ_DIM, device=x.device)
    block = 8192
    for i in range(0, x.shape[1], block):
        g = torch.Generator(device=x.device).manual_seed(zlib.crc32(f"{name}:{x.shape[1]}:{i}".encode()))
        r = torch.randn(min(block, x.shape[1] - i), PROJ_DIM, generator=g, device=x.device) / PROJ_DIM ** 0.5
        out += x[:, i:i + block] @ r
    return out.half().cpu().numpy()


class Taps:
    """Stage outputs of TurboVLA-family models (spiking base and ANN TurboVLA share the module layout)."""

    def __init__(self, model):
        self.model, self.store = model, {}

        def keep(name):
            def hook(module, inputs, output):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                self.store.setdefault(name, []).append(output.detach())
            return hook
        model.vision_encoder.register_forward_hook(keep("vision"))
        model.vision_language_interaction.register_forward_hook(keep("fused"))
        model.action_head.decoder.register_forward_hook(keep("decoder"))

    def run(self, policy, primaries, wrists, states, instruction):
        self.store = {}
        samples, state_t = policy._build_batch(primaries, wrists, states)
        samples, state_t = policy._prepare_model_inputs(samples, state_t)
        with torch.inference_mode():
            actions, features = self.model([instruction] * len(primaries), samples, state_t,
                                           return_distillation_features=True)
        stages = {k: torch.cat(v) for k, v in self.store.items()}
        stages["projected"] = features["visual"]
        n = len(primaries)
        for k, v in stages.items():
            if v.shape[0] != n:  # per-view rows [B*2, ...], each sample's views contiguous
                assert v.shape[0] % n == 0, (k, v.shape)
                stages[k] = v.reshape(n, -1)
        if "shapes" not in self.__dict__:
            self.shapes = {k: tuple(v.shape) for k, v in stages.items()}
            print("[counterfactual] stage shapes", self.shapes, flush=True)
        nvis = features["visual"].shape[1]
        if stages["fused"].dim() == 3 and stages["fused"].shape[1] > nvis:
            stages["fused"] = stages["fused"][:, :nvis]
        return actions.float().cpu().numpy(), {k: project(v, k) for k, v in stages.items()}


def main():
    argv = sys.argv[1:]
    i = argv.index("--record_path")
    record_path = Path(argv[i + 1])
    del argv[i:i + 2]
    sys.argv = [sys.argv[0], *argv]
    sink = {"episodes": []}
    current = {}
    make_env, run_episode = rollout._make_libero_env, rollout._run_episode

    def recording_make_env(task, cfg):
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
        task_id = [t.name for t in suite.tasks].index(task.name)
        current["states"] = suite.get_task_init_states(task_id)
        current["count"] = 0
        env, description = make_env(task, cfg)
        current["env"] = StateRecordingEnv(env, task, sink)
        return current["env"], description

    def branch_run(cfg, env, policy, description, state, grip, goal, budget, wait, rotate, dummy, objs):
        """Restore state (plate already moved), the gripper's integrated command and the controller goal recorded at
        that snapshot, optionally settle with dummy actions, run the policy closed-loop."""
        raw, inner = env._env, env._inner
        inner.timestep, inner.done = 0, False
        inner.robots[0].gripper.current_action = np.array(grip, dtype=np.float64)
        inner.sim.reset()  # fresh MjData (solver warm start, ctrl) so that every branch starts from the same data
        obs = raw.set_init_state(state)
        controller = inner.robots[0].controller
        controller.update(force=True)
        controller.goal_pos, controller.goal_ori = np.array(goal[0]), np.array(goal[1])
        for _ in range(wait):
            obs, _, _, _ = raw.step(dummy.tolist())
        start = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]][:2] for n in objs]).copy()
        success, queue, t = False, [], 0
        while t < budget:
            if not queue:
                queue = list(policy.predict_env_action_chunk(rotate(obs["agentview_image"]),
                                                             rotate(obs["robot0_eye_in_hand_image"]), description,
                                                             obs, execute_steps=cfg.num_open_loop_steps))
            obs, _, done, _ = raw.step(queue.pop(0).tolist())
            t += 1
            if done:
                success = True
                break
        end = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]][:2] for n in objs]).copy()
        return {"success": success, "steps": t, "start_xy": start, "end_xy": end}

    def recording_run_episode(cfg, env, policy, description, initial_state, get_libero_dummy_action,
                              rotate_libero_image, *args, **kwargs):
        index = current["count"]
        current["count"] += 1
        initial_state = current["states"][index]
        success, images, actions = run_episode(cfg, env, policy, description, initial_state, get_libero_dummy_action,
                                               rotate_libero_image, *args, **kwargs)
        ep = {"success": bool(success), "init_index": index, "actions": np.asarray(actions, np.float32),
              "wait_steps": cfg.num_steps_wait, **env.episode()}
        inner, raw = env._inner, env._env
        objs = sink["task"]["objects"]
        goal = sink["task"]["goal"][0]
        bowl, plate = objs.index(goal[1]), objs.index(goal[2])
        joint = inner.objects_dict[goal[2]].joints[0]
        qpos = inner.sim.model.get_joint_qpos_addr(joint)[0]
        dummy = np.asarray(get_libero_dummy_action(), np.float32)
        wait = cfg.num_steps_wait
        max_steps = rollout.TASK_MAX_STEPS[cfg.task_suite_name]
        z0 = ep["object_pos"][wait, bowl, 2]
        lifted = np.flatnonzero(ep["grasp"][wait:, bowl] & (ep["object_pos"][wait:, bowl, 2] - z0 >= LIFT_TRANSPORT))
        frames = {"t0": wait}
        if len(lifted):  # first chunk boundary (policy query) at or after the lift, so the unshifted re-run replays
            snap = wait + cfg.num_open_loop_steps * int(np.ceil(lifted[0] / cfg.num_open_loop_steps))
            if snap < len(ep["sim_states"]) - 1:
                frames["transport"] = snap
        if "approach" in FRAMES and "transport" in frames:
            start = frames["transport"]
            gap = np.linalg.norm(ep["object_pos"][start:, bowl, :2] - ep["object_pos"][start:, plate, :2], axis=-1)
            near = np.flatnonzero(ep["grasp"][start:, bowl] & (gap < APPROACH_XY))
            if len(near):
                snap = start + cfg.num_open_loop_steps * int(np.ceil(near[0] / cfg.num_open_loop_steps))
                if snap < len(ep["sim_states"]) - 1:
                    frames["approach"] = snap

        def shifted(state, dxy):
            s = state.copy()
            s[1 + qpos:1 + qpos + 2] += dxy  # flattened state = [time, qpos, qvel]
            return s

        def goal_at(snap):
            return ep["controller_goal_pos"][snap], ep["controller_goal_ori"][snap]

        ep["frames"], ep["branches"], ep["probes"] = frames, [], {}
        rng = np.random.default_rng(1000 + index)
        for name, snap in frames.items():
            for dxy in SHIFTS:
                if name == "t0":  # move the plate in the initial state, then the usual settle steps
                    r = branch_run(cfg, env, policy, description, shifted(initial_state, dxy),
                                   np.zeros_like(ep["gripper_command"][0]), goal_at(0), max_steps, wait,
                                   rotate_libero_image, dummy, objs)
                else:
                    r = branch_run(cfg, env, policy, description, shifted(ep["sim_states"][snap], dxy),
                                   ep["gripper_command"][snap], goal_at(snap), max_steps - (snap - wait), 0,
                                   rotate_libero_image, dummy, objs)
                ep["branches"].append({"frame": name, "nominal": dxy, **r})
            offsets = np.r_[np.zeros((1, 2)), rng.uniform(-PROBE_RANGE, PROBE_RANGE, (PROBE_OFFSETS, 2))]
            primaries, wrists, states, actual = [], [], [], []
            from turbovla.evaluation.policy import state_from_libero_obs
            for dxy in offsets:
                obs = raw.set_init_state(shifted(ep["sim_states"][snap], dxy))
                primaries.append(rotate_libero_image(obs["agentview_image"]))
                wrists.append(rotate_libero_image(obs["robot0_eye_in_hand_image"]))
                states.append(state_from_libero_obs(obs))
                actual.append(inner.sim.data.body_xpos[inner.obj_body_id[goal[2]]][:2].copy())
            if "taps" not in current:
                current["taps"] = Taps(policy.model)
            pred, feats = current["taps"].run(policy, primaries, wrists, states, description)
            ep["probes"][name] = {"plate_xy": np.array(actual), "actions": pred, "features": feats,
                                  "eef_xy": ep["eef"][snap, :2].copy(), "bowl_xy": ep["object_pos"][snap, bowl, :2]}
        sink["episodes"].append(ep)
        print(f"[counterfactual] init {index} success={success} frames={frames} "
              f"branches={[(b['frame'], b['nominal'], b['success']) for b in ep['branches']]}", flush=True)
        return success, images, actions

    rollout._make_libero_env, rollout._run_episode = recording_make_env, recording_run_episode
    rollout.main()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "wb") as f:
        pickle.dump(sink, f)


if __name__ == "__main__":
    main()
