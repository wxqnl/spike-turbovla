"""Native MT50 open-loop rollout used by the fixed public benchmark."""

def rollout_episode(policy, env, reset_seed):
    raw, _ = env.reset(reset_seed)
    initial = {
        "random_vector": env.env._last_rand_vec.tolist(),
        "target": env.env._target_pos.tolist(),
        "state4": raw[:4].tolist(),
    }
    chunk = None
    offset = 0
    success = False
    for step in range(500):
        if chunk is None or offset == 12:
            chunk = policy.predict(env.image(), raw, env.native_task_id)
            offset = 0
        action = policy.adapter.native_action(chunk[offset])
        offset += 1
        raw, reward, done, truncated, info = env.step(action)
        success = bool(info["is_success"])
        if done or truncated:
            break
    return {
        "native_task_id": env.native_task_id,
        "env_name": env.name,
        "reset_seed": reset_seed,
        "initial": initial,
        "success": success,
        "steps": step + 1,
    }
