"""How coarsely does the SpikeACT state encoder see the robot state?

Each normalized state coordinate drives ONE PLIF neuron (hard reset, scalar threshold shared by all
coordinates) for T = 4 steps; the spike count is all the network receives about it. We search the
neuron's decay and threshold for the best affine decoding of N(0, 1) inputs and convert the error to
physical units with the LIBERO state std.

    python proprio_quantization.py /path/to/Spike-driven-VLA/configs/libero_stats.json
"""
import json
import sys

import numpy as np


def run(current, T=4, a=0.5, th=0.5):
    u = np.zeros_like(current)
    out = np.zeros_like(current)
    for _ in range(T):
        v = u + a * (current - u)
        s = (v >= th).astype(float) - (v <= -th).astype(float)
        u = v * (1 - np.abs(s))
        out += s
    return out / T


stats = json.load(open(sys.argv[1]))["libero_all4_no_noops"]["state"]
std = np.array(stats["std"])
z = np.random.default_rng(0).standard_normal(200_000)
best = None
for a in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
    for th in np.geomspace(0.05, 3, 60):
        y = run(z, a=a, th=th)
        A = np.stack([y, np.ones_like(y)], 1)
        c, *_ = np.linalg.lstsq(A, z, rcond=None)
        err = float(np.sqrt(np.mean((A @ c - z) ** 2)))
        if best is None or err < best[0]:
            best = (err, a, th, len(np.unique(y)))
err, a, th, levels = best
print(f"best single-neuron code at T=4: {levels} levels, RMSE {err:.3f} std (decay {a}, threshold {th:.2f})")
print("end-effector x/y/z error (cm):", np.round(err * std[:3] * 100, 2))
