"""How often does top-k routing over ternary spike affinities hit ties, and which sources win them?

Sites follow the release config: (name, candidates N, k, head_dim d). Spikes are i.i.d.
ternary with firing rate p (half +1, half -1). Silent keys are excluded, as in routing.py.
Tie-break is stable argsort = lowest source index wins.
"""
import numpy as np

rng = np.random.default_rng(0)
sites = [("MWSF visual->language (k_v)", 21, 3, 256), ("MWSF language->visual (k_l)", 392, 8, 256),
         ("language self-attn", 21, 3, 64), ("SpikeACT cross (memory)", 421, 8, 32)]
print(f"{'site':30s} {'p':>5s} {'boundary tie':>13s} {'winners decided by index':>25s} {'winner idx / N':>15s}")
for name, N, k, d in sites:
    for p in (0.05, 0.1, 0.2):
        rows = 4000
        def spikes(n):
            u = rng.random((rows, n, d))
            return np.where(u < p / 2, 1, np.where(u < p, -1, 0)).astype(np.int16)
        q = spikes(1)[:, 0]
        k_ = spikes(N)
        s = np.einsum("rd,rnd->rn", q, k_).astype(float)
        live = (k_ != 0).any(-1)
        s[~live] = -np.inf
        order = np.argsort(-s, kind="stable", axis=1)
        ss = np.take_along_axis(s, order, 1)
        kth, nxt = ss[:, k - 1], ss[:, k]
        tie = (kth == nxt) & np.isfinite(kth)
        # winners whose score equals the k-th score are chosen among an equal set by index
        decided = ((ss[:, :k] == kth[:, None]) & tie[:, None]).sum(1) / k
        idx = order[:, :k].mean(1) / (N - 1)
        print(f"{name:30s} {p:5.2f} {tie.mean():13.1%} {decided.mean():25.1%} {idx.mean():15.2f}")
