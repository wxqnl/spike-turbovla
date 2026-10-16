"""SpikeACT cross-attention routing share by memory group under index vs random tie-breaking.

Memory order follows the release: [agent view 196 | wrist view 196 | instruction 21 | state 8],
k = 8 winners per query, head_dim = 32. Spikes are i.i.d. ternary with firing rate p, so this
isolates what the tie rule alone does; a trained checkpoint is needed for real statistics.
"""
import numpy as np

rng = np.random.default_rng(0)
N, k, d, rows = 421, 8, 32, 6000
groups = {"agent view": slice(0, 196), "wrist view": slice(196, 392),
          "instruction": slice(392, 413), "state": slice(413, 421)}
print("fair share:", "  ".join(f"{g}: {(s.stop - s.start) / N:.1%}" for g, s in groups.items()))
for p in (0.1, 0.2):
    u = rng.random((rows, N + 1, d))
    spikes = np.where(u < p / 2, 1, np.where(u < p, -1, 0)).astype(np.int16)
    q, keys = spikes[:, 0], spikes[:, 1:]
    score = np.einsum("rd,rnd->rn", q, keys).astype(float)
    score[~(keys != 0).any(-1)] = -np.inf  # silent keys cannot win
    for name, jitter in (("index tie-break", np.zeros_like(score)), ("random tie-break", rng.random(score.shape) * 0.5)):
        winners = np.argsort(-(score + jitter), kind="stable", axis=1)[:, :k]
        mask = np.zeros_like(score, bool)
        np.put_along_axis(mask, winners, True, 1)
        share = {g: mask[:, s].sum() / mask.sum() for g, s in groups.items()}
        print(f"p={p} {name:17s} " + "  ".join(f"{g}: {v:5.1%}" for g, v in share.items()))
