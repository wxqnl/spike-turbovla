"""Fast subset of head_analysis.py: plate-offset R^2 for a few key stages only, stages in parallel processes.
Usage: python key_r2.py <root> [--stages vterm0,ctxvis0,...] [--frames t0,transport]
"""
import argparse
import os
import sys

# One BLAS thread per worker: on the 128-core training host, 16 workers x 64 BLAS threads starved the training runs'
# data loaders (2026-10-13: both runs 2.4 -> 4.8 s/step). KEY_R2_WORKERS caps the worker count (default 8).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterfactual_analysis import load, ridge_cv  # noqa: E402

DEFAULT = "vvis0,rterm0,vterm0,ctxvis0,ctx0,ctxspk0,cross0,layer0,cross2,layer2,readout0,actions"


def fit(job):
    model, frame, stage, X, y, groups = job
    return model, frame, stage, float(ridge_cv(X, y, groups))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--stages", default=DEFAULT)
    p.add_argument("--frames", default="t0,transport")
    args = p.parse_args()
    jobs = []
    for model, eps in load(args.root).items():
        for frame in args.frames.split(","):
            probes = [(k, e["probes"][frame]) for k, e in enumerate(eps) if frame in e["probes"]]
            if len(probes) < 5:
                continue
            groups = np.concatenate([[k] * (len(pr["plate_xy"]) - 1) for k, pr in probes])
            y = np.concatenate([(pr["plate_xy"][1:] - pr["plate_xy"][0]) * 100 for _, pr in probes])
            for stage in args.stages.split(","):
                if stage == "actions":
                    X = np.concatenate([(pr["actions"][1:] - pr["actions"][:1]).reshape(len(pr["actions"]) - 1, -1)
                                        for _, pr in probes]).astype(np.float64)
                elif stage in probes[0][1]["features"]:
                    X = np.concatenate([pr["features"][stage][1:].astype(np.float64)
                                        - pr["features"][stage][:1].astype(np.float64) for _, pr in probes])
                else:
                    continue
                jobs.append((model, frame, stage, X, y, groups))
    results = {}
    with ProcessPoolExecutor(max_workers=min(int(os.environ.get("KEY_R2_WORKERS", 8)), len(jobs))) as pool:
        for model, frame, stage, r2 in pool.map(fit, jobs):
            results.setdefault((model, frame), {})[stage] = r2
    for (model, frame), row in sorted(results.items()):
        order = [s for s in args.stages.split(",") if s in row]
        print(f"{model:9s} {frame:9s} " + "  ".join(f"{s} {row[s]:.2f}" for s in order))


if __name__ == "__main__":
    main()
