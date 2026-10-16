"""Analysis of counterfactual_rollout.py records.

1. Determinism: the unshifted re-runs (t0, transport) must reproduce the official episode (success, landing).
2. Causal tracking gain, per branch frame (t0, transport): landing shift of the bowl vs actual plate shift, both
   relative to the unshifted branch from the same state; pooled least squares (2x2), reported as the mean of the
   diagonal with a bootstrap over episodes. Ideal 1.
3. Open-loop response: the predicted chunk's summed xy translation (env units) vs the rendered plate offset.
4. Where the plate offset is lost: for each stage (vision encoder output, projected visual tokens, fused visual
   tokens, action-decoder output, predicted actions), decode the plate offset from the feature change it causes
   (shifted minus unshifted render of the same state) with linear kernel ridge, cross-validated by episode. R^2.
Usage: python counterfactual_analysis.py <root> --stats <libero_all4_stats.json> [--stats_key libero_all4_no_noops]
       (root holds <model>/*.pkl or <model>.pkl)
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGES = ["vision", "projected", "fused", "decoder", "actions"]


def load(root):
    out = defaultdict(list)
    for f in sorted(Path(root).glob("*.pkl")) + sorted(Path(root).glob("*/**/*.pkl")):
        model = f.stem if f.parent == Path(root) else f.relative_to(root).parts[0]
        rec = pickle.loads(f.read_bytes())
        task = rec["task"]
        for ep in rec["episodes"]:
            ep["_task"], ep["_file"] = task, f.name
            out[model].append(ep)
    return out


def indices(ep):
    objs, goal = ep["_task"]["objects"], ep["_task"]["goal"][0]
    return objs.index(goal[1]), objs.index(goal[2])


def gains(eps, frame, rng, near=None):
    rows = []  # (episode id, d_plate, d_landing)
    for k, ep in enumerate(eps):
        bowl, plate = indices(ep)
        br = [b for b in ep["branches"] if b["frame"] == frame]
        if not br:
            continue
        ctrl = [b for b in br if tuple(b["nominal"]) == (0.0, 0.0)][0]
        for b in br:
            if b is ctrl:
                continue
            if near is not None and any(np.linalg.norm(x["end_xy"][bowl] - x["end_xy"][plate]) > near for x in (b, ctrl)):
                continue
            rows.append((k, b["start_xy"][plate] - ctrl["start_xy"][plate], b["end_xy"][bowl] - ctrl["end_xy"][bowl]))
    if not rows:
        return None
    ids = np.array([r[0] for r in rows])
    X, Y = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])

    def fit(i):
        coef, *_ = np.linalg.lstsq(X[i], Y[i], rcond=None)
        return np.trace(coef) / 2
    g = fit(np.arange(len(X)))
    uniq = np.unique(ids)
    boots = []
    for _ in range(1000):
        pick = rng.choice(uniq, len(uniq))
        boots.append(fit(np.concatenate([np.flatnonzero(ids == u) for u in pick])))
    return g, np.percentile(boots, [2.5, 97.5]), len(X)


def ridge_cv(X, y, groups, folds=5, lambdas=10.0 ** np.arange(-4, 4)):
    X = X / max(np.linalg.norm(X, axis=1).mean(), 1e-12)
    uniq = np.unique(groups)
    assign = {g: i % folds for i, g in enumerate(np.random.default_rng(0).permutation(uniq))}
    fold = np.array([assign[g] for g in groups])
    pred = np.zeros_like(y)
    for f in range(folds):
        tr, te = fold != f, fold == f
        # inner choice of lambda on the training groups
        inner = np.array([assign[g] for g in groups[tr]]) % (folds - 1)
        best, best_err = lambdas[0], np.inf
        Ktr = X[tr] @ X[tr].T
        for lam in lambdas:
            err = 0
            for j in range(folds - 1):
                a, b = inner != j, inner == j
                alpha = np.linalg.solve(Ktr[np.ix_(a, a)] + lam * np.eye(a.sum()), y[tr][a])
                err += ((Ktr[np.ix_(b, a)] @ alpha - y[tr][b]) ** 2).sum()
            if err < best_err:
                best, best_err = lam, err
        alpha = np.linalg.solve(Ktr + best * np.eye(tr.sum()), y[tr])
        pred[te] = X[te] @ X[tr].T @ alpha
    return 1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--stats", required=True)
    p.add_argument("--stats_key", default="libero_all4_no_noops")
    args = p.parse_args()
    stats = json.loads(Path(args.stats).read_text())[args.stats_key]
    a_min, a_max = np.asarray(stats["action"]["min"], np.float32), np.asarray(stats["action"]["max"], np.float32)
    data = load(args.root)
    rng = np.random.default_rng(0)
    report = {}
    for model, eps in data.items():
        rep = report[model] = {}
        # 1. determinism
        print(f"\n== {model}: {len(eps)} episodes, official success {sum(e['success'] for e in eps)}")
        for frame in ("t0", "transport"):
            same = []
            for ep in eps:
                bowl, _ = indices(ep)
                ctrl = [b for b in ep["branches"] if b["frame"] == frame and tuple(b["nominal"]) == (0.0, 0.0)]
                if ctrl:
                    same.append((ctrl[0]["success"] == ep["success"],
                                 float(np.linalg.norm(ctrl[0]["end_xy"][bowl] - ep["object_pos"][-1, bowl, :2]))))
            if same:
                rep[f"determinism {frame}"] = {"same_success": float(np.mean([s[0] for s in same])),
                                               "landing_diff_max_cm": 100 * max(s[1] for s in same), "n": len(same)}
                print(f"  unshifted {frame} re-run vs official: same success {np.mean([s[0] for s in same]):.2f}, "
                      f"landing diff max {100 * max(s[1] for s in same):.3f} cm (n={len(same)})")
        # 2. causal gain
        for frame in ("t0", "transport"):
            for near in (None, 0.05):
                r = gains(eps, frame, rng, near)
                if r is None:
                    continue
                g, ci, n = r
                tag = f"gain {frame}" + ("" if near is None else " (landed within 5 cm)")
                rep[tag] = {"gain": float(g), "ci": ci.tolist(), "n": n}
                print(f"  causal {tag:34s} {g:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  n={n}")
            br = [b for e in eps for b in e["branches"] if b["frame"] == frame and tuple(b["nominal"]) != (0.0, 0.0)]
            if br:
                print(f"  success under plate shifts ({frame}): {np.mean([b['success'] for b in br]):.2f} of {len(br)}")
        # 3-4. open-loop response and stage decodability
        for frame in ("t0", "transport"):
            probes = [(k, e["probes"][frame]) for k, e in enumerate(eps) if frame in e["probes"]]
            if not probes:
                continue
            groups = np.concatenate([[k] * (len(pr["plate_xy"]) - 1) for k, pr in probes])
            dplate = np.concatenate([(pr["plate_xy"][1:] - pr["plate_xy"][0]) * 100 for _, pr in probes])
            disp = []
            for _, pr in probes:
                env = 0.5 * (pr["actions"][..., :6] + 1) * (a_max[:6] - a_min[:6]) + a_min[:6]
                s = env[..., :2].sum(1)
                disp.append(s[1:] - s[0])
            disp = np.concatenate(disp)
            coef, *_ = np.linalg.lstsq(dplate, disp, rcond=None)
            line = {"open_loop_response_per_cm": float(np.trace(coef) / 2)}
            for stage in STAGES:
                if stage == "actions":
                    X = np.concatenate([(pr["actions"][1:] - pr["actions"][:1]).reshape(len(pr["actions"]) - 1, -1)
                                        for _, pr in probes]).astype(np.float64)
                elif stage in probes[0][1]["features"]:
                    X = np.concatenate([(pr["features"][stage][1:].astype(np.float64)
                                         - pr["features"][stage][:1].astype(np.float64)) for _, pr in probes])
                else:
                    continue
                line[stage] = float(ridge_cv(X, dplate, groups)) if len(np.unique(groups)) >= 5 else None
            rep[f"stages {frame}"] = line
            print(f"  {frame:9s} open-loop chunk response {line['open_loop_response_per_cm']:+.3f} env-units/cm; "
                  "decode plate offset R^2: " + "  ".join(
                      f"{s} {line[s]:.2f}" if line.get(s) is not None else f"{s} -" for s in STAGES if s in line))
    Path(args.root, "counterfactual_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
