"""Analysis of head_probe.py records.

Per model and frame (t0, transport, approach): R^2 of decoding the plate offset (cm) from the feature change it
causes at every stage, now including the stages inside the action head (cross-attention output and output of each
decoder layer, each readout linear), with the same grouped-by-episode kernel ridge as counterfactual_analysis.py;
the open-loop chunk response; and the cross-attention mass on visual / text / state tokens per decoder layer.
Usage: python head_analysis.py <root> --stats <libero_all4_stats.json> [--stats_key libero_all4_no_noops]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterfactual_analysis import load, ridge_cv  # noqa: E402


HEAD_ORDER = ["kvis", "vvis", "routevis", "where", "rterm", "vterm", "uterm", "mterm", "sterm", "usterm", "sread", "ctxvis", "ctx", "ctxspk", "cross", "layer"]  # per decoder layer


def stage_order(keys):
    head = sorted((k for k in keys if re.fullmatch(r"(%s)\d+" % "|".join(HEAD_ORDER), k)),
                  key=lambda k: (int(re.search(r"\d+$", k).group()), HEAD_ORDER.index(re.sub(r"\d+$", "", k))))
    readout = sorted((k for k in keys if re.fullmatch(r"readout\d+", k)), key=lambda k: int(k[7:]))
    return [k for k in ("vision", "projected", "fused") if k in keys] + head + readout + ["actions"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--stats", required=True)
    p.add_argument("--stats_key", default="libero_all4_no_noops")
    args = p.parse_args()
    stats = json.loads(Path(args.stats).read_text())[args.stats_key]
    a_min, a_max = np.asarray(stats["action"]["min"], np.float32), np.asarray(stats["action"]["max"], np.float32)
    report = {}
    for model, eps in load(args.root).items():
        print(f"\n== {model}: {len(eps)} episodes")
        for frame in ("t0", "transport", "approach"):
            probes = [(k, e["probes"][frame]) for k, e in enumerate(eps) if frame in e["probes"]]
            if not probes:
                continue
            groups = np.concatenate([[k] * (len(pr["plate_xy"]) - 1) for k, pr in probes])
            dplate = np.concatenate([(pr["plate_xy"][1:] - pr["plate_xy"][0]) * 100 for _, pr in probes])
            disp = []
            for _, pr in probes:
                s = (0.5 * (pr["actions"][..., :2] + 1) * (a_max[:2] - a_min[:2]) + a_min[:2]).sum(1)
                disp.append(s[1:] - s[0])
            coef, *_ = np.linalg.lstsq(dplate, np.concatenate(disp), rcond=None)
            line = {"episodes": len(probes), "open_loop_response_per_cm": float(np.trace(coef) / 2), "r2": {}}
            for stage in stage_order(probes[0][1]["features"]):
                if stage == "actions":
                    X = np.concatenate([(pr["actions"][1:] - pr["actions"][:1]).reshape(len(pr["actions"]) - 1, -1)
                                        for _, pr in probes]).astype(np.float64)
                else:
                    X = np.concatenate([pr["features"][stage][1:].astype(np.float64)
                                        - pr["features"][stage][:1].astype(np.float64) for _, pr in probes])
                line["r2"][stage] = float(ridge_cv(X, dplate, groups)) if len(np.unique(groups)) >= 5 else None
            mass = np.concatenate([pr["features"]["attention_mass"] for _, pr in probes]).mean(0)  # [layers, 3]
            line["attention_mass"] = mass.tolist()
            report.setdefault(model, {})[frame] = line
            print(f"  {frame:9s} n={len(probes):3d} open-loop {line['open_loop_response_per_cm']:+.3f} env-units/cm")
            print("    R^2  " + "  ".join(f"{s} {v:.2f}" if v is not None else f"{s} -" for s, v in line["r2"].items()))
            print("    cross-attention mass (visual/text/state): " + "  ".join(
                f"L{i} {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}" for i, m in enumerate(mass)))
            if "flips" in probes[0][1]["features"]:  # cross_probe.py: change vs the unshifted render, shifted only
                flips = np.concatenate([pr["features"]["flips"][1:] for _, pr in probes]).mean(0)
                line["flips"] = flips.tolist()
                print("    change vs unshifted (key spikes flipped / visual route L1 / context spikes flipped): " +
                      "  ".join(f"L{i} {f[0]:.3f}/{f[1]:.3f}/{f[2]:.3f}" for i, f in enumerate(flips)))
            if "split_norms" in probes[0][1]["features"]:  # read_split_probe.py: shifted renders only
                norms = np.concatenate([pr["features"]["split_norms"][1:] for _, pr in probes]).mean(0)
                line["split_norms"] = norms.tolist()
                print("    visual-read change, mean norm (route term / value term / total): " +
                      "  ".join(f"L{i} {f[0]:.3f}/{f[1]:.3f}/{f[2]:.3f}" for i, f in enumerate(norms)))
                if norms.shape[1] > 3:
                    print("    same, routing rows renormalized over image keys (route / value): " +
                          "  ".join(f"L{i} {f[3]:.3f}/{f[4]:.3f}" for i, f in enumerate(norms)))
    Path(args.root, "head_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
