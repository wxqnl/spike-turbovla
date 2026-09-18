"""CPU-only release regression checks; no dataset, checkpoint, CUDA or rollout."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from turbovla.models.action_head import TurboVLAActionHead
from turbovla.models.configuration import ActionHeadConfig, InteractionConfig, TurboVLAConfig, VisionEncoderConfig
from turbovla.models.full_state_action_head import FullStateSpikeActionHead
from turbovla.models.shiftmax_action_head import FullStateShiftmaxActionHead
from turbovla.models.minimal_cross_wta_head import MinimalCrossWTAActionHead
from turbovla.models.progressive_full_wta_head import ProgressiveFullWTAHead
from turbovla.models.components.wta_operators import BipolarPLIF, SpikeDotProductAttention
from turbovla.training.train_mixed import parse_args_with_mixed_suite_stats

ROOT = Path(__file__).resolve().parents[1]
RECIPES = json.loads((ROOT / "experiments/libero/configs/action_head_variants.json").read_text())["variants"]
torch.set_num_threads(1)


def configuration(recipe):
    config = ActionHeadConfig(
        head_type=recipe["head_type"], num_state_tokens=recipe["num_state_tokens"],
        spike_state_encoding=recipe["state_encoding"],
        spike_state_population=recipe["state_population"],
        spike_state_population_range=recipe.get("state_population_range", 4),
        continuous_state=recipe.get("continuous_state", False),
        spike_timesteps=8 if recipe["head_type"] == "progressive_full_wta" else 4,
    )
    return config


def head_for(recipe):
    config = configuration(recipe)
    interaction = InteractionConfig(backbone_type="spike_text_only", cross_attention_type="spike_sdsa")
    original = TurboVLAActionHead(config, 256, 8, 2048)
    kind = config.head_type
    if kind == "ann":
        return original
    if kind == "minimal_cross_wta":
        return MinimalCrossWTAActionHead(config, 256, 8, 2048, interaction, original)
    if kind == "progressive_full_wta":
        return ProgressiveFullWTAHead(original, config, interaction)
    cls = FullStateShiftmaxActionHead if "shiftmax" in kind else FullStateSpikeActionHead
    head = cls(config, 256, 8, 2048, interaction)
    head.load_common(original)
    return head


class ReleaseVariants(unittest.TestCase):
    def test_all_recipes_reach_real_training_parser(self):
        for name, recipe in RECIPES.items():
            with self.subTest(variant=name):
                command = [
                    sys.executable, "-m", "experiments.libero.train_action_head",
                    "--variant", name, "--dry-run", "--world-size", str(recipe["world_size"]),
                    "--run_dir", "/tmp/spike-release-preview/" + name,
                    "--vision_checkpoint", "/not-loaded/vision.pth",
                    "--text_projection_init", "/not-loaded/text.pth",
                    "--max_steps", str(recipe["max_steps"]),
                    "--batch_size", str(recipe["batch_per_gpu"]), "--grad_accum_steps", "1",
                ]
                preview = json.loads(subprocess.check_output(command, text=True))
                previous = sys.argv
                try:
                    sys.argv = ["train", *preview["training_arguments"]]
                    args = parse_args_with_mixed_suite_stats()
                finally:
                    sys.argv = previous
                self.assertEqual(args.action_head_type, recipe["head_type"])
                self.assertEqual(args.num_state_tokens, recipe["num_state_tokens"])
                self.assertEqual(args.continuous_state, recipe.get("continuous_state", False))
                self.assertEqual(args.text_attention_routing, recipe.get("text_attention_routing", "wta"))
                self.assertEqual(args.min_lr_ratio, 1)
                self.assertEqual(args.warmup_steps, recipe["warmup_steps"])
                self.assertEqual(args.lr_warmup_samples, recipe["lr_warmup_samples"])
                self.assertEqual(args.vision_gradient_checkpointing, recipe["vision_gradient_checkpointing"])
                # Validation runs in the dataclass constructor, not afterward.
                TurboVLAConfig(
                    action=configuration(recipe),
                    interaction=InteractionConfig(
                        backbone_type="spike_text_only", cross_attention_type="spike_sdsa",
                        text_attention_routing=recipe.get("text_attention_routing", "wta"),
                    ),
                    vision=VisionEncoderConfig(compute_precision="fp32"),
                )
                if name == "progressive_fullwta_t8":
                    self.assertEqual(args.extra_save_steps, [20, 25600, 27200])

    def test_full_size_heads_forward_backward(self):
        # Complete 3-layer, hidden256/FFN2048 heads; only batch/token count is small.
        for name, recipe in RECIPES.items():
            with self.subTest(variant=name):
                torch.manual_seed(42)
                head = head_for(recipe).train()
                if isinstance(head, ProgressiveFullWTAHead):
                    head.set_progress(26400, 32000)
                memory = torch.randn(1, 16, 256, requires_grad=True)
                state = torch.randn(1, 8)
                output = head(memory, state)
                self.assertEqual(tuple(output.shape), (1, 12, 7))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                gradients = [p.grad for p in head.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
                if name == "shiftmax_ann_state":
                    self.assertFalse(any(isinstance(m, BipolarPLIF) for m in head.state_projection.modules()))
                if name == "minimal_cross_wta":
                    self.assertFalse(any(isinstance(m, BipolarPLIF) for m in head.state_projection.modules()))
                    self.assertIsInstance(head.decoder.decoder.layers[0].self_attn, torch.nn.MultiheadAttention)

    def test_progressive_shared_weights_endpoints_and_buffers(self):
        recipe = RECIPES["progressive_fullwta_t8"]
        config = configuration(recipe)
        original = TurboVLAActionHead(config, 256, 8, 2048).eval()
        memory, state = torch.randn(1, 16, 256), torch.randn(1, 8)
        with torch.no_grad():
            expected = original(memory, state)
        head = ProgressiveFullWTAHead(original, config, InteractionConfig()).eval()
        self.assertIs(head.decoder, original.decoder)
        self.assertIs(head.state_projection, original.state_projection)
        for step, alpha in [(25600, 0), (26400, .5), (27200, 1), (32000, 1)]:
            self.assertAlmostEqual(head.set_progress(step, 32000), alpha)
            with torch.no_grad():
                actual = head(memory, state)
                if alpha == 0:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if alpha == 1:
                    torch.testing.assert_close(actual, head.spike_forward(memory, state), rtol=0, atol=0)
        clone = head_for(recipe).eval()
        clone.load_state_dict(head.state_dict(), strict=True)
        self.assertEqual(clone.transition_updates.item(), 32000)
        self.assertEqual(clone.transition_alpha.item(), 1)

    def test_sdsa_route_masks_without_topk(self):
        q = torch.tensor([[[[1., -1.], [0., 0.]]]])
        k = torch.tensor([[[[1., 0.], [0., 1.], [0., 0.]]]])
        allowed = torch.ones((1, 2, 3), dtype=torch.bool)
        route = SpikeDotProductAttention()(q @ k.transpose(-1, -2), allowed, q, k)
        torch.testing.assert_close(route, torch.tensor([[[[1., -1., 0.], [0., 0., 0.]]]]))

    def test_result_scopes(self):
        results = json.loads((ROOT / "docs/results/libero_20260918.json").read_text())
        for row in results["completed"]:
            self.assertEqual(sum(s["successes"] for s in row["suites"].values()), row["total_successes"])
            self.assertEqual(sum(s["episodes"] for s in row["suites"].values()), row["total_episodes"])
            if row["scope"] == "long_only_500":
                self.assertIsNone(row["overall_success_rate"])
                self.assertEqual(set(row["suites"]), {"libero_10"})


if __name__ == "__main__":
    unittest.main()
