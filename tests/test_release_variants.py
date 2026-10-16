"""CPU-only release regression checks; no dataset, checkpoint, CUDA or rollout."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from turbovla.models.action_head import TurboVLAActionHead
from turbovla.models.configuration import ActionHeadConfig, InteractionConfig, TurboVLAConfig, VisionEncoderConfig
from turbovla.models.full_state_action_head import FullStateSpikeActionHead
from turbovla.models.shiftmax_action_head import FullStateShiftmaxActionHead
from turbovla.models.minimal_cross_wta_head import MinimalCrossWTAActionHead
from turbovla.models.progressive_full_wta_head import ProgressiveFullWTAHead
from turbovla.models.extensions.minimal_cross_wta_head import MinimalCrossWTAActionHead as ExtensionMinimalCrossWTAActionHead
from turbovla.models.extensions.progressive_full_wta_head import ProgressiveFullWTAHead as ExtensionProgressiveFullWTAHead
from turbovla.models.components.wta_operators import BipolarPLIF, SpikeDotProductAttention
from turbovla.training.train_mixed import parse_args_with_mixed_suite_stats
from experiments.libero.train_action_head import RECIPES as RELEASE_RECIPES

ROOT = Path(__file__).resolve().parents[1]
RECIPES = RELEASE_RECIPES["variants"]
torch.set_num_threads(1)


def layout(recipe):
    return recipe.get("ifr_locate", False) or recipe.get("ifr_spatial", 0) or recipe.get("ifr_sign_field", False)


def configuration(recipe):
    config = ActionHeadConfig(
        head_type=recipe["head_type"], num_state_tokens=recipe["num_state_tokens"],
        spike_state_encoding=recipe["state_encoding"],
        spike_state_population=recipe["state_population"],
        spike_state_population_range=recipe.get("state_population_range", 4),
        continuous_state=recipe.get("continuous_state", False),
        spike_timesteps=8 if recipe["head_type"] == "progressive_full_wta" else 4,
        ifr_integrate=recipe.get("ifr_integrate", False), ifr_shift=recipe.get("ifr_shift", 0),
        ifr_split_state=recipe.get("ifr_split_state", False),
        ifr_locate=recipe.get("ifr_locate", False), ifr_spatial=recipe.get("ifr_spatial", 0),
        ifr_norm_bits=recipe.get("ifr_norm_bits", 0), ifr_sign_field=recipe.get("ifr_sign_field", False),
        ifr_context_levels=recipe.get("ifr_context_levels", 0),
        # the model sets the real layout (2 cameras x 14 x 14); 2 x 2 x 2 fits the small test memory
        ifr_locate_views=2 if layout(recipe) else 0,
        ifr_locate_grid=2 if layout(recipe) else 0,
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
    def test_extension_import_compatibility(self):
        self.assertIs(MinimalCrossWTAActionHead, ExtensionMinimalCrossWTAActionHead)
        self.assertIs(ProgressiveFullWTAHead, ExtensionProgressiveFullWTAHead)

    def test_base_recipe_and_legacy_name_match(self):
        self.assertEqual(RELEASE_RECIPES["legacy_variant_aliases"]["spike_driven_lightaug"], "spike-driven-vla")
        self.assertEqual(RECIPES["spike-driven-vla"]["head_type"], "full_spike_state8_shiftmax_continuous_inputs")
        self.assertEqual(RECIPES["spike-driven-vla"]["checkpoint_prefix"], "spike_driven_vla")

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
                    "--batch_size", str(recipe["batch_per_gpu"]),
                    "--grad_accum_steps", str(recipe.get("gradient_accumulation", 1)),
                ]
                backbone = None
                if recipe.get("vision_encoder") == "hf_vit":
                    backbone = tempfile.mkdtemp(prefix="hf-vit-")
                    check = recipe["vision_check"]
                    (Path(backbone) / "config.json").write_text(json.dumps({
                        "model_type": check["model_type"][0], "hidden_size": check["hidden_size"],
                        "patch_size": check["patch_size"]}))
                    (Path(backbone) / "preprocessor_config.json").write_text("{}")
                    command += ["--vision_backbone", backbone]
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
                self.assertEqual(args.libero_light_augmentation, recipe.get("libero_light_augmentation", False))
                self.assertEqual(args.min_lr_ratio, 1)
                self.assertEqual(args.warmup_steps, recipe["warmup_steps"])
                self.assertEqual(args.lr_warmup_samples, recipe["lr_warmup_samples"])
                self.assertEqual(args.vision_gradient_checkpointing, recipe["vision_gradient_checkpointing"])
                self.assertEqual((args.ifr_integrate, args.ifr_shift, args.ifr_split_state, args.ifr_locate,
                                  args.ifr_spatial, args.ifr_norm_bits, args.ifr_sign_field, args.ifr_context_levels),
                                 (recipe.get("ifr_integrate", False), recipe.get("ifr_shift", 0),
                                  recipe.get("ifr_split_state", False), recipe.get("ifr_locate", False),
                                  recipe.get("ifr_spatial", 0), recipe.get("ifr_norm_bits", 0),
                                  recipe.get("ifr_sign_field", False), recipe.get("ifr_context_levels", 0)))
                self.assertEqual(args.position_embedding, recipe.get("position_embedding", "view"))
                if "other_lr" in recipe:
                    self.assertEqual((args.lr, args.head_lr, args.vision_encoder_lr),
                                     (recipe["other_lr"], recipe["other_lr"], recipe["vision_lr"]))
                if recipe.get("distillation", True) is False:
                    self.assertEqual((args.distill_feature_weight, args.distill_action_weight), (0, 0))
                else:
                    self.assertEqual((args.distill_feature_weight, args.distill_action_weight), (0.1, 0.1))
                    self.assertTrue(args.teacher_checkpoint)
                if backbone is not None:
                    self.assertEqual((args.vision_encoder_type, args.vision_image_size), ("dinov3", 224))
                    self.assertEqual(Path(args.vision_model_path), Path(backbone).resolve())
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
                if name in {"spike_driven", "spike-driven-vla"}:
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

    def test_ifr_locate_position_values(self):
        from turbovla.models.shiftmax_action_head import IFRCrossAttention, ShiftmaxACTDecoder
        interaction = InteractionConfig(backbone_type="spike_text_only", cross_attention_type="spike_sdsa")
        kwargs = dict(continuous_inputs=True, integrate=True, shift=2, state_tokens=0)
        torch.manual_seed(0)
        plain = IFRCrossAttention(256, 8, interaction, **kwargs).eval()
        torch.manual_seed(0)
        locate = IFRCrossAttention(256, 8, interaction, locate_views=2, locate_grid=3, **kwargs).eval()
        missing, unexpected = locate.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual((missing, unexpected), (["locate_value"], []))  # nothing else differs, same initialization
        query, memory = torch.randn(2, 12, 256), torch.randn(2, 30, 256)
        valid = torch.ones(2, 30, 1, dtype=torch.bool)

        def run(module):
            state = {}
            with torch.no_grad():
                return torch.stack([module.forward_step(query, memory, valid, state, t) for t in range(4)])
        torch.testing.assert_close(run(locate), run(plain), rtol=0, atol=0)  # zero map: the original read
        per_view = 3 + 4 * len(IFRCrossAttention.LOCATE_FREQUENCIES)  # constant, row, column, sin/cos per frequency
        self.assertEqual(tuple(locate.locate_features.shape), (18, 2 * per_view))
        self.assertEqual(locate.locate_features[:9, per_view:].abs().sum().item(), 0)  # each camera its own block
        with torch.no_grad():
            locate.locate_value.normal_()
        self.assertGreater((run(locate) - run(plain)).abs().max().item(), 0)
        with self.assertRaises(ValueError):  # the head refuses locate without the model's visual layout
            ShiftmaxACTDecoder(ActionHeadConfig(head_type="full_spike_state8_shiftmax_continuous_inputs",
                                                num_state_tokens=8, ifr_locate=True), 256, 8, 2048, interaction)

    def test_ifr_spatial_scores(self):
        from turbovla.models.shiftmax_action_head import IFRCrossAttention, ShiftmaxACTDecoder
        interaction = InteractionConfig(backbone_type="spike_text_only", cross_attention_type="spike_sdsa")
        kwargs = dict(continuous_inputs=True, integrate=True, shift=2, state_tokens=0)
        torch.manual_seed(0)
        plain = IFRCrossAttention(256, 8, interaction, **kwargs).eval()
        for k in (3, 5):
            torch.manual_seed(0)
            spatial = IFRCrossAttention(256, 8, interaction, spatial=k, views=2, grid=3, **kwargs).eval()
            self.assertEqual(spatial.load_state_dict(plain.state_dict(), strict=True).missing_keys, [])  # no new parameters
            self.assertEqual(spatial.spatial_kernel.sum().item(), 1.0)
            nvis = 18
            flat = torch.arange(4.0).repeat_interleave(nvis + 5).view(1, 1, 4, nvis + 5)  # constant per row
            torch.testing.assert_close(spatial.spatial_scores(flat), flat, rtol=0, atol=0)  # constant map unchanged
            scores = torch.randn(2, 8, 12, nvis + 5)
            out = spatial.spatial_scores(scores)
            torch.testing.assert_close(out[..., nvis:], scores[..., nvis:], rtol=0, atol=0)  # text / state untouched
            bumped = scores.clone()
            bumped[..., 9:nvis] += 10.0  # second camera only
            torch.testing.assert_close(spatial.spatial_scores(bumped)[..., :9], out[..., :9], rtol=0, atol=0)
            self.assertLess(out[..., :nvis].std().item(), scores[..., :nvis].std().item())  # neighborhoods averaged
        query, memory = torch.randn(2, 12, 256), torch.randn(2, 30, 256)
        valid = torch.ones(2, 30, 1, dtype=torch.bool)
        state = {}
        with torch.no_grad():
            self.assertEqual(spatial.forward_step(query, memory, valid, state, 0).shape, (2, 12, 256))
        with self.assertRaises(ValueError):  # the head refuses spatial integration without the model's visual layout
            ShiftmaxACTDecoder(ActionHeadConfig(head_type="full_spike_state8_shiftmax_continuous_inputs",
                                                num_state_tokens=8, ifr_spatial=3), 256, 8, 2048, interaction)

    def test_ifr_fine_normalizer_and_sign_field(self):
        from turbovla.models.shiftmax_action_head import IFRCrossAttention, ShiftmaxACTDecoder, ShiftmaxRouting
        scores = torch.randn(3, 8, 12, 40) * 3
        allowed = torch.ones(3, 12, 40, dtype=torch.bool)
        original = ShiftmaxRouting()(scores, allowed)
        torch.testing.assert_close(ShiftmaxRouting(0)(scores, allowed), original, rtol=0, atol=0)
        fine = ShiftmaxRouting(4)(scores, allowed)
        sums = fine.sum(-1)
        self.assertTrue(bool(((sums > 2 ** (-1 / 16) - 1e-6) & (sums <= 1 + 1e-6)).all()))  # rows within one 1/16 step
        torch.testing.assert_close(fine / sums[..., None], original / original.sum(-1, keepdim=True))  # same shape
        scores.requires_grad_(True)
        ShiftmaxRouting(4)(scores, allowed).pow(2).sum().backward()
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        interaction = InteractionConfig(backbone_type="spike_text_only", cross_attention_type="spike_sdsa")
        kwargs = dict(continuous_inputs=True, integrate=True, shift=2, state_tokens=0)
        torch.manual_seed(0)
        plain = IFRCrossAttention(256, 8, interaction, **kwargs).eval()
        torch.manual_seed(0)
        signed = IFRCrossAttention(256, 8, interaction, views=2, grid=3, norm_bits=4, sign_field=True, **kwargs).eval()
        self.assertEqual(signed.load_state_dict(plain.state_dict(), strict=True).missing_keys, [])  # no new parameters
        pattern = signed.sign_pattern
        self.assertEqual(tuple(pattern.shape), (18, 32))
        self.assertEqual(set(pattern.unique().tolist()), {-1.0, 1.0})
        self.assertTrue(bool((pattern[:, :16] == 1).all()))  # half of the channels read the plain contents
        self.assertTrue(bool((pattern[:9] == pattern[9:]).all()))  # both cameras, each over its own grid
        self.assertGreater(len({tuple(c) for c in pattern[:9, 16:].T.tolist()}), 4)  # several distinct patterns
        v = torch.randn(2, 8, 25, 32)
        out = signed.read_values(v)
        torch.testing.assert_close(out[:, :, 18:], v[:, :, 18:], rtol=0, atol=0)  # text / state values untouched
        torch.testing.assert_close(out[:, :, :18].abs(), v[:, :, :18].abs(), rtol=0, atol=0)  # sign flips only
        query, memory = torch.randn(2, 12, 256), torch.randn(2, 30, 256)
        valid = torch.ones(2, 30, 1, dtype=torch.bool)
        with torch.no_grad():
            self.assertEqual(signed.forward_step(query, memory, valid, {}, 0).shape, (2, 12, 256))
        with self.assertRaises(ValueError):  # the head refuses the sign field without the model's visual layout
            ShiftmaxACTDecoder(ActionHeadConfig(head_type="full_spike_state8_shiftmax_continuous_inputs",
                                                num_state_tokens=8, ifr_sign_field=True), 256, 8, 2048, interaction)

    def test_integer_context_neuron(self):
        from turbovla.models.shiftmax_action_head import IFRCrossAttention, IntegerPLIF
        torch.manual_seed(0)
        plif = BipolarPLIF()
        neuron = IntegerPLIF(4)
        self.assertEqual(neuron.load_state_dict(plif.state_dict(), strict=True).missing_keys, [])  # same parameters
        threshold = float(neuron.log_threshold.exp())
        current = torch.randn(3, 5, 16) * 4
        spike, membrane = neuron(current)
        self.assertTrue(bool((spike == spike.round()).all()) and float(spike.abs().max()) <= 4)  # integer events
        voltage = neuron.decay_logit.sigmoid() * current
        torch.testing.assert_close(membrane, voltage - spike * threshold)  # soft reset keeps the residual
        inside = (voltage / threshold).abs() < 3.5
        self.assertTrue(bool(((voltage - spike * threshold).abs()[inside] <= threshold / 2 + 1e-6).all()))
        ramp = torch.linspace(-1, 1, 101) * threshold  # a slowly changing input: the event count over timesteps
        counts, state = torch.zeros_like(ramp), None  # follows the input more finely than BipolarPLIF's
        hard, hard_state = torch.zeros_like(ramp), None
        for _ in range(4):
            s, state = neuron(ramp, state)
            h, hard_state = plif(ramp, hard_state)
            counts, hard = counts + s, hard + h
        self.assertGreater(len(counts.unique()), len(hard.unique()))
        x = torch.randn(4, 16, requires_grad=True)
        neuron(x)[0].sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum()) > 0)
        valid = torch.tensor([[True], [False], [True], [True]])
        s, m = neuron(torch.randn(4, 16) * 3, valid=valid)
        self.assertTrue(bool((s[1] == 0).all() and (m[1] == 0).all()))
        interaction = InteractionConfig(backbone_type="spike_text_only", cross_attention_type="spike_sdsa")
        kwargs = dict(continuous_inputs=True, integrate=True, shift=2, state_tokens=0)
        torch.manual_seed(0)
        plain = IFRCrossAttention(256, 8, interaction, **kwargs).eval()
        torch.manual_seed(0)
        integer = IFRCrossAttention(256, 8, interaction, context_levels=4, **kwargs).eval()
        self.assertEqual(integer.load_state_dict(plain.state_dict(), strict=True).missing_keys, [])  # same init
        self.assertIsInstance(integer.spike_nodes["context"], IntegerPLIF)
        query, memory = torch.randn(2, 12, 256), torch.randn(2, 30, 256)
        with torch.no_grad():
            self.assertEqual(integer.forward_step(query, memory, torch.ones(2, 30, 1, dtype=torch.bool), {}, 0).shape,
                             (2, 12, 256))
        with self.assertRaises(ValueError):
            IntegerPLIF(0)

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
