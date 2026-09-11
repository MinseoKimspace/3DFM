from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import eval as evaluation
from diagnose_guidance import diagnose_batch
from fm.losses import fm_loss
from fm.sampler import sample_euler
from models.builder import build_model
from sample_intervention import choose_intervention, sample_mode_in_batches


class StubDiPT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


class StubSparse:
    def replace_feature(self, feat):
        return self


def prepare_point(model, z, t):
    return SimpleNamespace(
        feat=z.repeat(1, 1, model.channels // 3).reshape(-1, model.channels),
        condition=t.reshape(z.shape[0], 1).expand(-1, model.channels),
        sparse_conv_feat=StubSparse(),
    )


class SelfConditionPMATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Stub only the CUDA-dependent trunk; use real PMA, cross-attention and FM code.
        upstream, structure = ModuleType("third_party.dipt"), ModuleType("third_party.dipt.structure")
        upstream.DiffusionPointTransformer = StubDiPT
        structure.Point = SimpleNamespace
        spec = importlib.util.spec_from_file_location("tested_dipt", SRC / "models" / "dipt_backbone.py")
        cls.backbones = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"third_party.dipt": upstream, "third_party.dipt.structure": structure}):
            spec.loader.exec_module(cls.backbones)

    def setUp(self):
        torch.manual_seed(123)
        for p in (
            patch.dict(sys.modules, {"models.dipt_backbone": self.backbones}),
            patch.object(self.backbones.DiPTFlowBackbone, "_prepare_point", prepare_point),
            patch.object(self.backbones.DiPTFlowBackbone, "_run_blocks", lambda model, point, start, end: point),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.config = dict(arch="dipt_xhat_selfcond_pma", num_points=8, dipt_channels=12,
                           dipt_depth=3, dipt_num_heads=3, dipt_patch_size=[8] * 3,
                           early_layers=1, num_slots=3, knn_k=3)
        self.model = build_model(self.config).eval()
        self.z, self.previous = torch.randn(2, 8, 3), torch.randn(2, 8, 3)
        self.t = torch.full((2, 1, 1), 0.5)

    def record_forwards(self):
        calls = []

        def capture(module, args, kwargs, output):
            prior = kwargs.get("self_cond")
            calls.append(dict(x=args[0].detach().clone(), t=args[1].detach().clone(),
                              prior=None if prior is None else prior.clone(),
                              v=output.detach().clone(), grad=torch.is_grad_enabled(),
                              mode=kwargs.get("cond_mode", "normal")))

        handle = self.model.register_forward_hook(capture, with_kwargs=True)
        self.addCleanup(handle.remove)
        return calls

    def test_writer_uses_previous_coordinates_and_features(self):
        inputs = []
        handle = self.model.spatial_pma.register_forward_pre_hook(lambda module, args: inputs.append(args))
        self.addCleanup(handle.remove)
        previous = self.previous.requires_grad_()
        out = self.model(self.z, self.t, self_cond=previous)
        torch.testing.assert_close(inputs[0][0], previous)
        torch.testing.assert_close(inputs[0][1], self.model.self_cond_embed(previous.detach()))
        self.assertFalse(inputs[0][0].requires_grad)
        self.assertEqual(out.shape, self.z.shape)
        out.square().mean().backward()
        self.assertIsNone(previous.grad)
        for module in (self.model.self_cond_embed, self.model.spatial_pma, self.model.slot_cross_attn):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_missing_history_and_zero_bypass_the_whole_branch(self):
        with patch.object(self.model.spatial_pma, "forward", side_effect=AssertionError("writer called")), \
             patch.object(self.model.slot_cross_attn, "forward", side_effect=AssertionError("reader called")):
            absent = self.model(self.z, self.t)
            zero = self.model(self.z, self.t, self_cond=self.previous, cond_mode="zero")
        torch.testing.assert_close(absent, zero, rtol=0, atol=0)
        base = build_model({**self.config, "arch": "dipt_base"}).eval()
        base.load_state_dict({key: self.model.state_dict()[key] for key in base.state_dict()})
        torch.testing.assert_close(absent, base(self.z, self.t), rtol=0, atol=0)

    def test_shuffle_keeps_previous_geometry_and_features_paired(self):
        inputs = []
        handle = self.model.spatial_pma.register_forward_pre_hook(lambda module, args: inputs.append(args))
        self.addCleanup(handle.remove)
        self.model(self.z, self.t, self_cond=self.previous, cond_mode="shuffle")
        swapped = self.previous.flip(0)
        torch.testing.assert_close(inputs[0][0], swapped)
        torch.testing.assert_close(inputs[0][1], self.model.self_cond_embed(swapped))

    def test_training_uses_detached_first_pass_without_auxiliary_loss(self):
        self.model.train()
        calls = self.record_forwards()
        loss, metrics = fm_loss(self.model, self.z, self_condition_prob=1.0)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["prior"])
        self.assertFalse(calls[0]["grad"])
        self.assertFalse(calls[1]["prior"].requires_grad)
        torch.testing.assert_close(calls[1]["prior"], calls[0]["x"] + (1 - calls[0]["t"]) * calls[0]["v"])
        torch.testing.assert_close(calls[0]["x"], calls[1]["x"])
        self.assertEqual(metrics["aux_mse"].item(), 0.0)
        loss.backward()
        self.assertIsNotNone(self.model.self_cond_embed[0].weight.grad)
        calls.clear()
        fm_loss(self.model, self.z, self_condition_prob=0.0)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["prior"])
        with self.assertRaisesRegex(ValueError, "aux_weight"):
            fm_loss(self.model, self.z, aux_weight=0.1)

    def test_sampling_reuses_previous_endpoint_and_resets_history(self):
        calls = self.record_forwards()
        for _ in range(2):
            calls.clear()
            sample_euler(self.model, 2, 8, 4, "cpu", init=self.z)
            self.assertEqual(len(calls), 4)
            self.assertIsNone(calls[0]["prior"])
            for before, after in zip(calls, calls[1:]):
                torch.testing.assert_close(after["prior"], before["x"] + (1 - before["t"]) * before["v"])
                self.assertFalse(after["prior"].requires_grad)

    def test_eval_and_interventions_dispatch_condition_modes(self):
        self.assertEqual(evaluation.intervention_arg(self.model, self.config), "cond_mode")
        self.assertEqual(choose_intervention(self.model, self.config)[0], "cond_mode")
        expected = sample_euler(self.model, 2, 8, 4, "cpu", init=self.z)
        normal, _, stats = evaluation.sample_checkpoint(self.model, self.config, self.z, 4, 2,
                                                        torch.device("cpu"), "normal", False)
        torch.testing.assert_close(normal, expected, rtol=0, atol=0)
        self.assertEqual(stats, {})
        for mode in ("zero", "shuffle"):
            samples, _ = sample_mode_in_batches(self.model, self.z, 4, 2, torch.device("cpu"), "cond_mode", mode)
            self.assertEqual(samples.shape, self.z.shape)
            self.assertTrue(torch.isfinite(samples).all())
        for mode, steps in (("normal", 4), ("zero", 4), ("normal", 1)):
            _, _, stats = evaluation.sample_checkpoint(self.model, self.config, self.z, steps, 2,
                                                       torch.device("cpu"), mode, True)
            self.assertEqual(bool(stats), mode == "normal" and steps > 1)
            self.assertEqual(self.model.slot_cross_attn.last_stats, {})
        original = build_model({**self.config, "arch": "dipt_xhat_anchor_pma"})
        self.assertEqual(evaluation.intervention_arg(original, self.config), "slot_mode")
        self.assertEqual(choose_intervention(original, self.config)[0], "slot_mode")

    def test_diagnostic_tracks_normal_history_and_matches_sampler(self):
        calls = self.record_forwards()
        rows, actual = diagnose_batch(self.model, self.z, 4, torch.device("cpu"), self.z.dtype)
        self.assertEqual(len(calls), 12)
        self.assertEqual(rows[0]["diff_zero_sum"], 0.0)
        self.assertEqual(rows[0]["diff_shuffle_sum"], 0.0)
        for step in range(1, 4):
            previous = calls[(step - 1) * 3]
            expected = previous["x"] + (1 - previous["t"]) * previous["v"]
            for event in calls[step * 3:step * 3 + 3]:
                torch.testing.assert_close(event["prior"], expected)
                torch.testing.assert_close(event["x"], calls[step * 3]["x"])
        expected = sample_euler(self.model, 2, 8, 4, "cpu", init=self.z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_validation(self):
        restored = build_model(self.config).eval()
        restored.load_state_dict(self.model.state_dict(), strict=True)
        torch.testing.assert_close(restored(self.z, self.t, self_cond=self.previous),
                                   self.model(self.z, self.t, self_cond=self.previous), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "same shape"):
            self.model(self.z, self.t, self_cond=self.previous[:, :-1])
        with self.assertRaisesRegex(ValueError, "Unknown conditioning"):
            self.model(self.z, self.t, cond_mode="bad")


if __name__ == "__main__":
    unittest.main()
