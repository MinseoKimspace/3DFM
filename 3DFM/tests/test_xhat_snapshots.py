from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sample
from render_mitsuba_points import collect_jobs, make_scene


class ToyAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_with_aux(self, x, t):
        self.calls += 1
        return {"velocity": torch.full_like(x, 2.0), "x_hat1": x - 3.0 * (1.0 - t)}

    def forward(self, x, t):
        return self.forward_with_aux(x, t)["velocity"]


class XHatSnapshotTests(unittest.TestCase):
    def test_capture_matches_normal_sampling_and_preserves_all_batches(self):
        noise = torch.randn(5, 7, 3, generator=torch.Generator().manual_seed(123))
        normal, recorded = ToyAnchor().eval(), ToyAnchor().eval()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            expected, _ = sample.sample_in_batches(normal, noise, 16, 2, torch.device("cpu"))
            actual, _ = sample.sample_in_batches(recorded, noise, 16, 2, torch.device("cpu"), root / "xhat")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, noise + 2.0)
            self.assertEqual(normal.calls, 48)
            self.assertEqual(recorded.calls, normal.calls)
            manifest = json.loads((root / "xhat" / "manifest.json").read_text())
            self.assertEqual([r["step"] for r in manifest["snapshots"]], [0, 4, 8, 12, 15])
            for row in manifest["snapshots"]:
                self.assertEqual(row["t"], row["step"] / 16)
                state = torch.load(root / "xhat" / row["xt"], weights_only=True)
                prediction = torch.load(root / "xhat" / row["xhat1"], weights_only=True)
                self.assertEqual(state.shape, noise.shape)
                torch.testing.assert_close(state, noise + 2.0 * row["t"])
                torch.testing.assert_close(prediction, state - 3.0 * (1.0 - row["t"]))
            jobs = collect_jobs(SimpleNamespace(xhat_dir=str(root)))
            self.assertEqual(len(jobs), 11)
            self.assertIn("Current state", jobs[0].label)
            self.assertIn("Predicted final", jobs[1].label)
            self.assertEqual(jobs[-1].path, root / "samples.pt")

    def test_small_nfe_deduplicates_snapshots(self):
        for nfe in (1, 2, 3):
            with self.subTest(nfe=nfe), tempfile.TemporaryDirectory() as tmp:
                with contextlib.redirect_stdout(io.StringIO()):
                    sample.sample_in_batches(ToyAnchor(), torch.zeros(1, 2, 3), nfe, 1, torch.device("cpu"), Path(tmp))
                rows = json.loads((Path(tmp) / "manifest.json").read_text())["snapshots"]
                self.assertEqual([r["step"] for r in rows], list(range(nfe)))

    def test_non_anchor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "XHatAnchorPMA"):
            sample.XHatRecorder(torch.nn.Identity(), 16)

    def test_cli_records_checkpoint_setting_and_timing_warning(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as log:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            torch.save({"args": {"arch": "dipt_xhat_anchor_pma", "aux_weight": 0.0, "num_points": 7}}, checkpoint)
            argv = ["sample.py", "--checkpoint", str(checkpoint), "--out-dir", str(root / "out"),
                    "--num-samples", "3", "--batch-size", "2", "--nfe", "4", "--device", "cpu", "--save-xhat"]
            with patch.object(sys, "argv", argv), patch.object(sample, "build_model_from_checkpoint", return_value=ToyAnchor().eval()):
                sample.main()
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(summary["checkpoint_aux_weight"], 0.0)
            self.assertTrue(summary["times"]["4"]["includes_xhat_capture"])
            self.assertTrue((root / "out" / "nfe_004" / "samples.pt").exists())
            self.assertIn("checkpoint aux_weight: 0.0", log.getvalue())

    def test_shared_frame_preserves_camera_and_lighting(self):
        mi = SimpleNamespace(ScalarTransform4f=SimpleNamespace(look_at=lambda **kw: kw), load_dict=lambda scene: scene)
        args = SimpleNamespace(pad=1.12, view="side", camera_distance=3.2, integrator="path", fov=35,
                               spp=64, width=600, height=600, ambient=0.9, key_light=3,
                               color="#2563eb", height_color=True, radius=0.2)
        a = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        b = a + 4
        frame = torch.cat([a, b])
        first, second = make_scene(mi, a, args, frame), make_scene(mi, b, args, frame)
        self.assertEqual(first["sensor"], second["sensor"])
        self.assertEqual(first["key_light"], second["key_light"])
        self.assertNotEqual(first["point_0000"]["center"], second["point_0000"]["center"])


if __name__ == "__main__":
    unittest.main()
