import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.feature_experiment import fingerprint
from src.finetune_experiment import (
    load_level5_checkpoint,
    pretrain_source,
    train_target,
    verify_fingerprint,
)
from src.neural_network_experiment import inner_group_split


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [
                [-2.0, -1.0, 0.0],
                [-1.5, -0.5, 0.2],
                [-1.0, -1.5, 0.1],
                [-0.5, -2.0, 0.3],
                [0.5, 2.0, 0.7],
                [1.0, 1.5, 0.8],
                [1.5, 0.5, 0.9],
                [2.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

    def test_level5_checkpoint_loads_on_cpu(self):
        checkpoint = load_level5_checkpoint(Path.cwd())
        self.assertIn("model_state", checkpoint)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in checkpoint["model_state"].values()))

    def test_frozen_encoder_does_not_change(self):
        source, _, _ = pretrain_source(self.x, 42, epochs=5)
        state = copy.deepcopy(source.encoder.state_dict())
        model, _, _, _ = train_target(
            self.x,
            self.y,
            "frozen",
            42,
            state,
            epochs=5,
        )
        for name, tensor in state.items():
            self.assertTrue(torch.equal(tensor, model.encoder.state_dict()[name]))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_group_separation(self):
        frame = pd.DataFrame(self.x, columns=["a", "b", "c"])
        target = pd.Series(self.y)
        groups = pd.Series(["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4"])
        fit, stop = inner_group_split(frame, target, groups, 42)
        self.assertFalse(set(groups.iloc[fit]) & set(groups.iloc[stop]))

    def test_fingerprint_verification_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"
            path.write_text("trusted", encoding="utf-8")
            expected = fingerprint(path)
            verify_fingerprint(path, expected)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_fingerprint(path, expected)

    def test_same_seed_reproduces_history_and_state(self):
        first, history_a, epoch_a, _ = train_target(
            self.x,
            self.y,
            "scratch",
            42,
            None,
            epochs=12,
        )
        second, history_b, epoch_b, _ = train_target(
            self.x,
            self.y,
            "scratch",
            42,
            None,
            epochs=12,
        )
        self.assertEqual(epoch_a, epoch_b)
        self.assertEqual(history_a, history_b)
        for name, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()