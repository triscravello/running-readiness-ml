import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.neural_network_experiment import (
    BinaryMLP,
    CONFIGS,
    inner_group_split,
    train_fixed,
    train_with_early_stopping,
    validate_level5_contract,
)

class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([
            [-2.0, -1.0], [-1.5, -0.5], [-1.0, -1.5], [-0.5, -2.0],
            [0.5, 2.0], [1.0, 1.5], [1.5, 0.5], [2.0, 1.0],
        ], dtype=np.float32)
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        self.config = CONFIGS[0]

    def test_group_separation(self):
        frame = pd.DataFrame(self.x, columns=["a", "b"])
        target = pd.Series(self.y)
        groups = pd.Series(["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4"])
        fit, stop = inner_group_split(frame, target, groups, 42)
        self.assertFalse(set(groups.iloc[fit]) & set(groups.iloc[stop]))

    def test_schema_validation_rejects_unapproved_feature(self):
        handoff = {"handoff_version": "2.0.0", "next_level": "Level 5: Networks",
                   "selected_candidate_features": ["distance_km"], "selected_variant": "raw"}
        schema = {"candidate_feature_columns": ["distance_km"]}
        fold = {"fold_signature": "same"}
        report = {"selected_variant": "raw",
                  "cross_validation": {"identical_fold_signatures": ["same"] * 4},
                  "consumed_v1_holdout_read": False,
                  "results": {"raw": {"folds": [fold] * 4}, "engineered": {"folds": [fold] * 4}}}
        marker = {"forbidden_future_uses": ["feature_selection", "threshold_tuning",
                                            "model_comparison", "model_selection"]}
        validate_level5_contract(handoff, schema, report, marker)
        handoff["selected_candidate_features"] = ["future_weather"]
        with self.assertRaisesRegex(ValueError, "Approved features"):
            validate_level5_contract(handoff, schema, report, marker)

    def test_same_seed_reproduces_history_and_state(self):
        first, history_a, epoch_a = train_with_early_stopping(
            self.x[:6], self.y[:6], self.x[6:], self.y[6:], self.config, 42, 30, 5)
        second, history_b, epoch_b = train_with_early_stopping(
            self.x[:6], self.y[:6], self.x[6:], self.y[6:], self.config, 42, 30, 5)
        self.assertEqual(epoch_a, epoch_b)
        self.assertEqual(history_a, history_b)
        for key in first.state_dict():
            self.assertTrue(torch.equal(first.state_dict()[key], second.state_dict()[key]))

    def test_checkpoint_reload_preserves_predictions(self):
        model, _ = train_fixed(self.x, self.y, self.config, 42, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_state": model.state_dict(), "input_dim": 2,
                        "config": self.config}, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            reloaded = BinaryMLP(saved["input_dim"], saved["config"]["hidden_units"], saved["config"]["dropout"])
            reloaded.load_state_dict(saved["model_state"])
            model.eval()
            reloaded.eval()
            inputs = torch.tensor(self.x)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(inputs), reloaded(inputs)))


if __name__ == "__main__":
    unittest.main()