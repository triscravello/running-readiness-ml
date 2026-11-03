from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from torch import nn

try:
    from src.level7_runtime import CHECKPOINT_PATH, IntegrityError, ModelRuntime
    from src.level7_service import app
except ModuleNotFoundError:
    from level7_runtime import CHECKPOINT_PATH, IntegrityError, ModelRuntime
    from level7_service import app

ROOT = Path(__file__).resolve().parents[1]
VALID_REQUEST = {
    "weekly_km_to_date": 32.0,
    "sleep_hours_last_night": 7.0,
    "resting_hr_morning": 50.0,
    "soreness_score_evening": 4.0,
    "prior_injury": 0,
}

class ReferenceEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer2(torch.relu(self.layer1(inputs))))


class ReferenceTransferClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = ReferenceEncoder(input_dim)
        self.head = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs)).squeeze(1)

def reference_prediction(payload: dict) -> tuple[float, int, float]:
    handoff = json.loads(
        (ROOT / "artifacts/v4/level7_deploy_handoff.json").read_text(encoding="utf-8")
    )
    preprocessor = json.loads(
        (ROOT / "artifacts/v4/level6_preprocessor.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        ROOT / CHECKPOINT_PATH, map_location=torch.device("cpu"), weights_only=True
    )
    values = np.asarray(
        [[float(payload[name]) for name in handoff["feature_names"]]], dtype=np.float64
    )
    transformed = (
        (
            np.where(
                np.isnan(values),
                np.asarray(preprocessor["imputer_statistics"], dtype=np.float64),
                values,
            )
            - np.asarray(preprocessor["scaler_mean"], dtype=np.float64)
        )
        / np.asarray(preprocessor["scaler_scale"], dtype=np.float64)
    ).astype(np.float32)
    model = ReferenceTransferClassifier(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(torch.from_numpy(transformed))).item())
    threshold = float(checkpoint["threshold"])
    return probability, int(probability >= threshold), threshold

class Level7ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_documents_integrity_and_limitations(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["integrity_status"], "passed")
        self.assertEqual(body["device"], "cpu")
        self.assertEqual(body["selected_approach"], "partial")
        self.assertEqual(body["evidence_scope"], "development_only")
        self.assertIn("Synthetic educational", body["model_notice"])
        self.assertIn("not claim", body["inherited_v1_preprocessing_limitation"])

    def test_prediction_matches_independent_reference(self):
        probability, decision, threshold = reference_prediction(VALID_REQUEST)
        response = self.client.post("/predict", json=VALID_REQUEST)
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.json()["probability"], probability, places=7)
        self.assertEqual(response.json()["decision"], decision)
        self.assertEqual(response.json()["threshold"], threshold)

    def test_invalid_requests_are_rejected(self):
        cases = []
        missing = dict(VALID_REQUEST)
        missing.pop("sleep_hours_last_night")
        cases.append(("missing", missing))
        extra = dict(VALID_REQUEST, needs_recovery_next_day=1)
        cases.append(("extra", extra))
        wrong_type = dict(VALID_REQUEST, resting_hr_morning="50.0")
        cases.append(("wrong type", wrong_type))
        out_of_range = dict(VALID_REQUEST, soreness_score_evening=11.0)
        cases.append(("out of range", out_of_range))
        for label, payload in cases:
            with self.subTest(label=label):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)

    def test_nonfinite_value_is_rejected(self):
        payload = dict(VALID_REQUEST, weekly_km_to_date=float("nan"))
        response = self.client.post(
            "/predict",
            content=json.dumps(payload),
            headers={"content-type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)

    def test_startup_rejects_corrupted_temporary_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            for relative in (
                "serving-manifest.json",
                "artifacts/v4/level7_deploy_handoff.json",
                "artifacts/v4/level6_preprocessor.json",
                "artifacts/v4/level6_selected_model.pt",
            ):
                target = temporary_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            checkpoint = temporary_root / CHECKPOINT_PATH
            data = bytearray(checkpoint.read_bytes())
            data[len(data) // 2] ^= 0x01
            checkpoint.write_bytes(data)
            with self.assertRaises(IntegrityError):
                ModelRuntime(temporary_root)

    def test_bundle_excludes_historical_and_holdout_files(self):
        for relative in (
            "data/v1/processed/train.csv",
            "data/v1/processed/test.csv",
            "artifacts/v1/holdout_consumed.json",
            "reports/v1/level3_evaluation.json",
            "reports/v4/finetune_experiment.json",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)
        self.assertTrue((ROOT / "lineage-attestation.json").is_file())
        self.assertTrue((ROOT / "serving-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()