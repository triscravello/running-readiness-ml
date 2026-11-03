from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HANDOFF_PATH = Path("artifacts/v4/level7_deploy_handoff.json")
CHECKPOINT_PATH = Path("artifacts/v4/level6_selected_model.pt")
PREPROCESSOR_PATH = Path("artifacts/v4/level6_preprocessor.json")
MANIFEST_PATH = Path("serving-manifest.json")
SAMPLE_REQUEST = {
    "weekly_km_to_date": 32.0,
    "sleep_hours_last_night": 7.0,
    "resting_hr_morning": 50.0,
    "soreness_score_evening": 4.0,
    "prior_injury": 0,
}

class IntegrityError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_serving_artifacts(root: Path) -> dict:
    manifest_path = root / MANIFEST_PATH
    if not manifest_path.is_file():
        raise IntegrityError(f"Missing serving manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_text, expected in manifest["artifacts"].items():
        artifact = root / relative_text
        if not artifact.is_file():
            raise IntegrityError(f"Missing serving artifact: {artifact}")
        actual = file_sha256(artifact)
        if actual != expected:
            raise IntegrityError(
                f"Serving artifact mismatch for {artifact}: expected {expected}, got {actual}"
            )
    return manifest


class Encoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer2(torch.relu(self.layer1(inputs))))


class TransferClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim)
        self.head = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs)).squeeze(1)


class ModelRuntime:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = verify_serving_artifacts(root)
        self.integrity_status = "passed"
        self.handoff = json.loads((root / HANDOFF_PATH).read_text(encoding="utf-8"))
        self.preprocessor = json.loads(
            (root / PREPROCESSOR_PATH).read_text(encoding="utf-8")
        )
        checkpoint: dict[str, Any] = torch.load(
            root / CHECKPOINT_PATH,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        if checkpoint.get("model_class") != "TransferClassifier":
            raise IntegrityError("Unexpected Level 6 model class")
        if checkpoint.get("feature_names") != self.handoff["feature_names"]:
            raise IntegrityError("Checkpoint feature order differs from the handoff")
        if self.preprocessor.get("feature_names") != self.handoff["feature_names"]:
            raise IntegrityError("Preprocessor feature order differs from the handoff")
        if checkpoint.get("threshold") != self.handoff["threshold"]:
            raise IntegrityError("Checkpoint threshold differs from the handoff")
        if checkpoint.get("approach") != self.handoff["selected_approach"]:
            raise IntegrityError("Checkpoint approach differs from the handoff")
        if checkpoint.get("input_dim") != len(self.handoff["feature_names"]):
            raise IntegrityError("Checkpoint input dimension is inconsistent")

        self.model = TransferClassifier(checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        self.threshold = float(checkpoint["threshold"])
        self.feature_names = list(checkpoint["feature_names"])
        self.device = "cpu"

    def preprocess(self, payload: dict[str, float | int]) -> np.ndarray:
        values = np.asarray(
            [[float(payload[name]) for name in self.feature_names]], dtype=np.float64
        )
        statistics = np.asarray(
            self.preprocessor["imputer_statistics"], dtype=np.float64
        )
        means = np.asarray(self.preprocessor["scaler_mean"], dtype=np.float64)
        scales = np.asarray(self.preprocessor["scaler_scale"], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Inputs must be finite")
        if np.any(scales == 0.0):
            raise IntegrityError("Preprocessor contains a zero scale")
        imputed = np.where(np.isnan(values), statistics, values)
        return ((imputed - means) / scales).astype(np.float32)

    def predict(self, payload: dict[str, float | int]) -> dict[str, Any]:
        inputs = torch.from_numpy(self.preprocess(payload))
        with torch.inference_mode():
            probability = float(torch.sigmoid(self.model(inputs)).item())
        return {
            "probability": probability,
            "decision": int(probability >= self.threshold),
            "threshold": self.threshold,
            "device": self.device,
            "selected_approach": self.handoff["selected_approach"],
        }


def main() -> None:
    runtime = ModelRuntime(Path.cwd())
    print(json.dumps(runtime.predict(SAMPLE_REQUEST), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()