from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.feature_experiment import (
    FORBIDDEN_HOLDOUT_USES,
    LEVEL5_HANDOFF_PATH,
    TRAIN_PATH,
    V2_SCHEMA_PATH,
    fingerprint,
    fold_signature,
    read_json,
    write_json_exclusive,
)
from src.neural_network_experiment import (
    CHECKPOINT_PATH as LEVEL5_CHECKPOINT_PATH,
    LEVEL6_HANDOFF_PATH,
    PREPROCESSOR_PATH as LEVEL5_PREPROCESSOR_PATH,
    REPORT_PATH as LEVEL5_REPORT_PATH,
    SEED,
    THRESHOLD,
    approved_matrix,
    audit_level4,
    evaluate_model,
    fit_preprocessor,
    seed_everything,
    serialize_preprocessor,
    transform,
)
from src.train_models import SCHEMA_PATH, make_group_folds

REPORT_PATH = Path("reports/v4/finetune_experiment.json")
SECRET_REPORT_PATH = Path("reports/v4/catastrophic_forgetting.json")
PREVIEW_SOURCE_PATH = Path("artifacts/v4/level6_source_preview.pt")
SOURCE_CHECKPOINT_PATH = Path("artifacts/v4/level6_source_pretrained.pt")
SELECTED_CHECKPOINT_PATH = Path("artifacts/v4/level6_selected_model.pt")
PREPROCESSOR_PATH = Path("artifacts/v4/level6_preprocessor.json")
LEVEL7_HANDOFF_PATH = Path("artifacts/v4/level7_deploy_handoff.json")
DOC_PATH = Path("docs/level-6-finetune.md")
APPROACHES = ("scratch", "frozen", "partial")
SOURCE_EPOCHS = 60
SOURCE_MULTIPLIER = 8
SOURCE_NOISE = 0.15
TARGET_EPOCHS = 80
ENCODER_LEARNING_RATE = 0.001
HEAD_LEARNING_RATE = 0.01


def verify_fingerprint(path: Path, expected: str) -> None:
    if fingerprint(path) != expected:
        raise ValueError(f"Fingerprint mismatch: {path}")


def load_level5_checkpoint(root: Path) -> dict[str, Any]:
    return torch.load(
        root / LEVEL5_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )


def audit_level5(root: Path) -> dict[str, Any]:
    base = audit_level4(root)
    handoff = read_json(root / LEVEL6_HANDOFF_PATH)
    if handoff.get("handoff_version") != "3.0.0":
        raise ValueError("Expected Level 6 handoff version 3.0.0")
    if handoff.get("created_by") != "Level 5: Networks":
        raise ValueError("Unexpected Level 6 handoff creator")
    if handoff.get("next_level") != "Level 6: Fine-tune":
        raise ValueError("Handoff does not target Level 6: Fine-tune")
    if handoff.get("consumed_v1_holdout_read") is not False:
        raise ValueError("Level 5 handoff does not preserve zero holdout reads")
    if handoff.get("forbidden_uses") != list(FORBIDDEN_HOLDOUT_USES):
        raise ValueError("Consumed-holdout restrictions changed")
    paths = {
        "level4_handoff": LEVEL5_HANDOFF_PATH,
        "development_data": TRAIN_PATH,
        "feature_schema": V2_SCHEMA_PATH,
        "level5_report": LEVEL5_REPORT_PATH,
        "checkpoint": LEVEL5_CHECKPOINT_PATH,
        "preprocessor": LEVEL5_PREPROCESSOR_PATH,
    }
    for label, relative in paths.items():
        verify_fingerprint(root / relative, handoff["fingerprints"][label])
    checkpoint = load_level5_checkpoint(root)
    preprocessor = read_json(root / LEVEL5_PREPROCESSOR_PATH)
    if checkpoint["feature_names"] != handoff["approved_features"]:
        raise ValueError("Checkpoint features differ from the handoff")
    if preprocessor["feature_names"] != handoff["approved_features"]:
        raise ValueError("Preprocessor features differ from the handoff")
    for field in ("threshold", "seed", "best_epoch"):
        if checkpoint[field] != handoff[field]:
            raise ValueError(f"Checkpoint metadata mismatch: {field}")
    if checkpoint["config"] != handoff["selected_config"]:
        raise ValueError("Checkpoint configuration differs from the handoff")
    return {
        "status": "passed",
        "holdout_reads": base["holdout_reads"],
        "verified_fingerprints": sorted(paths),
        "checkpoint_device": "cpu",
        "approved_features": handoff["approved_features"],
    }


def inherited_data(root: Path):
    x, y, groups, _ = approved_matrix(root)
    frame = pd.read_csv(root / TRAIN_PATH)
    folds = make_group_folds(frame, read_json(root / SCHEMA_PATH))
    report = read_json(root / LEVEL5_REPORT_PATH)
    signatures = [fold_signature(fit, valid) for fit, valid in folds]
    if signatures != report["fold_signatures"]:
        raise ValueError("Inherited fold signatures changed")
    return x, y, groups, folds, signatures


def assess_transfer_support(root: Path) -> dict[str, Any]:
    audit_level5(root)
    checkpoint = load_level5_checkpoint(root)
    x, y, groups, folds, signatures = inherited_data(root)
    reasons: list[str] = []
    if x.shape[1] < 2:
        reasons.append("Fewer than two approved numeric features")
    if len(folds) != 4 or len(signatures) != 4:
        reasons.append("Exactly four inherited folds are required")
    for number, (fit_rows, validation_rows) in enumerate(folds, start=1):
        fit_groups = set(groups.iloc[fit_rows])
        validation_groups = set(groups.iloc[validation_rows])
        if fit_groups & validation_groups:
            reasons.append(f"Fold {number} has runner overlap")
        if y.iloc[fit_rows].nunique() < 2 or y.iloc[validation_rows].nunique() < 2:
            reasons.append(f"Fold {number} lacks both target classes")
    has_source_lineage = "source_task" in checkpoint
    return {
        "level5_checkpoint_is_pretrained": has_source_lineage,
        "level5_checkpoint_reason": (
            "A source-task lineage exists." if has_source_lineage else
            "The checkpoint records only readiness-target training and no separate source task."
        ),
        "controlled_transfer_supported": not reasons,
        "blocking_reasons": reasons,
        "approved_feature_count": x.shape[1],
        "runner_count": int(groups.nunique()),
        "fold_count": len(folds),
        "fallback_proposal": (
            None if not reasons else
            "Run a separate all-synthetic denoising-source and shifted-classification-target exercise; do not compare against readiness labels."
        ),
    }


class Encoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer2(torch.relu(self.layer1(inputs))))


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class TransferClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim)
        self.head = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs)).squeeze(1)


def source_arrays(x_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = max(len(x_train) * SOURCE_MULTIPLIER, 64)
    rows = rng.integers(0, len(x_train), size=count)
    clean = x_train[rows] + rng.normal(0.0, 0.03, size=(count, x_train.shape[1]))
    corrupted = clean + rng.normal(0.0, SOURCE_NOISE, size=clean.shape)
    return corrupted.astype(np.float32), clean.astype(np.float32)


def pretrain_source(
    x_train: np.ndarray,
    seed: int,
    epochs: int = SOURCE_EPOCHS,
) -> tuple[DenoisingAutoencoder, list[float], float]:
    seed_everything(seed)
    corrupted, clean = source_arrays(x_train, seed)
    tx = torch.tensor(corrupted, dtype=torch.float32)
    ty = torch.tensor(clean, dtype=torch.float32)
    model = DenoisingAutoencoder(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history: list[float] = []
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(tx) - ty).pow(2).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return model, history, time.perf_counter() - started


def build_target(
    input_dim: int,
    approach: str,
    seed: int,
    encoder_state: dict[str, torch.Tensor] | None,
) -> TransferClassifier:
    seed_everything(seed)
    model = TransferClassifier(input_dim)
    if approach != "scratch":
        if encoder_state is None:
            raise ValueError("Transferred approaches require source weights")
        model.encoder.load_state_dict(encoder_state)
    if approach == "frozen":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    elif approach == "partial":
        for parameter in model.encoder.layer1.parameters():
            parameter.requires_grad = False
    elif approach != "scratch":
        raise ValueError(f"Unknown approach: {approach}")
    return model


def optimizer_for(model: TransferClassifier, approach: str) -> torch.optim.Adam:
    groups: list[dict[str, Any]] = [
        {"params": model.head.parameters(), "lr": HEAD_LEARNING_RATE}
    ]
    if approach == "scratch":
        groups.insert(
            0,
            {"params": model.encoder.parameters(), "lr": ENCODER_LEARNING_RATE},
        )
    elif approach == "partial":
        groups.insert(
            0,
            {"params": model.encoder.layer2.parameters(), "lr": ENCODER_LEARNING_RATE},
        )
    elif approach != "frozen":
        raise ValueError(f"Unknown approach: {approach}")
    return torch.optim.Adam(groups)


def target_loss(model: TransferClassifier, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(nn.BCEWithLogitsLoss()(model(x), y).item())


def train_target(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    approach: str,
    seed: int,
    encoder_state: dict[str, torch.Tensor] | None,
    epochs: int = TARGET_EPOCHS,
) -> tuple[TransferClassifier, list[dict[str, float]], int, float]:
    model = build_target(x_fit.shape[1], approach, seed, encoder_state)
    optimizer = optimizer_for(model, approach)
    criterion = nn.BCEWithLogitsLoss()
    tx = torch.tensor(x_fit, dtype=torch.float32)
    ty = torch.tensor(y_fit, dtype=torch.float32)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tx), ty)
        loss.backward()
        optimizer.step()
        history.append(
            {"epoch": float(epoch), "train_loss": target_loss(model, tx, ty)}
        )
    return model, history, epochs, time.perf_counter() - started


def gap_values(train: dict[str, float], validation: dict[str, float]) -> dict[str, float]:
    return {
        metric: (
            validation[metric] - train[metric]
            if metric == "loss"
            else train[metric] - validation[metric]
        )
        for metric in train
    }


def evaluate_fold(root: Path, fold_number: int, fit_rows, validation_rows) -> dict[str, Any]:
    x, y, groups, _, _ = inherited_data(root)
    fit_groups = set(groups.iloc[fit_rows])
    validation_groups = set(groups.iloc[validation_rows])
    if fit_groups & validation_groups:
        raise ValueError("Outer runner leakage detected")
    imputer, scaler = fit_preprocessor(x.iloc[fit_rows])
    x_fit = transform(x.iloc[fit_rows], imputer, scaler)
    x_validation = transform(x.iloc[validation_rows], imputer, scaler)
    source, source_history, source_seconds = pretrain_source(
        x_fit, SEED + 200 + fold_number
    )
    results: dict[str, Any] = {}
    for approach in APPROACHES:
        model, _, downstream_epochs, downstream_seconds = train_target(
            x_fit,
            y.iloc[fit_rows].to_numpy(np.float32),
            approach,
            SEED + fold_number,
            None if approach == "scratch" else source.encoder.state_dict(),
            epochs=TARGET_EPOCHS,
        )
        train_metrics = evaluate_model(
            model, x_fit, y.iloc[fit_rows].to_numpy(np.float32)
        )
        validation_metrics = evaluate_model(
            model,
            x_validation,
            y.iloc[validation_rows].to_numpy(np.float32),
        )
        source_time = 0.0 if approach == "scratch" else source_seconds
        results[approach] = {
            "downstream_epochs": downstream_epochs,
            "train": train_metrics,
            "validation": validation_metrics,
            "generalization_gap": gap_values(train_metrics, validation_metrics),
            "source_pretraining_seconds": source_time,
            "downstream_training_seconds": downstream_seconds,
            "total_training_seconds": source_time + downstream_seconds,
        }
    return {
        "fold": fold_number,
        "fold_signature": fold_signature(fit_rows, validation_rows),
        "source_training_runners": sorted(str(value) for value in fit_groups),
        "target_validation_runners": sorted(str(value) for value in validation_groups),
        "source_validation_runner_overlap": [],
        "source_target_labels_used": False,
        "source_reconstruction_loss": {
            "first": source_history[0],
            "last": source_history[-1],
        },
        "approaches": results,
    }


def summarize(folds: list[dict[str, Any]], approach: str) -> dict[str, Any]:
    metrics = ("accuracy", "precision", "recall", "f1", "loss")
    return {
        "validation_mean": {
            metric: fmean(row["approaches"][approach]["validation"][metric] for row in folds)
            for metric in metrics
        },
        "validation_stdev": {
            metric: pstdev(row["approaches"][approach]["validation"][metric] for row in folds)
            for metric in metrics
        },
        "gap_mean": {
            metric: fmean(row["approaches"][approach]["generalization_gap"][metric] for row in folds)
            for metric in metrics
        },
        "source_pretraining_seconds_mean": fmean(
            row["approaches"][approach]["source_pretraining_seconds"] for row in folds
        ),
        "downstream_training_seconds_mean": fmean(
            row["approaches"][approach]["downstream_training_seconds"] for row in folds
        ),
        "total_training_seconds_mean": fmean(
            row["approaches"][approach]["total_training_seconds"] for row in folds
        ),
    }


def ensure_outputs_absent(root: Path, paths: tuple[Path, ...]) -> None:
    for relative in paths:
        if (root / relative).exists():
            raise FileExistsError(f"Refusing to overwrite {relative}")


def save_source_checkpoint(path: Path, model: DenoisingAutoencoder, feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_task": "denoise_corrupted_synthetic_feature_bootstraps",
            "pretraining_type": "self_supervised",
            "external_domain_pretraining": False,
            "target_labels_used": False,
            "encoder_state": model.encoder.state_dict(),
            "decoder_state": model.decoder.state_dict(),
            "input_dim": len(feature_names),
            "feature_names": feature_names,
            "seed": SEED,
            "epochs": SOURCE_EPOCHS,
            "noise_standard_deviation": SOURCE_NOISE,
        },
        path,
    )


def pretrain_preview(root: Path) -> dict[str, Any]:
    audit_level5(root)
    assessment = assess_transfer_support(root)
    if not assessment["controlled_transfer_supported"]:
        raise ValueError(assessment["fallback_proposal"])
    ensure_outputs_absent(root, (PREVIEW_SOURCE_PATH,))
    x, _, groups, folds, _ = inherited_data(root)
    fit_rows, validation_rows = folds[0]
    imputer, scaler = fit_preprocessor(x.iloc[fit_rows])
    x_fit = transform(x.iloc[fit_rows], imputer, scaler)
    model, history, seconds = pretrain_source(x_fit, SEED)
    save_source_checkpoint(root / PREVIEW_SOURCE_PATH, model, list(x.columns))
    overlap = set(groups.iloc[fit_rows]) & set(groups.iloc[validation_rows])
    return {
        "checkpoint": str(PREVIEW_SOURCE_PATH),
        "source_task": "denoise_corrupted_synthetic_feature_bootstraps",
        "pretraining_type": "self_supervised",
        "external_domain_pretraining": False,
        "target_labels_used": False,
        "source_validation_runner_overlap": sorted(str(value) for value in overlap),
        "first_reconstruction_loss": history[0],
        "last_reconstruction_loss": history[-1],
        "source_pretraining_seconds": seconds,
    }


def run_experiment(root: Path) -> dict[str, Any]:
    audit = audit_level5(root)
    assessment = assess_transfer_support(root)
    if not assessment["controlled_transfer_supported"]:
        raise ValueError(assessment["fallback_proposal"])
    outputs = (
        REPORT_PATH,
        SOURCE_CHECKPOINT_PATH,
        SELECTED_CHECKPOINT_PATH,
        PREPROCESSOR_PATH,
        LEVEL7_HANDOFF_PATH,
    )
    ensure_outputs_absent(root, outputs)
    x, y, _, folds, signatures = inherited_data(root)
    records = [
        evaluate_fold(root, number, fit_rows, validation_rows)
        for number, (fit_rows, validation_rows) in enumerate(folds, start=1)
    ]
    aggregates = {approach: summarize(records, approach) for approach in APPROACHES}
    selected = max(
        APPROACHES,
        key=lambda name: (
            aggregates[name]["validation_mean"]["f1"],
            -abs(aggregates[name]["gap_mean"]["f1"]),
            -aggregates[name]["downstream_training_seconds_mean"],
        ),
    )
    imputer, scaler = fit_preprocessor(x)
    x_all = transform(x, imputer, scaler)
    source, source_history, source_seconds = pretrain_source(x_all, SEED + 400)
    final_model, _, downstream_epochs, downstream_seconds = train_target(
        x_all,
        y.to_numpy(np.float32),
        selected,
        SEED,
        None if selected == "scratch" else source.encoder.state_dict(),
        epochs=TARGET_EPOCHS,
    )
    save_source_checkpoint(root / SOURCE_CHECKPOINT_PATH, source, list(x.columns))
    torch.save(
        {
            "model_state": final_model.state_dict(),
            "model_class": "TransferClassifier",
            "input_dim": x.shape[1],
            "feature_names": list(x.columns),
            "approach": selected,
            "threshold": THRESHOLD,
            "seed": SEED,
            "downstream_epochs": downstream_epochs,
            "source_task": None if selected == "scratch" else "denoise_corrupted_synthetic_feature_bootstraps",
        },
        root / SELECTED_CHECKPOINT_PATH,
    )
    write_json_exclusive(
        root / PREPROCESSOR_PATH,
        serialize_preprocessor(imputer, scaler, list(x.columns)),
    )
    report = {
        "report_version": "4.0.0",
        "evidence_scope": "development_only",
        "consumed_v1_holdout_read": False,
        "inherited_v1_preprocessing_limitation": (
            "The historical v1 process used development-wide mean imputation before splitting; Level 6 does not claim the full inherited history is leakage-free."
        ),
        "audit": audit,
        "assessment": assessment,
        "fair_comparison_contract": {
            "identical_fold_signatures": True,
            "identical_downstream_epochs": TARGET_EPOCHS,
            "identical_architecture": True,
            "identical_head_learning_rate": HEAD_LEARNING_RATE,
            "scratch_and_partial_encoder_learning_rate": ENCODER_LEARNING_RATE,
            "validation_used_for_fitting": False,
        },
        "source_task": {
            "name": "denoise_corrupted_synthetic_feature_bootstraps",
            "pretraining_type": "self_supervised",
            "external_domain_pretraining": False,
            "target": "clean standardized approved feature vector",
            "target_labels_used": False,
            "synthetic_multiplier": SOURCE_MULTIPLIER,
            "noise_standard_deviation": SOURCE_NOISE,
        },
        "target_task": "binary running readiness classification",
        "fold_signatures": signatures,
        "folds": records,
        "aggregates": aggregates,
        "selected_approach": selected,
        "final_development_fit": {
            "downstream_epochs": downstream_epochs,
            "source_first_loss": source_history[0],
            "source_last_loss": source_history[-1],
            "source_pretraining_seconds": source_seconds,
            "downstream_training_seconds": downstream_seconds,
            "train_metrics": evaluate_model(final_model, x_all, y.to_numpy(np.float32)),
        },
        "determinism": {
            "seed": SEED,
            "torch_use_deterministic_algorithms": True,
            "device": "cpu",
            "cross_platform_bitwise_guarantee": False,
        },
    }
    write_json_exclusive(root / REPORT_PATH, report)
    handoff = {
        "handoff_version": "4.0.0",
        "created_by": "Level 6: Fine-tune",
        "next_level": "Level 7: Deploy",
        "evidence_scope": "development_only",
        "consumed_v1_holdout_read": False,
        "forbidden_uses": list(FORBIDDEN_HOLDOUT_USES),
        "selected_approach": selected,
        "feature_names": list(x.columns),
        "threshold": THRESHOLD,
        "seed": SEED,
        "downstream_epochs": TARGET_EPOCHS,
        "limitations": [
            report["inherited_v1_preprocessing_limitation"],
            "The source task is controlled self-supervised synthetic denoising, not external-domain pretraining.",
            "Training times vary with host load even when model states and metrics reproduce in the same pinned environment.",
        ],
        "fingerprints": {
            "level6_input_handoff": fingerprint(root / LEVEL6_HANDOFF_PATH),
            "development_data": fingerprint(root / TRAIN_PATH),
            "level6_report": fingerprint(root / REPORT_PATH),
            "source_checkpoint": fingerprint(root / SOURCE_CHECKPOINT_PATH),
            "selected_checkpoint": fingerprint(root / SELECTED_CHECKPOINT_PATH),
            "preprocessor": fingerprint(root / PREPROCESSOR_PATH),
        },
    }
    write_json_exclusive(root / LEVEL7_HANDOFF_PATH, handoff)
    return {
        "selected_approach": selected,
        "report": str(REPORT_PATH),
        "checkpoint": str(SELECTED_CHECKPOINT_PATH),
        "handoff": str(LEVEL7_HANDOFF_PATH),
    }


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def reconstruction_loss(
    encoder: Encoder,
    decoder: nn.Module,
    clean: np.ndarray,
) -> float:
    encoder.eval()
    decoder.eval()
    tx = torch.tensor(clean, dtype=torch.float32)
    with torch.no_grad():
        return float((decoder(encoder(tx)) - tx).pow(2).mean().item())


def representation_similarity(before: Encoder, after: Encoder, x: np.ndarray) -> float:
    before.eval()
    after.eval()
    tx = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        left = before(tx)
        right = after(tx)
        values = torch.nn.functional.cosine_similarity(left, right, dim=1)
    return float(values.mean().item())


def train_full_finetune(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    encoder_state: dict[str, torch.Tensor],
    seed: int,
    encoder_lr: float,
    epochs: int = 40,
) -> TransferClassifier:
    seed_everything(seed)
    model = TransferClassifier(x_fit.shape[1])
    model.encoder.load_state_dict(encoder_state)
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.head.parameters(), "lr": HEAD_LEARNING_RATE},
        ]
    )
    criterion = nn.BCEWithLogitsLoss()
    tx = torch.tensor(x_fit, dtype=torch.float32)
    ty = torch.tensor(y_fit, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tx), ty)
        loss.backward()
        optimizer.step()
    return model


def run_secret(root: Path) -> dict[str, Any]:
    audit_level5(root)
    ensure_outputs_absent(root, (SECRET_REPORT_PATH,))
    x, y, _, folds, _ = inherited_data(root)
    fit_rows, validation_rows = folds[0]
    imputer, scaler = fit_preprocessor(x.iloc[fit_rows])
    x_fit = transform(x.iloc[fit_rows], imputer, scaler)
    x_validation = transform(x.iloc[validation_rows], imputer, scaler)
    source, _, _ = pretrain_source(x_fit, SEED + 500)
    immutable_encoder = copy.deepcopy(source.encoder).eval()
    immutable_decoder = copy.deepcopy(source.decoder).eval()
    immutable_encoder.requires_grad_(False)
    immutable_decoder.requires_grad_(False)
    immutable_hash_before = state_sha256(immutable_encoder)
    _, probe = source_arrays(x_fit, SEED + 500)
    probe_hash = array_sha256(probe)
    source_loss_before = reconstruction_loss(
        immutable_encoder, immutable_decoder, probe
    )
    rows: dict[str, Any] = {}
    for label, learning_rate in (("excessive", 0.1), ("controlled", 0.001)):
        model = train_full_finetune(
            x_fit,
            y.iloc[fit_rows].to_numpy(np.float32),
            immutable_encoder.state_dict(),
            SEED,
            learning_rate,
        )
        after_loss = reconstruction_loss(model.encoder, immutable_decoder, probe)
        similarity = representation_similarity(
            immutable_encoder, model.encoder, probe
        )
        reconstruction_change = after_loss - source_loss_before
        rows[label] = {
            "encoder_learning_rate": learning_rate,
            "target_validation": evaluate_model(
                model,
                x_validation,
                y.iloc[validation_rows].to_numpy(np.float32),
            ),
            "probe_sha256": probe_hash,
            "source_reconstruction_loss_before": source_loss_before,
            "source_reconstruction_loss_after": after_loss,
            "source_reconstruction_loss_change": reconstruction_change,
            "representation_cosine_similarity": similarity,
            "forgetting_supported_by_both_measures": (
                reconstruction_change > 1e-6 and similarity < 0.99
            ),
        }
    immutable_hash_after = state_sha256(immutable_encoder)
    if immutable_hash_after != immutable_hash_before:
        raise RuntimeError("Immutable pretrained encoder changed")
    excessive = rows["excessive"]
    controlled = rows["controlled"]
    excessive_more_forgetting = (
        excessive["source_reconstruction_loss_change"]
        > controlled["source_reconstruction_loss_change"]
        and excessive["representation_cosine_similarity"]
        < controlled["representation_cosine_similarity"]
    )
    payload = {
        "report_version": "4.0.0",
        "evidence_scope": "exploratory_development_only",
        "influences_selected_model": False,
        "consumed_v1_holdout_read": False,
        "immutable_encoder_sha256_before": immutable_hash_before,
        "immutable_encoder_sha256_after": immutable_hash_after,
        "immutable_encoder_unchanged": True,
        "probe_sha256": probe_hash,
        "same_probe_used_for_all_measurements": True,
        "comparison": rows,
        "excessive_rate_caused_more_forgetting_supported": excessive_more_forgetting,
        "interpretation_rule": (
            "Describe an arm as evidence of forgetting only when reconstruction loss increases and representation similarity falls. Describe the excessive rate as causing more forgetting only when both retention measures are worse than the controlled rate."
        ),
    }
    write_json_exclusive(root / SECRET_REPORT_PATH, payload)
    return {"report": str(SECRET_REPORT_PATH), "comparison": rows}


def document_experiment(root: Path) -> None:
    report = read_json(root / REPORT_PATH)
    selected = report["selected_approach"]
    rows = []
    for approach in APPROACHES:
        aggregate = report["aggregates"][approach]
        values = aggregate["validation_mean"]
        rows.append(
            f"| {approach} | {values['f1']:.3f} | {values['precision']:.3f} | "
            f"{values['recall']:.3f} | {values['loss']:.3f} | "
            f"{aggregate['source_pretraining_seconds_mean']:.3f} | "
            f"{aggregate['downstream_training_seconds_mean']:.3f} |"
        )
    text = f"""# Level 6: Fine-Tune

## Result

Selected approach: `{selected}` using development-only runner-safe evidence.

| Approach | F1 | Precision | Recall | Loss | Source seconds | Downstream seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Fair comparison contract

All three approaches use the same inherited folds, encoder and head architecture, loss, threshold, seeds, and `{TARGET_EPOCHS}` downstream epochs. Scratch and partial fine-tuning use the same encoder learning rate, and all three use the same head learning rate. Source pretraining and downstream fitting times are reported separately.

## What counts as pretrained

The Level 5 readiness checkpoint is not called pretrained because it learned the same supervised target. Level 6 creates pretrained weights only after the encoder learns a separately defined denoising source task without readiness labels.

This is controlled self-supervised pretraining on synthetic variants of fold-training data. It is not same-task warm-starting and it is not external-domain transfer learning.

Frozen-feature transfer keeps the source encoder unchanged and trains a new head. Partial fine-tuning keeps the first encoder layer frozen while adapting the later encoder layer. Scratch training initializes the same target architecture randomly.

## Leakage and evidence boundary

All three methods use the same four inherited runner-safe folds. Source synthesis, fold-local preprocessing, and downstream fitting remain inside each outer-training boundary. Target-validation runners and labels are used only for evaluation. The consumed v1 holdout is not read.

The inherited v1 process used development-wide mean imputation before splitting. Level 6 preprocessing is boundary-local, but the full historical pipeline is not retroactively called leakage-free.

## Reproducibility

The workflow uses seed `{report['determinism']['seed']}`, CPU execution, and deterministic algorithms where supported. PyTorch does not guarantee bitwise equality across releases or platforms. Training time is measured separately and can vary with host load.

## Artifacts

- `{REPORT_PATH}`
- `{SOURCE_CHECKPOINT_PATH}`
- `{SELECTED_CHECKPOINT_PATH}`
- `{PREPROCESSOR_PATH}`
- `{LEVEL7_HANDOFF_PATH}`

## Reproduce

```bash
docker build --platform linux/amd64 -f Dockerfile.level6 -t running-readiness-level6:2.15.0 .
docker run --rm --platform linux/amd64 --mount type=bind,src="$(pwd)",dst=/workspace -w /workspace running-readiness-level6:2.15.0 python -m unittest -v tests/test_finetune.py
```
"""
    (root / DOC_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / DOC_PATH).write_text(text, encoding="utf-8")
    readme = root / "README.md"
    marker = "<!-- LEVEL_6_FINETUNE -->"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# Running Readiness\n"
    line = f"{marker}\n- [Level 6: Fine-Tune]({DOC_PATH.as_posix()})\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n\n" + line
    else:
        existing = existing.rstrip() + "\n\n" + line
    readme.write_text(existing, encoding="utf-8")


def main() -> None:
    root = Path.cwd()
    command = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if command == "audit":
        payload = audit_level5(root)
    elif command == "assess":
        payload = assess_transfer_support(root)
    elif command == "pretrain":
        payload = pretrain_preview(root)
    elif command == "run":
        payload = run_experiment(root)
    elif command == "secret":
        payload = run_secret(root)
    elif command == "document":
        document_experiment(root)
        payload = {"documentation": str(DOC_PATH), "status": "written"}
    else:
                raise SystemExit("Use: audit | assess | pretrain | run | secret | document")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()