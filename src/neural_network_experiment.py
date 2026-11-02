from __future__ import annotations

import copy
import json
import os
import platform
import random
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.feature_experiment import (
    FORBIDDEN_HOLDOUT_USES,
    HOLDOUT_MARKER_PATH,
    LEVEL5_HANDOFF_PATH,
    RAW_FEATURES,
    TARGET_COLUMN,
    TRAIN_PATH,
    V2_REPORT_PATH,
    V2_SCHEMA_PATH,
    SchemaFeatureTransformer,
    audit_handoff,
    fingerprint,
    fold_signature,
    metric_values,
    read_json,
    write_json_exclusive,
)
from src.train_models import GROUP_COLUMN, SCHEMA_PATH, make_group_folds

REPORT_PATH = Path("reports/v3/neural_network_experiment.json")
CHECKPOINT_PATH = Path("artifacts/v3/level5_mlp_state.pt")
PREPROCESSOR_PATH = Path("artifacts/v3/level5_preprocessor.json")
LEVEL6_HANDOFF_PATH = Path("artifacts/v3/level6_finetune_handoff.json")
DOC_PATH = Path("docs/level-5-networks.md")
SEED = 42
THRESHOLD = 0.5
CONFIGS = (
    {"name": "compact", "hidden_units": 8, "dropout": 0.0,
     "learning_rate": 0.01, "weight_decay": 0.0},
    {"name": "regularized", "hidden_units": 16, "dropout": 0.25,
     "learning_rate": 0.01, "weight_decay": 0.001},
)


def _sha(root: Path, relative: Path) -> str:
    return fingerprint(root / relative)


def validate_level5_contract(
    handoff: dict[str, Any], feature_schema: dict[str, Any],
    report: dict[str, Any], marker: dict[str, Any]
) -> None:
    if handoff.get("handoff_version") != "2.0.0":
        raise ValueError("Expected Level 4 handoff version 2.0.0")
    if handoff.get("next_level") != "Level 5: Networks":
        raise ValueError("Handoff does not target Level 5: Networks")
    approved = handoff.get("selected_candidate_features", [])
    if not approved or not set(approved).issubset(feature_schema.get("candidate_feature_columns", [])):
        raise ValueError("Approved features do not match the v2 feature schema")
    if handoff.get("selected_variant") != report.get("selected_variant"):
        raise ValueError("Handoff and report disagree on selected variant")
    expected = report.get("cross_validation", {}).get("identical_fold_signatures", [])
    if len(expected) != 4:
        raise ValueError("Exactly four inherited fold signatures are required")
    for variant in ("raw", "engineered"):
        observed = [row["fold_signature"] for row in report["results"][variant]["folds"]]
        if observed != expected:
            raise ValueError(f"Level 4 {variant} folds changed")
    if report.get("consumed_v1_holdout_read") is not False:
        raise ValueError("Level 4 report does not prove zero holdout reads")
    if set(marker.get("forbidden_future_uses", [])) != FORBIDDEN_HOLDOUT_USES:
        raise ValueError("Consumed-holdout restrictions changed")


def audit_level4(root: Path) -> dict[str, Any]:
    base = audit_handoff(root)
    handoff = read_json(root / LEVEL5_HANDOFF_PATH)
    feature_schema = read_json(root / V2_SCHEMA_PATH)
    report = read_json(root / V2_REPORT_PATH)
    marker = read_json(root / HOLDOUT_MARKER_PATH)
    validate_level5_contract(handoff, feature_schema, report, marker)
    checks = {
        "feature_schema": (V2_SCHEMA_PATH, handoff["feature_schema"]["sha256"]),
        "feature_report": (V2_REPORT_PATH, handoff["feature_experiment"]["sha256"]),
        "development_data": (TRAIN_PATH, handoff["development_data"]["sha256"]),
    }
    for label, (path, expected) in checks.items():
        if _sha(root, path) != expected:
            raise ValueError(f"Fingerprint mismatch: {label}")
    return {
        "status": "passed",
        "holdout_reads": base["holdout_read"],
        "selected_variant": handoff["selected_variant"],
        "approved_features": handoff["selected_candidate_features"],
        "fold_signatures": report["cross_validation"]["identical_fold_signatures"],
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def approved_matrix(root: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    handoff = read_json(root / LEVEL5_HANDOFF_PATH)
    schema = read_json(root / V2_SCHEMA_PATH)
    frame = pd.read_csv(root / TRAIN_PATH)
    transformer = SchemaFeatureTransformer(
        mode=handoff["selected_variant"],
        engineered_features=tuple(schema["engineered_feature_columns"]),
    )
    transformed = transformer.fit_transform(frame[list(RAW_FEATURES)])
    approved = handoff["selected_candidate_features"]
    missing = sorted(set(approved) - set(transformed.columns))
    if missing:
        raise ValueError(f"Approved features unavailable: {missing}")
    return transformed[approved], frame[TARGET_COLUMN], frame[GROUP_COLUMN], handoff


def fit_preprocessor(x_fit: pd.DataFrame) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="mean")
    imputed = imputer.fit_transform(x_fit)
    scaler = StandardScaler()
    scaler.fit(imputed)
    return imputer, scaler


def transform(x: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(imputer.transform(x)).astype(np.float32)


def inner_group_split(x: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    fit_idx, stop_idx = next(splitter.split(x, y, groups))
    fit_groups = set(groups.iloc[fit_idx])
    stop_groups = set(groups.iloc[stop_idx])
    if fit_groups & stop_groups:
        raise ValueError("Inner runner leakage detected")
    return fit_idx, stop_idx


def _tensors(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _loss(model: BinaryMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(nn.BCEWithLogitsLoss()(model(x), y).item())


def train_with_early_stopping(
    x_fit: np.ndarray, y_fit: np.ndarray, x_stop: np.ndarray, y_stop: np.ndarray,
    config: dict[str, Any], seed: int, max_epochs: int = 200, patience: int = 15,
) -> tuple[BinaryMLP, list[dict[str, float]], int]:
    seed_everything(seed)
    tx, ty = _tensors(x_fit, y_fit)
    sx, sy = _tensors(x_stop, y_stop)
    model = BinaryMLP(tx.shape[1], config["hidden_units"], config["dropout"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    history: list[dict[str, float]] = []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tx), ty)
        loss.backward()
        optimizer.step()
        train_loss = _loss(model, tx, ty)
        stop_loss = _loss(model, sx, sy)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": stop_loss})
        if stop_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Early stopping never recorded a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_fixed(
    x_fit: np.ndarray, y_fit: np.ndarray, config: dict[str, Any], seed: int,
    epochs: int, x_watch: np.ndarray | None = None, y_watch: np.ndarray | None = None,
) -> tuple[BinaryMLP, list[dict[str, float]]]:
    seed_everything(seed)
    tx, ty = _tensors(x_fit, y_fit)
    model = BinaryMLP(tx.shape[1], config["hidden_units"], config["dropout"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history: list[dict[str, float]] = []
    watch = _tensors(x_watch, y_watch) if x_watch is not None and y_watch is not None else None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tx), ty)
        loss.backward()
        optimizer.step()
        if watch is not None:
            history.append({"epoch": epoch, "train_loss": _loss(model, tx, ty),
                            "validation_loss": _loss(model, watch[0], watch[1])})
    return model, history


def evaluate_model(model: BinaryMLP, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    tx, ty = _tensors(x, y)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tx)).cpu().numpy()
    values = metric_values(y.astype(int), (probabilities >= THRESHOLD).astype(int))
    values["loss"] = _loss(model, tx, ty)
    return values


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("accuracy", "precision", "recall", "f1", "loss")
    return {
        "validation_mean": {m: fmean(row["validation"][m] for row in folds) for m in metrics},
        "validation_stdev": {m: pstdev(row["validation"][m] for row in folds) for m in metrics},
        "gap_mean": {m: fmean(row["generalization_gap"][m] for row in folds) for m in metrics},
    }


def evaluate_config(
    x: pd.DataFrame, y: pd.Series, groups: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]], config: dict[str, Any],
) -> dict[str, Any]:
    records = []
    for fold_number, (fit_rows, validation_rows) in enumerate(folds, start=1):
        outer_fit_groups = set(groups.iloc[fit_rows])
        outer_validation_groups = set(groups.iloc[validation_rows])
        if outer_fit_groups & outer_validation_groups:
            raise ValueError("Outer runner leakage detected")
        x_outer, y_outer, g_outer = x.iloc[fit_rows], y.iloc[fit_rows], groups.iloc[fit_rows]
        inner_fit, inner_stop = inner_group_split(x_outer, y_outer, g_outer, SEED + fold_number)
        inner_imp, inner_scale = fit_preprocessor(x_outer.iloc[inner_fit])
        _, history, best_epoch = train_with_early_stopping(
            transform(x_outer.iloc[inner_fit], inner_imp, inner_scale), y_outer.iloc[inner_fit].to_numpy(np.float32),
            transform(x_outer.iloc[inner_stop], inner_imp, inner_scale), y_outer.iloc[inner_stop].to_numpy(np.float32),
            config, SEED + fold_number,
        )
        outer_imp, outer_scale = fit_preprocessor(x_outer)
        model, _ = train_fixed(
            transform(x_outer, outer_imp, outer_scale), y_outer.to_numpy(np.float32),
            config, SEED + fold_number, best_epoch,
        )
        train_metrics = evaluate_model(model, transform(x_outer, outer_imp, outer_scale), y_outer.to_numpy(np.float32))
        validation_metrics = evaluate_model(
            model, transform(x.iloc[validation_rows], outer_imp, outer_scale),
            y.iloc[validation_rows].to_numpy(np.float32),
        )
        gap = {m: (validation_metrics[m] - train_metrics[m] if m == "loss" else train_metrics[m] - validation_metrics[m])
               for m in train_metrics}
        records.append({
            "fold": fold_number, "fold_signature": fold_signature(fit_rows, validation_rows),
            "outer_runner_overlap": [], "inner_runner_overlap": [], "best_epoch": best_epoch,
            "loss_history": history, "train": train_metrics,
            "validation": validation_metrics, "generalization_gap": gap,
        })
    return {"config": config, "folds": records, "aggregate": summarize_folds(records)}


def dependency_versions() -> dict[str, str]:
    return {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch")}


def serialize_preprocessor(imputer: SimpleImputer, scaler: StandardScaler, features: list[str]) -> dict[str, Any]:
    return {
        "feature_names": features, "imputation": "mean", "feature_selection": "none",
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
    }


def run_experiment(root: Path) -> dict[str, Any]:
    audit = audit_level4(root)
    for relative in (REPORT_PATH, CHECKPOINT_PATH, PREPROCESSOR_PATH, LEVEL6_HANDOFF_PATH):
        if (root / relative).exists():
            raise FileExistsError(f"Refusing to overwrite {relative}")
    x, y, groups, handoff = approved_matrix(root)
    folds = make_group_folds(pd.read_csv(root / TRAIN_PATH), read_json(root / SCHEMA_PATH))
    signatures = [fold_signature(fit, valid) for fit, valid in folds]
    if signatures != audit["fold_signatures"]:
        raise ValueError("Current runner-safe folds do not match Level 4")
    results = {config["name"]: evaluate_config(x, y, groups, folds, config) for config in CONFIGS}
    selected_name = max(
        results,
        key=lambda name: (
            results[name]["aggregate"]["validation_mean"]["f1"],
            -abs(results[name]["aggregate"]["gap_mean"]["f1"]),
            name == "regularized",
        ),
    )
    selected = results[selected_name]["config"]
    inner_fit, inner_stop = inner_group_split(x, y, groups, SEED)
    inner_imp, inner_scale = fit_preprocessor(x.iloc[inner_fit])
    _, final_history, best_epoch = train_with_early_stopping(
        transform(x.iloc[inner_fit], inner_imp, inner_scale), y.iloc[inner_fit].to_numpy(np.float32),
        transform(x.iloc[inner_stop], inner_imp, inner_scale), y.iloc[inner_stop].to_numpy(np.float32),
        selected, SEED,
    )
    imputer, scaler = fit_preprocessor(x)
    model, _ = train_fixed(transform(x, imputer, scaler), y.to_numpy(np.float32), selected, SEED, best_epoch)
    checkpoint = {
        "model_state": model.state_dict(), "input_dim": x.shape[1], "config": selected,
        "feature_names": list(x.columns), "threshold": THRESHOLD,
        "seed": SEED, "best_epoch": best_epoch,
    }
    (root / CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, root / CHECKPOINT_PATH)
    write_json_exclusive(root / PREPROCESSOR_PATH, serialize_preprocessor(imputer, scaler, list(x.columns)))
    level4_report = read_json(root / V2_REPORT_PATH)
    report = {
        "report_version": "3.0.0", "evidence_scope": "development_only",
        "consumed_v1_holdout_read": False, "threshold": THRESHOLD,
        "determinism": {"seed": SEED, "torch_use_deterministic_algorithms": True,
                        "cross_platform_bitwise_guarantee": False},
        "dependencies": dependency_versions(), "runtime": {"python": sys.version, "platform": platform.platform(),
                        "base_image": os.environ.get("LEVEL5_BASE_IMAGE", "unknown")},
        "approved_features": list(x.columns), "feature_selection": "none",
        "fold_signatures": signatures,
        "classical_development_controls": {
            variant: level4_report["results"][variant]["aggregate"] for variant in ("raw", "engineered")
        },
        "mlp_results": results, "selected_config": selected_name,
        "final_development_fit": {"best_epoch": best_epoch, "loss_history": final_history,
                                  "train_metrics": evaluate_model(model, transform(x, imputer, scaler), y.to_numpy(np.float32))},
        "limitations": [
            "All Level 5 comparisons are development-only because the v1 holdout is already consumed.",
            "Level 5 fits imputation and scaling inside each relevant training boundary.",
            "The inherited v1 process used development-wide mean imputation before splitting, so the historical pipeline is not claimed to be leakage-free.",
            "PyTorch does not guarantee identical results across releases, commits, or platforms even with deterministic settings.",
        ],
    }
    write_json_exclusive(root / REPORT_PATH, report)
    handoff6 = {
        "handoff_version": "3.0.0", "created_by": "Level 5: Networks",
        "next_level": "Level 6: Fine-tune", "consumed_v1_holdout_read": False,
        "forbidden_uses": list(FORBIDDEN_HOLDOUT_USES),
        "approved_features": list(x.columns), "selected_config": selected,
        "threshold": THRESHOLD, "seed": SEED, "best_epoch": best_epoch,
        "fingerprints": {
            "level4_handoff": fingerprint(root / LEVEL5_HANDOFF_PATH),
            "development_data": fingerprint(root / TRAIN_PATH),
            "feature_schema": fingerprint(root / V2_SCHEMA_PATH),
            "level5_report": fingerprint(root / REPORT_PATH),
            "checkpoint": fingerprint(root / CHECKPOINT_PATH),
            "preprocessor": fingerprint(root / PREPROCESSOR_PATH),
        },
        "limitations": report["limitations"],
    }
    write_json_exclusive(root / LEVEL6_HANDOFF_PATH, handoff6)
    return {"selected_config": selected_name, "best_epoch": best_epoch,
            "report": str(REPORT_PATH), "handoff": str(LEVEL6_HANDOFF_PATH)}


def concept_demo(root: Path) -> dict[str, Any]:
    audit_level4(root)
    seed_everything(SEED)
    inputs = torch.tensor([[0.5, -1.0]], dtype=torch.float32)
    target = torch.tensor([1.0], dtype=torch.float32)
    neuron = nn.Linear(2, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(neuron.parameters(), lr=0.1)
    before_logit = neuron(inputs).squeeze(1)
    before_loss = criterion(before_logit, target)
    optimizer.zero_grad(set_to_none=True)
    before_loss.backward()
    gradients = {name: parameter.grad.detach().tolist() for name, parameter in neuron.named_parameters()}
    optimizer.step()
    after_logit = neuron(inputs).squeeze(1)
    after_loss = criterion(after_logit, target)
    return {"tensor_shape": list(inputs.shape), "neuron_equation": "z = xW^T + b",
            "before_logit_not_probability": float(before_logit.item()),
            "gradients": gradients, "before_loss": float(before_loss.item()),
            "after_logit_not_probability": float(after_logit.item()), "after_loss": float(after_loss.item())}


def document_experiment(root: Path) -> None:
    report = read_json(root / REPORT_PATH)
    selected = report["selected_config"]
    aggregate = report["mlp_results"][selected]["aggregate"]
    metrics = aggregate["validation_mean"]
    text = f"""# Level 5: Networks

## Result

Selected configuration: `{selected}` on development-only runner-safe folds.

| Metric | Mean outer validation |
| --- | ---: |
| F1 | {metrics['f1']:.3f} |
| Precision | {metrics['precision']:.3f} |
| Recall | {metrics['recall']:.3f} |
| Loss | {metrics['loss']:.3f} |

## Architecture and training

The model is `Linear -> ReLU -> Dropout -> Linear`. Training uses raw logits with binary cross-entropy, explicit backpropagation, Adam updates, nested runner-safe early stopping, and threshold `{report['threshold']}` only for metrics.

## Evidence boundary

The v1 holdout is consumed and was not read. These are development-only comparisons against Level 4's raw and engineered controls. Level 5 preprocessing is fold-local, but v1 historically used development-wide mean imputation before splitting. The full historical pipeline is not described as leakage-free.

## Artifacts

- `{REPORT_PATH}`
- `{CHECKPOINT_PATH}`
- `{PREPROCESSOR_PATH}`
- `{LEVEL6_HANDOFF_PATH}`

## Reproduce

```bash
docker build --platform linux/amd64 -f Dockerfile.level5 -t running-readiness-level5:2.15.0 .
docker run --rm --platform linux/amd64 --mount type=bind,src="$(pwd)",dst=/workspace -w /workspace running-readiness-level5:2.15.0 python -m src.neural_network_experiment audit
docker run --rm --platform linux/amd64 --mount type=bind,src="$(pwd)",dst=/workspace -w /workspace running-readiness-level5:2.15.0 python -m unittest -v tests/test_neural_networks.py
```
"""
    (root / DOC_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / DOC_PATH).write_text(text, encoding="utf-8")
    readme = root / "README.md"
    marker = "<!-- LEVEL_5_NETWORKS -->"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# Running Readiness\n"
    line = f"{marker}\n- [Level 5: Networks]({DOC_PATH.as_posix()})\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n\n" + line
    else:
        existing = existing.rstrip() + "\n\n" + line
    readme.write_text(existing, encoding="utf-8")


def main() -> None:
    root = Path.cwd()
    command = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if command == "audit":
        payload = audit_level4(root)
    elif command == "concepts":
        payload = concept_demo(root)
    elif command == "run":
        payload = run_experiment(root)
    elif command == "document":
        document_experiment(root)
        payload = {"documentation": str(DOC_PATH), "status": "written"}
    else:
        raise SystemExit("Use: audit | concepts | run | document")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()