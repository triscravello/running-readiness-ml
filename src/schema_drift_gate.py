from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_pipeline import DATASET_VERSION, validate_schema


def run_schema_drift_demo(project_root: Path) -> dict[str, object]:
    contract_path = (
        project_root
        / "contracts"
        / DATASET_VERSION
        / "schema_contract.json"
    )
    valid_batch_path = (
        project_root
        / "data"
        / DATASET_VERSION
        / "processed"
        / "test.csv"
    )
    evidence_path = (
        project_root
        / "evidence"
        / DATASET_VERSION
        / "schema_drift_evidence.json"
    )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    valid_batch = pd.read_csv(valid_batch_path)
    expected_columns = contract["required_columns"]
    valid_result = validate_schema(valid_batch, expected_columns)

    malformed_columns = [
        column
        for column in valid_batch.columns
        if column != "sleep_hours_last_night"
    ]
    malformed_batch = valid_batch[malformed_columns].copy()
    malformed_batch["next_day_resting_hr"] = 65.0
    malformed_result = validate_schema(
        malformed_batch, expected_columns
    )

    evidence = {
        "valid_batch": {
            "accepted": bool(valid_result["valid"]),
            "validation": valid_result,
        },
        "malformed_batch": {
            "rejected": bool(not malformed_result["valid"]),
            "validation": malformed_result,
        },
    }

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    return evidence


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    evidence = run_schema_drift_demo(project_root)
    print(json.dumps(evidence, indent=2))

    if not (
        evidence["valid_batch"]["accepted"]
        and evidence["malformed_batch"]["rejected"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()