from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET_VERSION = "v1"
SCHEMA_VERSION = "1.0.0"
RANDOM_STATE = 42
GROUP_COLUMN = "runner_id"
TARGET_COLUMN = "needs_recovery_next_day"
KEY_COLUMNS = ["runner_id", "observation_date"]
FEATURE_COLUMNS = [
    "weekly_km_to_date",
    "sleep_hours_last_night",
    "resting_hr_morning",
    "soreness_score_evening",
    "prior_injury",
]
ALL_COLUMNS = [
    "runner_id",
    "observation_date",
    "prediction_timestamp",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    "target_available_timestamp",
]
VALID_RANGES = {
    "weekly_km_to_date": (0.0, 200.0),
    "sleep_hours_last_night": (0.0, 14.0),
    "resting_hr_morning": (30.0, 220.0),
    "soreness_score_evening": (0.0, 10.0),
    "prior_injury": (0.0, 1.0),
}


def build_schema_contract() -> dict[str, object]:
    roles = {
        "runner_id": "group_identifier",
        "observation_date": "context",
        "prediction_timestamp": "context",
        TARGET_COLUMN: "target",
        "target_available_timestamp": "target_metadata",
    }
    data_types = {
        "runner_id": "string",
        "observation_date": "string",
        "prediction_timestamp": "string",
        "weekly_km_to_date": "float",
        "sleep_hours_last_night": "float",
        "resting_hr_morning": "float",
        "soreness_score_evening": "float",
        "prior_injury": "integer",
        TARGET_COLUMN: "integer",
        "target_available_timestamp": "string",
    }

    columns = []
    for name in ALL_COLUMNS:
        role = roles.get(name, "feature")
        columns.append(
            {
                "name": name,
                "role": role,
                "data_type": data_types[name],
                "available_at_prediction": role
                in {"group_identifier", "context", "feature"},
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "prediction_definition": {
            "entity": "runner-day",
            "prediction_timestamp": "20:00 local time on observation day D",
            "target": TARGET_COLUMN,
            "prediction_horizon": "next calendar day D+1",
            "target_available_timestamp": (
                "08:00 local time on next calendar day D+1"
            ),
        },
        "primary_key": KEY_COLUMNS,
        "group_column": GROUP_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "required_columns": ALL_COLUMNS,
        "columns": columns,
    }


def build_raw_dataset() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for runner_number in range(1, 25):
        baseline_hr = 45 + (runner_number % 10)
        prior_injury = int(runner_number % 7 == 0)

        for day in range(1, 7):
            weekly_km = 18.0 + (runner_number % 6) * 3.0 + day * 2.0
            sleep_hours = 5.5 + ((runner_number + day) % 6) * 0.5
            resting_hr = baseline_hr + (day % 3) * 2
            soreness = (runner_number * 2 + day) % 11
            needs_recovery = int(
                sleep_hours < 6.5
                or soreness >= 8
                or resting_hr >= baseline_hr + 4
                or weekly_km >= 45.0
                or (prior_injury == 1 and soreness >= 6)
            )

            records.append(
                {
                    "runner_id": f"runner-{runner_number:02d}",
                    "observation_date": f"2026-01-{day:02d}",
                    "prediction_timestamp": (
                        f"2026-01-{day:02d}T20:00:00"
                    ),
                    "weekly_km_to_date": weekly_km,
                    "sleep_hours_last_night": sleep_hours,
                    "resting_hr_morning": resting_hr,
                    "soreness_score_evening": soreness,
                    "prior_injury": prior_injury,
                    TARGET_COLUMN: needs_recovery,
                    "target_available_timestamp": (
                        f"2026-01-{day + 1:02d}T08:00:00"
                    ),
                }
            )

    records[7]["sleep_hours_last_night"] = None
    records[61]["resting_hr_morning"] = None
    records[103]["soreness_score_evening"] = None
    records[15]["weekly_km_to_date"] = -3.0
    records[95]["sleep_hours_last_night"] = 16.0
    records.append(records[20].copy())

    return pd.DataFrame(records, columns=ALL_COLUMNS)


def validate_schema(
    frame: pd.DataFrame,
    expected_columns: list[str] | None = None,
) -> dict[str, object]:
    required = expected_columns or ALL_COLUMNS
    expected = set(required)
    observed = set(frame.columns)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    return {
        "valid": not missing and not unexpected,
        "missing": missing,
        "unexpected": unexpected,
    }


def quality_report(frame: pd.DataFrame) -> dict[str, int | float]:
    missing_features = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    duplicate_records = int(frame.duplicated(subset=KEY_COLUMNS).sum())
    invalid_values = 0

    for column, (minimum, maximum) in VALID_RANGES.items():
        invalid_values += int(
            ((frame[column] < minimum) | (frame[column] > maximum)).sum()
        )

    checked_values = frame.shape[0] * len(FEATURE_COLUMNS)
    temporal_violations = int(
        (
            frame["prediction_timestamp"]
            >= frame["target_available_timestamp"]
        ).sum()
    )

    return {
        "row_count": int(frame.shape[0]),
        "runner_count": len(set(frame[GROUP_COLUMN])),
        "missing_feature_values": missing_features,
        "feature_completeness_ratio": round(
            1 - missing_features / checked_values, 4
        ),
        "duplicate_runner_days": duplicate_records,
        "invalid_values": invalid_values,
        "temporal_order_violations": temporal_violations,
        "target_positive_rate": round(float(frame[TARGET_COLUMN].mean()), 4),
    }


def clean_structural_issues(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.drop_duplicates(subset=KEY_COLUMNS, keep="first").copy()

    for column, (minimum, maximum) in VALID_RANGES.items():
        out_of_range = (cleaned[column] < minimum) | (
            cleaned[column] > maximum
        )
        cleaned.loc[out_of_range, column] = np.nan

    return cleaned


def split_by_runner(
    cleaned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(
        splitter.split(
            cleaned,
            cleaned[TARGET_COLUMN],
            groups=cleaned[GROUP_COLUMN],
        )
    )
    return (
        cleaned.iloc[train_indices].copy(),
        cleaned.iloc[test_indices].copy(),
    )


def fit_preprocessing(frame: pd.DataFrame) -> pd.Series:
    return frame[FEATURE_COLUMNS].mean()


def apply_preprocessing(
    frame: pd.DataFrame, fill_values: pd.Series
) -> pd.DataFrame:
    ready = frame.copy()
    ready[FEATURE_COLUMNS] = ready[FEATURE_COLUMNS].fillna(fill_values)
    return ready


def as_float_dict(values: pd.Series) -> dict[str, float]:
    return {name: float(value) for name, value in values.items()}


def demonstrate_preprocessing_leakage(
    cleaned: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> dict[str, object]:
    leaky_global_values = fit_preprocessing(cleaned)
    train_only_values = fit_preprocessing(train_raw)

    perturbed_test = test_raw.copy()
    perturbed_test.loc[:, "sleep_hours_last_night"] = 14.0
    perturbed_all = pd.concat(
        [train_raw, perturbed_test], ignore_index=True
    )
    perturbed_global_values = fit_preprocessing(perturbed_all)
    train_values_after_test_change = fit_preprocessing(train_raw)

    global_before = float(
        leaky_global_values["sleep_hours_last_night"]
    )
    global_after = float(
        perturbed_global_values["sleep_hours_last_night"]
    )
    train_before = float(train_only_values["sleep_hours_last_night"])
    train_after = float(
        train_values_after_test_change["sleep_hours_last_night"]
    )

    return {
        "wrong_order": "fit preprocessing on all rows, then split",
        "correct_order": "split runners, fit on train, apply to both",
        "leaky_global_fill_values": as_float_dict(leaky_global_values),
        "train_only_fill_values": as_float_dict(train_only_values),
        "global_sleep_mean_before": round(global_before, 4),
        "global_sleep_mean_after_test_perturbation": round(
            global_after, 4
        ),
        "train_sleep_mean_before": round(train_before, 4),
        "train_sleep_mean_after_test_perturbation": round(
            train_after, 4
        ),
        "global_fit_changed_after_test_perturbation": bool(
            abs(global_before - global_after) > 1e-9
        ),
        "train_fit_changed_after_test_perturbation": bool(
            abs(train_before - train_after) > 1e-9
        ),
    }


def build_outputs() -> dict[str, object]:
    contract = build_schema_contract()
    raw = build_raw_dataset()
    cleaned = clean_structural_issues(raw)
    train_raw, test_raw = split_by_runner(cleaned)
    leakage_demo = demonstrate_preprocessing_leakage(
        cleaned, train_raw, test_raw
    )

    train_fill_values = fit_preprocessing(train_raw)
    train_ready = apply_preprocessing(train_raw, train_fill_values)
    test_ready = apply_preprocessing(test_raw, train_fill_values)

    train_runners = set(train_ready[GROUP_COLUMN])
    test_runners = set(test_ready[GROUP_COLUMN])
    runner_overlap = sorted(train_runners.intersection(test_runners))

    preprocessing_artifact = {
        "artifact_version": "1.0.0",
        "dataset_version": DATASET_VERSION,
        "strategy": "mean imputation",
        "fit_scope": "training runners only",
        "feature_columns": FEATURE_COLUMNS,
        "fill_values": as_float_dict(train_fill_values),
    }

    report = {
        "dataset_version": DATASET_VERSION,
        "schema_validation": validate_schema(
            raw, contract["required_columns"]
        ),
        "prediction_definition": contract["prediction_definition"],
        "raw_quality": quality_report(raw),
        "cleaned_quality": quality_report(cleaned),
        "train_quality": quality_report(train_ready),
        "test_quality": quality_report(test_ready),
        "split": {
            "method": "GroupShuffleSplit",
            "group_column": GROUP_COLUMN,
            "random_state": RANDOM_STATE,
            "test_group_fraction": 0.25,
            "train_runner_count": len(train_runners),
            "test_runner_count": len(test_runners),
            "runner_overlap": runner_overlap,
        },
        "leakage_demo": leakage_demo,
    }

    return {
        "raw": raw,
        "cleaned": cleaned,
        "train": train_ready,
        "test": test_ready,
        "preprocessing": preprocessing_artifact,
        "report": report,
        "contract": contract,
    }


def run_pipeline(project_root: Path) -> dict[str, object]:
    outputs = build_outputs()
    paths = {
        "raw": project_root
        / "data"
        / DATASET_VERSION
        / "raw"
        / "run_readiness_observations.csv",
        "cleaned": project_root
        / "data"
        / DATASET_VERSION
        / "cleaned"
        / "run_readiness_observations.csv",
        "train": project_root
        / "data"
        / DATASET_VERSION
        / "processed"
        / "train.csv",
        "test": project_root
        / "data"
        / DATASET_VERSION
        / "processed"
        / "test.csv",
        "preprocessing": project_root
        / "artifacts"
        / DATASET_VERSION
        / "preprocessing.json",
        "report": project_root
        / "reports"
        / DATASET_VERSION
        / "data_quality_report.json",
        "contract": project_root
        / "contracts"
        / DATASET_VERSION
        / "schema_contract.json",
    }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    for name in ("raw", "cleaned", "train", "test"):
        outputs[name].to_csv(paths[name], index=False)

    for name in ("preprocessing", "report", "contract"):
        paths[name].write_text(
            json.dumps(outputs[name], indent=2), encoding="utf-8"
        )

    return outputs


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs = run_pipeline(project_root)
    print(json.dumps(outputs["report"], indent=2))


if __name__ == "__main__":
    main()

