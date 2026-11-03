from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_USES = frozenset(
    {"feature_selection", "model_comparison", "model_selection", "threshold_tuning"}
)


class VerificationError(RuntimeError):
    pass


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_file(path: Path, expected: str) -> str:
    if not path.is_file():
        raise VerificationError(f"Missing required artifact: {path}")
    actual = sha256(path)
    if actual != expected:
        raise VerificationError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def resolved_path(
    root: Path, relative: Path, overrides: Mapping[Path, Path] | None
) -> Path:
    return overrides[relative] if overrides and relative in overrides else root / relative


def add_check(checks: dict[Path, str], relative: Path, expected: str) -> None:
    if relative in checks and checks[relative] != expected:
        raise VerificationError(f"Conflicting expected hashes for {relative}")
    checks[relative] = expected


def audit_lineage(
    root: Path = PROJECT_ROOT,
    overrides: Mapping[Path, Path] | None = None,
) -> dict:
    level3 = read_json(root / "artifacts/v1/level3_evaluation_contract.json")
    level4 = read_json(root / "artifacts/v1/level4_features_contract.json")
    level6_input = read_json(root / "artifacts/v3/level6_finetune_handoff.json")
    level7 = read_json(root / "artifacts/v4/level7_deploy_handoff.json")
    checks: dict[Path, str] = {}
    for record in level3["input_fingerprints"].values():
        add_check(checks, Path(record["path"]), record["sha256"])
    add_check(
        checks,
        Path(level4["holdout_policy"]["marker_path"]),
        level4["holdout_policy"]["marker_sha256"],
    )
    add_check(
        checks,
        Path(level4["evaluation_evidence"]["json_report"]["path"]),
        level4["evaluation_evidence"]["json_report"]["sha256"],
    )

    level7_paths = {
        "development_data": Path("data/v1/processed/train.csv"),
        "level6_input_handoff": Path("artifacts/v3/level6_finetune_handoff.json"),
        "level6_report": Path("reports/v4/finetune_experiment.json"),
        "source_checkpoint": Path("artifacts/v4/level6_source_pretrained.pt"),
        "selected_checkpoint": Path("artifacts/v4/level6_selected_model.pt"),
        "preprocessor": Path("artifacts/v4/level6_preprocessor.json"),
    }
    for label, relative in level7_paths.items():
        add_check(checks, relative, level7["fingerprints"][label])
    verified = {}
    for relative, expected in sorted(checks.items(), key=lambda item: str(item[0])):
        verified[relative.as_posix()] = verify_file(
            resolved_path(root, relative, overrides), expected
        )

    marker = read_json(
        resolved_path(root, Path(level4["holdout_policy"]["marker_path"]), overrides)
    )
    level6_report = read_json(root / "reports/v4/finetune_experiment.json")
    if level4["holdout_policy"]["holdout_consumed"] is not True:
        raise VerificationError("Level 4 no longer marks the holdout consumed")
    if marker.get("holdout_consumed") is not True:
        raise VerificationError("Local holdout marker does not mark it consumed")
    if set(marker.get("forbidden_future_uses", [])) != FORBIDDEN_USES:
        raise VerificationError("Local holdout restrictions changed")
    if set(level4["holdout_policy"]["forbidden_future_uses"]) != FORBIDDEN_USES:
        raise VerificationError("Level 4 holdout restrictions changed")
    if set(level6_input["forbidden_uses"]) != FORBIDDEN_USES:
        raise VerificationError("Level 6 input restrictions changed")
    if set(level7["forbidden_uses"]) != FORBIDDEN_USES:
        raise VerificationError("Level 7 handoff restrictions changed")
    if level7.get("consumed_v1_holdout_read") is not False:
        raise VerificationError("Level 7 handoff records a holdout read")
    if level6_report.get("consumed_v1_holdout_read") is not False:
        raise VerificationError("Level 6 report records a holdout read")
    if level7.get("next_level") != "Level 7: Deploy":
        raise VerificationError("Unexpected Level 7 handoff destination")
    if level7.get("threshold") != 0.5:
        raise VerificationError("The locked Level 6 threshold changed")

    return {
        "status": "passed",
        "verified": verified,
        "holdout_consumed": True,
        "consumed_v1_holdout_read": False,
        "forbidden_uses": sorted(FORBIDDEN_USES),
    }


def main() -> None:
    print(json.dumps(audit_lineage(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()