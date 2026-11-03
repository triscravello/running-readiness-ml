from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Mapping

from src.level7_lineage import PROJECT_ROOT, audit_lineage, read_json, sha256

CONTEXT_PATH = Path("build/level7-context")
IMAGE = "running-readiness-level7:4.0.0"
LEVEL6_IMAGE = "running-readiness-level6:2.15.0"
HANDOFF_RELATIVE = Path("artifacts/v4/level7_deploy_handoff.json")
CHECKPOINT_RELATIVE = Path("artifacts/v4/level6_selected_model.pt")
PREPROCESSOR_RELATIVE = Path("artifacts/v4/level6_preprocessor.json")
ALLOWLIST = {
    Path("Dockerfile.level7"): Path("Dockerfile"),
    Path("requirements-level7.txt"): Path("requirements-level7.txt"),
    Path("requirements-level7-test.txt"): Path("requirements-level7-test.txt"),
    Path("src/level7_runtime.py"): Path("level7_runtime.py"),
    Path("src/level7_service.py"): Path("level7_service.py"),
    Path("tests/test_level7_service.py"): Path("test_level7_service.py"),
    CHECKPOINT_RELATIVE: CHECKPOINT_RELATIVE,
    PREPROCESSOR_RELATIVE: PREPROCESSOR_RELATIVE,
    HANDOFF_RELATIVE: HANDOFF_RELATIVE,
}

def run_level6_audit(root: Path, runner=subprocess.run) -> None:
    runner(
        [
            "docker", "run", "--rm", "--platform", "linux/amd64", "--mount",
            f"type=bind,src={root},dst=/workspace", "-w", "/workspace",
            LEVEL6_IMAGE, "python", "-m", "src.finetune_experiment", "audit",
        ],
        cwd=root,
        check=True,
    )


def create_serving_manifest(root: Path) -> dict:
    handoff = read_json(root / HANDOFF_RELATIVE)
    level5 = read_json(root / "artifacts/v2/level5_features_handoff.json")
    return {
        "artifacts": {
            CHECKPOINT_RELATIVE.as_posix(): handoff["fingerprints"]["selected_checkpoint"],
            PREPROCESSOR_RELATIVE.as_posix(): handoff["fingerprints"]["preprocessor"],
            HANDOFF_RELATIVE.as_posix(): sha256(root / HANDOFF_RELATIVE),
        },
        "evidence_scope": handoff["evidence_scope"],
        "model_notice": level5["scientific_status"],
        "inherited_v1_preprocessing_limitation": handoff["limitations"][0],
    }


def stage_context(
    root: Path = PROJECT_ROOT,
    overrides: Mapping[Path, Path] | None = None,
    runner=subprocess.run,
) -> Path:
    attestation = audit_lineage(root, overrides)
    run_level6_audit(root, runner)
    context = root / CONTEXT_PATH
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)

    for source_relative, target_relative in ALLOWLIST.items():
        source = root / source_relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing allowlisted package file: {source}")
        target = context / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    (context / "lineage-attestation.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (context / "serving-manifest.json").write_text(
        json.dumps(create_serving_manifest(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return context


def build_image(root: Path = PROJECT_ROOT, runner=subprocess.run) -> None:
    context = stage_context(root, runner=runner)
    runner(
        ["docker", "build", "--platform", "linux/amd64", "-t", IMAGE, str(context)],
        cwd=root,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "stage", "build"))
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit_lineage(), indent=2, sort_keys=True))
    elif args.command == "stage":
        print(stage_context())
    else:
        build_image()
        print(IMAGE)


if __name__ == "__main__":
    main()