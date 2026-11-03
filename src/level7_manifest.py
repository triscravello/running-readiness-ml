from __future__ import annotations

import json
from pathlib import Path

from src.level7_lineage import PROJECT_ROOT, audit_lineage, read_json, sha256

HANDOFF = Path("artifacts/v4/level7_deploy_handoff.json")
MANIFEST = Path("serving-manifest.json")
# Serving artifact -> fingerprint label in the Level 6 handoff.
FROM_HANDOFF = {
    Path("artifacts/v4/level6_selected_model.pt"): "selected_checkpoint",
    Path("artifacts/v4/level6_preprocessor.json"): "preprocessor",
}


def build_manifest(root: Path = PROJECT_ROOT) -> dict:
    """Write serving-manifest.json only after the lineage audit passes.

    Checkpoint and preprocessor hashes come from the Level 6 handoff (already
    verified against disk by the audit), never from re-hashing unverified files.
    The handoff itself is not self-fingerprinted, so its hash is pinned here.
    """
    target = root / MANIFEST
    if target.exists():
        raise FileExistsError(f"{target} exists; refusing to overwrite")
    audit_lineage(root)
    fingerprints = read_json(root / HANDOFF)["fingerprints"]
    artifacts = {p.as_posix(): fingerprints[k] for p, k in FROM_HANDOFF.items()}
    artifacts[HANDOFF.as_posix()] = sha256(root / HANDOFF)
    manifest = {"artifacts": dict(sorted(artifacts.items()))}
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    print(json.dumps(build_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
