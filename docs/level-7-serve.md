# Level 7: Serving runtime, integrity checks, and limits

This page describes the Level 7 CPU inference service as it exists in the repository and in the verified image `running-readiness-level7:4.0.0`. It separates what is **verified** (executed during the F0A baseline on 2026-11-23) from what is **by code reading** and what is a **recorded assertion**. Environment and command details are in [environments.md](environments.md).

## 1. Model status and limitations (read first)

* **Synthetic educational model.** The service's own notice reads: *"Synthetic educational feature experiment, not a validated coaching system."* Predictions must not be used for real training or health decisions.
* **Development-only evidence.** `evidence_scope` is `development_only`. All model-quality evidence comes from group-safe validation on development data. The consumed v1 holdout was used once, for the historical Level 3 decision-tree evaluation, and is **not** evidence for this model.
* **Inherited preprocessing limitation.** *"The historical v1 process used development-wide mean imputation before splitting; Level 6 does not claim the full inherited history is leakage-free."*
* **Locked decisions.** Selected approach `partial` (partial fine-tuning of a denoising-pretrained encoder), threshold **0.5**, seed 42, 80 downstream epochs. The holdout is consumed and may not be used for feature selection, model selection, model comparison, or threshold tuning.
* The sample request in this repository yields a probability of about 0.498, close to the 0.5 threshold, so a single prediction is a weak signal.

## 2. Runtime architecture

```text
client ──HTTP──> level7_service.py (FastAPI, Pydantic V2)
                      │  validated payload
                      ▼
                 level7_runtime.ModelRuntime
                      │ 1. verify_serving_artifacts()   SHA-256 of 3 files vs serving-manifest.json
                      │ 2. load handoff + preprocessor  JSON
                      │ 3. torch.load(weights_only=True, map_location=cpu)
                      │ 4. metadata checks              class, feature order, threshold, approach, input_dim
                      │ 5. TransferClassifier.load_state_dict(strict=True); eval()
                      ▼
              preprocess → torch.inference_mode → sigmoid → probability ≥ threshold
```

**Model.** `TransferClassifier` = `Encoder` (Linear 5→16, ReLU, Linear 16→8, ReLU) + `Linear(8→1)` head, reconstructed in `level7_runtime.py` and loaded with `strict=True`. Device is always CPU.

**Preprocessing** (from `artifacts/v4/level6_preprocessor.json`): imputer statistics, then `(x − mean) / scale` per feature, cast to float32. Feature order is `weekly_km_to_date`, `sleep_hours_last_night`, `resting_hr_morning`, `soreness_score_evening`, `prior_injury`.

**Request schema** (`PredictionRequest`; `extra="forbid"`, `strict=True`, `allow_inf_nan=False`):

| Field | Type | Range |
| :--- | :--- | :--- |
| `weekly_km_to_date` | finite float | 0 – 200 |
| `sleep_hours_last_night` | finite float | 0 – 14 |
| `resting_hr_morning` | finite float | 30 – 220 |
| `soreness_score_evening` | finite float | 0 – 10 |
| `prior_injury` | strict integer | 0 or 1 (a float like `0.0` is rejected) |

Unknown keys (for example `needs_recovery_next_day`), missing keys, wrong types, out-of-range values, and NaN/Infinity return **422**. The error handler returns only error type, location, and message, and never echoes client input. Because NaN is rejected at the boundary, the imputation branch in `preprocess` is unreachable through the API (by code reading).

**Endpoints.** `GET /health` returns `status`, `integrity_status`, `device`, `selected_approach`, `evidence_scope`, `model_notice`, and the preprocessing limitation. `POST /predict` returns `probability`, `decision` (`probability >= threshold`), `threshold`, `device`, `selected_approach`.

**Serving root.** The service builds `ModelRuntime` from a *serving root* when `level7_service` is imported:

* `SERVING_ROOT` unset or empty: the directory containing `level7_service.py`. This is the original behaviour. In the image it is `/app`, next to the manifest and artifacts. In the repository it is `src/`, where no manifest exists, so a bare import fails closed with `Missing serving manifest: …/src/serving-manifest.json`.
* `SERVING_ROOT=<path>`: that path, resolved to an absolute path (a relative path resolves against the process's working directory).

The same manifest and artifact checks (§3) run against whichever root is chosen, so a missing directory, a missing artifact, a modified artifact, or a mismatched manifest all stop startup (**verified** by subprocess tests). The root is read once, at import; the Dockerfile does not set it. The tests use the override to serve a disposable root from the repository root (see [environments.md](environments.md) §8). There is still no server entry point: `python -m src.level7_service` starts nothing, so run the service from the container.

## 3. SHA-256 artifact checks and fail-closed behaviour

At construction, `ModelRuntime` reads `serving-manifest.json` and recomputes SHA-256 for every file listed under `artifacts`:

| File | SHA-256 (from the manifest) |
| :--- | :--- |
| `artifacts/v4/level6_preprocessor.json` | `d59626ec299ed8389a43b41ba2d0a70b55d66652aede7792bf99d8e219a9caa9` |
| `artifacts/v4/level6_selected_model.pt` | `ad5d50467e39e809aa49f1a6778f270072e6ca09e8ea3baa22b28cd0c5bb9bfb` |
| `artifacts/v4/level7_deploy_handoff.json` | `e498bf697174c58e2c8fc341622c5f4deafc58fd4c8111b0adac6194be66a952` |

A missing file or a hash mismatch raises `IntegrityError`. Because the runtime is built when `level7_service` is imported, the server never starts. Checkpoint, feature-order, threshold, approach, and input-dimension mismatches against the handoff also raise `IntegrityError`.

**Verified** (F0A, in the container, using modified *copies* mounted read-only):

| Scenario | Result |
| :--- | :--- |
| One bit flipped in the checkpoint, original manifest | Startup fails, exit code 1, `Serving artifact mismatch` |
| Preprocessor changed, original manifest | Startup fails, exit code 1 |
| `tests/test_level7_service.py::test_startup_rejects_corrupted_temporary_checkpoint` | Passes |
| `SERVING_ROOT` pointing to a missing directory (subprocess, F1) | Import fails, `Missing serving manifest: <path>` |
| `SERVING_ROOT` with a missing artifact, a corrupted artifact (each of the three), or a manifest hash that does not match (subprocess, F1) | Import fails closed with `Missing serving artifact` or `Serving artifact mismatch` |
| `ModelRuntime` on a missing manifest, missing artifacts, corrupted artifacts, an altered manifest hash, or a manifest entry for an absent file (F1) | `IntegrityError` with the matching message |
| Rebuilt disposable image, checkpoint replaced by a bit-flipped copy (F1) | Startup fails, exit code 1 |
| Packaging gate: corrupted checkpoint override, fake Docker runner (`tests/test_level7_corruption.py`) | `VerificationError` is raised and zero runner calls are made |

## 4. The serving manifest is not independently authenticated

**State this plainly: the manifest is data, not a trust anchor.** Nothing signs it, and nothing records its own hash anywhere the runtime checks.

* **Verified:** with the preprocessor changed *and* the manifest edited to carry the new hash, the container starts and serves (exit 0, prediction unchanged). The check catches accidental corruption and a change to any listed file made without editing the manifest. It does not stop someone who can change both.
* The deploy handoff's own hash is pinned only inside the manifest, so a consistent edit of the handoff and manifest is also accepted (by code reading).
* **Not covered by any hash at runtime:** `level7_runtime.py`, `level7_service.py`, installed Python packages, and `lineage-attestation.json`.
* **`SERVING_ROOT` does not change this.** It selects which directory is checked; it adds no authentication. Anyone who can set the container's environment and supply a consistent bundle can serve that bundle, just as anyone who can replace the image's files can.
* The tests do pin the manifest *bytes* (`a007e449a7728f632d63ba3709109997db88e0bfe52e35b7f4263d1fc4d2b12a`, the manifest inside the existing image), so a change to what the manifest generator produces fails the test suite. That protects the repository, not a running service.
* **`lineage-attestation.json` is not read by the runtime.** It is copied into the image, and a test only checks that it exists (**verified by code reading**).
* The manifest's `evidence_scope`, `model_notice`, and `inherited_v1_preprocessing_limitation` text is displayed by `/health` and in the API description. These strings are unauthenticated too.
* **Two different manifests exist.**
  * The tracked `serving-manifest.json` at the repository root holds only the three hashes above. It lacks the three descriptive keys the service reads, so the service cannot use it as its root (by code reading).
  * The manifest inside the image is generated by `src.level7_package` and also carries the descriptive keys. The generated copy is in the gitignored `build/level7-context/`.

Because of this, the word "cryptographic" in this project means **hash-checked integrity of listed files**, not authenticated provenance. Anchoring the manifest (a pinned hash or signature held outside the bundle) is a possible later improvement and is not implemented.

## 5. Historical lineage verification vs public serving verification

| | Historical lineage verification | Public serving verification |
| :--- | :--- | :--- |
| Command | `python -m src.level7_lineage` (also the first step of `src.level7_package`) | Automatic at service start (`ModelRuntime`) |
| Files | **Ten** lineage files: development data `train.csv`, `preprocessing.json`, `schema_contract.json`, the Level 6 handoff, report, source and selected checkpoints, preprocessor, `level3_evaluation.json`, and the consumed-holdout marker | **Three** serving artifacts listed in the manifest |
| Needs gitignored local files | **Yes**: `data/v1/processed/train.csv`, `reports/v1/level3_evaluation.json`, `artifacts/v1/holdout_consumed.json` | No. Works in the image and from tracked files |
| Holdout | Checks the marker's recorded restrictions and handoff flags. **Never opens, parses, or hashes `test.csv`** | Not applicable; the image contains no data files (**verified**) |
| Where it can pass | Owner's machine only (verified natively and in Docker). A fresh clone cannot pass; a partial public mode is only proposed | Anywhere the image runs |
| What it proves | The inherited files match the hashes recorded in the tracked contracts and handoffs | The three served files match the bundled manifest |
| What it does not prove | That the holdout was never read. `consumed_v1_holdout_read: false` is a hash-pinned *recorded assertion* | That the manifest itself is genuine (§4) |

A third layer is the frozen-file guard, `tests/test_frozen_artifacts.py`, which compares protected files with `frozen-hashes.json` and verifies the holdout's recorded metadata without opening the holdout. See [environments.md](environments.md) §7.

## 6. What the image contains (verified)

`/app/`: `artifacts/v4/level6_selected_model.pt`, `artifacts/v4/level6_preprocessor.json`, `artifacts/v4/level7_deploy_handoff.json`, `serving-manifest.json`, `lineage-attestation.json`, `level7_runtime.py`, `level7_service.py`, and an empty `level7-tests-passed` marker. A filesystem search found no `train.csv`, `test.csv`, or `holdout_consumed.json`. The runtime image has no `httpx` or test files; the tests ran in a separate build stage. Since F1 that stage runs `tests/test_level7_service.py` and `tests/test_level7_bundle.py` with `LEVEL7_REQUIRE_BUNDLE_TESTS=1`, so the bundle-isolation tests cannot be skipped there, and the build fails if the bundle holds historical data or holdout files (verified with a disposable rebuild and with deliberately broken builds). The existing image was built before this change. The service runs as root with no `HEALTHCHECK`.

## 7. Known limitations (documented, not changed in this milestone)

* The service has no server entry point of its own, and `uvicorn` is not installed in the native environments (§2).
* The existing `running-readiness-level7:4.0.0` image predates the `SERVING_ROOT` change and still contains the earlier `level7_service.py` (`142a76a3…`; the repository now holds `022c275f…`). It behaves identically when `SERVING_ROOT` is unset. It has not been rebuilt or retagged. A disposable `--no-cache` rebuild from the current sources succeeded and differs from the existing image only in that file and three unpinned transitive packages (see [environments.md](environments.md) §3).
* Manifest unauthenticated; attestation unused at runtime (§4).
* The Level 7 image's transitive Python packages and the base image are not pinned by hash or digest; clean rebuilds are not byte-identical (see [environments.md](environments.md) §3).
* Root user, no `HEALTHCHECK`, and a `StarletteDeprecationWarning` that recommends `httpx2` for the test client.
