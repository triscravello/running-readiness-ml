# Secure Running-Readiness Predictor: A Group-Safe, Leakage-Resistant ML System

An end-to-end, production-grade machine learning system designed to predict an athlete's next-day running-readiness. This repository tracks the evolution of the prediction pipeline from raw data contracts (Level 1) to a secure, containerized CPU inference service protected by multi-stage Docker builds, strict input-validation schemas, and cryptographic fail-closed lineage verification (Level 7).

---

## System Architecture

```text
[ Raw Data Source ]
        |
        v
[ Level 1: Time-Aware Boundary ] ---> At 8:00 PM UTC (20:00) local time
        |
        v
[ Level 2: Group-Safe Splitting ] ---> Isolated by Runner ID (GroupShuffleSplit)
        |
        v
[ Level 3: Evaluated & Locked ] ---> GroupKFold CV (Decision Tree vs. Majority Baseline)
        |
        v
[ Level 4: Feature Gate Transformer ] ---> Targets and unapproved features stripped
        |
        v
[ Level 5 & 6: Neural Network ] ---> Self-Supervised Pretraining & Denoising Transfer Learning
        |
        v
[ Level 7: Secure CPU Serving Edge ] ---> FastAPI & Pydantic V2 input gating (extra="forbid")
        ^
        | [Lineage Verification Gate]
        +--- SHA-256 validation of 10 lineage files (packaging) and 3 serving artifacts (startup)
```

---

## Detailed Development Lifecycle

### Level 1: Time-Aware Data Contracts & Leakage Auditing
* **The Prediction Boundary:** Predictions are made at exactly 8:00 PM UTC (20:00) local time on observation day D for the running-readiness label on day D+1 (observable at 8:00 AM UTC on D+1).
* **Feature Schema:** Inputs are limited strictly to information available before 8:00 PM UTC (accumulated distance, sleep duration, morning resting heart rate, evening soreness, and prior injury history). Runner IDs, dates, and target labels are treated as context/supervision metadata and stripped from the model inputs.
* **The Imputation Leakage Experiment:** Proved that global preprocessing (imputing missing features before splitting) causes representation leakage. Refactored the data pipeline to fit preprocessors on the training fold only, freezing and serializing the learned states into `artifacts/v1/preprocessing.json`.

### Level 2 & 3: Group-Safe Splitting, Evaluation & Model Selection Lock
* **Group Isolation:** Repeated observations from the same runners represent highly correlated states. To prevent data contamination, we split the data by Runner ID using a group-aware division (`GroupShuffleSplit`).
* **Evaluation Benchmarks:** Evaluated on November 1, 2026, at 9:37 PM UTC. The final validation-selected decision tree model was locked and logged to `artifacts/v1/level3_selection_lock.json` after outperforming the majority baseline.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **`decision_tree`** | **0.833** | **0.913** | **0.840** | **0.875** |
| `majority_baseline` | 0.694 | 0.694 | 1.000 | 0.820 |

* **Selected Confusion Matrix** `[[TN, FP], [FN, TP]]`: `[[9, 2], [4, 21]]`

### Level 4: Feature Gate Transformers
* **Target Proxy Auditing:** Constructed custom schema-gated transformers to exclude target-derived proxies and unapproved, prediction-time-ineligible fields.
* **Cryptographic Handoffs:** Documented and frozen state contracts using SHA-256 fingerprints to ensure downstream components ingest verified features.

### Level 5 & 6: Multi-Layer Neural Networks & Denoising Transfer Learning
* **Self-Supervised Pretraining:** Trained a robust source encoder using an auxiliary denoising feature reconstruction task to capture low-dimensional athletic patterns.
* **Transfer Learning:** Transferred the pretrained encoder weights to a downstream classifier, applying strict learning rate regularization to prevent catastrophic forgetting of raw athletic representation bounds.

### Level 7: Secure CPU Inference Container & Fail-Closed API Boundary
* **Lineage Verification (First Gate):** A pre-packaging gate (`python -m src.level7_lineage`, run automatically by `src.level7_package`) validates the SHA-256 hashes of **ten** inherited lineage files (development data, contracts, handoffs, reports, checkpoints, and the consumed-holdout marker) against hashes recorded in the tracked contracts and handoffs, and checks the recorded holdout restrictions. It **never opens, parses, or hashes `test.csv`**: the holdout's non-use rests on the hash-pinned marker and the handoff flags (`consumed_v1_holdout_read: false`), which are recorded assertions, not independent proof. Three of the ten files are gitignored (`data/v1/processed/train.csv`, `reports/v1/level3_evaluation.json`, `artifacts/v1/holdout_consumed.json`), so this gate can only pass where those local files exist.
* **Inference Reconstruction:** Reconstructs the Level 6 classifier architecture in PyTorch on CPU. At startup the runtime recomputes the SHA-256 of the three artifacts listed in `serving-manifest.json` (checkpoint, preprocessor, deploy handoff) and refuses to start on any mismatch. The manifest itself is not independently authenticated; see [docs/level-7-serve.md](docs/level-7-serve.md).
* **Schema-Gated API Boundary:** Replaced a permissive, vulnerable dictionary-based API route with a strict FastAPI service backed by Pydantic V2 schemas. The validation layer enforces:
  * `extra="forbid"`: Rejecting prediction-time-ineligible keys (like `needs_recovery_next_day`) with an instant `422 Unprocessable Entity` response.
  * `strict=True` and range validation (e.g., resting heart rate restricted to `[30.0, 220.0]`).
  * `allow_inf_nan=False`: Blocking mathematically unstable values.
* **Multi-Stage Containerization:** Docker images target `linux/amd64`. The Level 7 build stages an explicit allowlist context (`src/level7_package.py`), runs the service tests and the bundle-isolation tests in a test stage (the bundle tests cannot be skipped there), and produces a runtime image that holds only the three serving artifacts, the manifest, the lineage attestation, and the two service modules. Inspection of the existing image found no training, test, or holdout files. The model is a synthetic educational experiment with development-only evidence, not a validated coaching system.

---

## Security & Verification Guarantees

### 1. The Zero-Call Supply-Chain Proof
To prove that our deployment pipeline fails closed under tampering, we run automated tests that inject a single-byte corruption (using an XOR mask `^= 0x01` at the file midpoint) into the serialized model checkpoint `level6_selected_model.pt`. 
* **The Result:** The lineage validator instantly intercepts the hash mismatch, raises a `VerificationError`, and terminates the build process.
* **The Proof:** Our mock runner asserts that zero external Docker or build shell commands are dispatched, ensuring a corrupted artifact can never be staged or packaged. This test passes natively and in the canonical Docker environment.
* **Scope:** It covers the pre-packaging lineage gate only. The runtime startup check is separate, and the serving manifest is not independently authenticated (an artifact and a manifest edited together are accepted). See [docs/level-7-serve.md](docs/level-7-serve.md).

### 2. Holdout Handling
* The `test.csv` holdout still exists locally under the gitignored `data/v1/processed/` directory. It is excluded from the Level 7 build context by an explicit allowlist (there is no `.dockerignore`), and the existing runtime image was inspected and contains no `train.csv`, `test.csv`, or holdout marker.
* The lineage gate never opens it (see above). `tests/test_frozen_artifacts.py` verifies the holdout's *recorded* metadata through `artifacts/v1/holdout_consumed.json` and excludes the file from `frozen-hashes.json`; it also never opens it.
* Consumed-holdout restrictions: no feature selection, model selection, model comparison, or threshold tuning (the locked threshold is 0.5).

---

## Running Locally

### Which environment to trust

| | Canonical: Docker | Native (Intel Mac) |
| :--- | :--- | :--- |
| Platform | `linux/amd64`, Debian 13, Python 3.15.2 | macOS x86_64, Python 3.12 |
| Key packages | torch 2.15.0+cpu, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 | **Two incompatible partial environments** (below) |
| Scope | All audits and tests were verified here | Partial only |

PyTorch 2.15.0 has no wheel for Intel macOS, and pandas 3.1.0 refuses to import under NumPy 1.26.4, the version paired with the native torch 2.2.2 (torch 2.2.2 with NumPy 2 was not tested). **No verified native environment can run the whole suite**, and `python -m unittest discover tests/` cannot pass natively. Use Docker for full verification: see [docs/environments.md](docs/environments.md) for the exact read-only commands, the verified package versions, and the commands that must *not* be run (several overwrite locked artifacts or read the consumed holdout).

### 1. Prerequisites
Docker Desktop (linux/amd64) for the canonical environment, and [uv](https://docs.astral.sh/uv/) with Python 3.12 for the optional native environments.

### 2. Clone
```bash
git clone https://github.com/triscravello/running-readiness-ml.git
cd running-readiness-ml
```
A fresh clone does **not** contain the gitignored lineage files (`data/`, `models/*.joblib`, `artifacts/v1/holdout_consumed.json`, `reports/v1/level3_evaluation.*`, `reports/v1/level3_errors.csv`). The consumed holdout is never distributed. Without those files the lineage audit cannot pass, and the frozen-file test reports them as unavailable rather than passing them.

### 3. Optional native environments (partial)
Create these outside the repository so they are never mistaken for project files.

**A. Data and scikit-learn tests (no torch).** Verified: 34 tests pass.
```bash
uv venv --python 3.12 ~/.venvs/running-readiness-data
uv pip install --python ~/.venvs/running-readiness-data/bin/python -r requirements.txt
~/.venvs/running-readiness-data/bin/python -m unittest \
  tests.test_data_pipeline tests.test_evaluation tests.test_features \
  tests.test_training tests.test_leakage_proxy tests.test_schema_drift_gate
```

**B. Serving and torch checks (no pandas or scikit-learn).** The install command resolves to torch 2.2.2 and NumPy 1.26.4 (dry-run verified); the existing local `.venv` has this shape.
```bash
uv venv --python 3.12 ~/.venvs/running-readiness-torch
uv pip install --python ~/.venvs/running-readiness-torch/bin/python \
  "fastapi==0.162.0" "pydantic==2.14.2" "httpx==0.28.1" "numpy==1.26.4" torch
```

### 4. Verify lineage and run the model locally
These work natively in environment B on the owner's machine (they need the gitignored lineage files):
```bash
# SHA-256 verification of the ten lineage files (never opens test.csv)
python -m src.level7_lineage

# Single-prediction smoke test against the verified artifacts
python -m src.level7_runtime

# Level 7 service tests, bundle-isolation tests (skipped natively, see below),
# packaging-corruption test, and the frozen-file regression guard
python -m unittest tests.test_level7_service tests.test_level7_bundle \
  tests.test_level7_corruption tests.test_frozen_artifacts
```
The prediction for the built-in sample request (`probability 0.49793776869773865`, decision 0) is identical in the Docker images and in environment B.

**Serving root.** The service reads `serving-manifest.json` and the three serving artifacts from its *serving root*. By default that is the directory containing `level7_service.py`: `/app` in the container (unchanged), and `src/` in the repository, which has no manifest, so a bare `import src.level7_service` still fails closed. Set `SERVING_ROOT=<directory>` to use another root; an empty value counts as unset. The same SHA-256 checks run against whichever root is chosen, and a missing or modified file stops startup. The service tests build a disposable root in a temporary directory from the verified artifacts and the existing manifest generator, so they run from the repository root without touching any tracked file. `tests/test_level7_bundle.py` checks the container bundle and is skipped (visibly) outside it.

**Still container-first.** `python -m src.level7_service` does not start a server (the module only defines the FastAPI `app`), and `uvicorn` is not installed in environment B. Run the service from its container.

---

## Building and Running the Secure Docker Container

There is **no root `Dockerfile`**. The images form a local chain, and the image names and tags are:

| Image | Built from | Purpose |
| :--- | :--- | :--- |
| `running-readiness-level5:2.15.0` | `Dockerfile.level5` | Python 3.15.2, pinned requirements, torch 2.15.0+cpu |
| `running-readiness-level6:2.15.0` | `Dockerfile.level6` (`FROM` level5) | Level 6 audits and tests |
| `running-readiness-level7:4.0.0` | `python -m src.level7_package build` (uses `Dockerfile.level7` inside a staged context) | Serving image |

The existing local images are the **verified reference**. They are the only exact copies: a clean rebuild of Level 5 was observed to be functionally equivalent (same packages for every pinned dependency, same audit and test results, same prediction) but **not byte-identical** (newer Debian base layer; three unpinned transitive packages drifted). Do not delete or retag the originals casually. A disposable clean rebuild of the Level 7 image (from a copy of the repository, under a throwaway tag, no cache) succeeded and matched the original except for the intended `level7_service.py` change and three drifted unpinned transitive packages; the Level 6 rebuild was not re-run. **The existing `running-readiness-level7:4.0.0` image predates the `SERVING_ROOT` change** and still contains the earlier `level7_service.py` (behaviourally identical when `SERVING_ROOT` is unset); it has not been rebuilt or retagged.

### 1. Build (only if you need to; network required)
```bash
# Level 5: use a disposable context so the repository (including data/ and .venv) is never sent to Docker.
# Use a different tag unless you intend to replace the verified image.
CTX="$(mktemp -d)" && cp Dockerfile.level5 requirements-level5.txt "$CTX"/
docker build --platform linux/amd64 -f "$CTX/Dockerfile.level5" -t running-readiness-level5:rebuild-check "$CTX"
```
`Dockerfile.level6` has a hard-coded `FROM running-readiness-level5:2.15.0`. The Level 7 packager is **not a read-only command**: it runs the lineage audit (needs the local lineage files), runs the Level 6 audit through a read-write bind mount of the repository, deletes and recreates the gitignored `build/level7-context/` (regenerating its serving manifest and lineage attestation), builds, and tags `running-readiness-level7:4.0.0`, replacing the existing tag. See [docs/environments.md](docs/environments.md) before running it.

### 2. Launch the Containerized API
```bash
docker run --rm --read-only --tmpfs /tmp -p 127.0.0.1:8000:8000 running-readiness-level7:4.0.0
```
Startup fails closed (non-zero exit) if a manifest-listed artifact does not match its recorded SHA-256. In the Docker build, the test stage runs `tests/test_level7_service.py` and `tests/test_level7_bundle.py` with `LEVEL7_REQUIRE_BUNDLE_TESTS=1`, under which the bundle tests fail instead of skipping. See [docs/environments.md](docs/environments.md) for the exact verification commands.

### 3. Curl Test (Enforcing the Schema Gate)

**Valid Input (200 OK):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weekly_km_to_date": 45.2,
    "sleep_hours_last_night": 7.5,
    "resting_hr_morning": 58.0,
    "soreness_score_evening": 2.0,
    "prior_injury": 0
  }'
```
`prior_injury` must be the integer `0` or `1`. A float such as `0.0` is rejected with a 422 because the schema is strict.

**Invalid Input (422 Unprocessable Entity - Blocked by `extra="forbid"`):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weekly_km_to_date": 45.2,
    "sleep_hours_last_night": 7.5,
    "resting_hr_morning": 58.0,
    "soreness_score_evening": 2.0,
    "prior_injury": 0,
    "needs_recovery_next_day": 1.0
  }'
```

