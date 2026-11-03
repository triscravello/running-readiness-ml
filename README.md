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
        +--- Cryptographic SHA-256 validation of all artifacts & configurations
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
* **Lineage Verification (First Gate):** A pre-packaging execution script validates the SHA-256 hashes of all eight inherited files (configs, models, contracts) against expected hashes, asserting that the v1 holdout dataset (`test.csv`) has not been modified or read.
* **Inference Reconstruction:** Reconstructs the Level 6 classifier architecture in PyTorch, executing on local device CPU with a fail-closed check that terminates startup if the local model manifest is compromised.
* **Schema-Gated API Boundary:** Replaced a permissive, vulnerable dictionary-based API route with a strict FastAPI service backed by Pydantic V2 schemas. The validation layer enforces:
  * `extra="forbid"`: Rejecting prediction-time-ineligible keys (like `needs_recovery_next_day`) with an instant `422 Unprocessable Entity` response.
  * `strict=True` and range validation (e.g., resting heart rate restricted to `[30.0, 220.0]`).
  * `allow_inf_nan=False`: Blocking mathematically unstable values.
* **Multi-Stage Containerization:** Docker builds are locked to native `linux/amd64` compilation. A clean, multi-stage layout separates build dependencies, runs verification test suites during image building, and produces a minimal, production-ready runtime container completely devoid of development data.

---

## Security & Verification Guarantees

### 1. The Zero-Call Supply-Chain Proof
To prove that our deployment pipeline fails closed under tampering, we run automated tests that inject a single-byte corruption (using an XOR mask `^= 0x01` at the file midpoint) into the serialized model checkpoint `level6_selected_model.pt`. 
* **The Result:** The lineage validator instantly intercepts the hash mismatch, raises a `VerificationError`, and terminates the build process.
* **The Proof:** Our mock runner asserts that zero external Docker or build shell commands are dispatched, ensuring a corrupted artifact can never be staged or packaged.

### 2. Isolated Holdout Integrity
* The `test.csv` holdout file is completely isolated outside of the build environment. The serving image ignore policies and verification gates ensure that raw evaluation labels never leak into the production container.

---

## Running Locally

### 1. Prerequisites (macOS / Linux)
Ensure you are using Python 3.12 (highly recommended for PyTorch pre-compiled wheel compatibility on Intel Mac hosts) and have Docker Desktop installed.

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone 
cd level-1-data

# Recreate and activate a clean virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install exact production, preprocessing, and test dependencies
uv pip install fastapi httpx torch numpy==1.26.4 pandas scikit-learn
```

### 3. Verify Local Lineage and Start the Service
```bash
# Verify the cryptographic hashes of the local pipeline artifacts
python3 -m src.level7_lineage

# Launch the schema-gated FastAPI production service
python3 -m src.level7_service
```

### 4. Run the Full Validation and Parity Suite
```bash
python3 -m unittest discover -v tests/
```

---

## Building and Running the Secure Docker Container

### 1. Build the Gated Production Image
```bash
# Build the container (executes tests internally during multi-stage compilation)
docker build --platform linux/amd64 -t running-readiness-serving:latest .
```

### 2. Launch the Containerized API
```bash
docker run --rm -p 8000:8000 running-readiness-serving:latest
```

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
    "prior_injury": 0.0
  }'
```

**Invalid Input (422 Unprocessable Entity - Blocked by `extra="forbid"`):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weekly_km_to_date": 45.2,
    "sleep_hours_last_night": 7.5,
    "resting_hr_morning": 58.0,
    "soreness_score_evening": 2.0,
    "prior_injury": 0.0,
    "needs_recovery_next_day": 1.0
  }'
```

