# Environments, verification commands, and commands to avoid

This page records what was **verified** on 2026-11-23 by the F0A read-only Docker baseline, what was **not** verified, and which commands must never be run when you only want to verify. Statements are labelled:

* **Verified**: executed and observed.
* **By code reading**: derived from the source; not executed.
* **Not verified**: not run.

## 1. Canonical environment (Docker, verified)

Three local images form a chain: `running-readiness-level5:2.15.0` → `running-readiness-level6:2.15.0` → `running-readiness-level7:4.0.0`.

| Item | Observed value (all three images) |
| :--- | :--- |
| Platform | `linux/amd64`, Docker Desktop 29.5.3 on an Intel Mac host |
| OS | Debian GNU/Linux 13 (trixie), Linux 6.12.76-linuxkit |
| Python | 3.15.2 |
| torch | 2.15.0+cpu |
| numpy | 2.5.4 |
| pandas / scikit-learn / scipy / joblib | 3.1.0 / 1.9.1 / 1.18.1 / 1.6.0 |
| Level 7 adds | fastapi 0.162.0, pydantic 2.14.2, starlette 1.10.0, uvicorn 0.54.0 (httpx is *not* in the runtime image) |
| Seed | `PYTHONHASHSEED=42`; code seeds 42 and sets `torch.use_deterministic_algorithms(True)` |
| Not pinned | torch thread count (observed 8, host-dependent), base-image digest |
| Container user | root; no `HEALTHCHECK` |

| Image | ID |
| :--- | :--- |
| level5 | `sha256:87f3e46f2e825def609c8e4033445e367523f65e7566b944daf5dda9c75e6b1d` |
| level6 | `sha256:d19cca78ec6191afcff7e6936614b110466f830ff27a81b656653aee1b24dcdc` |
| level7 | `sha256:2335bca6985513d1db7079f3cf43391da91b193dd475fe5f7303dadde9c2d099` |

The complete package lists are in [`requirements-canonical.freeze`](../requirements-canonical.freeze). That file is an **inventory of what the images contain, not an installation lockfile**.

## 2. Verified behaviour of the existing images

* Level 5 audit, Level 6 audit, and the lineage audit all pass inside the containers (**verified**).
* Every test module passes inside the containers: 44 reported runs, 38 unique tests (`test_leakage_proxy` re-imports the six `test_features` tests) (**verified**).
* The pre-F1 `tests/test_level7_service.py` passed (6 of 6) in the image layout, including prediction parity against an independent reference (**verified** in F0A). The current, portable test files are covered in §8.
* The running service answers `/health` and `/predict`; invalid, extra-key, wrong-type, and NaN inputs return 422 (**verified**).
* The Level 7 image contains only the three serving artifacts, `serving-manifest.json`, `lineage-attestation.json`, `level7_runtime.py`, `level7_service.py`, and an empty `level7-tests-passed` marker. No `train.csv`, `test.csv`, or holdout marker is present (**verified**). The marker file is empty: it shows the test stage ran during the original build, nothing more.
* Startup fails closed with exit code 1 when the checkpoint or preprocessor is changed and the manifest is not (**verified**). A preprocessor and manifest edited *together* are accepted (**verified**; see [level-7-serve.md](level-7-serve.md)).
* The sample prediction `probability 0.49793776869773865`, decision 0, is identical in all three images, in a rebuilt Level 5 image, and in the native environment with torch 2.2.2 / NumPy 1.26.4 (**verified**).

## 3. Clean rebuilds: what is and is not verified

| Question | Answer |
| :--- | :--- |
| Level 5 rebuild from `Dockerfile.level5` + `requirements-level5.txt` | **Verified** to succeed (about 90 s, network required) from a disposable context. |
| Byte-identical to the existing image? | **No.** A different Debian base snapshot, and the unpinned transitive packages `cloudpickle` (3.1.2 → 3.1.3), `narwhals` (2.28.0 → 2.29.0), and `threadpoolctl` (3.7.0 → 3.8.0) changed. |
| Functionally equivalent? | **Verified** for Level 5: the pinned packages are unchanged, the Level 5 audit and `test_neural_networks` pass, and the runtime prediction is identical. |
| Level 6 rebuild | **Not verified.** `Dockerfile.level6` only adds `WORKDIR`/`CMD` on top of the hard-coded `running-readiness-level5:2.15.0` tag. |
| Level 7 clean rebuild | **Verified in F1**, in a disposable copy of the repository (without `test.csv`), using the real packager `stage` step and `docker build --no-cache` under a throwaway tag. It succeeded in about 40 s; the build's test stage ran 36 tests, including the three bundle tests, with `OK (skipped=6)`. The result matched the existing image except for the intended `level7_service.py` change (`142a76a3…` → `022c275f…`): `level7_runtime.py`, the three artifacts, `serving-manifest.json` (`a007e449…`), and `lineage-attestation.json` are byte-identical, and the sample prediction is identical. Unpinned transitives drifted: `annotated-doc` 0.0.5 → 0.0.7, `click` 8.5.0 → 8.5.1, `typing-inspection` 0.4.4 → 0.4.5. Running the packager outside a disposable copy is still unsafe (§5, §6). |
| Do rebuilds change protected artifacts? | **No.** The images read `artifacts/` through a mount or copy three of them into the Level 7 image. Nothing in the Level 5 rebuild touched the repository. |
| Base images and requirement files | Not pinned by digest or hash. Hash-locked dependencies are a later milestone. |

Keep the existing images: they are the only exact copies.

## 4. Safe, read-only verification commands

These are the commands used for the baseline. Run them from the repository root. **bash or zsh.**

```bash
REPO="$PWD"
MASK="$(mktemp)"   # empty file shown to containers in place of the holdout

# Read-only repo mount, no network, read-only root filesystem, holdout masked.
RO=(--rm --network none --platform linux/amd64 --read-only --tmpfs /tmp
    -e PYTHONDONTWRITEBYTECODE=1
    --mount "type=bind,src=$REPO,dst=/workspace,readonly"
    --mount "type=bind,src=$MASK,dst=/workspace/data/v1/processed/test.csv,readonly"
    -w /workspace)
```

Drop the second `--mount` line if `data/v1/processed/test.csv` does not exist (a fresh clone): a file mount needs an existing target. The mask means that if any command tried to read the consumed holdout it would see an empty file and fail visibly.

```bash
# Versions inside an image (no repo mount needed)
docker run --rm --network none --platform linux/amd64 running-readiness-level5:2.15.0 \
  python -c "import sys,torch,numpy;print(sys.version.split()[0],torch.__version__,numpy.__version__)"

# Audits (read-only by code reading and verified by execution)
docker run "${RO[@]}" running-readiness-level5:2.15.0 python -m src.neural_network_experiment audit
docker run "${RO[@]}" running-readiness-level6:2.15.0 python -m src.finetune_experiment audit
docker run "${RO[@]}" running-readiness-level6:2.15.0 python -m src.level7_lineage

# Tests (each module builds its own temporary roots; none needs test.csv)
for t in test_neural_networks test_data_pipeline test_evaluation test_features \
         test_training test_leakage_proxy test_schema_drift_gate \
         test_level7_corruption test_frozen_artifacts; do
  docker run "${RO[@]}" running-readiness-level5:2.15.0 python -m unittest -v tests.$t
done
docker run "${RO[@]}" running-readiness-level6:2.15.0 python -m unittest -v tests.test_finetune
```

Pass `-e FROZEN_REQUIRE_LOCAL=1` to `docker run` to make unavailable gitignored lineage files a failure instead of a skip.

**Level 7 service tests** need `httpx`, which the runtime image lacks, so use a disposable test layer over the *existing* image (network needed once, for `pip`). Build it in a temporary directory, never in the repository. The same layer serves both layouts:

```bash
# 1. Disposable layer: adds only httpx (from requirements-level7-test.txt)
T="$(mktemp -d)"; cp requirements-level7-test.txt "$T"/
cat > "$T/Dockerfile" <<'EOF'
FROM running-readiness-level7:4.0.0
COPY requirements-level7-test.txt /tmp/requirements-level7-test.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-level7-test.txt
EOF
docker build --platform linux/amd64 -t rr-level7-httpx:disposable "$T"

# 2. Repository layout on the canonical stack (uses the RO array from above: repo read-only, holdout masked)
docker run "${RO[@]}" rr-level7-httpx:disposable python -m unittest -v \
  tests.test_level7_service tests.test_level7_bundle tests.test_level7_corruption tests.test_frozen_artifacts
# expect: Ran 45 tests ... OK (skipped=3). The 3 skips are the bundle tests, reported as NOT APPLICABLE.

# 3. Image layout: overlay the current service and tests into the flat /app layout,
#    as the Docker test stage does (the existing image still holds the pre-F1 service module)
T2="$(mktemp -d)"; cp src/level7_service.py tests/test_level7_service.py tests/test_level7_bundle.py "$T2"/
cat > "$T2/Dockerfile" <<'EOF'
FROM rr-level7-httpx:disposable
COPY level7_service.py /app/level7_service.py
RUN mkdir -p /app/tests
COPY test_level7_service.py /app/tests/test_level7_service.py
COPY test_level7_bundle.py /app/tests/test_level7_bundle.py
EOF
docker build --platform linux/amd64 -t rr-level7-overlay:disposable "$T2"
docker run --rm --network none --platform linux/amd64 --read-only --tmpfs /tmp \
  -e LEVEL7_REQUIRE_BUNDLE_TESTS=1 -w /app rr-level7-overlay:disposable \
  python -m unittest -v tests/test_level7_service.py tests/test_level7_bundle.py
# expect: Ran 36 tests ... OK (skipped=6). The 6 skips are layout-only tests; none is a bundle test.

docker rmi rr-level7-overlay:disposable rr-level7-httpx:disposable      # clean up
```

**Disposable clean rebuild of the Level 7 image.** Only in a copy: the packager's `stage` step deletes and recreates `build/level7-context/` and runs the Level 6 audit through a read-write bind mount of whatever repository it lives in. Leave `test.csv` out of the copy, and use `docker build` directly, never `python -m src.level7_package build`, which retags `running-readiness-level7:4.0.0`:

```bash
C="$(mktemp -d)"
rsync -a --exclude .git --exclude .venv --exclude build --exclude __pycache__ \
  --exclude 'data/v1/processed/test.csv' ./ "$C"/
(cd "$C" && python3 -m src.level7_package stage)
docker build --no-cache --platform linux/amd64 -t rr-level7-rebuild:disposable "$C/build/level7-context"
docker rmi rr-level7-rebuild:disposable                                  # clean up
```

**Live service smoke test** (the repository is not mounted):

```bash
docker run -d --name rr-live --platform linux/amd64 --read-only --tmpfs /tmp \
  -p 127.0.0.1:18000:8000 running-readiness-level7:4.0.0
curl -s http://127.0.0.1:18000/health
curl -s -X POST http://127.0.0.1:18000/predict -H 'content-type: application/json' \
  -d '{"weekly_km_to_date":32.0,"sleep_hours_last_night":7.0,"resting_hr_morning":50.0,"soreness_score_evening":4.0,"prior_injury":0}'
docker stop rr-live && docker rm rr-live
```

To prove tamper detection, mount a modified **copy** (made in a temporary directory) read-only over `/app/artifacts/v4/level6_selected_model.pt`. Never modify the artifact in the repository.

### Native commands (partial)

| Environment | What was verified to work |
| :--- | :--- |
| A: numpy 2.5.4 from `requirements.txt`, no torch | `test_data_pipeline`, `test_evaluation`, `test_features`, `test_training`, `test_leakage_proxy`, `test_schema_drift_gate` (34 tests) |
| B: torch 2.2.2, numpy 1.26.4, fastapi 0.162.0 (the existing `.venv`) | `python -m src.level7_lineage`, `python -m src.level7_runtime`, `tests.test_level7_service` (from the repo root, using a disposable serving root), `tests.test_level7_corruption`, `tests.test_frozen_artifacts`, and `tests.test_level7_bundle` (skipped as NOT APPLICABLE: 45 tests, `OK (skipped=3)` together) |
| Neither | `test_neural_networks`, `test_finetune` (need pandas and torch together), `python -m src.level7_service` (starts no server), `python -m unittest discover tests/` |

Native `python -m unittest discover -v tests/` is expected to report import errors; that is an environment limit, not a code regression. Torch 2.15.0 has no Intel-mac wheel; NumPy 1.26.4 cannot import pandas 3.1.0 (**verified**).

## 5. Commands that read the holdout or write protected files: do not run to verify

Several modules compute the repository root from `__file__`, so they write into the working tree even when started from another directory. Most use exclusive-create (`"x"`) writes and fail on a second run, but several **overwrite**, and a failure after a partial write still leaves a changed file. Commands are listed as **by code reading**; none was executed.

| Command | Why it must not be run for verification |
| :--- | :--- |
| `python -m src.evaluate_models evaluate` | Reads the **consumed holdout** (`test.csv`) and writes the consumed marker, reports, and the Level 4 contract. `lock` writes the selection lock. `document` rewrites `README.md`. |
| `python -m src.schema_drift_gate` | Reads the real `data/v1/processed/test.csv` as its "valid batch" and writes `evidence/v1/schema_drift_evidence.json`. (Its *tests* use temporary roots and are safe.) |
| `python -m src.data_pipeline` | **Overwrites** the cleaned data and the train and test splits (including the holdout file), `artifacts/v1/preprocessing.json`, and the data-quality report (`to_csv` / `write_text`). |
| `python -m src.train_models` | **Overwrites** `models/v1/*.joblib`, the training report, and `artifacts/v1/level3_evaluation_contract.json` (`joblib.dump` / `write_text`). |
| `python -m src.capacity_experiment` | **Overwrites** `reports/v1/tree_capacity_experiment.json` (`write_text`). |
| `python -m src.threshold_analysis`, `src.leakage_proxy_experiment`, `src.overfit_experiment` | Write reports under `reports/` (exclusive create; fail if present). |
| `python -m src.feature_experiment run` / `document` | Writes the v2 schema, report, and Level 5 handoff; `document` rewrites `README.md`. (`audit` and `plan` are read-only by code reading.) |
| `python -m src.neural_network_experiment run` / `document` | `run` calls `torch.save` on `artifacts/v3/level5_mlp_state.pt` (**overwrite**) before exclusive-create writes that then fail, so it can corrupt a locked checkpoint. `document` rewrites `README.md`. |
| `python -m src.finetune_experiment pretrain` / `run` / `secret` / `document` | `pretrain` and `run` **overwrite** `artifacts/v4/*.pt` checkpoints; `secret` and `run` write reports and handoffs; `document` rewrites `README.md`. |
| `python -m src.level7_manifest` | Writes `serving-manifest.json` (refuses to overwrite, but would create the file if absent). |
| `python -m src.level7_package stage` / `build` | Deletes and recreates `build/level7-context/` (regenerating its manifest and attestation), runs the Level 6 audit through a **read-write** bind mount of the repository, and `build` retags `running-readiness-level7:4.0.0`. Requires the local lineage files. `audit` only prints the lineage audit (read-only by code reading, not executed). |
| `docker run` of an image with the repository bind-mounted **without** `readonly` | Turns any of the above into writes to your working tree. Always use `readonly`. |

Note that the Level 5 and Level 6 images have `audit` as their default `CMD`, which is read-only, but any other argument runs that subcommand.

Read-only by code reading, **not executed**: `python -m src.feature_experiment audit|plan`, `python -m src.neural_network_experiment concepts`, `python -m src.finetune_experiment assess`, `python -m src.level7_package audit`.

## 6. Rules for Docker builds

* There is **no `.dockerignore`**. Build Level 5 from a disposable context containing only `Dockerfile.level5` and `requirements-level5.txt`, so `data/` (including the holdout) and `.venv` never enter a build context.
* Build verification rebuilds under a different tag (for example `:rebuild-check`) and delete them afterwards. Do not retag `running-readiness-level5:2.15.0`, `…level6:2.15.0`, or `…level7:4.0.0` unless you intend to replace the verified reference.
* The Level 7 serving image is produced by `python -m src.level7_package build` through a staged allowlist context. There is no root `Dockerfile`; `Dockerfile.level7` is valid only inside that staged context.

## 7. Frozen-file guard

`tests/test_frozen_artifacts.py` compares protected files against [`frozen-hashes.json`](../frozen-hashes.json). It detects modified or deleted files, files added under the protected paths, and tampered holdout-marker metadata. It never opens, parses, or hashes `data/v1/processed/test.csv`. Gitignored lineage files missing from a fresh clone are reported as **UNAVAILABLE (skipped)**, never as passed. How to approve new versioned artifacts is described in `frozen-hashes.json` under `approval_procedure`.

## 8. F1: service portability and test isolation (verified)

`src/level7_service.py` now reads an optional `SERVING_ROOT` environment variable. Unset or empty keeps the previous default (the directory containing the module), so container behaviour is unchanged. `src/level7_runtime.py` is unchanged and runs the same SHA-256 checks against whichever root is chosen. Tests live in `tests/test_level7_service.py` (portable between the repository and `/app`) and `tests/test_level7_bundle.py` (container isolation).

| Tier | What ran | Result |
| :--- | :--- | :--- |
| Native env B, repository layout | `test_level7_service`, `test_level7_bundle`, `test_level7_corruption`, `test_frozen_artifacts` | 45 tests, `OK (skipped=3)`; the skips are the bundle tests, reported as NOT APPLICABLE |
| Canonical stack (Python 3.15.2, torch 2.15.0), existing image + httpx layer, repo mounted read-only, holdout masked | Same four modules | 45 tests, `OK (skipped=3)` |
| Image layout (overlay: current service and tests in flat `/app`), `LEVEL7_REQUIRE_BUNDLE_TESTS=1` | `test_level7_service`, `test_level7_bundle` | 36 tests, `OK (skipped=6)`; **the 3 bundle tests ran and passed**; the skips are layout-only tests |
| Disposable `--no-cache` rebuild, the build's own test stage | Same two files | Build succeeded; 36 tests, `OK (skipped=6)`; bundle tests `ok` in the build log |
| Negative: sentinel set in the repository layout (native and Docker) | `test_level7_bundle` | 3 failures, not skips |
| Negative: Dockerfile with the bundle `COPY` removed | `docker build` | Build fails |
| Negative: bundle containing `data/` | `docker build` | Build fails at `test_bundle_excludes_*` |
| Negative: bundle containing `data/` and the sentinel removed | `docker build` | Build fails (the bundle tests do not skip in the image layout either) |
| New tests against the *pre-F1* image service | `test_level7_service` | 7 override tests fail, so the tests detect a service without the override; the default-root test passes there |

Why the bundle tests cannot be skipped in the build: they skip only if the repository layout is positively detected (`src/level7_service.py` exists, which the bundle never has) **and** `LEVEL7_REQUIRE_BUNDLE_TESTS` is not `1`. The Dockerfile test stage sets it, and `DockerfileContractTests` checks that the stage still runs both files with it before creating the marker.

The test fixture (repository layout only) builds a temporary serving root from the three tracked artifacts and the existing `create_serving_manifest` generator, then pins its bytes to `a007e449a7728f632d63ba3709109997db88e0bfe52e35b7f4263d1fc4d2b12a`, the manifest inside the existing image. Nothing tracked is written.

The golden reference for the sample request is `probability 0.49793776869773865`, checked with an absolute tolerance of `1e-6` (a few float32 ulps; observed bit-identical on torch 2.2.2 and 2.15.0), with decision `0` and threshold `0.5` asserted separately.

**The existing `running-readiness-level7:4.0.0` image contains the pre-F1 service module** (`142a76a3…`). The repository now holds `022c275f…`. The image has not been rebuilt or retagged.
