# Level 5: Networks

## Result

Selected configuration: `regularized` on development-only runner-safe folds.

| Metric | Mean outer validation |
| --- | ---: |
| F1 | 0.819 |
| Precision | 0.808 |
| Recall | 0.840 |
| Loss | 0.498 |

## Architecture and training

The model is `Linear -> ReLU -> Dropout -> Linear`. Training uses raw logits with binary cross-entropy, explicit backpropagation, Adam updates, nested runner-safe early stopping, and threshold `0.5` only for metrics.

## Evidence boundary

The v1 holdout is consumed and was not read. These are development-only comparisons against Level 4's raw and engineered controls. Level 5 preprocessing is fold-local, but v1 historically used development-wide mean imputation before splitting. The full historical pipeline is not described as leakage-free.

## Artifacts

- `reports/v3/neural_network_experiment.json`
- `artifacts/v3/level5_mlp_state.pt`
- `artifacts/v3/level5_preprocessor.json`
- `artifacts/v3/level6_finetune_handoff.json`

## Reproduce

```bash
docker build --platform linux/amd64 -f Dockerfile.level5 -t running-readiness-level5:2.15.0 .
docker run --rm --platform linux/amd64 --mount type=bind,src="$(pwd)",dst=/workspace -w /workspace running-readiness-level5:2.15.0 python -m src.neural_network_experiment audit
docker run --rm --platform linux/amd64 --mount type=bind,src="$(pwd)",dst=/workspace -w /workspace running-readiness-level5:2.15.0 python -m unittest -v tests/test_neural_networks.py
```
