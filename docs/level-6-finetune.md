# Level 6: Fine-Tune

## Result

Selected approach: `partial` using development-only runner-safe evidence.

| Approach | F1 | Precision | Recall | Loss | Source seconds | Downstream seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 0.827 | 0.712 | 0.988 | 0.550 | 0.000 | 0.118 |
| frozen | 0.837 | 0.720 | 1.000 | 0.520 | 0.065 | 0.100 |
| partial | 0.838 | 0.729 | 0.986 | 0.503 | 0.065 | 0.116 |

## Fair comparison contract

All three approaches use the same inherited folds, encoder and head architecture, loss, threshold, seeds, and `80` downstream epochs. Scratch and partial fine-tuning use the same encoder learning rate, and all three use the same head learning rate. Source pretraining and downstream fitting times are reported separately.

## What counts as pretrained

The Level 5 readiness checkpoint is not called pretrained because it learned the same supervised target. Level 6 creates pretrained weights only after the encoder learns a separately defined denoising source task without readiness labels.

This is controlled self-supervised pretraining on synthetic variants of fold-training data. It is not same-task warm-starting and it is not external-domain transfer learning.

Frozen-feature transfer keeps the source encoder unchanged and trains a new head. Partial fine-tuning keeps the first encoder layer frozen while adapting the later encoder layer. Scratch training initializes the same target architecture randomly.

## Leakage and evidence boundary

All three methods use the same four inherited runner-safe folds. Source synthesis, fold-local preprocessing, and downstream fitting remain inside each outer-training boundary. Target-validation runners and labels are used only for evaluation. The consumed v1 holdout is not read.

The inherited v1 process used development-wide mean imputation before splitting. Level 6 preprocessing is boundary-local, but the full historical pipeline is not retroactively called leakage-free.

## Reproducibility

The workflow uses seed `42`, CPU execution, and deterministic algorithms where supported. PyTorch does not guarantee bitwise equality across releases or platforms. Training time is measured separately and can vary with host load.

## Artifacts

- `reports/v4/finetune_experiment.json`
- `artifacts/v4/level6_source_pretrained.pt`
- `artifacts/v4/level6_selected_model.pt`
- `artifacts/v4/level6_preprocessor.json`
- `artifacts/v4/level7_deploy_handoff.json`

## Reproduce

```bash
docker build --platform linux/amd64 -f Dockerfile.level6 -t running-readiness-level6:2.15.0 .
docker run --rm --platform linux/amd64 --mount type=bind,src="$(pwd)",dst=/workspace -w /workspace running-readiness-level6:2.15.0 python -m unittest -v tests/test_finetune.py
```
