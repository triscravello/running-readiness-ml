# Level 4: Features

This experiment compares raw and schema-gated engineered features using identical grouped development folds. The consumed v1 holdout was not read.

| Variant | Validation accuracy | Precision 1 | Recall 1 | F1 1 | Accuracy gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.769 | 0.838 | 0.841 | 0.837 | 0.231 |
| engineered | 0.746 | 0.829 | 0.816 | 0.820 | 0.254 |

Selected variant: `raw`

Mean pairwise feature stability:

- Raw: `0.8`
- Engineered: `0.577778`

Skipped unsupported transformations: `prior_week_load_ratio`

The training-load ratios are experimental synthetic predictors. They are not validated coaching rules. The inherited v1 processed input remains subject to its documented Level 1 preprocessing history.

Level 5 handoff: `artifacts/v2/level5_features_handoff.json`
Selected Level 5 variant: `raw`
