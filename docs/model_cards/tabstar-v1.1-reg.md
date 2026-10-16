---
library_name: tabstar
tags:
- tabular
- regression
---

# TabSTAR v1.1 regression

`alana89/TabSTAR-v1.1-reg-20261009`, the regression checkpoint behind
`tabstar.tabstarv1p1_model.TabSTARv1p1Regressor`.

```python
from tabstar.tabstarv1p1_model import TabSTARv1p1Regressor

model = TabSTARv1p1Regressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

The checkpoint is resolved automatically; `pretrain_dataset_or_path` overrides it.

## Pretraining

- **Task type**: regression only, a separate model per task type
- **Pool**: 200,013 regression tables, decontaminated against the evaluation suites
- **Dataset sampling**: 50,000 datasets per epoch, resampled from the full pool at the start of every epoch
- **Learning rate**: 1e-4, **weight decay**: 1e-3, linear warmup over the first 10% of steps
- **Epochs**: 25 planned, early-stopping patience 5; the run completed all 25
- **Batch size**: 512 global (micro-batch 32, no gradient accumulation)
- **Rows per dataset**: up to 1,024 train and 64 validation rows
- **Text encoder**: `intfloat/e5-small-v2`, top 6 layers unfrozen, text batch size 1024
- **Tabular backbone**: 6 interaction layers, `d_model` 384
- **Numeric features**: a 6-dimensional engineered-statistics vector through an MLP
- **Regression head**: a scalar sub-head plus a distributional sub-head over 200 bins on z-scored
  targets clipped to +/-10
- **Loss**: 0.5 * MSE (scalar) + 0.5 * distributional loss, itself 0.5 * cross-entropy + 0.5 * CRPS
  over the 200 bin centers
- **Checkpoint selection**: downstream finetuning Elo, measured every epoch, rather than pretraining loss

## Downstream finetuning

The package defaults, which are also what the checkpoint was selected under, none of them tuned per
dataset:

- LoRA rank 32, alpha 2x rank, dropout 0.1, learning rate 1e-3, weight decay 0
- Batch size 128, up to 50 epochs, early-stopping patience 5
- Predictions averaged over the three best-validation-loss epochs

## Differences from the original TabSTAR (v1)

- **Pretraining data**: filtered by content-overlap sketches over column value sets and schemas,
  which drop duplicate tables and copies of evaluation tables
- **Scale**: 50,000 datasets per epoch, resampled each epoch from a pool of ~200K tables
- **Task separation**: classification and regression are pretrained as separate models, not one
  mixed model
- **Global batch 512**: larger and more diverse per step than v1
- **Selection metric**: downstream finetuning Elo rather than pretraining loss
- **Fusion**: each column's text and numeric embeddings are layer-normalised and summed, with a
  learned scalar weight on the text stream, rather than a fixed combination
- **Regression head and loss**: a dual scalar plus distributional head (200 bins, z-clip +/-10)
  trained with MSE plus a CE/CRPS blend, instead of a single scalar head
- **Text encoder and width** are unchanged from v1 (E5-small-v2, `d_model` 384)

## Caveats

- Epoch-to-epoch selection scores are noisy, so the chosen epoch is partly selection luck.
- Selection scores are computed separately per task type; this checkpoint is only evaluated on
  regression.
