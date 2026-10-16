---
library_name: tabstar
tags:
- tabular
- classification
---

# TabSTAR v1.1 classification

`alana89/TabSTAR-v1.1-cls-20261011`, the classification checkpoint behind
`tabstar.tabstarv1p1_model.TabSTARv1p1Classifier`.

```python
from tabstar.tabstarv1p1_model import TabSTARv1p1Classifier

model = TabSTARv1p1Classifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

The checkpoint is resolved automatically; `pretrain_dataset_or_path` overrides it.

## Pretraining

- **Task type**: classification only, a separate model per task type
- **Pool**: 200,142 classification tables, decontaminated against the evaluation suites
- **Dataset sampling**: 50,000 datasets per epoch, resampled from the full pool at the start of every epoch
- **Learning rate**: 1e-4, **weight decay**: 1e-3, linear warmup over the first 10% of steps
- **Epochs**: 25 planned, early-stopping patience 5
- **Batch size**: 512 global (micro-batch 32, no gradient accumulation)
- **Rows per dataset**: up to 1,024 train and 64 validation rows
- **Text encoder**: `intfloat/e5-small-v2`, top 6 layers unfrozen, text batch size 1024
- **Tabular backbone**: 6 interaction layers, `d_model` 384
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
- **Text encoder and width** are unchanged from v1 (E5-small-v2, `d_model` 384)

## Caveats

- Epoch-to-epoch selection scores are noisy, so the chosen epoch is partly selection luck.
- Selection scores are computed separately per task type; this checkpoint is only evaluated on
  classification.
