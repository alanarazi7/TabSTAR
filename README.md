<img src="tabstar_logo.png" alt="TabSTAR Logo" width="50%">

Welcome to the TabSTAR repository! You can use it in two modes: production mode for fitting TabSTAR on your own dataset, and research mode to pretrain TabSTAR and replicate our work in the paper. 

🚧 The repository is under construction: We'll appreciate your feedback! 🚧

---

### 📚 Resources

* **Paper**: [TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations](https://arxiv.org/abs/2505.18125)
* **Project Website**: [TabSTAR](https://eilamshapira.com/TabSTAR/)

---

## Production Mode

Use this section when you want to fit a pretrained TabSTAR model to your own dataset.

### Installation

```bash
source init.sh
```

### Inference Example

```python
from pandas import DataFrame, Series
from tabstar.inference.inference import for_downstream

# --- USER-PROVIDED INPUTS ---
x = None        # TODO: load your feature DataFrame here
y = None        # TODO: load your target Series here
is_cls = None   # TODO: True for classification, False for regression
x_test = None   # TODO Optional: load your test feature DataFrame (or leave as None)
y_test = None   # TODO Optional: load your test target Series (or leave as None)
# -----------------------------

# Sanity checks
assert isinstance(x, DataFrame), "x should be a pandas DataFrame"
assert isinstance(y, Series),  "y should be a pandas Series"
assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"
assert isinstance(x_test, (DataFrame, type(None))), "x_test should be a pandas DataFrame or None"
assert isinstance(y_test, (Series, type(None))), "y_test should be a pandas Series or None"

# Run inference (optionally, a train/val split will be created if x_test or y_test is None)
y_pred = for_downstream(x=x, y=y, is_cls=is_cls, x_test=x_test, y_test=y_test)
```

**Explanation of usage:**

1. **Import and prepare your data.**

   * `x` must be a `pandas.DataFrame` containing all features.
   * `y` must be a `pandas.Series` containing the corresponding labels (for regression or classification).
   * Set `is_cls=True` if you are solving a classification problem; `is_cls=False` for regression.
   * If you already have a separate test split, provide it via `x_test` and `y_test`.  If you leave `x_test`/`y_test` as `None`, we'll automatically split off a portion of `(x, y)` for validation.
2. **Call `for_downstream(...)`.**

   * This function loads a pretrained TabSTAR model over 400 datasets from OpenML and Kaggle.
   * It returns `y_pred`, a NumPy array of predictions on `x_test`.

That's it! From there, you can compute accuracy, MSE, or other metrics as desired.

🔜 We still need to support a few features:
- Allowing storing your model for later use.
- Reverting regression predictions to the original scale (currently, they are z‐scored).
- Anything else you need? Please open an issue!

---

## Research Mode

Use this section when you want to pretrain, finetune, or run baselines on TabSTAR. It assumes you are actively working on model development, experimenting with different datasets, or comparing against other methods.

### Prerequisites

After cloning the repo, run:

```bash
source init.sh
```

This will install all necessary dependencies, set up your environment, and download any example data needed to get started.

### Pretraining

To pretrain TabSTAR on a specified number of datasets:

```bash
python do_pretrain.py --n_datasets=256
```

* `--n_datasets`: How many datasets to sample for the multi‐dataset pretraining loop.
* You can reduce this number for quick debugging, but note that fewer datasets will harm downstream performance.

### Finetuning

Once pretraining finishes, note the printed `<PRETRAINED_EXP>` identifier. Then run:

```bash
python do_finetune.py --pretrain_exp=<PRETRAINED_EXP> --dataset_id=46655
```

* `--dataset_id`: An ID for the downstream task you want to finetune on. Currently, these datasets must come from a closed list of datasets. 

### Baseline Comparison

If you want to compare TabSTAR against a classic baseline (e.g., random forest):

```bash
python do_baseline.py --model=rf --dataset_id=46655
```

* `--model=rf` chooses a random forest from scikit‐learn; you can also try other names supported by `do_baseline.py` (check the script for details).
* The script will load features from OpenML and fit the chosen baseline to the same train/val split used by TabSTAR.

### License

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

### Citation

If you use TabSTAR in your research, please cite:

```bibtex
@article{arazi2025tabstarf,
  title   = {TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations},
  author  = {Alan Arazi and Eilam Shapira and Roi Reichart},
  journal = {arXiv preprint arXiv:2505.18125},
  year    = {2025},
}
```
