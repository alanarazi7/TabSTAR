<img src="https://raw.githubusercontent.com/alanarazi7/TabSTAR/main/figures/tabstar_logo.png" alt="TabSTAR Logo" width="50%">

[![PyPI version](https://badge.fury.io/py/tabstar.svg)](https://badge.fury.io/py/tabstar)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/tabstar/)
[![Downloads](https://pepy.tech/badge/tabstar/month)](https://pepy.tech/project/tabstar)
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2505.18125-blue.svg)](https://arxiv.org/pdf/2505.18125)


**Welcome to the TabSTAR repository! 👋**  
You can either use TabSTAR as a package for your own tabular data tasks, or explore the full repository for research purposes, including customized pretraining and replication of paper results.

🔜 Pending Features:
- [ ] Fixed seed support for reproducibility
- [ ] Automatic task type detection (classification/regression)
- [ ] Hyperparameter tuning support

🚧 The repository is under construction: Any bugs or feature request? Please open an issue! 🚧

---

### 📚 Resources

* **Paper**: [TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations](https://arxiv.org/abs/2505.18125)
* **Project Website**: [TabSTAR](https://eilamshapira.com/TabSTAR/)

<img src="https://raw.githubusercontent.com/alanarazi7/TabSTAR/main/figures/tabstar_arch.png" alt="TabSTAR Logo" width="200%">

---

## Package Mode

Use this mode if you want to fit a pretrained TabSTAR model to your own dataset.

### Installation

```bash
pip install tabstar
```

### Inference Example

Using TabSTAR is as simple as this:

```python
from importlib.resources import files
import pandas as pd
from sklearn.model_selection import train_test_split

from tabstar.tabstar_model import TabSTARClassifier

csv_path = files("tabstar").joinpath("resources", "imdb.csv")
x = pd.read_csv(csv_path)
y = x.pop('Genre_is_Drama')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
tabstar = TabSTARClassifier()
tabstar.fit(x_train, y_train)
# tabstar.save("my_model_path.pkl")
# tabstar = TabSTARClassifier.load("my_model_path.pkl")
# y_pred = tabstar.predict(x_test)
metric = tabstar.score(X=x_test, y=y_test)
print(f"AUC: {metric:.4f}")
```

Below is a template you can use to quickly get started with TabSTAR with your own data.

```python
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

# --- USER-PROVIDED INPUTS ---
x_train = None  # TODO: load your feature DataFrame here
y_train = None  # TODO: load your target Series here
is_cls = None   # TODO: True for classification, False for regression
x_test = None   # TODO Optional: load your test feature DataFrame (or leave as None)
y_test = None   # TODO Optional: load your test target Series (or leave as None)
# -----------------------------

# Sanity checks
assert isinstance(x_train, DataFrame), "x should be a pandas DataFrame"
assert isinstance(y_train, Series), "y should be a pandas Series"
assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

if x_test is None:
    assert y_test is None, "If x_test is None, y_test must also be None"
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

assert isinstance(x_test, DataFrame), "x_test should be a pandas DataFrame"
assert isinstance(y_test, Series), "y_test should be a pandas Series"

tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
tabstar = tabstar_cls()
tabstar.fit(x_train, y_train)
# tabstar.save("my_model_path.pkl")
# tabstar = TabSTARClassifier.load("my_model_path.pkl")
y_pred = tabstar.predict(x_test)
# metric = tabstar.score(X=x_test, y=y_test)
```

---

## Research Mode

Use this section when you want to pretrain, finetune, or run baselines on TabSTAR. It assumes you are actively working on model development, experimenting with different datasets, or comparing against other methods.

### Installation

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

`--n_datasets` determines how many datasets to use for pretraining. You can reduce this number for quick debugging, but note this will harm downstream performance.

### Finetuning

Once pretraining finishes, note the printed `<PRETRAINED_EXP>` identifier. Then run:

```bash
python do_finetune.py --pretrain_exp=<PRETRAINED_EXP> --dataset_id=46655
```

`--dataset_id` is an ID for the downstream task you want to evaluate yourself on. Only the 400 datasets in the paper are supported.  

### Baseline Comparison

If you want to compare TabSTAR against a classic baseline (e.g., random forest):

```bash
python do_baseline.py --model=rf --dataset_id=46655
```

You can also try other names models supported by `do_baseline.py` (check the script for details).

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

### License

MIT © Alan Arazi et al.
