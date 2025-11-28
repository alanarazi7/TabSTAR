# TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields

<img src="https://raw.githubusercontent.com/alanarazi7/TabSTAR/main/figures/tabstar_logo.png" alt="TabSTAR Logo" width="50%">

[![PyPI version](https://badge.fury.io/py/tabstar.svg)](https://badge.fury.io/py/tabstar)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/tabstar/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tabstar?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/tabstar)
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2505.18125-blue.svg)](https://arxiv.org/pdf/2505.18125)
[![NeurIPS](https://img.shields.io/badge/NeurIPS_2025-OpenReview-blue)](https://openreview.net/forum?id=FrXHdcTEzE)

**Welcome to the TabSTAR repository! 👋**  
You can either use TabSTAR as a package for your own tabular data tasks, or explore the full repository for research purposes, including customized pretraining and replication of paper results.

---

### 📚 Resources

* **Paper**: [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](https://arxiv.org/abs/2505.18125)
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

Use this section when you want to pretrain, finetune, or evaluate TabSTAR on benchmarks. It assumes you are actively working on model development, experimenting with different datasets, or comparing against other methods.

### Installation

After cloning the repo, run:

```bash
source init.sh
```

This will install all necessary dependencies, set up your environment, and download any example data needed to get started.

### Benchmark Evaluation

If you want to evaluate TabSTAR on public datasets belonging to the paper's benchmark:

```bash
python tabstar_paper/do_benchmark.py --model=tabstar --dataset_id=<DATASET_ID>
```

This script can over the 400 datasets in the paper, both for TabSTAR and other baselines presented in the paper.
The `--dataset_id` argument should be selected as the value of datasets appearing at `tabstar/datasets/all_datasets.py`.

### Pretraining

To pretrain TabSTAR on a specified number of datasets:

```bash
python tabstar_paper/do_pretrain.py --n_datasets=256
```

`--n_datasets` determines how many datasets to use for pretraining. You can reduce this number for quick debugging, but note this will harm downstream performance.

### Finetuning

Once pretraining finishes, note the printed `<PRETRAINED_EXP>` identifier. Then run:

```bash
python tabstar_paper/do_finetune.py --pretrain_exp=<PRETRAINED_EXP> --dataset_id=46655
```

`--dataset_id` is an ID for the downstream task you want to evaluate yourself on. Only the 400 datasets in the paper are supported.  

### Citation

If you use TabSTAR in your research, please cite:

```bibtex
@inproceedings{
arazi2025tabstar,
title={Tab{STAR}: A Tabular Foundation Model for Tabular Data with Text Fields},
author={Alan Arazi and Eilam Shapira and Roi Reichart},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=FrXHdcTEzE}
}
```

### License

MIT

## ❤️ Contributors

[![langflow contributors](https://contrib.rocks/image?repo=alanarazi7/TabSTAR)](https://github.com/alanarazi7/TabSTAR/graphs/contributors)
