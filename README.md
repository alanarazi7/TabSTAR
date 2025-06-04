<img src="tabstar_logo.png" alt="TabSTAR Logo" width="50%">

Welcome to the TabSTAR repository! You can use it in two modes: production mode for fitting TabSTAR on your own dataset, and research mode to pretrain TabSTAR and replicate our work in the paper. 

🚧 The repository is under construction: Any bugs or feature request? Please open an issue! 🚧

---

### 📚 Resources

* **Paper**: [TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations](https://arxiv.org/abs/2505.18125)
* **Project Website**: [TabSTAR](https://eilamshapira.com/TabSTAR/)

---

## Production Mode

Use this mode if you want to fit a pretrained TabSTAR model to your own dataset.
Note that currently we still don't support reloading that model for later use, but this is coming soon! 🔜

### Installation

```bash
source init.sh
```

### Inference Example

**Explanation of usage:**

A template for using TabSTAR in production mode is provided in `do_tabstar.py`.

1. **Import and prepare your data.**

   * `x_train` must be a `pandas.DataFrame` containing all features.
   * `y_train` must be a `pandas.Series` containing the corresponding labels (for regression or classification).
   * Set `is_cls=True` if you are solving a classification problem; `is_cls=False` for regression.
   * Provide your test split via `x_test` and `y_test`, or we'll automatically do it for you.
2. **Call `for_downstream(...)`.**

   * This function loads a pretrained TabSTAR model over 400 datasets from OpenML and Kaggle.
   * It returns `y_pred`, a NumPy array of predictions on `x_test`.

That's it!

**Full Example** 

Here we show a working example of TabSTAR over an OpenML public dataset, [`imdb_genre_prediction`](https://www.openml.org/search?type=data&id=46667).
(Note that the pretrained model has already seen thi dataset, it's just an example of how to use the code.)

```python
from do_tabstar_openml import do_tabstar_example

do_tabstar_example()
```

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
