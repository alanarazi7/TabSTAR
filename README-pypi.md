<img src="https://raw.githubusercontent.com/alanarazi7/TabSTAR/main/figures/tabstar_logo.png" alt="TabSTAR Logo" width="50%">

📚 [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](https://arxiv.org/abs/2505.18125)

---

## 🚀 TabSTAR v1.1 is on its way

We are releasing **TabSTAR v1.1**, a new generation of pretrained TabSTAR models. It ships alongside the original models: existing code keeps working unchanged.

We are releasing the model first. A detailed writeup of what changed, together with the v1.1 pretraining code, will follow.

---

## Install

To fit a pretrained TabSTAR model to your own dataset, install the package using:

```bash
pip install tabstar
```
---

## Quickstart Example

You can quickly get started with TabSTAR using the following example.

```python
from importlib.resources import files
import pandas as pd
from sklearn.model_selection import train_test_split

from tabstar.tabstar_model import TabSTARClassifier

csv_path = files("tabstar").joinpath("resources", "imdb.csv")
x = pd.read_csv(csv_path)
y = x.pop('Genre_is_Drama')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# For regression tasks, replace `TabSTARClassifier` with `TabSTARRegressor`.
tabstar = TabSTARClassifier()
tabstar.fit(x_train, y_train)
# tabstar.save("my_model_path.pkl")
# tabstar = TabSTARClassifier.load("my_model_path.pkl")
# y_pred = tabstar.predict(x_test)
metric = tabstar.score(X=x_test, y=y_test)
print(f"AUC: {metric:.4f}")
```

For paper replication, TabSTAR evaluation on benchmarks, custom pretraining or research purposes, see:

🔗 [TabSTAR Research Repository](https://github.com/alanarazi7/TabSTAR)
---

## Citation

If you use TabSTAR in your work, please cite:

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

---

## License

MIT © Alan Arazi et al.