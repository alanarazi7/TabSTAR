<img src="https://raw.githubusercontent.com/alanarazi7/TabSTAR/main/figures/tabstar_logo.png" alt="TabSTAR Logo" width="50%">

📚 [TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations](https://arxiv.org/abs/2505.18125)

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

🔜 Pending Features:
- [ ] Fixed seed support for reproducibility
- [ ] Automatic task type detection (classification/regression)
- [ ] Hyperparameter tuning support


For paper replication, custom pretraining or research purposes, see:

🔗 [TabSTAR Research Repository](https://github.com/alanarazi7/TabSTAR)
---

## Citation

If you use TabSTAR in your work, please cite:

```bibtex
@article{arazi2025tabstarf,
  title   = {TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations},
  author  = {Alan Arazi and Eilam Shapira and Roi Reichart},
  journal = {arXiv preprint arXiv:2505.18125},
  year    = {2025},
}
```

---

## License

MIT © Alan Arazi et al.