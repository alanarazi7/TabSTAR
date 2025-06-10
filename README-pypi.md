# tabstar

<img src="https://raw.githubusercontent.com/alanarazi7/TabSTAR/main/figures/tabstar_logo.png" alt="TabSTAR Logo" width="50%">

**TabSTAR**: A Foundation Tabular Model With Semantically Target-Aware Representations

Use TabSTAR as a package for your own tabular data tasks.

---

## Install

```bash
pip install tabstar
```
---

## Quickstart Example

```python
from importlib.resources import files
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tabstar.tabstar_model import TabSTARClassifier

csv_path = files("tabstar").joinpath("resources", "imdb.csv")
x = pd.read_csv(csv_path)
y = x.pop('Genre_is_Drama')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
tabstar = TabSTARClassifier()
tabstar.fit(x_train, y_train)
y_pred = tabstar.predict(x_test)
print(classification_report(y_test, y_pred))
```

🔜 Pending Features:
- [ ] Reloading fitted model for later use
- [ ] Fixed seed support for reproducibility
- [ ] Automatic task type detection (classification/regression)


---

## Documentation & Research Mode

For the full repository README (advanced usage, research-mode scripts, custom pretraining), see:

🔗 [TabSTAR Research Repository](https://github.com/alanarazi7/TabSTAR)
📚 [Paper](https://arxiv.org/abs/2505.18125)
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