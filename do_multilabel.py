from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

from tabstar.tabstar_model import TabSTARMultilabelClassifier

x, y = make_multilabel_classification(
    n_samples=100,
    n_features=3,
    n_classes=5,
    n_labels=2,
    allow_unlabeled=False,
    random_state=42,
    return_distributions=False,
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
tabstar = TabSTARMultilabelClassifier(debug=True)
tabstar.fit(x_train, y_train)