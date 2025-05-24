import openml

from tabstar.processing.splits import split_to_test
from tabstar.tabstar_cls import TabSTARClassifier

openml_id = 41078
dataset = openml.datasets.get_dataset(openml_id, download_data=True, download_features_meta_data=True)
x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

x_train, x_test, y_train, y_test = split_to_test(x, y)

tabstar = TabSTARClassifier()
tabstar.fit(x_train, y_train)