import openml

from tabstar.preprocessing.splits import split_to_test, split_to_val
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

openml_id = 46667
# TODO: automatically detect regression vs classification?
is_cls = True

dataset = openml.datasets.get_dataset(openml_id, download_data=True, download_features_meta_data=True)
x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

x_train, x_test, y_train, y_test = split_to_test(x, y, is_cls=is_cls)
x_train, x_val, y_train, y_val = split_to_val(x_train, y_train, is_cls=is_cls)

tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
tabstar = tabstar_cls()
tabstar.fit(x_train, y_train)