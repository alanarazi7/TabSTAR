import openml
from sklearn.model_selection import train_test_split

from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
from tabular.datasets.tabular_datasets import OpenMLDatasetID

for name in OpenMLDatasetID:
    print(f"Processing dataset: {name.name} (ID: {name.value})")
    openml_id = name.value
    is_cls = not name.name.startswith("REG")
    dataset = openml.datasets.get_dataset(openml_id, download_data=True, download_features_meta_data=True)
    x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
    tabstar = tabstar_cls()
    try:
        tabstar.fit(x_train, y_train)
        y_pred = tabstar.predict(x_test)
    except RuntimeError:
        print(f"⚠️ RuntimeError while processing dataset {name.name}. Skipping.")
        continue