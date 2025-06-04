import openml
from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
from tabstar.training.metrics import calculate_metric

def do_tabstar_example():
    openml_id = 46667
    is_cls = True
    dataset = openml.datasets.get_dataset(openml_id, download_data=True, download_features_meta_data=True)
    x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    x_train, x_test, y_train, y_test = split_to_test(x, y, is_cls=is_cls)
    tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
    tabstar = tabstar_cls()
    tabstar.fit(x_train, y_train)
    y_pred = tabstar.predict(x_test)
    metric = calculate_metric(y_true=y_test.to_numpy(), y_pred=y_pred, d_output=tabstar.preprocessor_.d_output)
    metric_name = "AUROC" if is_cls else "R^2"
    print(f"Test metric {metric_name}: {metric:.4f}")

if __name__ == "__main__":
    do_tabstar_example()