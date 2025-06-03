from pandas import DataFrame, Series

from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
from tabstar.training.metrics import calculate_metric

# TODO: add support for custom test-sets
def from_x_y(x: DataFrame, y: Series, is_cls: bool):
    x_train, x_test, y_train, y_test = split_to_test(x, y, is_cls=is_cls)
    tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
    tabstar = tabstar_cls()
    tabstar.fit(x_train, y_train)
    y_pred = tabstar.predict(x_test)
    metric = calculate_metric(y_true=y_test.to_numpy(), y_pred=y_pred, d_output=tabstar.preprocessor_.d_output)
    metric_name = "AUROC" if is_cls else "R^2"
    print(f"Test metric {metric_name}: {metric:.4f}")