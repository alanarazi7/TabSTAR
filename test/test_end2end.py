import openml
from tabstar.inference.inference import from_x_y

def test_openml_example():
    # This should take a few minutes
    openml_id = 46667
    is_cls = True
    dataset = openml.datasets.get_dataset(openml_id, download_data=True, download_features_meta_data=True)
    x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    from_x_y(x=x, y=y, is_cls=is_cls, x_test=None, y_test=None)