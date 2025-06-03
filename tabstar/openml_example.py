import openml

from tabstar.inference.inference import from_x_y

openml_id = None        # TODO: set your OpenML dataset ID here
is_cls = None           # TODO: set True for classification or False for regression

assert isinstance(openml_id, int), "openml_id should be an integer representing the OpenML dataset ID"
assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

dataset = openml.datasets.get_dataset(openml_id, download_data=True, download_features_meta_data=True)
x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

from_x_y(x=x, y=y, is_cls=is_cls)