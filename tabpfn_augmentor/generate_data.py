import torch
from pandas import DataFrame, Series
from tabpfn import TabPFNRegressor

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment


# adapted from: https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/generate_data.py
def generate_with_tabpfn(x_train: DataFrame,
                         y_train: Series,
                         is_cls: bool,
                         augmenting_factor: int = 3,
                         temperature: float = 1.0) -> GenerateSyntheticDataExperiment:
    clf = TabPFNClassifier(n_estimators=3)
    reg = TabPFNRegressor(n_estimators=3)
    model_unsupervised = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg,)
    attribute_names = x_train.columns.tolist()
    x_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_dtype = torch.long if is_cls else torch.float32
    y_tensor = torch.tensor(y_train.values, dtype=y_dtype)
    current_n_samples = x_train.shape[0]
    exp_synthetic = GenerateSyntheticDataExperiment(task_type="unsupervised")
    exp_synthetic.run(
        tabpfn=model_unsupervised,
        X=x_tensor,
        y=y_tensor,
        attribute_names=attribute_names,
        temp=temperature,
        n_samples=current_n_samples * augmenting_factor,
    )
    return exp_synthetic