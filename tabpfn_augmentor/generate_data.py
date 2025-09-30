from typing import Tuple

import pandas as pd
import torch
from pandas import DataFrame, Series
from tabpfn import TabPFNRegressor

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment


AUGMENT_CACHE = {}


# adapted from: https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/generate_data.py
def augment_with_tabpfn(x_train: DataFrame,
                        y_train: Series,
                        is_cls: bool,
                        cache_key: str,
                        augmenting_factor: int = 3,
                        temperature: float = 1.0) -> Tuple[DataFrame, Series]:
    if cache_key in AUGMENT_CACHE:
        return AUGMENT_CACHE[cache_key]
    clf = TabPFNClassifier(n_estimators=3)
    reg = TabPFNRegressor(n_estimators=3)
    model_unsupervised = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg,)
    attribute_names = x_train.columns.tolist()
    x_tensor = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    y_dtype = torch.long if is_cls else torch.float32
    y_tensor = torch.tensor(y_train.to_numpy(), dtype=y_dtype)
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
    synth_y = predict_y_with_tabpfn(exp=exp_synthetic, is_cls=is_cls)
    synth_x = DataFrame(data=exp_synthetic.synthetic_X, columns=attribute_names)
    synth_y = Series(data=synth_y, name=y_train.name)
    aug_x = pd.concat([x_train, synth_x], ignore_index=True)
    aug_y = pd.concat([y_train, synth_y], ignore_index=True)
    AUGMENT_CACHE[cache_key] = (aug_x, aug_y)
    return aug_x, aug_y



def predict_y_with_tabpfn(exp: GenerateSyntheticDataExperiment, is_cls: bool):
    synth_x = exp.synthetic_X
    x = exp.X
    y = exp.y
    if is_cls:
        clf = TabPFNClassifier()
        clf.fit(x, y)
        synth_y = clf.predict(synth_x)
    else:
        reg = TabPFNRegressor()
        reg.fit(x, y)
        synth_y = reg.predict(synth_x)
    return synth_y

