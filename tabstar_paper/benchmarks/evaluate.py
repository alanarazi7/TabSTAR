from typing import Type, Optional

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARClassifier, BaseTabSTAR, TabSTARRegressor
from tabstar.training.metrics import Metrics
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.datasets.downloading import download_dataset
from tabstar_paper.preprocessing.sampling import subsample_dataset
from tabstar_paper.utils.logging import log_calls

DOWNSTREAM_EXAMPLES = 10_000


@log_calls
def evaluate_on_dataset(model_cls: Type[TabularModel],
                        dataset_id: TabularDatasetID,
                        run_num: int,
                        train_examples: int = DOWNSTREAM_EXAMPLES,
                        device: Optional[str] = None,
                        verbose: bool = False) -> Metrics:
    is_tabstar = issubclass(model_cls, BaseTabSTAR)
    name = "TabSTAR ⭐" if is_tabstar else model_cls.MODEL_NAME
    print(f"Running model {name} over dataset {dataset_id} with run number {run_num}")
    dataset = download_dataset(dataset_id=dataset_id)
    is_cls = dataset.is_cls
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=is_cls, train_examples=train_examples, seed=run_num)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, seed=run_num, train_examples=train_examples)
    if is_tabstar:
        tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
        model = tabstar_cls(pretrain_dataset=dataset_id, device=device, verbose=verbose)
    else:
        model = model_cls(is_cls=is_cls, verbose=verbose)
    model.fit(x_train, y_train)
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    print(f"Scored {metrics.score:.4f} on dataset {dataset.dataset_id}.")
    return metrics