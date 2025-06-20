import argparse
from typing import Optional

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
from tabstar_paper.datasets.downloading import download_dataset, get_dataset_from_arg
from tabstar_paper.preprocessing.sampling import subsample_dataset

DOWNSTREAM_EXAMPLES = 10_000


def eval_tabstar_on_dataset(dataset_id: TabularDatasetID,
                            run_num: int,
                            train_examples: int,
                            device: Optional[str] = None,
                            verbose: bool = False) -> float:
    dataset = download_dataset(dataset_id=dataset_id)
    is_cls = dataset.is_cls
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=is_cls, train_examples=train_examples, seed=run_num)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, seed=run_num, train_examples=train_examples)
    tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
    tabstar = tabstar_cls(pretrain_dataset=dataset_id, device=device, verbose=verbose)
    tabstar.fit(x_train, y_train)
    metric = tabstar.score(X=x_test, y=y_test)
    print(f"Scored {metric:.4f} on dataset {dataset.dataset_id}.")
    return metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', required=True)
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)

    args = parser.parse_args()
    tabular_dataset_id = get_dataset_from_arg(args.dataset_id)
    eval_tabstar_on_dataset(dataset_id=tabular_dataset_id, run_num=args.run_num, train_examples=args.train_examples)
