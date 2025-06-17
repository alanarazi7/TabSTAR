import argparse

from tabstar.datasets.all_datasets import TabularDatasetID, OpenMLDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
from tabstar_paper.datasets.downloading import download_dataset, get_dataset_from_arg

DOWNSTREAM_EXAMPLES = 10_000


def eval_tabstar_on_dataset(dataset_id: TabularDatasetID, run_num: int, train_examples: int) -> float:
    dataset = download_dataset(dataset_id=dataset_id)
    # TODO: we'll need a 'train examples' split here, to run over a subset of the dataset.
    x_train, x_test, y_train, y_test = split_to_test(x=dataset.x, y=dataset.y, is_cls=dataset.is_cls, seed=run_num)
    tabstar_cls = TabSTARClassifier if dataset.is_cls else TabSTARRegressor
    tabstar = tabstar_cls(pretrain_dataset=dataset_id)
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
