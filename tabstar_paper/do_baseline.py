import argparse
from typing import Type

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.datasets.downloading import download_dataset
from tabular.datasets.tabular_datasets import get_dataset_from_arg
from tabular.evaluation.constants import DOWNSTREAM_EXAMPLES
from tabular.trainers.finetune import do_finetune_run

BASELINES = [CatBoost]

SHORT2MODELS = {model.SHORT_NAME: model for model in BASELINES}

def eval_baseline_on_dataset(model: Type[TabularModel], dataset_id: TabularDatasetID, run_num: int, train_examples: int) -> float:
    dataset = download_dataset(dataset_id=dataset_id)
    # TODO: we'll need a 'train examples' split here, to run over a subset of the dataset.
    x_train, x_test, y_train, y_test = split_to_test(x=dataset.x, y=dataset.y, is_cls=dataset.is_cls, seed=run_num)
    model = model(is_cls=dataset.is_cls)

    tabstar_cls = TabSTARClassifier if dataset.is_cls else TabSTARRegressor
    tabstar = tabstar_cls(pretrain_dataset=dataset_id)
    tabstar.fit(x_train, y_train)
    y_pred = tabstar.predict(x_test)
    metric = calculate_metric(y_test, y_pred, d_output=tabstar.preprocessor_.d_output)
    print(f"Scored {metric:.4f} on dataset {dataset.dataset_id}.")
    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=SHORT2MODELS.keys(), required=True)
    parser.add_argument('--dataset_id', required=True)
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    args = parser.parse_args()
    tabular_dataset_id = get_dataset_from_arg(args.dataset_id)
    model = SHORT2MODELS[args.model]

    do_finetune_run(model=model, dataset=tabular_dataset_id, run_num=args.run_num, train_examples=args.train_examples)