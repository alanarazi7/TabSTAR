import argparse
from typing import Type

from tabstar.datasets.all_datasets import TabularDatasetID, OpenMLDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar.training.metrics import calculate_metric
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.datasets.downloading import download_dataset, get_dataset_from_arg
from tabstar_paper.do_benchmark import DOWNSTREAM_EXAMPLES
from tabstar_paper.preprocessing.sampling import subsample_dataset

BASELINES = [CatBoost]

SHORT2MODELS = {model.SHORT_NAME: model for model in BASELINES}

def eval_baseline_on_dataset(model: Type[TabularModel], dataset_id: TabularDatasetID, run_num: int, train_examples: int) -> float:
    dataset = download_dataset(dataset_id=dataset_id)
    is_cls = dataset.is_cls
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=is_cls, train_examples=train_examples, seed=run_num)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, seed=run_num)
    model = model(is_cls=is_cls)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metric = calculate_metric(y_test, y_pred, d_output=model.d_output)
    print(f"Scored {metric:.4f} on dataset {dataset.dataset_id}.")
    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=SHORT2MODELS.keys(),
                        default=CatBoost.SHORT_NAME)
    parser.add_argument('--dataset_id', default=OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION.value)
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    args = parser.parse_args()
    tabular_dataset_id = get_dataset_from_arg(args.dataset_id)
    model = SHORT2MODELS[args.model]

    eval_baseline_on_dataset(model=model, dataset_id=tabular_dataset_id, run_num=args.run_num,
                             train_examples=args.train_examples)