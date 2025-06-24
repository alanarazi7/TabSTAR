import argparse
import logging

from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.datasets.downloading import get_dataset_from_arg
from tabstar_paper.baselines.utils import log_calls

BASELINES = [CatBoost, XGBoost]

SHORT2MODELS = {model.SHORT_NAME: model for model in BASELINES}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s') # as a default, will only print warnings and errors. \
# locally, you can set it to DEBUG or INFO to see more details.


@log_calls
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), default=None)
    parser.add_argument('--dataset_id', default=OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION.value)
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    args = parser.parse_args()
    tabular_dataset_id = get_dataset_from_arg(args.dataset_id)

    models_to_run = BASELINES if args.model is None else [SHORT2MODELS[args.model]]
    for model in models_to_run:
        evaluate_on_dataset(model_cls=model, dataset_id=tabular_dataset_id, run_num=args.run_num,
                            train_examples=args.train_examples)


if __name__ == "__main__":
    main()