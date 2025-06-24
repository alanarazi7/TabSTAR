import argparse

from tabstar.tabstar_model import BaseTabSTAR
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.datasets.downloading import get_dataset_from_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', required=True)
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)

    args = parser.parse_args()
    tabular_dataset_id = get_dataset_from_arg(args.dataset_id)
    evaluate_on_dataset(model_cls=BaseTabSTAR, dataset_id=tabular_dataset_id, run_num=args.run_num, train_examples=args.train_examples)
