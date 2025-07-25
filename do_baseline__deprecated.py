import argparse

from tabstar.training.devices import get_device
from tabstar_paper.constants import GPU
from tabular.datasets.tabular_datasets import get_dataset_from_arg
from tabular.models.competitors.carte import CARTE
from tabular.models.competitors.catboost import CatBoostOptuna
from tabular.models.competitors.tabpfn2 import TabPFNv2
from tabular.models.competitors.xg_boost import XGBoostOptuna
from tabular.trainers.finetune import do_finetune_run

# We are refactoring this code, use `do_benchmark.py`
BASELINES = [TabPFNv2, CARTE, CatBoostOptuna, XGBoostOptuna]

SHORT2MODELS = {model.SHORT_NAME: model for model in BASELINES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='default_baseline_experiment')
    parser.add_argument('--model', type=str, default='rf', choices=SHORT2MODELS.keys())
    parser.add_argument('--dataset_id', default=46667)
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--n_examples', type=int, default=10_000)
    parser.add_argument('--carte_lr_index', type=int, default=None)
    args = parser.parse_args()
    # with warning print
    print(f"⚠️ Running baseline in legacy flow. Move to `tabstar_paper/do_baseline.py` if possible. ⚠️")
    print(f"🧹 Running {args.exp} with {args.model} on dataset {args.dataset_id} for run {args.run_num}")

    model = SHORT2MODELS[args.model]
    if model == CARTE and not 0 <= args.carte_lr_index <= 5:
        raise ValueError(f"Invalid CARTE lr index: {args.carte_lr_index}. Should be between 0 and 5.")

    dataset = get_dataset_from_arg(args.dataset_id)
    n_examples = args.n_examples

    device = get_device(device=GPU)
    do_finetune_run(exp_name=args.exp, dataset=dataset, model=model, run_num=args.run_num, train_examples=n_examples,
                    device=device)