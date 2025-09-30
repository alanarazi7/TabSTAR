from dataclasses import dataclass
from typing import Type

from tabpfn_augmentor.evaluator import evaluate_on_augmented_dataset
from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar.training.devices import get_device
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.realmlp import RealMLP
from tabstar_paper.baselines.tabpfnv2 import TabPFNv2
from tabstar_paper.baselines.xgboost import XGBoost

augment_benchmark = [
    OpenMLDatasetID.MUL_NATURE_YEAST_PROTEIN,
    OpenMLDatasetID.BIN_HEALTHCARE_BREAST_CANCER_WISCONSIN,
    OpenMLDatasetID.BIN_COMPUTERS_IMAGE_BANK_NOTE_AUTHENTICATION,

    # TODO: the code runs into a lot of edge-case errors when using some well known datasets... doing something wrong?
    # OpenMLDatasetID.REG_SPORTS_MONEYBALL,
    # OpenMLDatasetID.BIN_FINANCIAL_CREDIT_GERMAN,
    # OpenMLDatasetID.MUL_NATURE_EUCALYPTUS_SEED,
    '''
      File "/Users/alanarazi/tabstar_code/TabSTAR/tabpfn_augmentor/generate_data.py", line 34, in augment_with_tabpfn
        exp_synthetic.run(
      File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/experiments.py", line 132, in run
        self.synthetic_X = tabpfn.generate_synthetic_data(n_samples=n_samples, t=temp)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/unsupervised.py", line 410, in generate_synthetic_data
        return self.impute_(
               ^^^^^^^^^^^^^
      File "/Users/alanarazi/tabstar_code/TabSTAR/.TabSTAR-env/lib/python3.11/site-packages/tabpfn_extensions/unsupervised/unsupervised.py", line 172, in impute_
        pred = torch.stack([d for d in densities]).mean(dim=0)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray
    '''
]

models = [XGBoost, RealMLP, TabPFNv2]

num_folds = 5

augmentations = [True, False]

max_examples = 500
max_features = 3

@dataclass
class TabularTask:
    dataset: OpenMLDatasetID
    model: Type[TabularModel]
    fold: int
    do_augment: bool


tabular_tasks = [TabularTask(dataset=d, model=m, fold=f, do_augment=a)
                 for d in augment_benchmark
                 for m in models
                 for f in range(num_folds)
                 for a in augmentations]


def eval_augmented_benchmark():
    results = []
    device = get_device()
    for t in tabular_tasks:
        res = evaluate_on_augmented_dataset(model_cls=t.model, dataset_id=t.dataset,
                                            do_augment=t.do_augment, fold=t.fold, device=device,
                                            train_examples=max_examples, max_features=max_features)
        results.append(res)


if __name__ == "__main__":
    eval_augmented_benchmark()
