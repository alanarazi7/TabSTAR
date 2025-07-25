from dataclasses import dataclass, asdict
from os.path import exists
from typing import Type, Optional

import torch

from tabstar_paper.utils.io_handlers import load_json, dump_json
from tabular.datasets.tabular_datasets import TabularDatasetID, get_sid
from tabular.models.abstract_model import TabularModel
from tabular.preprocessing.splits import DataSplit
from tabular.utils.paths import create_dir, train_results_path


@dataclass
class RunMetadata:
    dataset_id: str
    model_cls: str
    exp_name: str
    test_score: float
    run_num: int
    train_examples: int
    seed: Optional[int] = None
    dev_loss: Optional[float] = None


    @classmethod
    def from_json(cls, path: str):
        data = load_json(path)
        return cls(**data)


class ModelTrainer:

    def __init__(self,
                 dataset_id: TabularDatasetID,
                 model_cls: Type[TabularModel],
                 exp_name: str,
                 device: torch.device,
                 run_num: int,
                 train_examples: int,
                 carte_lr_idx: Optional[int] = None):
        self.model_cls = model_cls
        self.dataset_id = dataset_id
        self.exp_name = exp_name
        self.train_examples = train_examples
        self.run_num = run_num
        self.device = device
        self.res_path = train_results_path(self.run_name)
        # Hacky that it is here, but, oh well
        self.carte_lr_idx = carte_lr_idx

    def existing_score(self) -> Optional[RunMetadata]:
        if exists(self.res_path):
            return RunMetadata.from_json(self.res_path)

    def run(self) -> RunMetadata:
        create_dir(self.res_path, is_file=True)
        model = self.model_cls(run_name=self.run_name, dataset_ids=[self.dataset_id], device=self.device,
                               run_num=self.run_num, train_examples=self.train_examples,
                               carte_lr_index=self.carte_lr_idx)
        model.initialize_model()
        dev_loss = model.train()
        test_results = model.test()
        metadata = RunMetadata(dataset_id=self.dataset_id.value,
                               model_cls=self.model_cls.MODEL_NAME,
                               exp_name=self.exp_name,
                               dev_loss=dev_loss,
                               test_score=float(test_results[DataSplit.TEST].score),
                               run_num=self.run_num,
                               train_examples=self.train_examples)
        dump_json(asdict(metadata), self.res_path)
        return metadata

    @property
    def run_name(self) -> str:
        strings = [self.exp_name,
                   f"sid_{get_sid(self.dataset_id)}",
                   f"run_{self.run_num}",
                   f"examples_{self.train_examples}"]
        if self.model_cls is not None:
            strings.append(self.model_cls.MODEL_NAME)
        return "__".join(strings)