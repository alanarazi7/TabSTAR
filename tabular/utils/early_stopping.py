from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs

FINETUNE_PATIENCE = 5
PRETRAIN_PATIENCE = 3

class EarlyStopping:

    def __init__(self, args: PretrainArgs | FinetuneArgs):
        self.metric: float = float('-inf')
        self.epochs_without_improvement: int = 0
        assert isinstance(args, (PretrainArgs, FinetuneArgs)), "args must be of type PretrainArgs or FinetuneArgs"
        if isinstance(args, PretrainArgs):
            self.patience = PRETRAIN_PATIENCE
        elif isinstance(args, FinetuneArgs):
            self.patience = args.patience
        else:
            raise ValueError("Invalid argument type for EarlyStopping")

    def update(self, metric: float):
        if metric > self.metric:
            self.metric = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    @property
    def is_best(self) -> bool:
        return self.epochs_without_improvement == 0

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience