FINETUNE_PATIENCE = 5

class EarlyStopping:

    def __init__(self):
        self.metric: float = float('-inf')
        self.failed: int = 0
        self.patience = FINETUNE_PATIENCE

    def update(self, metric: float):
        if metric > self.metric:
            self.metric = metric
            self.failed = 0
        else:
            self.failed += 1

    @property
    def is_best(self) -> bool:
        return self.failed == 0

    @property
    def should_stop(self) -> bool:
        return self.failed >= self.patience