from typing import Dict


class EarlyStopping:

    def __init__(self, patience: int):
        self.metric: float = float('-inf')
        self.failed: int = 0
        self.patience = patience

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

    def state_dict(self) -> Dict:
        return {'metric': self.metric, 'failed': self.failed, 'patience': self.patience}

    def load_state_dict(self, state_dict: Dict):
        self.metric = state_dict['metric']
        self.failed = state_dict['failed']
        self.patience = state_dict['patience']