"""Picks all of TabArena's datasets for the finetunability signal, smallest-first by total cells
(rows x features) so scoring a checkpoint stays as cheap as possible.

Requires `tabarena` installed in this environment.
"""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FinetunabilityDatasets:
    cls: List[str]
    reg: List[str]

    @property
    def all(self) -> List[str]:
        return [*self.cls, *self.reg]


def select_finetunability_datasets() -> FinetunabilityDatasets:
    """All TabArena classification and regression datasets, ranked smallest-first by total
    cells. Names match TabArena's own `dataset_name`.
    """
    from tabarena.contexts import TabArenaContext

    metadata = TabArenaContext().task_metadata
    total_cells = metadata["num_instances"] * metadata["n_features"]
    metadata = metadata.assign(total_cells=total_cells).sort_values("total_cells")
    cls = metadata.loc[metadata["is_classification"], "dataset_name"].tolist()
    reg = metadata.loc[~metadata["is_classification"], "dataset_name"].tolist()
    return FinetunabilityDatasets(cls=cls, reg=reg)
