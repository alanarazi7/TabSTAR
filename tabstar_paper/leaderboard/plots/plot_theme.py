from dataclasses import dataclass
from typing import Tuple

@dataclass
class PlotTheme:
    tabstar_color: str = '#FFBE7D'
    tabstar_light_color: str = '#FDEBD0'
    baseline_light_col: str = 'paleturquoise'
    baseline_dark_col:   str = '#A9CCE3'
    half_h: float = 0.35                    # height of each split segment
    outside_threshold: float = 0.3
    nbins: int = 10
    figsize: Tuple[int, int] = (6, 4)
    base_fs: int = 13
    big_fs: int = 15
    pad: float = 0.01                       # padding for text labels
    avg: str = 'avg'
    low: str = 'low'
    high: str = 'high'

    @property
    def full_h(self) -> float:
        return 2 * self.half_h

    @property
    def split_dy(self) -> float:
        # vertical offset of split segments
        return self.half_h / 2