import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

from tabstar_paper.leaderboard.data.keys import MODEL, UNLIMIT, BASE_MODEL
from tabstar_paper.leaderboard.filters.tasks import TabularTask, TASK2PRETTY
from tabstar_paper.leaderboard.plots.main_10k import PlotTheme
from tabstar_paper.leaderboard.plots.utils import add_errorbar


## TODO: multiple functions could be merge with the 10K version

def is_unlimited_label(model_name: str) -> bool:
    return UNLIMIT in str(model_name)

def pack_rows_by_limit(data: pd.DataFrame, mean_col: str) -> list[tuple[str, pd.Series | None, pd.Series | None]]:
    """
    For each BASE_MODEL, pick at most one '10K' row (no UNLIMIT in MODEL)
    and one 'Unlimited' row (UNLIMIT in MODEL). If multiple exist (e.g. tuned/untuned),
    take the one with the best mean_col.
    Returns: [(base, row_10k, row_unlim), ...] ordered by best available mean.
    """
    order = (data.groupby(BASE_MODEL)[mean_col]
                .max()
                .sort_values(ascending=False)
                .index.tolist())

    rows = []
    for base in order:
        block = data[data[BASE_MODEL] == base]

        tenk_block   = block[~block[MODEL].map(is_unlimited_label)]
        unlim_block  = block[ block[MODEL].map(is_unlimited_label)]

        tenk_row  = tenk_block.loc[tenk_block[mean_col].idxmax()] if len(tenk_block)  else None
        unlim_row = unlim_block.loc[unlim_block[mean_col].idxmax()] if len(unlim_block) else None

        rows.append((base, tenk_row, unlim_row))
    return rows


def plot_grouped_by_limit(df: pd.DataFrame, num_datasets: int, task: TabularTask):
    assert any(UNLIMIT in m for m in set(df[MODEL])), "This plot is for 100K"
    pt = PlotTheme(baseline_dark_col="thistle", baseline_light_col="lavender")
    data = df.copy()

    packed = pack_rows_by_limit(data, mean_col=pt.avg)

    n_rows = len(packed)
    fig, ax = plt.subplots(figsize=pt.figsize)

    # y positions (best on top)
    y_centers = np.arange(n_rows)[::-1]

    def draw_seg(row, y, color):
        m  = float(row[pt.avg])
        lo = float(m - row[pt.low])
        hi = float(row[pt.high] - m)
        ax.barh(y=y, width=m, height=pt.half_h, color=color, edgecolor=color, linewidth=0)
        add_errorbar(ax, x=m, y=y, lower=lo, upper=hi)  # black by default
        return m, hi

    for i, (base, row_10k, row_unlim) in enumerate(packed):
        y = y_centers[i]
        is_tabstar = base.startswith('TabSTAR')

        # colors
        c_10k   = pt.tabstar_light_color if is_tabstar else pt.baseline_light_col
        c_unlim = pt.tabstar_color if is_tabstar else pt.baseline_dark_col

        widths, uppers = [], []

        if row_10k is not None and row_unlim is not None:
            m0, h0 = draw_seg(row_10k,   y - pt.split_dy, c_10k)
            m1, h1 = draw_seg(row_unlim, y + pt.split_dy, c_unlim)
            widths.extend([m0, m1]); uppers.extend([h0, h1])
        else:
            row = row_10k if row_10k is not None else row_unlim
            m  = float(row[pt.avg])
            lo = float(m - row[pt.low])
            hi = float(row[pt.high] - m)
            col = c_10k if row_10k is not None else c_unlim
            ax.barh(y=y, width=m, height=pt.full_h, color=col, edgecolor=col, linewidth=0)
            add_errorbar(ax, x=m, y=y, lower=lo, upper=hi)
            widths.append(m); uppers.append(hi)

        # inside/outside placement
        width_max = max(widths); upper_max = max(uppers)
        pad = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if ax.get_xlim()[1] > ax.get_xlim()[0] else 0.02
        if width_max < pt.outside_threshold:
            tx = width_max + upper_max + pad
        else:
            tx = pad
        ax.text(tx, y, base, va='center', ha='left', color='black', fontsize=pt.big_fs)

    # axes cosmetics
    ax.set_yticks([])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=pt.nbins))
    ax.tick_params(axis='x', labelsize=pt.big_fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, color='0.9', linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)

    ax.set_title(f"{TASK2PRETTY[task]} - Above 10K examples ({num_datasets} datasets)", fontsize=pt.big_fs)
    ax.set_xlabel("Normalized Score", fontsize=pt.big_fs)

    legend_handles = build_legend_handles(pt=pt)
    ax.legend(handles=list(legend_handles), loc='lower right', frameon=False, fontsize=pt.base_fs)

    fig.tight_layout()
    return fig


def build_legend_handles(pt):
    handles = [Patch(color=pt.tabstar_light_color, label="10K"),
               Patch(color=pt.tabstar_color, label="Unlimited"),
               Patch(color=pt.baseline_light_col, label="10K"),
               Patch(color=pt.baseline_dark_col, label="Unlimited"),]
    return handles