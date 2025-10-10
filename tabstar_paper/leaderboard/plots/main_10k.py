import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tabstar_paper.leaderboard.data.keys import MODEL, UNLIMIT, BASE_MODEL, IS_TUNED
from tabstar_paper.leaderboard.filters.tasks import TabularTask, TASK2PRETTY
from tabstar_paper.leaderboard.plots.plot_theme import PlotTheme
from tabstar_paper.leaderboard.plots.utils import add_errorbar, draw_segment


## TODO: this code is very legacy and AI assisted, could be cleaned up a lot

def plot_grouped_models(df: pd.DataFrame, num_datasets: int, task: TabularTask,
                        base_model_key: str = BASE_MODEL) -> plt.Figure:
    assert not any(UNLIMIT in m for m in set(df[MODEL])), "This plot is for 10K only, no UNLIMIT models allowed!"
    data = df.copy()
    pt = PlotTheme()

    # Order rows by the best available mean across the pair
    order = data.groupby(base_model_key)[pt.avg].max().sort_values(ascending=False).index.tolist()
    # Pack default/tuned rows per base
    packed = []
    for b in order:
        block = data[data[base_model_key] == b]
        default = block[~block[IS_TUNED]].head(1)  # may be empty
        tuned   = block[ block[IS_TUNED]].head(1)  # may be empty
        default = default.iloc[0] if len(default) else None
        tuned   = tuned.iloc[0]   if len(tuned)   else None
        packed.append((b, default, tuned))

    n_rows = len(packed)
    fig, ax = plt.subplots(figsize=pt.figsize)

    # vertical placement: best at top (descending), so reverse y sequence
    y_centers = np.arange(n_rows)[::-1]

    # Threshold for moving label outside
    outside_threshold = 0.3

    for i, (base, default_row, tuned_row) in enumerate(packed):
        y = y_centers[i]
        # TODO: hacky, but colours TabSTAR experiments as well
        is_tabstar = base.startswith(('TabSTAR', 'Scale', 'Unfreeze', 'Numerical'))

        # choose colors (TabSTAR stays orange for both; baselines use two blues)
        tuned_color   = pt.tabstar_color if is_tabstar else pt.baseline_dark_col
        default_color = pt.tabstar_color if is_tabstar else pt.baseline_light_col

        # draw the appropriate configuration
        widths_for_label = []
        uperrs_for_label = []

        if default_row is not None and tuned_row is not None:
            m0, hi0 = draw_segment(ax, default_row, y - pt.split_dy, color=default_color)
            m1, hi1 = draw_segment(ax, tuned_row,   y + pt.split_dy, color=tuned_color)
            widths_for_label.extend([m0, m1])
            uperrs_for_label.extend([hi0, hi1])
        else:
            row = default_row if default_row is not None else tuned_row
            m  = float(row[pt.avg])
            lo = float(m - row[pt.low])
            hi = float(row[pt.high] - m)

            ax.barh(y=y, width=m, height=pt.full_h,
                    color=(default_color if default_row is not None else tuned_color),
                    linewidth=0.7)
            add_errorbar(ax, x=m, y=y, lower=lo, upper=hi)
            widths_for_label.append(m)
            uperrs_for_label.append(hi)

        # If all widths are tiny, move label outside past the larger xerr
        width_for_logic = max(widths_for_label)
        upper_for_logic = max(uperrs_for_label)

        if width_for_logic < outside_threshold:
            text_x = width_for_logic + upper_for_logic + pt.pad
        else:
            text_x = pt.pad

        ax.text(text_x, y, base, va='center', ha='left', color='black', fontsize=pt.big_fs)

    # Remove default y-axis ticks/labels (we draw text labels manually)
    ax.set_yticks([])

    # X axis ticks & formatting
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=pt.nbins))
    ax.tick_params(axis='x', labelsize=pt.big_fs)

    # Aesthetics: spines, grid, z-order
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, color='0.9', linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)

    ax.set_title(f"{TASK2PRETTY[task]} - Up to 10K examples ({num_datasets} datasets)")
    ax.set_xlabel("Normalized Score", fontsize=pt.big_fs)
    legend_handles = build_legend_handles()
    ax.legend(handles=list(legend_handles), loc='lower right', frameon=False, fontsize=pt.base_fs)

    fig.tight_layout()
    return fig


def build_legend_handles():
    pt = PlotTheme()
    handles = [Patch(color=pt.tabstar_color, label="Default"),
               Patch(color=pt.baseline_light_col, label="Default"),
               Patch(color=pt.baseline_dark_col, label="Tuned"),]
    return handles
