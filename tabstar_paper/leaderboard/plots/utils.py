from io import BytesIO

from matplotlib import pyplot as plt
import streamlit as st

from tabstar_paper.leaderboard.plots.plot_theme import PlotTheme


def download_st_fig(fig: plt.Figure, fig_name: str, is_two: bool = False):
    pt = PlotTheme()
    w = pt.figsize[0]
    if is_two:
        w *= 2
    fig.set_size_inches(w, pt.figsize[1])

    # Apply tight layout so labels/titles aren't clipped
    fig.tight_layout()

    # Save with a high DPI and a tight bounding box (crop extra white space)
    buf = BytesIO()
    fig.savefig(buf, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    st.download_button(label="Download Plot as PDF", data=buf, file_name=f"{fig_name}.pdf", mime="application/pdf")


def draw_segment(ax, row, y, color):
    pt = PlotTheme()
    m = float(row[pt.avg])
    lo = float(m - row[pt.low])
    hi = float(row[pt.high] - m)
    ax.barh(y=y, width=m, height=pt.half_h, color=color, linewidth=0.7)
    add_errorbar(ax, x=m, y=y, lower=lo, upper=hi)

    return m, hi  # for label placement logic


def add_errorbar(ax: plt.Axes, x: float, y: float, lower: float, upper: float):
    ax.errorbar(x=x, y=y, xerr=[[lower], [upper]], fmt='none', ecolor='black', capsize=3, linewidth=0.7)
