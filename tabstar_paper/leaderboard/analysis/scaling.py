import matplotlib.pyplot as plt
import streamlit as st
from pandas import DataFrame

from tabstar_paper.leaderboard.data.keys import MODEL, DATASET, TEST_SCORE, TASK
from tabstar_paper.leaderboard.filters.tasks import add_task_col
from tabstar_paper.leaderboard.plots.utils import download_st_fig


def do_scaling_laws_plot(df: DataFrame):
    st.markdown("## Analysis of Scaling Laws ⚖️")
    df = preprocess_df(df)
    st.divider()
    fig = plot_df(df)
    st.pyplot(fig)
    download_st_fig(fig, fig_name="section_6_analysis_scaling_laws", is_two=False)
    st.divider()


def preprocess_df(df: DataFrame) -> DataFrame:
    df = df.copy()
    df = add_task_col(df)
    df = df[df[MODEL].apply(lambda m: m.startswith("Scale"))]
    df = df.groupby([MODEL, DATASET, TASK]).agg({TEST_SCORE: 'mean'})
    df = df.groupby([MODEL, TASK]).mean().reset_index()
    df[MODEL] = df[MODEL].apply(lambda x: int(x.replace("Scale-", "")))
    return df


# Like all plots code, this was done much with AI assistance
def plot_df(df) -> plt.Figure:
    order = [0, 16, 64, 256]
    pv = df.pivot_table(index=MODEL, columns=TASK, values=TEST_SCORE).reindex(order)

    x_vals = list(pv.index)
    x_labels = [str(m) for m in x_vals]
    x_plot = [m if m > 0 else 1 for m in x_vals]  # avoid log(0) while keeping label "0"

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "figure.dpi": 300
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # --- Classification ---
    axes[0].plot(x_plot, pv["CLS"], color="steelblue", marker="o", linewidth=3, markersize=9)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([1, 16, 64, 256])
    axes[0].set_xticklabels(x_labels)
    axes[0].set_title("Classification")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.8, 0.9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Regression ---
    axes[1].plot(x_plot, pv["REG"], color="seagreen", marker="s", linewidth=3, markersize=9)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([1, 16, 64, 256])
    axes[1].set_xticklabels(x_labels)
    axes[1].set_title("Regression")
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(0.7, 0.8)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # --- Shared X label ---
    fig.text(0.5, 0.02, "Pretraining datasets", ha="center", fontsize=18)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    return fig