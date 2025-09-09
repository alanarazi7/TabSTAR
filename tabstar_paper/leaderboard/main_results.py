import pandas as pd
import streamlit as st
from pandas import DataFrame

from tabstar_paper.leaderboard.data.ci import df_ci_for_model
from tabstar_paper.leaderboard.data.highlighting import st_dataframe_highlight_extremes
from tabstar_paper.leaderboard.data.keys import (MODEL, DATASET, FOLD, TEST_SCORE, IS_TUNED,
                                                 BASE_MODEL, DATASET_SIZE, NORM_SCORE)
from tabstar_paper.leaderboard.data.normalizing import add_norm_score
from tabstar_paper.leaderboard.filters.condition import filter_condition, Condition
from tabstar_paper.leaderboard.filters.models import filter_models
from tabstar_paper.leaderboard.filters.shared import filter_shared_datasets
from tabstar_paper.leaderboard.filters.tasks import filter_by_task
from tabstar_paper.leaderboard.plots.main_100k import plot_grouped_by_limit
from tabstar_paper.leaderboard.plots.main_10k import plot_grouped_models
from tabstar_paper.leaderboard.plots.utils import download_st_fig


def display_main_results(df: DataFrame):
    df = df.copy()
    df, task = filter_by_task(df)
    df, condition = filter_condition(df, key="main")
    df = filter_models(df, condition=condition, key="main")
    df = df[[MODEL, DATASET, DATASET_SIZE, TEST_SCORE, FOLD, BASE_MODEL, IS_TUNED]]
    assert len(df) == len(df[[MODEL, DATASET, DATASET_SIZE, FOLD]].drop_duplicates()), "Duplicate runs found!"
    df = add_norm_score(df)
    df = filter_shared_datasets(df)
    num_datasets = len(df[DATASET].unique())
    final_df = df_ci_for_model(df, score_key=NORM_SCORE)

    if condition == Condition.TEN_K:
        fig = plot_grouped_models(final_df, num_datasets, task=task)
    else:
        fig = plot_grouped_by_limit(final_df, num_datasets, task=task)
    st.pyplot(fig)
    fig_name = f"section_5_results_{task.lower()}_{condition.lower()}"
    download_st_fig(fig, fig_name=fig_name)
    st.divider()
    show_main_dataframe(df)
    st.divider()
    show_dataset_level_breakdown(df)

def show_main_dataframe(df: DataFrame):
    model_df_ci = df_ci_for_model(df, score_key=NORM_SCORE)[[MODEL, 'low', 'high']]
    model_df = df.groupby([MODEL]).agg(score=(TEST_SCORE, 'mean'),
                                       norm_score=(NORM_SCORE, 'mean'),
                                       total_runs=(FOLD, 'count')).reset_index()
    model_df = model_df.merge(model_df_ci, on=MODEL)
    model_df = model_df.sort_values(by=NORM_SCORE, ascending=False)
    model_df = model_df[[MODEL, NORM_SCORE, 'low', 'high', 'total_runs', 'score']]
    st.dataframe(model_df)

def show_dataset_level_breakdown(df: pd.DataFrame):
    st.markdown("## Dataset Level Breakdown 🔍")
    c1, c2, c3 = st.columns(3)
    with c1:
        normalize = st.checkbox("Normalize Scores", value=False)
    with c2:
        show_by_run = st.checkbox("Show by run", value=False)
    with c3:
        remove_nulls = st.checkbox("Remove Nulls", value=False)
    indices = [DATASET]
    if show_by_run:
        indices.append(FOLD)
    score_field = NORM_SCORE if normalize else TEST_SCORE
    pivot_df = pd.pivot_table(df, index=indices, columns=MODEL, values=score_field).reset_index()
    if remove_nulls:
        pivot_df = pivot_df.dropna(how='any')
    st_dataframe_highlight_extremes(pivot_df, exclude_columns=[FOLD])
