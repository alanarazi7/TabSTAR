import pandas as pd
import streamlit as st
from pandas import DataFrame

from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.lgbm import LightGBM
from tabstar_paper.baselines.random_forest import RandomForest
from tabstar_paper.baselines.realmlp import RealMLP
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_BIG
from tabstar_paper.leaderboard.data.ci import df_ci_for_model
from tabstar_paper.leaderboard.data.highlighting import st_dataframe_highlight_extremes
from tabstar_paper.leaderboard.data.keys import (MODEL, DATASET, FOLD, IS_CLS, TEST_SCORE, IS_TUNED,
                                                 BASE_MODEL, DATASET_SIZE, NORM_SCORE)
from tabstar_paper.leaderboard.data.loading import load_leaderboard_data
from tabstar_paper.leaderboard.plots.main_100k import plot_grouped_by_limit
from tabstar_paper.leaderboard.plots.main_10k import plot_grouped_models
from tabstar_paper.leaderboard.plots.utils import download_st_fig


def display_main_results():
    df = load_leaderboard_data()
    possible_tasks = ['CLS', 'REG']
    task = st.selectbox("Task", possible_tasks, index=0)
    condition = st.selectbox("Condition", ["10K", "100K"], index=0)
    if condition == "10K":
        df = df[df['dataset_size'] == 10_000]
    else:
        text_names = {d.name for d in TEXTUAL_BIG}
        df = df[df[DATASET].apply(lambda d: d in text_names)]
    df = df[df[IS_CLS]] if task == 'CLS' else df[~df[IS_CLS]]
    model_options = sorted(set(df[MODEL]))
    if condition == "100K":
        default_models = {m.MODEL_NAME for m in [CatBoost, XGBoost, LightGBM, RealMLP, RandomForest]}
        model_options = [m for m in model_options if m not in default_models]
    models = st.multiselect("Models", model_options, default=model_options)
    df = df[df[MODEL].isin(models)]
    df = df[[MODEL, DATASET, DATASET_SIZE, TEST_SCORE, FOLD, BASE_MODEL, IS_TUNED]]
    assert len(df) == len(df[[MODEL, DATASET, DATASET_SIZE, FOLD]].drop_duplicates()), "Duplicate runs found!"
    df = _add_norm_score(df)
    only_shared_datasets = st.checkbox("Only Shared Datasets", value=False)
    if only_shared_datasets:
        df = _get_only_shared_datasets(df)
    num_datasets = len(df[DATASET].unique())
    final_df = df_ci_for_model(df, score_key=NORM_SCORE)

    if condition == "10K":
        fig = plot_grouped_models(final_df, task, num_datasets)
    else:
        fig = plot_grouped_by_limit(final_df, task, num_datasets)
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


def _add_norm_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for f in ['min', 'max']:
        df[f] = df.groupby([DATASET, FOLD])[TEST_SCORE].transform(f)
    df[NORM_SCORE] = (df[TEST_SCORE] - df['min']) / (df['max'] - df['min'])
    df.drop(columns=['max', 'min'], inplace=True)
    if not min(df[NORM_SCORE]) == 0 and max(df[NORM_SCORE]) == 1:
        st.warning("Warning: Normalization might have failed. Check the min and max values.")
    return df


def _get_only_shared_datasets(dataset_df: pd.DataFrame) -> pd.DataFrame:
    num_models = len(set(dataset_df[MODEL]))
    # count, for each dataset, how many distinct models have it
    ds_model_counts = (dataset_df.groupby(DATASET)[MODEL].nunique().reset_index(name="model_count"))
    # only keep datasets whose model_count == total number of models
    shared_ds = ds_model_counts.loc[ds_model_counts["model_count"] == num_models, DATASET]
    dataset_df = dataset_df[dataset_df[DATASET].isin(shared_ds)]
    return dataset_df
