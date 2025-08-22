import pandas as pd
import streamlit as st

from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.lgbm import LightGBM
from tabstar_paper.baselines.realmlp import RealMLP
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.leaderboard.ci import df_ci_for_model
from tabstar_paper.leaderboard.data import load_leaderboard_data, MODEL, DATASET, FOLD, IS_CLS, TEST_SCORE, IS_TUNED, \
    BASE_MODEL, DATASET_SIZE
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
    df = df[df[IS_CLS]] if task == 'CLS' else df[~df[IS_CLS]]
    model_options = sorted(set(df[MODEL]))
    if condition == "100K":
        irrelevant_models = {CatBoost.MODEL_NAME, XGBoost.MODEL_NAME, LightGBM.MODEL_NAME, RealMLP.MODEL_NAME}
        model_options = [m for m in model_options if m not in irrelevant_models]
    models = st.multiselect("Models", model_options, default=model_options)
    df = df[df[MODEL].isin(models)]
    df = df[[MODEL, DATASET, DATASET_SIZE, TEST_SCORE, FOLD, BASE_MODEL, IS_TUNED]]
    assert len(df) == len(df[[MODEL, DATASET, DATASET_SIZE, FOLD]].drop_duplicates()), "Duplicate runs found!"
    df = _add_norm_score(df)
    only_shared_datasets = st.checkbox("Only Shared Datasets", value=False)
    if only_shared_datasets:
        df = _get_only_shared_datasets(df)
    num_datasets = len(df[DATASET].unique())
    final_df = df_ci_for_model(df, score_key='norm')

    if condition == "10K":
        fig = plot_grouped_models(final_df, task, num_datasets)
    else:
        fig = plot_grouped_by_limit(final_df, task, num_datasets)
    st.pyplot(fig)
    fig_name = f"section_5_results_{task.lower()}_{condition.lower()}"
    download_st_fig(fig, fig_name=fig_name)
    return df, task

def _add_norm_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for f in ['min', 'max']:
        df[f] = df.groupby([DATASET, FOLD])[TEST_SCORE].transform(f)
    df['norm'] = (df[TEST_SCORE] - df['min']) / (df['max'] - df['min'])
    df.drop(columns=['max', 'min'], inplace=True)
    if not min(df['norm']) == 0 and max(df['norm']) == 1:
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
