from enum import StrEnum
from os import listdir
from os.path import dirname, join
from typing import Tuple

import streamlit as st
from pandas import DataFrame, read_csv, concat

from tabstar_paper.leaderboard.data.ci import df_ci_for_model
from tabstar_paper.leaderboard.data.keys import DATASET_SIZE, MODEL, DATASET, TEST_SCORE, FOLD, NORM_SCORE, IS_TUNED, \
    BASE_MODEL
from tabstar_paper.leaderboard.data.loading import get_base_model
from tabstar_paper.leaderboard.data.normalizing import add_norm_score
from tabstar_paper.leaderboard.filters.condition import Condition
from tabstar_paper.leaderboard.filters.models import filter_models
from tabstar_paper.leaderboard.filters.shared import do_shared_filtering
from tabstar_paper.leaderboard.filters.tasks import filter_by_task
from tabstar_paper.leaderboard.main_results import show_main_dataframe
from tabstar_paper.leaderboard.plots.main_10k import plot_grouped_models
from tabstar_paper.leaderboard.plots.utils import download_st_fig

EXPERIMENT_KEY = "Experiment"

class Experiment(StrEnum):
    SCALING = "Scale"
    NUMERICAL = "Numerical"
    UNFREEZE = "Unfreeze"
    BASELINE = "Baseline"

@st.cache_data
def load_experiments_data() -> DataFrame:
    return get_experiments_data()


def get_experiments_data():
    curr_dir = str(dirname(__file__))
    result_dir = curr_dir.replace('leaderboard/analysis', 'leaderboard/results_analysis')
    paths = [join(result_dir, f) for f in listdir(result_dir)]
    dfs = []
    for f in paths:
        df = read_csv(f)
        df[IS_TUNED] = df[MODEL].apply(lambda x: 'Tuned' in x)
        df[BASE_MODEL] = df[MODEL].apply(get_base_model)
        dfs.append(df)
    df = concat(dfs)
    return df


def get_experiment(model_name: str) -> str:
    for e in Experiment:
        if model_name.startswith(str(e.value)):
            return str(e.value)
    return Experiment.BASELINE

def filter_by_experiment(df: DataFrame) -> Tuple[DataFrame, str]:
    df[EXPERIMENT_KEY] = df[MODEL].apply(get_experiment)
    experiments = [e.value for e in Experiment if e != Experiment.BASELINE]
    experiment = st.selectbox("Select experiment", options=experiments, index=0)
    df = df[df[EXPERIMENT_KEY].apply(lambda x: x == experiment or x == Experiment.BASELINE)]
    return df, experiment


def display_experiments_info(leaderboard_df: DataFrame):
    st.markdown("## Experiments 🧪")
    leaderboard_df = leaderboard_df.copy()
    leaderboard_df = leaderboard_df[leaderboard_df[DATASET_SIZE] == 10000]
    experiment_df = load_experiments_data()
    df = concat([leaderboard_df, experiment_df], axis=0)
    df, task = filter_by_task(df, key="analysis")
    df = filter_models(df, condition=Condition.TEN_K, key="analysis")
    df[MODEL] = df[MODEL].apply(lambda x: x.split()[0])
    assert len(df) == len(df[[MODEL, DATASET, DATASET_SIZE, FOLD, IS_TUNED]].drop_duplicates()), "Duplicate runs found!"
    df, experiment = filter_by_experiment(df)
    df = df[[MODEL, DATASET, TEST_SCORE, FOLD, DATASET_SIZE, IS_TUNED, BASE_MODEL, EXPERIMENT_KEY]]
    df = add_norm_score(df)
    df = do_shared_filtering(df)
    num_datasets = len(df[DATASET].unique())
    final_df = df_ci_for_model(df, score_key=NORM_SCORE)
    fig = plot_grouped_models(final_df, num_datasets, task=task, base_model_key=BASE_MODEL)
    st.pyplot(fig)
    fig_name = f"section_6_analysis_{task.lower()}_{experiment.lower()}"
    download_st_fig(fig, fig_name=fig_name)
    st.divider()
    show_main_dataframe(df)