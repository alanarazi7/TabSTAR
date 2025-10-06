from os import listdir
from os.path import join, dirname
from typing import List

import streamlit as st
from pandas import DataFrame, read_csv, concat

from tabstar_paper.leaderboard.analysis.feature_runtime import get_dataset_feature_group
from tabstar_paper.leaderboard.data.keys import MODEL, DATASET, FOLD

TRAIN_TIME = "Train Time"
TRAIN_GPU = "Train GPU"
TRAIN_CPU = "Train CPU"
INFER_TIME = "Infer Time"
INFER_GPU = "Infer GPU"
INFER_CPU = "Infer CPU"

FEAT_GRP = "Feature Group"

@st.cache_data
def load_costs_data() -> DataFrame:
    return get_costs_data(new_folder="results_costs")

@st.cache_data
def load_feature_costs_data() -> DataFrame:
    return get_costs_data(new_folder="results_features")


def get_costs_data(new_folder: str):
    curr_dir = str(dirname(__file__))
    result_dir = curr_dir.replace('leaderboard/analysis', f'leaderboard/{new_folder}')
    paths = [join(result_dir, f) for f in listdir(result_dir)]
    dfs = []
    for f in paths:
        df = read_csv(f)
        dfs.append(df)
    df = concat(dfs)
    df = df.rename(columns={'train_wall_time_s': TRAIN_TIME,
                            'train_peak_gpu_gb': TRAIN_GPU,
                            'train_peak_cpu_gb': TRAIN_CPU,
                            'inference_wall_time_s': INFER_TIME,
                            'inference_peak_gpu_gb': INFER_GPU,
                            'inference_peak_cpu_gb': INFER_CPU
                            })
    return df


def display_cost_info():
    st.markdown("## Costs 💰")
    df = load_costs_data()
    agg_df = get_median_performance(df, grp_fields=[MODEL])
    agg_df = agg_df.round(2)
    st.dataframe(agg_df)
    st.divider()
    st.markdown("## Run Time ⏱️")
    df = df[[MODEL, DATASET, TRAIN_TIME, FOLD]]
    df = df[df[MODEL].apply(lambda m: 'cpu' not in m.lower())]
    pivot_df = df.pivot_table(index=[DATASET], columns=[MODEL], values=TRAIN_TIME, aggfunc='median').reset_index()
    st.dataframe(pivot_df)
    st.divider()
    st.markdown("## TabSTAR Feature Number Analysis 📊")
    df = load_feature_costs_data()
    df[FEAT_GRP] = df[DATASET].apply(get_dataset_feature_group)
    df = get_median_performance(df, grp_fields=[MODEL, DATASET, FEAT_GRP])
    df = get_median_performance(df, grp_fields=[FEAT_GRP])
    st.dataframe(df)


def get_median_performance(df: DataFrame, grp_fields: List[str]) -> DataFrame:
    agg_dict = {k: 'median' for k in [TRAIN_TIME, TRAIN_GPU, TRAIN_CPU, INFER_TIME, INFER_GPU, INFER_CPU]}
    return df.groupby(grp_fields).agg(agg_dict).reset_index()
