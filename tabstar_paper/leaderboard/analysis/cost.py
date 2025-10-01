from os import listdir
from os.path import join, dirname

import streamlit as st
from pandas import DataFrame, read_csv, concat

from tabstar_paper.leaderboard.data.keys import MODEL, DATASET, FOLD

TRAIN_TIME = "Train Time"
TRAIN_GPU = "Train GPU"
TRAIN_CPU = "Train CPU"
INFER_TIME = "Infer Time"
INFER_GPU = "Infer GPU"
INFER_CPU = "Infer CPU"


@st.cache_data
def load_costs_data() -> DataFrame:
    return get_costs_data()


def get_costs_data():
    curr_dir = str(dirname(__file__))
    result_dir = curr_dir.replace('leaderboard/analysis', 'leaderboard/results_costs')
    paths = [join(result_dir, f) for f in listdir(result_dir)]
    dfs = []
    for f in paths:
        df = read_csv(f)
        dfs.append(df)
    df = concat(dfs)
    return df


def display_cost_info():
    st.markdown("## Costs 💰")
    df = load_costs_data()
    df = df.rename(columns={'train_wall_time_s': TRAIN_TIME,
                            'train_peak_gpu_gb': TRAIN_GPU,
                            'train_peak_cpu_gb': TRAIN_CPU,
                            'inference_wall_time_s': INFER_TIME,
                            'inference_peak_gpu_gb': INFER_GPU,
                            'inference_peak_cpu_gb': INFER_CPU
                            })
    agg_df = df.groupby(MODEL).agg({TRAIN_TIME: 'median',
                                    TRAIN_GPU: 'median',
                                    TRAIN_CPU: 'median',
                                    INFER_TIME: 'median',
                                    INFER_GPU: 'median',
                                    INFER_CPU: 'median'
                                    }).reset_index()
    agg_df = agg_df.round(2)
    st.dataframe(agg_df)
    st.divider()
    st.markdown("## Run Time ⏱️")
    df = df[[MODEL, DATASET, TRAIN_TIME, FOLD]]
    df = df[df[MODEL].apply(lambda m: 'cpu' not in m.lower())]
    pivot_df = df.pivot_table(index=[DATASET], columns=[MODEL], values=TRAIN_TIME, aggfunc='median').reset_index()
    st.dataframe(pivot_df)