from os import listdir
from os.path import dirname, join

from pandas import concat, DataFrame, read_csv
import streamlit as st

from tabstar_paper.leaderboard.data.keys import MODEL, DATASET, OPTUNA_TRIALS
from tabstar_paper.leaderboard.data.loading import add_suffix_if_unlimited


def load_trial_data() -> DataFrame:
    curr_dir = str(dirname(__file__))
    tuned_dir = join(curr_dir, 'results_tuned')
    paths = [join(tuned_dir, f) for f in listdir(tuned_dir)]
    dfs = []
    for f in paths:
        df = read_csv(f)
        add_suffix_if_unlimited(df)
        dfs.append(df)
    df = concat(dfs)
    return df


def display_metadata_info():
    st.markdown("## Number of Hyperparameter Trials 🧪")
    df = load_trial_data()
    df = df.pivot_table(index=[DATASET], columns=[MODEL], values=[OPTUNA_TRIALS])
    st.dataframe(df)