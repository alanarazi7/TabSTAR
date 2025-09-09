import json
from os import listdir
from os.path import dirname, join

import numpy as np
import pandas as pd
import streamlit as st

from tabstar_paper.leaderboard.data.keys import IS_TUNED, BASE_MODEL, MODEL, DATASET_SIZE, DATASET, UNLIMIT, \
    FOLD, BEST_VAL_LOSS


@st.cache_data
def load_leaderboard_data() -> pd.DataFrame:
    return get_results_data()


def get_results_data():
    curr_dir = str(dirname(__file__))
    result_dir = curr_dir.replace('leaderboard/data', 'leaderboard/results')
    paths = [join(result_dir, f) for f in listdir(result_dir)]
    dfs = []
    for f in paths:
        df = pd.read_csv(f)
        if 'carte' in f:
            df = aggregate_carte_data(df)
        df[IS_TUNED] = df[MODEL].apply(lambda x: 'Tuned' in x)
        df[BASE_MODEL] = df[MODEL].apply(_get_base_model)
        if set(df[DATASET_SIZE]) == {100000}:
            df[MODEL] = df[MODEL].apply(_add_unlimit_suffix)
        dfs.append(df)
    df = pd.concat(dfs)
    return df

def _add_unlimit_suffix(s: str) -> str:
    model_name, emoji = s.split()
    return f"{model_name}{UNLIMIT} {emoji}"

def _get_base_model(s: str) -> str:
    model_name, emoji = s.split()
    return model_name.replace("-Tuned", "")

def aggregate_carte_data(df: pd.DataFrame) -> pd.DataFrame:
    assert len(df) == len(df[[DATASET, FOLD, 'carte_lr_idx']].drop_duplicates())
    df[BEST_VAL_LOSS] = df[BEST_VAL_LOSS].apply(lambda l: np.mean(json.loads(l)))
    best_df = df.loc[df.groupby([DATASET, FOLD])[BEST_VAL_LOSS].idxmin()]
    assert len(best_df) == len(best_df[[DATASET, FOLD]].drop_duplicates())
    return best_df
