import pandas as pd
import streamlit as st

from tabstar_paper.leaderboard.data.keys import DATASET, MODEL


def filter_shared_datasets(df: pd.DataFrame) -> pd.DataFrame:
    only_shared_datasets = st.checkbox("Only Shared Datasets", value=False)
    if not only_shared_datasets:
        return df
    num_models = len(set(df[MODEL]))
    # count, for each dataset, how many distinct models have it
    ds_model_counts = (df.groupby(DATASET)[MODEL].nunique().reset_index(name=f"{MODEL}_count"))
    # only keep datasets whose model_count == total number of models
    shared_ds = ds_model_counts.loc[ds_model_counts["model_count"] == num_models, DATASET]
    df = df[df[DATASET].isin(shared_ds)]
    return df
