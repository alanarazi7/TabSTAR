import pandas as pd
import streamlit as st

from tabstar_paper.leaderboard.data.keys import DATASET, FOLD, NORM_SCORE, TEST_SCORE


def add_norm_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for f in ['min', 'max']:
        df[f] = df.groupby([DATASET, FOLD])[TEST_SCORE].transform(f)
    df[NORM_SCORE] = (df[TEST_SCORE] - df['min']) / (df['max'] - df['min'])
    df.drop(columns=['max', 'min'], inplace=True)
    if not min(df[NORM_SCORE]) == 0 and max(df[NORM_SCORE]) == 1:
        st.warning("Warning: Normalization might have failed. Check the min and max values.")
    return df
