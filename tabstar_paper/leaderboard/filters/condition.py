from enum import StrEnum
from typing import Tuple

from pandas import DataFrame
import streamlit as st

from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_BIG
from tabstar_paper.leaderboard.data.keys import DATASET


class Condition(StrEnum):
    TEN_K = "10K"
    UNLIMITED = "100K"



def filter_condition(df: DataFrame) -> Tuple[DataFrame, Condition]:
    condition = st.selectbox("Condition", [Condition.TEN_K, Condition.UNLIMITED], index=0)
    if condition == Condition.TEN_K:
        df = df[df['dataset_size'] == 10_000]
    else:
        text_names = {d.name for d in TEXTUAL_BIG}
        df = df[df[DATASET].apply(lambda d: d in text_names)]
    return df, condition