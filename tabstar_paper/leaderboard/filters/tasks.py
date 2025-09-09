from enum import StrEnum
from typing import Tuple

import streamlit as st
from pandas import DataFrame

from tabstar_paper.leaderboard.data.keys import DATASET


class TabularTask(StrEnum):
    CLS = "CLS"
    REG = "REG"


TASK2PRETTY = {
    TabularTask.CLS: "Classification",
    TabularTask.REG: "Regression"
}

def filter_by_task(df: DataFrame) -> Tuple[DataFrame, TabularTask]:
    reg_cond = df[DATASET].apply(lambda x: x.startswith(TabularTask.REG))
    task = st.selectbox("Task", [TabularTask.CLS, TabularTask.REG], index=0)
    if task == TabularTask.REG:
        df = df[reg_cond]
    else:
        df = df[~reg_cond]
    return df, task