from enum import StrEnum
from typing import Tuple

import streamlit as st
from pandas import DataFrame

from tabstar_paper.leaderboard.data.keys import DATASET, TASK


class TabularTask(StrEnum):
    CLS = "CLS"
    REG = "REG"


TASK2PRETTY = {
    TabularTask.CLS: "Classification",
    TabularTask.REG: "Regression"
}

def filter_by_task(df: DataFrame) -> Tuple[DataFrame, TabularTask]:
    df = add_task_col(df)
    task = st.selectbox("Task", [TabularTask.CLS, TabularTask.REG], index=0)
    df = df[df[TASK] == str(task)]
    return df, task


def add_task_col(df: DataFrame) -> DataFrame:
    df = df.copy()
    df[TASK] = df[DATASET].apply(lambda x: str(TabularTask.REG if x.startswith("REG") else TabularTask.CLS))
    return df