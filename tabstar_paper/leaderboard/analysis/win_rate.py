import pandas as pd
import streamlit as st
from pandas import DataFrame, Series

from tabstar_paper.leaderboard.data.ci import get_var_ci
from tabstar_paper.leaderboard.data.keys import MODEL, DATASET, TEST_SCORE, FOLD, TASK
from tabstar_paper.leaderboard.filters.condition import filter_condition
from tabstar_paper.leaderboard.filters.models import filter_models
from tabstar_paper.leaderboard.filters.tasks import add_task_col, TabularTask

REF_SCORE = "ref_score"
WIN = "win"

def do_win_rate_analysis(df: DataFrame):
    st.markdown("## Win Rate Analysis ⚔️")
    df = df.copy()
    df = add_task_col(df)
    df, condition = filter_condition(df, key="win")
    df = filter_models(df, condition=condition, key="win")
    available_models = sorted(set(df[MODEL]))
    # TODO: make this not hardcoded
    sota_model = "TabSTAR-Unlimit 🌟" if "TabSTAR-Unlimit 🌟" in available_models else "TabSTAR 🌟"
    index = available_models.index(sota_model) if sota_model in available_models else 0
    reference = st.selectbox("Select reference model", available_models, index=index)
    df = df[[DATASET, TEST_SCORE, FOLD, MODEL, TASK]]
    ref_df = df[df[MODEL] == reference]
    ref_df = ref_df.rename(columns={TEST_SCORE: REF_SCORE}).drop(columns=[MODEL])
    other_models = [m for m in available_models if m != reference]
    rows = []
    for model in other_models:
        for task in TabularTask:
            model_df = df[df[MODEL] == model][[DATASET, TEST_SCORE, FOLD, TASK]]
            model_df = model_df[model_df[TASK] == str(task)]
            compare_df = pd.merge(model_df, ref_df, on=[DATASET, FOLD, TASK])
            compare_df[WIN] = compare_df.apply(get_win_score, axis=1)
            if len(compare_df) == 0:
                continue
            ci = get_var_ci(compare_df, score_key=WIN)
            avg = ci.avg * 100
            half = ci.half * 100
            ci_str = f"{avg:.1f} ± {half:.1f}"
            if ci_str == "100.0 ± 0.0":
                ci_str = "100 ± 0.0"
            rows.append({MODEL: model, TASK: task, 'ci_str': ci_str})
    df = pd.DataFrame(rows)
    pivot_df = df.pivot(index=MODEL, columns=TASK, values='ci_str').reset_index()
    pivot_df = pivot_df.sort_values(MODEL).fillna("")
    st.dataframe(pivot_df)


def get_win_score(row: Series) -> float:
    if row[REF_SCORE] > row[TEST_SCORE]:
        return 1.0
    elif row[REF_SCORE] == row[TEST_SCORE]:
        return 0.5
    else:
        return 0.0