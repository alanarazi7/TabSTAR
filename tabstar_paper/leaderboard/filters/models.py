import streamlit as st
from pandas import DataFrame

from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.lgbm import LightGBM
from tabstar_paper.baselines.random_forest import RandomForest
from tabstar_paper.baselines.realmlp import RealMLP
from tabstar_paper.baselines.tabm import TabM
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.leaderboard.data.keys import MODEL
from tabstar_paper.leaderboard.filters.condition import Condition


DEFAULT_MODELS = {m.MODEL_NAME for m in [CatBoost, XGBoost, LightGBM, RealMLP, RandomForest, TabM]}



def filter_models(df: DataFrame, condition: Condition, key: str) -> DataFrame:
    model_options = sorted(set(df[MODEL]))
    if condition == Condition.UNLIMITED:
        model_options = [m for m in model_options if m not in DEFAULT_MODELS]
    models = st.multiselect("Models", model_options, default=model_options,
                            key=f"model_filter_{key}")
    df = df[df[MODEL].isin(models)]
    return df