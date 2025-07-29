import os
# TODO: understand whether this flag can make pre-training code slower
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock

from typing import Dict, Set

import pandas as pd
import torch
from pandas import DataFrame
from skrub import TextEncoder

from tabstar.arch.config import E5_SMALL


# TODO: there's a little gotcha in the way this is implemented, as there's no decoupling between encoding and PCA.
# In realistic scenarios where we tune the model over and over, the efficient way would be to fit the encoder once,
# cache the embeddings, and then apply PCA on the cached embeddings over and over for every train split in every run.
def fit_text_encoders(x: DataFrame, text_features: Set[str], device: torch.device) -> Dict[str, TextEncoder]:
    text_encoders = {}
    for col, dtype in x.dtypes.items():
        if col not in text_features:
            continue
        encoder = TextEncoder(model_name=E5_SMALL, device=device)
        encoder.fit(x[col])
        text_encoders[str(col)] = encoder
    return text_encoders

def transform_text_features(x: DataFrame, text_encoders: Dict[str, TextEncoder]) -> DataFrame:
    for text_col, text_encoder in text_encoders.items():
        embedding_df = text_encoder.transform(x[text_col])
        cols_before = len(x.columns)
        x = x.drop(columns=text_col)
        embedding_df = embedding_df.set_index(x.index)
        x = pd.concat([x, embedding_df], axis=1)
        assert len(x.columns) == cols_before + text_encoder.n_components - 1
    return x
