import os

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from tabstar.constants import SEED

# TODO: understand whether this flag can make pre-training code slower
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from typing import Dict, Set, Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame
import torch

from tabstar.arch.config import E5_SMALL, D_MODEL

PCA_COMPONENTS = 30

class E5EmbeddingModel:

    def __init__(self):
        self.e5_model: Optional[SentenceTransformer] = None
        self.cached_embeddings: Dict[str, np.ndarray] = {}

    def load(self, device: torch.device):
        if self.e5_model is None:
            self.e5_model = SentenceTransformer(E5_SMALL, device=str(device))

    def embed(self, texts: List[str], device: torch.device) -> np.ndarray:
        self.encode_texts_in_batches(texts=texts, device=device)
        embeddings = np.array([self.cached_embeddings[text] for text in texts])
        assert embeddings.shape == (len(texts), D_MODEL), f"Got {embeddings.shape}, expected ({len(texts)}, {D_MODEL})"
        return embeddings

    def encode_texts_in_batches(self, texts: List[str], device: torch.device):
        self.load(device)
        new_texts = sorted(set(texts).difference(set(self.cached_embeddings)))
        embeddings = self.e5_model.encode(new_texts, normalize_embeddings=True)
        for text, embedding in zip(new_texts, embeddings):
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (D_MODEL,), f"Got {embedding.shape}, expected ({D_MODEL},)"
            self.cached_embeddings[text] = embedding


E5_CACHED_MODEL = E5EmbeddingModel()


def fit_text_encoders(x: DataFrame, text_features: Set[str], device: torch.device) -> Dict[str, PCA]:
    text_encoders = {}
    for col in text_features:
        # TODO: perhaps we should pre-compute all possible texts before the 'fit' call, for both train and test
        texts = x[col].astype(str).tolist()
        embeddings = E5_CACHED_MODEL.embed(texts=texts, device=device)
        pca_encoder = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        pca_encoder.fit(embeddings)
        text_encoders[str(col)] = pca_encoder
    return text_encoders


def transform_text_features(x: DataFrame, text_encoders: Dict[str, PCA], device: torch.device) -> DataFrame:
    for text_col, text_pca_encoder in text_encoders.items():
        texts = x[text_col].astype(str).tolist()
        embeddings = E5_CACHED_MODEL.embed(texts=texts, device=device)
        pca_cols = [f"{text_col}_pca_{i}" for i in range(text_pca_encoder.n_components)]
        pca_vec = text_pca_encoder.transform(embeddings)
        pca_df = pd.DataFrame(pca_vec, index=x.index, columns=pca_cols)
        cols_before = len(x.columns)
        x = x.drop(columns=[text_col])
        x = pd.concat([x, pca_df], axis=1)
        assert len(x.columns) == cols_before + PCA_COMPONENTS - 1
    return x
