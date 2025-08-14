import os

from sklearn.decomposition import PCA

from tabstar.constants import SEED

# TODO: understand whether this flag can make pre-training code slower
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from typing import Dict, Set, Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from transformers import BertModel, BertTokenizerFast, AutoModel, AutoTokenizer

from tabstar.arch.config import E5_SMALL, D_MODEL

PCA_COMPONENTS = 30
BATCH_SIZE = 32

class E5EmbeddingModel:

    def __init__(self):
        self.model: Optional[BertModel] = None
        self.tokenizer: Optional[BertTokenizerFast] = None
        self.cached_embeddings: Dict[str, np.ndarray] = {}

    def load(self):
        if self.model is None:
            self.model = AutoModel.from_pretrained(E5_SMALL)
            self.model.eval()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(E5_SMALL)

    def embed(self, texts: List[str], device: torch.device) -> np.ndarray:
        self.encode_texts_in_batches(texts=texts, device=device)
        embeddings = np.array([self.cached_embeddings[text] for text in texts])
        assert embeddings.shape == (len(texts), D_MODEL), f"Got {embeddings.shape}, expected ({len(texts)}, {D_MODEL})"
        return embeddings

    def encode_texts_in_batches(self, texts: List[str], device: torch.device):
        self.load()
        self.model.to(device)
        new_texts = sorted(set(texts).difference(set(self.cached_embeddings)))
        for i in range(0, len(new_texts), BATCH_SIZE):
            batch_texts = new_texts[i:i + BATCH_SIZE]
            tokenized = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model(**tokenized)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            for text, embedding in zip(batch_texts, batch_embeddings):
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (D_MODEL,), f"Got {embedding.shape}, expected ({D_MODEL},)"
                self.cached_embeddings[text] = embedding


E5_CACHED_MODEL = E5EmbeddingModel()


def fit_text_encoders(x: DataFrame, text_features: Set[str], device: torch.device) -> Dict[str, PCA]:
    text_encoders = {}
    for col, dtype in x.dtypes.items():
        if col not in text_features:
            continue
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
