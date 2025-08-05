import torch
from skrub import TextEncoder

from tabstar.arch.config import E5_SMALL
from tabular.datasets.raw_dataset import RawDataset
from tabular.utils.processing import pd_concat_cols


def transform_texts_to_embeddings(raw: RawDataset, device: torch.device):
    # TODO: when PCA is on, we should actually to this with train-test-dev, otherwise we are leaking
    # this becomes very inconvenient when we are using Optuna with k-fold as it requires GPU, so neglecting
    # This actually might have inflated the results for the tree baselines, but unlikely to be dramatic
    model = E5_SMALL
    print(f"👅 Transforming text columns with {model}")
    encoder = TextEncoder(model_name=model, device=device)
    for col in raw.textual:
        text_col = raw.x[col]
        embedding_df = encoder.fit_transform(text_col)
        cols_before = len(raw.x.columns)
        raw.x.drop(columns=col, inplace=True)
        raw.x = pd_concat_cols([raw.x, embedding_df])
        assert len(raw.x.columns) == cols_before + 29
