from typing import List

import numpy as np
import torch
from torch import Tensor
from transformers import BertModel, AutoTokenizer

from tabstar.arch.config import E5_SMALL
from tabstar.training.devices import clear_cuda_cache


class TabSTAREmbedder:

    def __init__(self, text_encoder: BertModel, device: torch.device):
        self.text_encoder = text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(E5_SMALL)
        self.device = device
        self.d_model = text_encoder.config.hidden_size

    def embed(self, x_txt: List[str]) -> Tensor:
        orig_len = len(x_txt)
        x_txt = np.array(x_txt).reshape(1, -1)
        assert x_txt.shape == (1, orig_len), f"Unexpected shape: {x_txt.shape}, expected (1, {len(x_txt)})"
        text_batch_size = 1024
        while text_batch_size > 1:
            try:
                return self.get_textual_embedding_in_batches(x_txt, text_batch_size=text_batch_size)
            except torch.cuda.OutOfMemoryError:
                text_batch_size //= 2
                clear_cuda_cache()
                print(f"🤯 Reducing batch size to {text_batch_size} due to OOM")
        raise RuntimeError(f"🤯 OOM even with batch size 1!")

    def get_textual_embedding_in_batches(self, x_txt: np.array, text_batch_size: int) -> Tensor:
        # Get unique texts and mapping indices
        unique_texts, inverse_indices = np.unique(x_txt, return_inverse=True)
        num_unique_texts = len(unique_texts)
        embeddings = []
        for i in range(0, num_unique_texts, text_batch_size):
            batch_texts = unique_texts[i:i + text_batch_size].tolist()
            inputs = self.tokenizer(batch_texts, padding=True, return_tensors='pt', truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.text_encoder(**inputs)
            # Take the [CLS] token representation
            embeddings.append(outputs.last_hidden_state[:, 0, :])
        embeddings = torch.cat(embeddings, dim=0)
        inverse_indices = torch.tensor(inverse_indices, dtype=torch.long, device=embeddings.device)
        # Map the unique embeddings back to the original positions and reshape to match the original dimension
        batch_size, seq_len = x_txt.shape
        embeddings = embeddings[inverse_indices].view(batch_size, seq_len, -1)
        if not tuple(embeddings.shape) == (batch_size, seq_len, self.d_model):
            raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
        return embeddings

