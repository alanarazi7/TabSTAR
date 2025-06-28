import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedModel

from tabstar.arch.config import TabStarConfig, E5_SMALL
from tabstar.arch.interaction import InteractionEncoder
from tabstar.arch.fusion import NumericalFusion
from tabstar.arch.prediction import PredictionHead
from tabstar.training.devices import clear_cuda_cache


class TabStarModel(PreTrainedModel):
    config_class = TabStarConfig

    def __init__(self, config: TabStarConfig):
        super().__init__(config)
        self.text_encoder = AutoModel.from_pretrained(E5_SMALL)
        self.tokenizer = AutoTokenizer.from_pretrained(E5_SMALL)
        self.numerical_fusion = NumericalFusion()
        self.tabular_encoder = InteractionEncoder()
        self.cls_head = PredictionHead()
        self.reg_head = PredictionHead()
        self.post_init()

    def forward(self, x_txt: np.ndarray, x_num: np.ndarray, d_output: int) -> Tensor:
        textual_embeddings = self.get_textual_embedding(x_txt)
        embeddings = self.numerical_fusion(textual_embeddings=textual_embeddings, x_num=x_num)
        encoded = self.tabular_encoder(embeddings)
        target_tokens = encoded[:, :d_output]
        if d_output == 1:
            target_scores = self.reg_head(target_tokens)
        else:
            target_scores = self.cls_head(target_tokens)
        target_scores = target_scores.squeeze(dim=-1)
        assert tuple(target_scores.shape) == (x_txt.shape[0], d_output)
        return target_scores

    def get_textual_embedding(self, x_txt: np.array) -> Tensor:
        text_batch_size = 128
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
        if not tuple(embeddings.shape) == (batch_size, seq_len, self.config.d_model):
            raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
        return embeddings
