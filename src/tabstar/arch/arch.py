from typing import Dict

import torch
from torch import Tensor
from transformers import AutoModel, PreTrainedModel

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
        self.numerical_fusion = NumericalFusion()
        self.tabular_encoder = InteractionEncoder()
        self.cls_head = PredictionHead()
        self.reg_head = PredictionHead()
        self.post_init()

    def forward(self, x_txt: Tensor, x_num: Tensor, tokenized: Dict[str, Tensor], d_output: int) -> Tensor:
        textual_embeddings = self.get_textual_embedding(x_txt, tokenized=tokenized)
        x_num = torch.tensor(x_num, dtype=textual_embeddings.dtype, device=textual_embeddings.device)
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

    def get_textual_embedding(self, x_txt: Tensor, tokenized: Dict[str, Tensor]) -> Tensor:
        text_batch = 128
        unique_texts = tokenized['input_ids'].size(0)
        x_txt_shape = x_txt.shape
        while text_batch > 1:
            try:
                return self.get_textual_embedding_in_batches(x_txt, tokenized=tokenized, text_batch=text_batch)
            except torch.cuda.OutOfMemoryError:
                text_batch //= 2
                clear_cuda_cache()
                print(f"🤯 Reduced text batch size to {text_batch} due to OOM for {x_txt_shape=}, {unique_texts=}")
        raise RuntimeError(f"🤯 OOM even with batch size 1! Consider reducing the model's batch size")

    def get_textual_embedding_in_batches(self, x_txt: Tensor, tokenized: Dict[str, Tensor], text_batch: int) -> Tensor:
        num_unique_texts = tokenized['input_ids'].size(0)
        embeddings = []
        for i in range(0, num_unique_texts, text_batch):
            batch_inputs = {k: v[i:i + text_batch] for k, v in tokenized.items()}
            outputs = self.text_encoder(**batch_inputs)
            # Take the [CLS] token representation
            embeddings.append(outputs.last_hidden_state[:, 0, :])
        embeddings = torch.cat(embeddings, dim=0)
        batch_size, seq_len = x_txt.shape
        embeddings = embeddings[x_txt]
        if not tuple(embeddings.shape) == (batch_size, seq_len, self.config.d_model):
            raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
        return embeddings
