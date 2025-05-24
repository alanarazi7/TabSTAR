from typing import Dict

import numpy as np
import torch
from tabular.tabstar.params.constants import E5_SMALL, E5_LAYERS
from torch import Tensor
from transformers import AutoModel, PreTrainedModel

from tabular.tabstar.arch.numerical_fusion import NumericalFusion
from tabular.tabstar.arch.prediction_head import TabularPredictionHead
from tabular.tabstar.arch.encoder_backbone import TabularEncoderBackbone
from tabular.tabstar.params.config import TabStarConfig
from tabular.preprocessing.tokenization import tokenize
from tabular.utils.deep import get_last_layers_num
from tabular.utils.utils import cprint, verbose_print


# TODO: Consider merging "NumericalFusion, TabularEncoderBackbone" into this class directly
# We need to Ensure that all custom modules (e.g. numerica fusion) are either integrated
# into our main repository or properly registered so that they’re available when loading from the Hub.
class TabStarModel(PreTrainedModel):
    config_class = TabStarConfig

    def __init__(self, config: TabStarConfig):
        super().__init__(config)
        self.text_encoder = AutoModel.from_pretrained(E5_SMALL)
        assert config.d_model == self.text_encoder.config.hidden_size, "Block mismatch!"

        self.numerical_fusion = NumericalFusion(config=config)
        self.tabular_encoder = TabularEncoderBackbone(num_layers=config.num_layers, d_model=config.d_model)
        self.cls_head = TabularPredictionHead(input_size=config.d_model)
        self.reg_head = TabularPredictionHead(input_size=config.d_model)

        self.text_encoder_batch_size: Dict[str, int] = {}

        # TODO: (Optional: call post_init hook if you have any weight initialization logic)
        self.post_init()

    def forward(self, x_txt: np.ndarray, x_num: np.ndarray, sid: str, d_output: int) -> Tensor:
        # TODO: we should use Tensors and not np.arrays as inputs
        textual_embeddings = self.get_textual_embedding(x_txt, sid=sid)
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

    def get_textual_embedding(self, x_txt: np.array, sid: str) -> Tensor:
        text_batch_size = self.text_encoder_batch_size.setdefault(sid, 128)
        while text_batch_size > 1:
            try:
                return self.get_textual_embedding_in_batches(x_txt, text_batch_size=text_batch_size)
            except torch.cuda.OutOfMemoryError as oom:
                text_batch_size //= 2
                self.text_encoder_batch_size[sid] = text_batch_size
                torch.cuda.empty_cache()
                cprint(f"Reducing batch size to {text_batch_size} for dataset {sid} due to OOM: {oom}")
        raise RuntimeError(f"OOM even with batch size 1 for {sid}!")

    def get_textual_embedding_in_batches(self, x_txt: np.array, text_batch_size: int) -> Tensor:
        # Get unique texts and mapping indices
        unique_texts, inverse_indices = np.unique(x_txt, return_inverse=True)
        num_unique_texts = len(unique_texts)
        embeddings = []
        verbose_print(f"Unique texts: {num_unique_texts}, Text Batch size: {text_batch_size}, Input: {x_txt.shape}")
        for i in range(0, num_unique_texts, text_batch_size):
            batch_texts = unique_texts[i:i + text_batch_size].tolist()
            inputs = tokenize(batch_texts, device=self.device)
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

    def unfreeze_textual_encoder_layers(self):
        blocks = [block_name for block_name, _ in self.text_encoder.named_children()]
        assert blocks == ['embeddings', 'encoder', 'pooler'], f"Unexpected block structure: {blocks}"
        assert self.text_encoder.config.num_hidden_layers == E5_LAYERS
        assert 0 <= self.config.unfreeze_layers <= E5_LAYERS
        if self.config.unfreeze_layers == 0:
            self._freeze_the_whole_encoder()
        else:
            self._unfreeze_pooler_and_num_layers()
        total_params = sum(p.numel() for p in self.text_encoder.parameters())
        unfrozen_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        cprint(f"🥶 Unfroze {unfrozen_params:,} out of {total_params:,} parameters!")

    def _freeze_the_whole_encoder(self):
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        cprint(f"🥶 Freezing the whole text encoder - including the pooling!")

    def _unfreeze_pooler_and_num_layers(self):
        assert 0 <= self.config.unfreeze_layers <= E5_LAYERS
        for name, param in self.text_encoder.pooler.named_parameters():
            param.requires_grad = True
        for name, param in self.text_encoder.embeddings.named_parameters():
            param.requires_grad = False
        to_unfreeze = get_last_layers_num(to_unfreeze=self.config.unfreeze_layers)
        cprint(f"🥶 Planning to unfreeze {self.config.unfreeze_layers}/{E5_LAYERS} layers: {to_unfreeze}!")
        layer_prefixes = [f'layer.{i}.' for i in to_unfreeze]
        for name, param in self.text_encoder.encoder.named_parameters():
            if any(name.startswith(prefix) for prefix in layer_prefixes):
                param.requires_grad = True
            else:
                param.requires_grad = False

