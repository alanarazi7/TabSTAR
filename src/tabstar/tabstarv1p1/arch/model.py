import torch
from torch import nn, Tensor
from transformers import AutoModel, BatchEncoding, PreTrainedModel

from tabstar.tabstarv1p1.training.devices import clear_cuda_cache

from tabstar.tabstarv1p1.arch.interaction import InteractionEncoder
from tabstar.tabstarv1p1.arch.naive_fusion import NaiveFusion
from tabstar.tabstarv1p1.arch.prediction import DualRegressionHead, PredictionHead
from tabstar.tabstarv1p1.arch.config import TabStarConfig
from tabstar.tabstarv1p1.preprocessing.tokenization import trim_batch_to_max_size
from tabstar.tabstarv1p1.arch.numerical_encoder import NumericalEncoder
from tabstar.tabstarv1p1.training.loss import mask_predictions
from tabstar.tabstarv1p1.training.batching import TabSTARBatch
from tabstar.tabstarv1p1.training.devices import get_precision


class TabSTARModel(PreTrainedModel):
    config_class = TabStarConfig

    def __init__(self, config: TabStarConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.d_num = config.d_num
        self.text_batch_size = config.text_batch_size
        self.text_encoder = AutoModel.from_pretrained(self.config.text_model)
        self.numerical_encoder = NumericalEncoder(d_model=self.d_model, d_num=self.d_num)
        self.fusion = NaiveFusion(d_model=self.d_model)
        self.tabular_encoder = InteractionEncoder(num_layers=config.interaction_layers, d_model=self.d_model)
        self.cls_head = PredictionHead(d_model=self.d_model)
        self.reg_dist_bins = config.reg_dist_bins
        self.reg_head = DualRegressionHead(d_model=self.d_model, reg_dist_bins=config.reg_dist_bins)
        self.post_init()

    def forward(self, batch: TabSTARBatch) -> Tensor:
        text_embeddings = self.get_textual_embeddings(x_txt=batch.x_txt, tokenized=batch.tokenized)
        numerical_embeddings = self.numerical_encoder(x_num=batch.x_num)
        embeddings = self.fusion(textual_embeddings=text_embeddings, num_embeddings=numerical_embeddings)
        encoded = self.tabular_encoder(embeddings, feat_mask=batch.feature_mask)
        targets = self.get_predictions(batch=batch, encoded=encoded)
        targets = mask_predictions(targets, d_outputs=batch.d_outputs)
        return targets

    def get_textual_embeddings(self, x_txt: Tensor, tokenized: BatchEncoding) -> Tensor:
        # Local to this call, not persisted on self: an OOM here must not permanently shrink
        # every later call on this model (e.g. a reused base model scoring many datasets).
        text_batch_size = self.text_batch_size
        while True:
            try:
                return self.get_textual_embeddings_in_batches(x_txt, tokenized=tokenized, text_batch_size=text_batch_size)
            except torch.cuda.OutOfMemoryError:
                # Still an OOM, so the trainer can split the batch's rows, whose text activations it keeps for backward.
                if text_batch_size == 1:
                    raise
            # Retried outside the except block, so the failed attempt's tensors can be freed.
            text_batch_size //= 2
            clear_cuda_cache()
            print(f"🤯 Reducing batch size to {text_batch_size} due to OOM")

    def get_textual_embeddings_in_batches(self, x_txt: Tensor, tokenized: BatchEncoding, text_batch_size: int) -> Tensor:
        n_examples, elements = x_txt.shape
        num_unique_texts = tokenized["input_ids"].size(0)
        embeddings = []
        for i in range(0, num_unique_texts, text_batch_size):
            j = min(i + text_batch_size, num_unique_texts)
            batch_input = {k: v[i:j] for k, v in tokenized.items()}
            batch_input = trim_batch_to_max_size(batch_input)
            batch_input = {k: v.to(self.device, non_blocking=True) for k, v in batch_input.items()}
            batch_embeddings = self.text_encoder(**batch_input)
            batch_embeddings = batch_embeddings.last_hidden_state[:, 0, :].to(dtype=get_precision(device=self.device))
            embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings[x_txt]
        assert embeddings.shape == (n_examples, elements, self.d_model)
        return embeddings

    def get_predictions(self, batch: TabSTARBatch, encoded: Tensor) -> Tensor:
        batch_size, elements, d_num = batch.x_num.shape
        max_d_output = max(batch.d_outputs)
        targets = encoded[:, :max_d_output]
        head = self.cls_head if batch.is_cls else self.reg_head
        targets = head(targets)
        if batch.is_cls:
            targets = targets.squeeze(dim=-1)
            expected_shape = (batch_size, max_d_output)
        else:
            expected_shape = (batch_size, max_d_output, 1 + self.reg_dist_bins)
        if targets.shape != expected_shape:
            raise ValueError(f"Expected output shape {expected_shape}, got {targets.shape}")
        return targets