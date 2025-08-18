from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from transformers.utils import ModelOutput
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
        if not isinstance(x_num, Tensor):
            # TODO: this is a bug, it should always be a Tensor
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

@dataclass
class TabStarOutput(ModelOutput):
    logits: Optional[Tensor] = None                      # (B, d_output)
    embeddings: Optional[Tensor] = None                  # (B, seq_len, d_model) fused (text+num), pre-tabular
    text_hidden_states: Optional[Tuple[Tensor, ...]] = None  # tuple of length L: each (B, seq_len, d_model)

class TabStarEmbeddingModel(PreTrainedModel):
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

    @torch.no_grad()
    def forward(
        self,
        x_txt: np.ndarray,
        x_num: np.ndarray,
        d_output: int,
        *,
        return_embeddings: bool = False,
        return_text_hidden_states: bool = False,
        cls_only: bool = True,  # how to pool hidden states per layer (True = use CLS token, False = mean-pool with mask)
    ) -> Union[Tensor, TabStarOutput]:
        """
        If no extras requested, returns target_scores: (B, d_output).
        Otherwise returns TabStarOutput(logits, embeddings, text_hidden_states).
        """
        # Text side
        if return_text_hidden_states:
            textual_embeddings, text_hidden_states = self.get_textual_embedding(
                x_txt, return_hidden_states=True, cls_only=cls_only
            )
        else:
            textual_embeddings = self.get_textual_embedding(x_txt)
            text_hidden_states = None

        # Numeric side
        if not isinstance(x_num, Tensor):
            # TODO: this is a bug, it should always be a Tensor
            x_num = torch.tensor(x_num, dtype=textual_embeddings.dtype, device=textual_embeddings.device)

        # Fuse text + numeric
        embeddings = self.numerical_fusion(textual_embeddings=textual_embeddings, x_num=x_num)

        # Encode tabular sequence and score heads
        encoded = self.tabular_encoder(embeddings)
        target_tokens = encoded[:, :d_output]
        if d_output == 1:
            target_scores = self.reg_head(target_tokens)
        else:
            target_scores = self.cls_head(target_tokens)
        target_scores = target_scores.squeeze(dim=-1)
        assert tuple(target_scores.shape) == (x_txt.shape[0], d_output)

        # Fast path (back-compat): just return scores
        if not (return_embeddings or return_text_hidden_states):
            return target_scores

        # Otherwise, return a structured bundle
        out = TabStarOutput(
            logits=target_scores,
            embeddings=embeddings if return_embeddings else None,
            text_hidden_states=text_hidden_states,
        )
        return out

    # ------- Textual embeddings API (now with hidden states) ---------------------

    @torch.no_grad()
    def get_textual_embedding(
        self,
        x_txt: np.ndarray,
        *,
        return_hidden_states: bool = False,
        cls_only: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]:
        """
        Returns:
            embeddings: (B, seq_len, d_model)

        If return_hidden_states:
            returns (embeddings, hidden_states)
            where hidden_states is a tuple of length L, each tensor shaped (B, seq_len, d_model),
            corresponding to the (optionally CLS/pooled) representation from each transformer layer.
        """
        text_batch_size = 128
        while text_batch_size > 1:
            try:
                return self.get_textual_embedding_in_batches(
                    x_txt,
                    text_batch_size=text_batch_size,
                    return_hidden_states=return_hidden_states,
                    cls_only=cls_only,
                )
            except torch.cuda.OutOfMemoryError:
                text_batch_size //= 2
                clear_cuda_cache()
                print(f"🤯 Reducing batch size to {text_batch_size} due to OOM")
        # final attempt at BS=1
        return self.get_textual_embedding_in_batches(
            x_txt, text_batch_size=1, return_hidden_states=return_hidden_states, cls_only=cls_only
        )

    @torch.no_grad()
    def get_textual_embedding_in_batches(
        self,
        x_txt: np.ndarray,
        text_batch_size: int,
        *,
        return_hidden_states: bool = False,
        cls_only: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]:
        """
        Batched text encoding with de-duplication.
        """
        # Get unique texts and mapping indices
        unique_texts, inverse_indices = np.unique(x_txt, return_inverse=True)
        num_unique_texts = len(unique_texts)
        batch_size, seq_len = x_txt.shape

        # Storage for last-layer CLS embeddings
        last_layer_cls_chunks: List[Tensor] = []

        # Storage for per-layer pooled embeddings (if requested)
        per_layer_cls_chunks: Optional[List[List[Tensor]]] = None
        if return_hidden_states:
            # We'll build a list (per layer) of chunk tensors, then cat at the end
            per_layer_cls_chunks = []

        # Iterate batches of unique texts
        for i in range(0, num_unique_texts, text_batch_size):
            batch_texts = unique_texts[i:i + text_batch_size].tolist()

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                return_tensors='pt',
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Request hidden states if needed (saves overhead when not needed)
            outputs = self.text_encoder(**inputs, output_hidden_states=return_hidden_states)

            # ---- Last-layer -> CLS (your existing behavior) ---------------------
            # Take the [CLS] token representation
            last_layer_cls = outputs.last_hidden_state[:, 0, :]
            last_layer_cls_chunks.append(last_layer_cls)

            # ---- Optional: collect all hidden states ---------------------------
            if return_hidden_states:
                # outputs.hidden_states is a tuple: (emb_0, emb_1, ..., emb_L)
                # each: (U, seq_len_sub, hidden)
                hs = outputs.hidden_states  # type: Tuple[Tensor, ...]
                # Initialize holders at first chunk
                if not per_layer_cls_chunks:
                    per_layer_cls_chunks = [[] for _ in range(len(hs))]

                if cls_only:
                    # CLS at index 0 for each layer
                    pooled_per_layer = [h[:, 0, :] for h in hs]
                else:
                    # Mean-pool across tokens with attention mask (excluding padding)
                    # (U, T, H) -> (U, H)
                    attn_mask = inputs["attention_mask"].unsqueeze(-1)  # (U, T, 1)
                    # avoid division by zero
                    denom = attn_mask.sum(dim=1).clamp(min=1)
                    pooled_per_layer = [
                        (h * attn_mask).sum(dim=1) / denom for h in hs
                    ]

                # Append each layer’s pooled chunk
                for l, chunk in enumerate(pooled_per_layer):
                    per_layer_cls_chunks[l].append(chunk)

        # Concatenate along the unique-text dimension
        embeddings_unique = torch.cat(last_layer_cls_chunks, dim=0)  # (num_unique, H)

        # Map back to original positions and reshape
        inverse_indices = torch.tensor(inverse_indices, dtype=torch.long, device=embeddings_unique.device)
        embeddings = embeddings_unique[inverse_indices].view(batch_size, seq_len, -1)

        if not tuple(embeddings.shape) == (batch_size, seq_len, self.config.d_model):
            raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

        if not return_hidden_states:
            return embeddings

        # Build per-layer tensors, map back and reshape to (B, seq_len, H)
        assert per_layer_cls_chunks is not None
        per_layer_unique: List[Tensor] = [torch.cat(chunks, dim=0) for chunks in per_layer_cls_chunks]  # L x (num_unique, H)
        per_layer_full: Tuple[Tensor, ...] = tuple(
            layer_u[inverse_indices].view(batch_size, seq_len, -1)
            for layer_u in per_layer_unique
        )

        # Sanity-check shapes
        for li, h in enumerate(per_layer_full):
            if not tuple(h.shape) == (batch_size, seq_len, self.config.d_model):
                raise RuntimeError(f"Unexpected hidden state shape at layer {li}: {h.shape}")

        return embeddings, per_layer_full

    @classmethod
    def from_peft(
        cls,
        peft_model: "PeftModel",
        *,
        merge_adapters: bool = True,
        freeze_text_encoder: bool = False,
        device: Optional[torch.device] = None,
    ) -> "TabStarEmbeddingModel":
        """
        Create a TabStarEmbeddingModel from a PEFT-wrapped TabStarModel.
        The PEFT base model is assumed to expose the same submodules:
        text_encoder, numerical_fusion, tabular_encoder, cls_head, reg_head, (optional) tokenizer.

        Args:
            peft_model: A peft.PeftModel that wraps a TabStarModel (the version that doesn't return embeddings).
            merge_adapters: If True, attempts to merge adapters for pure inference weights.
            freeze_text_encoder: If True, sets requires_grad=False for the text encoder params.
            device: If provided, moves the returned model to this device.

        Returns:
            TabStarEmbeddingModel initialized with weights copied from the PEFT base model.
        """

        # --- Helper: get underlying base model no matter the PEFT wrapping details ---
        def _unwrap_base_model(m):
            # 1) Best case: dedicated helper
            if hasattr(m, "get_base_model"):
                try:
                    return m.get_base_model()
                except Exception:
                    pass
            # 2) Common attribute paths in PeftModel implementations
            for path in ("base_model.model", "base_model", "model"):
                cur = m
                ok = True
                for part in path.split("."):
                    if hasattr(cur, part):
                        cur = getattr(cur, part)
                    else:
                        ok = False
                        break
                if ok:
                    return cur
            # 3) Fallback: return as-is
            return m

        base = peft_model

        # Optionally bake adapters into the base weights for clean inference
        if merge_adapters and hasattr(base, "merge_and_unload"):
            try:
                base = base.merge_and_unload()
            except Exception:
                # non-fatal—continue without merging
                pass

        base = _unwrap_base_model(base)

        # Ensure we have a config to bootstrap this class
        config = getattr(base, "config", None)
        if config is None:
            raise ValueError("The provided PEFT model doesn't expose a `.config`. "
                             "Please pass a PeftModel wrapping a TabStarModel.")

        # Create a fresh TabStarEmbeddingModel
        self = cls(config)

        # Copy submodules from the (no-embedding) TabStarModel
        required = ["text_encoder", "numerical_fusion", "tabular_encoder", "cls_head", "reg_head"]
        for name in required:
            if not hasattr(base, name):
                raise AttributeError(f"Base model is missing required submodule `{name}`.")
            setattr(self, name, getattr(base, name))

        # Prefer tokenizer from the base model if present, else keep current
        if hasattr(base, "tokenizer") and getattr(base, "tokenizer") is not None:
            self.tokenizer = base.tokenizer

        # Optionally freeze text encoder params
        if freeze_text_encoder and hasattr(self.text_encoder, "parameters"):
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # Move to device if requested
        if device is not None:
            self.to(device)

        return self
