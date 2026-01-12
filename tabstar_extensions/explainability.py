"""Attention-based explainability for TabStar models."""
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
import logging
from typing import Optional, Tuple, List, Dict, Union, Literal
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from pandas import DataFrame
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AttentionCapturingEncoderLayer(nn.Module):
    """Transformer encoder layer that captures and returns attention weights."""
    
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        activation: str = 'relu',
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x_in = self.norm1(src) if self.norm_first else src
        attn_output, attn_weights = self.self_attn(
            x_in, x_in, x_in,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        x = src + self.dropout1(attn_output)
        if not self.norm_first:
            x = self.norm1(x)

        x_ff = self.norm2(x) if self.norm_first else x
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x_ff)))))
        if not self.norm_first:
            x = self.norm2(x)

        return (x, attn_weights) if return_attention else x


class ExplainableInteractionEncoder(nn.Module):
    """Modified InteractionEncoder that captures attention weights from all layers."""
    
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 384,
        num_heads_factor: int = 64,
        ffn_d_hidden_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        dim_feedforward = d_model * ffn_d_hidden_multiplier
        num_heads = d_model // num_heads_factor
        
        self.layers = nn.ModuleList([
            AttentionCapturingEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: Tensor,
        return_attention: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        attention_weights = []
        for layer in self.layers:
            result = layer(x, return_attention=return_attention)
            if return_attention:
                x, attn = result
                attention_weights.append(attn)
            else:
                x = result
        return (x, attention_weights) if return_attention else x
    
    @classmethod
    def from_pretrained_encoder(
        cls,
        original_encoder: nn.Module,
        device: Optional[torch.device] = None,
    ) -> "ExplainableInteractionEncoder":
        """Create ExplainableInteractionEncoder by copying weights from pretrained encoder."""
        orig_layer = original_encoder.encoder.layers[0]
        explainable_encoder = cls(
            num_layers=len(original_encoder.encoder.layers),
            d_model=orig_layer.self_attn.embed_dim,
            num_heads_factor=orig_layer.self_attn.embed_dim // orig_layer.self_attn.num_heads,
            ffn_d_hidden_multiplier=orig_layer.linear1.out_features // orig_layer.self_attn.embed_dim,
            dropout=0.0,
        )

        for orig, new in zip(original_encoder.encoder.layers, explainable_encoder.layers):
            for attr in ['in_proj_weight', 'in_proj_bias']:
                getattr(new.self_attn, attr).data.copy_(getattr(orig.self_attn, attr).data)
            new.self_attn.out_proj.weight.data.copy_(orig.self_attn.out_proj.weight.data)
            new.self_attn.out_proj.bias.data.copy_(orig.self_attn.out_proj.bias.data)
            for module_name in ['linear1', 'linear2', 'norm1', 'norm2']:
                for param in ['weight', 'bias']:
                    getattr(getattr(new, module_name), param).data.copy_(
                        getattr(getattr(orig, module_name), param).data
                    )

        return explainable_encoder.to(device) if device else explainable_encoder


@dataclass
class LocalExplanation:
    """Local (per-sample) feature importance explanation."""
    feature_names: List[str]
    importance: np.ndarray
    attention_weights: Optional[List[np.ndarray]] = None

    def __repr__(self) -> str:
        return f"LocalExplanation(n_samples={self.importance.shape[0]}, n_features={self.importance.shape[1]})"


@dataclass
class GlobalExplanation:
    """Global (dataset-wide) feature importance explanation."""
    feature_names: List[str]
    importance: np.ndarray
    std: np.ndarray

    def __repr__(self) -> str:
        return f"GlobalExplanation(n_features={len(self.importance)})"

    def to_dict(self) -> Dict[str, float]:
        return {name: float(imp) for name, imp in zip(self.feature_names, self.importance)}


class TabStarExplainer:
    """Attention-based explainability for TabStar models."""
    
    def __init__(
        self,
        model: TabSTARClassifier | TabSTARRegressor,
        preprocessor=None,
        aggregation: Literal['mean', 'last', 'rollout', 'weighted'] = 'mean',
        device: Optional[torch.device] = None,
    ):
        if hasattr(model, 'model_') and hasattr(model, 'preprocessor_'):
            preprocessor = preprocessor or model.preprocessor_
            model = model.model_

        if preprocessor is None:
            raise ValueError("preprocessor is required")

        self.original_model = model
        self.preprocessor = preprocessor
        self.aggregation = aggregation
        self.device = device or getattr(model, 'device', None) or next(model.parameters(), torch.tensor(0)).device

        self.base_model = self._unwrap_model(model)
        for attr in ['tabular_encoder', 'text_encoder', 'numerical_fusion', 'get_textual_embedding']:
            if not hasattr(self.base_model, attr) or (attr == 'get_textual_embedding' and not callable(getattr(self.base_model, attr))):
                raise AttributeError(f"Base model missing: {attr}")

        self.base_model.to(self.device).eval()
        self.explainable_encoder = ExplainableInteractionEncoder.from_pretrained_encoder(
            self.base_model.tabular_encoder, self.device
        ).eval()
        self._setup_feature_names()
    
    def _unwrap_model(self, model):
        """Unwrap a PeftModel to get the base TabStarModel."""
        def _has_encoder(obj):
            try:
                return getattr(obj, 'tabular_encoder', None) is not None
            except Exception:
                return False

        if type(model).__name__ == 'TabStarModel' and _has_encoder(model):
            return model

        for path in ['base_model.base_model', 'base_model.model', 'base_model', 'model']:
            obj = model
            for part in path.split('.'):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj and _has_encoder(obj):
                return obj

        if hasattr(model, 'get_base_model'):
            base = model.get_base_model()
            if base and _has_encoder(base):
                return base

        if _has_encoder(model):
            return model

        raise ValueError(f"Could not unwrap model to find tabular_encoder. Model type: {type(model).__name__}")
    
    def _setup_feature_names(self):
        """Extract feature names from the preprocessor."""
        self.num_cols = sorted(self.preprocessor.numerical_transformers.keys())
        self.d_output = self.preprocessor.d_output
        self.target_names = (
            [f"[TARGET:{v}]" for v in self.preprocessor.y_values]
            if self.preprocessor.is_cls and self.preprocessor.y_values
            else [f"[TARGET:{self.preprocessor.y_name}]"]
        )
        self.constant_columns = list(getattr(self.preprocessor, 'constant_columns', []))
        self.feature_names = []
        self._feature_names_initialized = False
        self._active_feature_count = 0
        self._constant_col_names = []
    
    def _initialize_feature_names(self, X: DataFrame, data):
        """Initialize feature names from the input DataFrame."""
        self._active_feature_count = data.x_txt.shape[1] - self.d_output
        constant_cols = [c for c in self.constant_columns if c in X.columns]
        non_constant = [c for c in X.columns if c not in self.constant_columns]
        text_cols = [c for c in non_constant if c not in self.num_cols]
        active_features = text_cols + self.num_cols

        if len(active_features) != self._active_feature_count:
            active_features = [f"feature_{i}" for i in range(self._active_feature_count)]

        self.feature_names = active_features + constant_cols
        self._constant_col_names = constant_cols
    
    def _get_fused_embeddings(self, data) -> Tensor:
        """Get fused embeddings (text + numerical) from the model."""
        with torch.no_grad():
            text_emb = self.base_model.get_textual_embedding(data.x_txt)
            x_num = data.x_num if isinstance(data.x_num, Tensor) else torch.tensor(
                data.x_num, dtype=text_emb.dtype, device=text_emb.device
            )
            return self.base_model.numerical_fusion(textual_embeddings=text_emb, x_num=x_num)
    
    def _extract_attention(self, X: DataFrame, batch_size: int = 32) -> List[np.ndarray]:
        """Extract attention weights for all samples in X."""
        from tabstar.training.dataloader import get_dataloader

        data = self.preprocessor.transform(X, y=None)
        if not self._feature_names_initialized:
            self._initialize_feature_names(X, data)
            self._feature_names_initialized = True

        all_attn = [[] for _ in range(self.explainable_encoder.num_layers)]
        self.explainable_encoder.eval()
        with torch.no_grad():
            for batch in get_dataloader(data, is_train=False, batch_size=batch_size):
                _, layer_attn = self.explainable_encoder(self._get_fused_embeddings(batch), return_attention=True)
                for i, attn in enumerate(layer_attn):
                    all_attn[i].append(attn.cpu().numpy())

        return [np.concatenate(layer, axis=0) for layer in all_attn]
    
    def _aggregate_attention(self, attention_weights: List[np.ndarray]) -> np.ndarray:
        """Aggregate attention weights across layers and heads."""
        stacked = np.stack(attention_weights, axis=0)

        if self.aggregation == 'mean':
            return stacked.mean(axis=(0, 2))
        elif self.aggregation == 'last':
            return attention_weights[-1].mean(axis=1)
        elif self.aggregation == 'rollout':
            layer_attn = stacked.mean(axis=2)
            seq_len = layer_attn.shape[-1]
            identity = np.eye(seq_len)[np.newaxis, np.newaxis, :, :]
            layer_attn = 0.5 * layer_attn + 0.5 * identity
            result = layer_attn[0]
            for i in range(1, len(layer_attn)):
                result = np.matmul(result, layer_attn[i])
            return result
        elif self.aggregation == 'weighted':
            weights = np.arange(1, len(attention_weights) + 1, dtype=np.float32)
            weights /= weights.sum()
            return np.tensordot(weights, stacked.mean(axis=2), axes=([0], [0]))
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _compute_feature_importance(self, aggregated_attention: np.ndarray) -> np.ndarray:
        """Compute feature importance from aggregated attention."""
        target_to_feature = aggregated_attention[:, :self.d_output, self.d_output:]
        active_imp = target_to_feature.mean(axis=1)
        active_imp = active_imp / (active_imp.sum(axis=1, keepdims=True) + 1e-10)

        n_constant = len(self._constant_col_names)
        if n_constant > 0:
            return np.concatenate([active_imp, np.zeros((len(active_imp), n_constant))], axis=1)
        return active_imp
    
    def explain_local(self, X: DataFrame, batch_size: int = 32, return_attention: bool = False) -> LocalExplanation:
        """Compute local (per-sample) feature importance."""
        attn = self._extract_attention(X, batch_size)
        importance = self._compute_feature_importance(self._aggregate_attention(attn))
        return LocalExplanation(
            feature_names=self.feature_names.copy(),
            importance=importance,
            attention_weights=attn if return_attention else None,
        )

    def explain_global(self, X: DataFrame, batch_size: int = 32) -> GlobalExplanation:
        """Compute global (dataset-wide) feature importance."""
        local_exp = self.explain_local(X, batch_size, return_attention=False)
        return GlobalExplanation(
            feature_names=self.feature_names.copy(),
            importance=local_exp.importance.mean(axis=0),
            std=local_exp.importance.std(axis=0),
        )
    
    def get_attention_heatmap_data(
        self,
        X: DataFrame,
        sample_idx: int = 0,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Get attention matrix data for visualization."""
        attn = self._extract_attention(X.iloc[[sample_idx]], batch_size=1)[layer_idx]
        attn_matrix = attn[0, head_idx] if head_idx is not None else attn[0].mean(axis=0)
        return attn_matrix, self.target_names + self.feature_names
    
    def plot_feature_importance(
        self,
        explanation: Union[GlobalExplanation, LocalExplanation],
        top_k: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        show_std: bool = True,
        sample_idx: int = 0,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot feature importance as a horizontal bar chart."""
        is_global = isinstance(explanation, GlobalExplanation)
        importance = explanation.importance if is_global else explanation.importance[sample_idx]
        std = explanation.std if is_global and show_std else None
        default_title = "Global Feature Importance (Attention-based)" if is_global else f"Local Feature Importance (Sample {sample_idx})"

        sorted_idx = np.argsort(importance)[::-1][:top_k]
        sorted_imp = importance[sorted_idx]
        sorted_names = [explanation.feature_names[i] for i in sorted_idx]
        sorted_std = std[sorted_idx] if std is not None else None

        if figsize is None:
            figsize = (10, max(4, min(len(sorted_names) * 0.4, 20)))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_imp, xerr=sorted_std, capsize=3 if sorted_std is not None else 0,
                color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title or default_title)
        plt.tight_layout()
        return fig
    
    def plot_attention_heatmap(
        self,
        X: DataFrame,
        sample_idx: int = 0,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 10),
        title: Optional[str] = None,
        cmap: str = 'Blues',
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot attention matrix as a heatmap."""
        attn_matrix, token_names = self.get_attention_heatmap_data(X, sample_idx, layer_idx, head_idx)
        display_names = [name[:20] + '...' if len(name) > 20 else name for name in token_names]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        im = ax.imshow(attn_matrix, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Attention Weight')
        ax.set_xticks(np.arange(len(display_names)))
        ax.set_yticks(np.arange(len(display_names)))
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(display_names, fontsize=8)
        ax.set_xlabel('Key (Attended To)')
        ax.set_ylabel('Query (Attending From)')

        layer_str = f"Layer {layer_idx}" if layer_idx >= 0 else "Last Layer"
        head_str = f"Head {head_idx}" if head_idx is not None else "All Heads (avg)"
        ax.set_title(title or f"Attention Heatmap - Sample {sample_idx}, {layer_str}, {head_str}")
        plt.tight_layout()
        return fig
    
    def plot_importance_distribution(
        self,
        local_explanation: LocalExplanation,
        top_k: int = 15,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot the distribution of feature importance across samples as box plots."""
        importance = local_explanation.importance
        sorted_idx = np.argsort(importance.mean(axis=0))[::-1][:top_k]
        sorted_names = [local_explanation.feature_names[i] for i in sorted_idx]
        sorted_data = [importance[:, i] for i in sorted_idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        bp = ax.boxplot(sorted_data, vert=False, labels=sorted_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)

        ax.set_xlabel('Importance')
        ax.set_title(title or "Feature Importance Distribution Across Samples")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig


def diagnose_model(model, max_depth: int = 4) -> None:
    """Print the model structure to help diagnose unwrapping issues."""
    def _traverse(obj, prefix="", depth=0):
        if depth > max_depth:
            return
        markers = []
        if hasattr(obj, 'tabular_encoder'):
            markers.append("HAS tabular_encoder")
        if hasattr(obj, 'model_'):
            markers.append("has model_")
        if hasattr(obj, 'preprocessor_'):
            markers.append("has preprocessor_")
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        print(f"{'  ' * depth}{prefix}{type(obj).__name__}{marker_str}")

        for attr in ['model_', 'base_model', 'model']:
            if hasattr(obj, attr) and not callable(getattr(obj, attr, None)):
                child = getattr(obj, attr)
                if child is not None and child is not obj and depth < max_depth:
                    _traverse(child, f".{attr} -> ", depth + 1)

    print("=" * 60)
    print("Model Structure Diagnosis")
    print("=" * 60)
    _traverse(model)
    print("=" * 60)


def explain_tabstar(
    model,
    X: DataFrame,
    preprocessor=None,
    aggregation: str = 'mean',
    top_k: Optional[int] = None,
    plot: bool = True,
) -> GlobalExplanation:
    """Convenience function to explain a TabStar model on a dataset."""
    explainer = TabStarExplainer(model, preprocessor, aggregation=aggregation)
    global_exp = explainer.explain_global(X)
    if plot:
        explainer.plot_feature_importance(global_exp, top_k=top_k)
        plt.show()
    return global_exp
