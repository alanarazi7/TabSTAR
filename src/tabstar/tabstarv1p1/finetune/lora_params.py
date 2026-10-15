from typing import List, Optional

from tabstar.tabstarv1p1.arch.model import TabSTARModel

TO_AVOID = ['bias',
            'LayerNorm',
            'text_encoder.embeddings',
            'output.dense',                 # Text-Encoder
            'intermediate.dense',           # Text-Encoder
            'norm1.weight',
            'norm2.weight',
            'fusion.ln_text',
            'fusion.ln_num',
            'text_weight'
            ]

TO_INCLUDE = [
    # Text Encoder
    'attention.self.query',
    'attention.self.key',
    'attention.self.value',
    'text_encoder.pooler.dense',
    # Numerical Encoder / Fusion
    'numerical_encoder.scalar_embedder',
    'numerical_fusion.scalar_embedder',
    # Interaction Encoder
    'self_attn.in_proj',
    'self_attn.out_proj',
    'linear1',
    'linear2',
    # Prediction
    'cls_head',
    'reg_head',
]

def print_all_modules(model: TabSTARModel):
    print("🦜 Printing all model modules, with parameter count:")
    for name, param in model.named_parameters():
        print(f" - {name}: {param.numel()} params")


def get_all_lora_modules(model: TabSTARModel, to_unfreeze: int = 6, is_cls: Optional[bool] = None) -> List[str]:
    """`is_cls=None` wraps both heads (e.g. `assert_lora_coverage`'s static architecture check);
    otherwise only the head the forward pass actually uses for this task is wrapped, since the
    other one gets no gradient either way.
    """
    inactive_head = None
    if is_cls is True:
        inactive_head = 'reg_head'
    elif is_cls is False:
        inactive_head = 'cls_head'
    modules = []
    for name, param in model.named_parameters():
        if is_lora_layer(name=name, to_unfreeze=to_unfreeze):
            if inactive_head is not None and inactive_head in name:
                continue
            if name.endswith('.weight'):
                name = name.replace('.weight', '')
            modules.append(name)
    return modules


def is_lora_layer(name: str, to_unfreeze: int) -> bool:
    assert to_unfreeze > 0, "Must unfreeze at least one text encoder layer"
    layer_to_freeze = list(range(12 - to_unfreeze))
    frozen_to_avoid = [f"text_encoder.encoder.layer.{i}." for i in layer_to_freeze]
    to_avoid = TO_AVOID + frozen_to_avoid
    if any(s in name for s in to_avoid):
        return False
    if any(s in name for s in TO_INCLUDE):
        return True
    raise NotImplementedError(f"Unknown layer type for LoRA: {name}")


def assert_lora_coverage(model: TabSTARModel) -> None:
    """Called once before a pretrain run starts (mirrors `assert_finetunability_installed`):
    a parameter an architecture change adds without registering it in TO_AVOID/TO_INCLUDE makes
    every finetunability check's LoRA-wrapping raise for every dataset, every epoch -- silently,
    since `run_finetunability_check` never propagates that failure, so it would otherwise burn a
    full run with zero finetunability signal before anyone notices. Checked with `to_unfreeze=12`
    (no text-encoder layer frozen): a smaller `to_unfreeze` only avoids more parameters, never
    fewer, so this is the strictest classification any real run could hit.
    """
    uncovered = []
    for name, _ in model.named_parameters():
        try:
            is_lora_layer(name=name, to_unfreeze=12)
        except NotImplementedError:
            uncovered.append(name)
    if uncovered:
        raise NotImplementedError(
            f"{len(uncovered)} parameter(s) not classified in lora_params.py's TO_AVOID/"
            f"TO_INCLUDE: {uncovered}. Register every new layer an architecture change adds "
            f"there before pretraining, or finetunability tracking silently fails every epoch."
        )


'''
 - text_encoder.embeddings.word_embeddings.weight: 11720448 params
 - text_encoder.embeddings.position_embeddings.weight: 196608 params
 - text_encoder.embeddings.token_type_embeddings.weight: 768 params
 - text_encoder.embeddings.LayerNorm.weight: 384 params
 - text_encoder.embeddings.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.0.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.0.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.0.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.0.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.0.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.0.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.0.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.0.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.0.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.0.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.0.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.0.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.0.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.0.output.dense.bias: 384 params
 - text_encoder.encoder.layer.0.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.0.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.1.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.1.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.1.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.1.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.1.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.1.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.1.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.1.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.1.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.1.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.1.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.1.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.1.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.1.output.dense.bias: 384 params
 - text_encoder.encoder.layer.1.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.1.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.2.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.2.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.2.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.2.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.2.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.2.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.2.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.2.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.2.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.2.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.2.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.2.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.2.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.2.output.dense.bias: 384 params
 - text_encoder.encoder.layer.2.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.2.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.3.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.3.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.3.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.3.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.3.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.3.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.3.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.3.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.3.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.3.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.3.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.3.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.3.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.3.output.dense.bias: 384 params
 - text_encoder.encoder.layer.3.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.3.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.4.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.4.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.4.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.4.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.4.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.4.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.4.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.4.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.4.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.4.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.4.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.4.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.4.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.4.output.dense.bias: 384 params
 - text_encoder.encoder.layer.4.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.4.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.5.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.5.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.5.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.5.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.5.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.5.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.5.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.5.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.5.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.5.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.5.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.5.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.5.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.5.output.dense.bias: 384 params
 - text_encoder.encoder.layer.5.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.5.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.6.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.6.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.6.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.6.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.6.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.6.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.6.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.6.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.6.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.6.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.6.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.6.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.6.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.6.output.dense.bias: 384 params
 - text_encoder.encoder.layer.6.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.6.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.7.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.7.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.7.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.7.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.7.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.7.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.7.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.7.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.7.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.7.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.7.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.7.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.7.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.7.output.dense.bias: 384 params
 - text_encoder.encoder.layer.7.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.7.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.8.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.8.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.8.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.8.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.8.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.8.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.8.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.8.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.8.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.8.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.8.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.8.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.8.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.8.output.dense.bias: 384 params
 - text_encoder.encoder.layer.8.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.8.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.9.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.9.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.9.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.9.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.9.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.9.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.9.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.9.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.9.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.9.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.9.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.9.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.9.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.9.output.dense.bias: 384 params
 - text_encoder.encoder.layer.9.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.9.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.10.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.10.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.10.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.10.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.10.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.10.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.10.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.10.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.10.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.10.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.10.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.10.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.10.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.10.output.dense.bias: 384 params
 - text_encoder.encoder.layer.10.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.10.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.11.attention.self.query.weight: 147456 params
 - text_encoder.encoder.layer.11.attention.self.query.bias: 384 params
 - text_encoder.encoder.layer.11.attention.self.key.weight: 147456 params
 - text_encoder.encoder.layer.11.attention.self.key.bias: 384 params
 - text_encoder.encoder.layer.11.attention.self.value.weight: 147456 params
 - text_encoder.encoder.layer.11.attention.self.value.bias: 384 params
 - text_encoder.encoder.layer.11.attention.output.dense.weight: 147456 params
 - text_encoder.encoder.layer.11.attention.output.dense.bias: 384 params
 - text_encoder.encoder.layer.11.attention.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.11.attention.output.LayerNorm.bias: 384 params
 - text_encoder.encoder.layer.11.intermediate.dense.weight: 589824 params
 - text_encoder.encoder.layer.11.intermediate.dense.bias: 1536 params
 - text_encoder.encoder.layer.11.output.dense.weight: 589824 params
 - text_encoder.encoder.layer.11.output.dense.bias: 384 params
 - text_encoder.encoder.layer.11.output.LayerNorm.weight: 384 params
 - text_encoder.encoder.layer.11.output.LayerNorm.bias: 384 params
 - text_encoder.pooler.dense.weight: 147456 params
 - text_encoder.pooler.dense.bias: 384 params
 - numerical_encoder.scalar_embedder.0.weight: 768 params
 - numerical_encoder.scalar_embedder.0.bias: 768 params
 - numerical_encoder.scalar_embedder.3.weight: 294912 params
 - numerical_encoder.scalar_embedder.3.bias: 384 params
 - numerical_fusion.fusion_block.self_attn.in_proj_weight: 442368 params
 - numerical_fusion.fusion_block.self_attn.in_proj_bias: 1152 params
 - numerical_fusion.fusion_block.self_attn.out_proj.weight: 147456 params
 - numerical_fusion.fusion_block.self_attn.out_proj.bias: 384 params
 - numerical_fusion.fusion_block.linear1.weight: 589824 params
 - numerical_fusion.fusion_block.linear1.bias: 1536 params
 - numerical_fusion.fusion_block.linear2.weight: 589824 params
 - numerical_fusion.fusion_block.linear2.bias: 384 params
 - numerical_fusion.fusion_block.norm1.weight: 384 params
 - numerical_fusion.fusion_block.norm1.bias: 384 params
 - numerical_fusion.fusion_block.norm2.weight: 384 params
 - numerical_fusion.fusion_block.norm2.bias: 384 params
 - tabular_encoder.encoder.layers.0.self_attn.in_proj_weight: 442368 params
 - tabular_encoder.encoder.layers.0.self_attn.in_proj_bias: 1152 params
 - tabular_encoder.encoder.layers.0.self_attn.out_proj.weight: 147456 params
 - tabular_encoder.encoder.layers.0.self_attn.out_proj.bias: 384 params
 - tabular_encoder.encoder.layers.0.linear1.weight: 589824 params
 - tabular_encoder.encoder.layers.0.linear1.bias: 1536 params
 - tabular_encoder.encoder.layers.0.linear2.weight: 589824 params
 - tabular_encoder.encoder.layers.0.linear2.bias: 384 params
 - tabular_encoder.encoder.layers.0.norm1.weight: 384 params
 - tabular_encoder.encoder.layers.0.norm1.bias: 384 params
 - tabular_encoder.encoder.layers.0.norm2.weight: 384 params
 - tabular_encoder.encoder.layers.0.norm2.bias: 384 params
 - tabular_encoder.encoder.layers.1.self_attn.in_proj_weight: 442368 params
 - tabular_encoder.encoder.layers.1.self_attn.in_proj_bias: 1152 params
 - tabular_encoder.encoder.layers.1.self_attn.out_proj.weight: 147456 params
 - tabular_encoder.encoder.layers.1.self_attn.out_proj.bias: 384 params
 - tabular_encoder.encoder.layers.1.linear1.weight: 589824 params
 - tabular_encoder.encoder.layers.1.linear1.bias: 1536 params
 - tabular_encoder.encoder.layers.1.linear2.weight: 589824 params
 - tabular_encoder.encoder.layers.1.linear2.bias: 384 params
 - tabular_encoder.encoder.layers.1.norm1.weight: 384 params
 - tabular_encoder.encoder.layers.1.norm1.bias: 384 params
 - tabular_encoder.encoder.layers.1.norm2.weight: 384 params
 - tabular_encoder.encoder.layers.1.norm2.bias: 384 params
 - tabular_encoder.encoder.layers.2.self_attn.in_proj_weight: 442368 params
 - tabular_encoder.encoder.layers.2.self_attn.in_proj_bias: 1152 params
 - tabular_encoder.encoder.layers.2.self_attn.out_proj.weight: 147456 params
 - tabular_encoder.encoder.layers.2.self_attn.out_proj.bias: 384 params
 - tabular_encoder.encoder.layers.2.linear1.weight: 589824 params
 - tabular_encoder.encoder.layers.2.linear1.bias: 1536 params
 - tabular_encoder.encoder.layers.2.linear2.weight: 589824 params
 - tabular_encoder.encoder.layers.2.linear2.bias: 384 params
 - tabular_encoder.encoder.layers.2.norm1.weight: 384 params
 - tabular_encoder.encoder.layers.2.norm1.bias: 384 params
 - tabular_encoder.encoder.layers.2.norm2.weight: 384 params
 - tabular_encoder.encoder.layers.2.norm2.bias: 384 params
 - tabular_encoder.encoder.layers.3.self_attn.in_proj_weight: 442368 params
 - tabular_encoder.encoder.layers.3.self_attn.in_proj_bias: 1152 params
 - tabular_encoder.encoder.layers.3.self_attn.out_proj.weight: 147456 params
 - tabular_encoder.encoder.layers.3.self_attn.out_proj.bias: 384 params
 - tabular_encoder.encoder.layers.3.linear1.weight: 589824 params
 - tabular_encoder.encoder.layers.3.linear1.bias: 1536 params
 - tabular_encoder.encoder.layers.3.linear2.weight: 589824 params
 - tabular_encoder.encoder.layers.3.linear2.bias: 384 params
 - tabular_encoder.encoder.layers.3.norm1.weight: 384 params
 - tabular_encoder.encoder.layers.3.norm1.bias: 384 params
 - tabular_encoder.encoder.layers.3.norm2.weight: 384 params
 - tabular_encoder.encoder.layers.3.norm2.bias: 384 params
 - tabular_encoder.encoder.layers.4.self_attn.in_proj_weight: 442368 params
 - tabular_encoder.encoder.layers.4.self_attn.in_proj_bias: 1152 params
 - tabular_encoder.encoder.layers.4.self_attn.out_proj.weight: 147456 params
 - tabular_encoder.encoder.layers.4.self_attn.out_proj.bias: 384 params
 - tabular_encoder.encoder.layers.4.linear1.weight: 589824 params
 - tabular_encoder.encoder.layers.4.linear1.bias: 1536 params
 - tabular_encoder.encoder.layers.4.linear2.weight: 589824 params
 - tabular_encoder.encoder.layers.4.linear2.bias: 384 params
 - tabular_encoder.encoder.layers.4.norm1.weight: 384 params
 - tabular_encoder.encoder.layers.4.norm1.bias: 384 params
 - tabular_encoder.encoder.layers.4.norm2.weight: 384 params
 - tabular_encoder.encoder.layers.4.norm2.bias: 384 params
 - tabular_encoder.encoder.layers.5.self_attn.in_proj_weight: 442368 params
 - tabular_encoder.encoder.layers.5.self_attn.in_proj_bias: 1152 params
 - tabular_encoder.encoder.layers.5.self_attn.out_proj.weight: 147456 params
 - tabular_encoder.encoder.layers.5.self_attn.out_proj.bias: 384 params
 - tabular_encoder.encoder.layers.5.linear1.weight: 589824 params
 - tabular_encoder.encoder.layers.5.linear1.bias: 1536 params
 - tabular_encoder.encoder.layers.5.linear2.weight: 589824 params
 - tabular_encoder.encoder.layers.5.linear2.bias: 384 params
 - tabular_encoder.encoder.layers.5.norm1.weight: 384 params
 - tabular_encoder.encoder.layers.5.norm1.bias: 384 params
 - tabular_encoder.encoder.layers.5.norm2.weight: 384 params
 - tabular_encoder.encoder.layers.5.norm2.bias: 384 params
 - cls_head.layers.0.weight: 589824 params
 - cls_head.layers.0.bias: 1536 params
 - cls_head.layers.2.weight: 1536 params
 - cls_head.layers.2.bias: 1 params
 - reg_head.layers.0.weight: 589824 params
 - reg_head.layers.0.bias: 1536 params
 - reg_head.layers.2.weight: 1536 params
 - reg_head.layers.2.bias: 1 params
 '''