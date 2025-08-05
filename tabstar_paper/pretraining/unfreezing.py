from typing import List

from transformers import BertModel

E5_SMALL_LAYERS = 12

def unfreeze_text_encoder(text_encoder: BertModel, layers_to_unfreeze: int):
    blocks = [block_name for block_name, _ in text_encoder.named_children()]
    assert blocks == ['embeddings', 'encoder', 'pooler'], f"Unexpected block structure: {blocks}"
    assert text_encoder.config.num_hidden_layers == E5_SMALL_LAYERS
    assert 0 <= layers_to_unfreeze <= E5_SMALL_LAYERS
    if layers_to_unfreeze == 0:
        freeze_the_whole_encoder(text_encoder=text_encoder)
    else:
        unfreeze_pooler_and_num_layers(text_encoder=text_encoder, num_layers=layers_to_unfreeze)
    total_params = sum(p.numel() for p in text_encoder.parameters())
    unfrozen_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    print(f"🥶 Unfroze {unfrozen_params:,} out of {total_params:,} parameters!")


def freeze_the_whole_encoder(text_encoder: BertModel):
    for name, param in text_encoder.named_parameters():
        param.requires_grad = False
    print(f"🥶 Freezing the whole text encoder - including the pooling!")


def unfreeze_pooler_and_num_layers(text_encoder: BertModel, num_layers: int):
    for name, param in text_encoder.pooler.named_parameters():
        param.requires_grad = True
    for name, param in text_encoder.embeddings.named_parameters():
        param.requires_grad = False
    to_unfreeze = get_last_layers_num(to_unfreeze=num_layers)
    print(f"🥶 Planning to unfreeze {num_layers}/{E5_SMALL_LAYERS} layers: {to_unfreeze}!")
    layer_prefixes = [f'layer.{i}.' for i in to_unfreeze]
    for name, param in text_encoder.encoder.named_parameters():
        if any(name.startswith(prefix) for prefix in layer_prefixes):
            param.requires_grad = True
        else:
            param.requires_grad = False

def get_last_layers_num(to_unfreeze: int, total_layers: int = E5_SMALL_LAYERS) -> List[int]:
    layers_reversed = list(reversed(range(total_layers)))
    unfrozen = layers_reversed[:to_unfreeze]
    return unfrozen