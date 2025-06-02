from peft import LoraConfig, get_peft_model

from tabstar.arch.arch import TabStarModel
from tabstar.arch.config import LORA_R


def load_model_with_lora():
    model = TabStarModel.from_pretrained("alana89/TabSTAR")
    # TODO: probably best if this is written more generic and not so hard-coded
    lora_modules = ["query", "key", "value", "out_proj", "linear1", "linear2",
                    "cls_head.layers.0", "reg_head.layers.0"]
    to_freeze = range(6)
    prefixes = tuple(f"text_encoder.encoder.layer.{i}." for i in to_freeze)
    to_exclude = [name for name, _ in model.named_modules() if name.startswith(prefixes)]
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        target_modules=lora_modules,
        exclude_modules=to_exclude,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model