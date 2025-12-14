from os.path import exists

from peft import LoraConfig, get_peft_model, PeftModel

from tabstar.arch.arch import TabStarModel


def load_pretrained(model_version: str, lora_r: int, lora_alpha: int, dropout: float) -> PeftModel:
    print(f"🤩 Loading pretrained model version: {model_version}")
    model = TabStarModel.from_pretrained(model_version, local_files_only=True)
    # TODO: probably best if this is written more generic and not so hard-coded
    lora_modules = ["query", "key", "value", "out_proj", "linear1", "linear2",
                    "cls_head.layers.0", "reg_head.layers.0"]
    to_freeze = range(6)
    prefixes = tuple(f"text_encoder.encoder.layer.{i}." for i in to_freeze)
    to_exclude = [name for name, _ in model.named_modules() if name.startswith(prefixes)]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * lora_alpha,
        target_modules=lora_modules,
        exclude_modules=to_exclude,
        lora_dropout=dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model


def load_finetuned(save_dir: str, tabstar_version: str) -> PeftModel:
    if not exists(save_dir):
        raise FileNotFoundError(f"Checkpoint path {save_dir} does not exist.")
    base_model = TabStarModel.from_pretrained(tabstar_version, local_files_only=True)
    model = PeftModel.from_pretrained(base_model, save_dir, device_map='cpu', local_files_only=True)
    return model