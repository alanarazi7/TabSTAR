from os.path import basename, exists
from typing import List, Optional

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer

from tabstar.tabstarv1p1.constants import TEXT_MODEL as E5_SMALL
from tabstar.tabstarv1p1.finetune.hyperparams import LORA_ALPHA
from tabstar.tabstarv1p1.finetune.lora_params import get_all_lora_modules
from tabstar.tabstarv1p1.arch.model import TabSTARModel
from tabstar.tabstarv1p1.arch.config import TabStarConfig
from tabstar.tabstarv1p1.utils.summary import summarize_model


def load_pretrained_base(model_version: str) -> TabSTARModel:
    """Loads the base model once so a caller finetuning many datasets from the same starting
    checkpoint can reuse it instead of reloading from disk each time. Safe to reuse across LoRA
    finetunes: the base weights stay frozen and unchanged throughout.
    """
    print(f"🤩 Loading pretrained model version: {model_version}")
    config = TabStarConfig.from_pretrained(model_version)
    model = TabSTARModel.from_pretrained(pretrained_model_name_or_path=model_version, config=config)
    model.tokenizer = AutoTokenizer.from_pretrained(E5_SMALL)
    return model


def wrap_with_lora(model: TabSTARModel, lora_r: int, dropout: float, lora_alpha: int = LORA_ALPHA,
                    to_unfreeze: int = 6, is_cls: Optional[bool] = None) -> PeftModel:
    target_modules = get_all_lora_modules(model=model, to_unfreeze=to_unfreeze, is_cls=is_cls)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * lora_alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # print_all_modules(model=model)
    summarize_model(model=model.base_model.model)
    return model


def load_finetuned(adapter_dirs: List[str], tabstar_version: str) -> PeftModel:
    """One base model carrying every adapter, each named after its directory."""
    for adapter_dir in adapter_dirs:
        if not exists(adapter_dir):
            raise FileNotFoundError(f"Checkpoint path {adapter_dir} does not exist.")
    base_model = TabSTARModel.from_pretrained(tabstar_version, device_map="cpu")
    base_model.tokenizer = AutoTokenizer.from_pretrained(E5_SMALL)
    first, *rest = adapter_dirs
    model = PeftModel.from_pretrained(base_model, first, adapter_name=basename(first), device_map='cpu')
    for adapter_dir in rest:
        model.load_adapter(adapter_dir, adapter_name=basename(adapter_dir))
    return model
