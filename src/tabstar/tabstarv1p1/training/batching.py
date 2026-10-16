from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from transformers import BatchEncoding


@dataclass
class TabSTARBatch:
    x_num: Tensor
    x_txt: Tensor
    tokenized: Optional[BatchEncoding]
    y: Tensor
    feature_mask: Tensor
    d_outputs: Tensor
    is_cls: bool

    def to_device(self, device: torch.device, dtype: torch.dtype):
        # 'tokenized' isn't moved to device here, as it is done in the model forward pass in chunks
        self.x_txt = self.x_txt.to(device, non_blocking=True)
        self.x_num = self.x_num.to(device, non_blocking=True, dtype=dtype)
        y_dtype = torch.long if self.is_cls else torch.float32
        self.y = self.y.to(device, dtype=y_dtype, non_blocking=True)
        self.feature_mask = self.feature_mask.to(device, non_blocking=True)
        self.d_outputs = self.d_outputs.to(device, non_blocking=True)

    def slice(self, start: int, end: int) -> "TabSTARBatch":
        # 'tokenized' holds the batch's unique texts, so keep only those these rows use: sharing all of them would
        # encode the whole batch's texts for every slice, and splitting the rows on OOM would never shrink the text
        # encoder's activations.
        x_txt, tokenized = self.x_txt[start:end], self.tokenized
        if tokenized is not None:
            used, x_txt = torch.unique(x_txt, return_inverse=True)
            tokenized = BatchEncoding({k: v[used.to(v.device)] for k, v in tokenized.items()})
        return TabSTARBatch(
            x_num=self.x_num[start:end],
            x_txt=x_txt,
            tokenized=tokenized,
            y=self.y[start:end],
            feature_mask=self.feature_mask[start:end],
            d_outputs=self.d_outputs[start:end],
            is_cls=self.is_cls,
        )