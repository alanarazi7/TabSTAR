import datetime
from os import makedirs
from os.path import join, basename
from typing import List, Dict, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class CheckpointManager:

    def __init__(self, do_average: bool, output_dir: Optional[str]):
        output_dir = output_dir or ".tabstar_checkpoint/"
        self.save_dir = join(output_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        makedirs(self.save_dir, exist_ok=True)
        self.avg_dir = join(self.save_dir, "averaged_model")
        self.best_dir = join(self.save_dir, "best_model")
        self.to_load_dir = self.best_dir
        self.cp_paths: List[str] = []
        self.val_losses: List[float] = []
        self.do_average = do_average
        self.avg_metric: Optional[float] = None

    def save_checkpoint(self, model: nn.Module, epoch: int, val_loss: float):
        """Save a checkpoint for later averaging (only model weights needed)"""
        if not self.do_average:
            return
        checkpoint_path = join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        checkpoint = {'model_dict': model.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        self.cp_paths.append(checkpoint_path)
        self.val_losses.append(val_loss)
        assert len(self.cp_paths) == len(self.val_losses) == epoch

    def average_checkpoints(self, model: nn.Module, evaluator: Callable, val_loader: DataLoader):
        if not self.do_average:
            return
        if len(self.cp_paths) < 2:
            print(f"⚠️ Only {len(self.cp_paths)} checkpoint(s) available, skipping averaging")
            return
        best_loss = min(self.val_losses)
        best_idx = int(np.argmin(self.val_losses))
        best_epoch = best_idx + 1
        print(f"🏆 Best checkpoint: Epoch {best_epoch} with loss {best_loss:.4f}")
        threshold = self.adam_smooth_minmax(best_loss)
        print(f"Threshold: {threshold:.4f} was chosen for best val loss of {best_loss:.4f}")
        cps_to_avg = [path for loss, path in zip(self.val_losses, self.cp_paths) if loss <= threshold]
        if len(cps_to_avg) < 1:
            print(f"⚠️ No checkpoints selected for averaging, skipping")
            return
        self.to_load_dir = self.avg_dir
        print(f"📊 Averaging {len(cps_to_avg)} checkpoints:")
        for cp_path in cps_to_avg:
            cp_idx = self.cp_paths.index(cp_path)
            marker = " 🏆" if cp_idx == best_epoch else ""
            print(f"- {basename(cp_path)} (val_loss={self.val_losses[cp_idx]:.4f}){marker}")

        averaged_weights = self.average_lora_checkpoints(cps_to_avg)
        averaged_checkpoint_path = join(self.save_dir, "checkpoint_averaged.pt")
        self.save_averaged_lora_checkpoint(base_checkpoint_path=cps_to_avg[0], averaged_lora_weights=averaged_weights,
                                           output_path=averaged_checkpoint_path)

        # Load averaged weights into model and save as a model
        model.load_state_dict(torch.load(averaged_checkpoint_path)['model_dict'], strict=False)
        averaged_model_dir = join(self.save_dir, "averaged_model")
        model.save_pretrained(averaged_model_dir)
        print(f"✅ Saved averaged model to {averaged_model_dir}")
        avg_val_loss, avg_val_metric = evaluator(val_loader)
        print(f"📈 Averaged checkpoint || Val Loss: {avg_val_loss:.4f} || Val Metric: {avg_val_metric:.4f}")
        self.avg_metric = avg_val_metric

    @classmethod
    def adam_smooth_minmax(cls, x):
        # Thanks to Adam Nathaniel for the "MinMax" idea
        return x + 0.02 + 0.03 * (x / (x + 0.3))


    @classmethod
    def extract_lora_weights(cls, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """
        Extract only LoRA adapter weights from a checkpoint.

        LoRA weights are named like:
        - base_model.model.{module_name}.lora_A.weight
        - base_model.model.{module_name}.lora_B.weight
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_dict = checkpoint['model_dict']

        lora_weights = {}
        for key, value in model_dict.items():
            if 'lora_A' in key or 'lora_B' in key:
                lora_weights[key] = value
        return lora_weights

    @classmethod
    def average_lora_checkpoints(cls, checkpoint_paths: List[str]) -> Dict[str, torch.Tensor]:
        """
        LoRA adds two low-rank matrices A and B for each adapter layer:
        ΔW = B @ A  # Update to original weight matrix W
        W_new = W_frozen + α * (B @ A)
        B_avg = (B₁ + B₂ + ... + Bₙ) / n
        A_avg = (A₁ + A₂ + ... + Aₙ) / n

        Average LoRA adapter weights across multiple checkpoints.We naively average A and B matrices separately.
        """
        all_lora_weights = [cls.extract_lora_weights(path) for path in checkpoint_paths]
        # Verify all checkpoints have the same keys
        keys = set(all_lora_weights[0].keys())
        for i, weights in enumerate(all_lora_weights[1:], 1):
            if set(weights.keys()) != keys:
                raise ValueError(f"Checkpoint {i} has different LoRA parameters!")
        averaged_weights = {}
        for key in keys:
            stacked = torch.stack([weights[key] for weights in all_lora_weights])
            averaged_weights[key] = stacked.mean(dim=0)
        return averaged_weights

    @classmethod
    def save_averaged_lora_checkpoint(cls, base_checkpoint_path: str, averaged_lora_weights: Dict[str, torch.Tensor],
                                      output_path: str):
        """
        Save a new checkpoint with averaged LoRA weights.

        Args:
            base_checkpoint_path: Path to one of the original checkpoints (for metadata)
            averaged_lora_weights: Dictionary of averaged LoRA weights
            output_path: Path to save the averaged checkpoint
        """
        # Load the base checkpoint to get structure and non-LoRA components
        checkpoint = torch.load(base_checkpoint_path, map_location='cpu')

        # Replace LoRA weights with averaged ones
        model_dict = checkpoint['model_dict']
        for key in averaged_lora_weights:
            model_dict[key] = averaged_lora_weights[key]
        checkpoint['model_dict'] = model_dict

        # Save the new checkpoint
        torch.save(checkpoint, output_path)
        print(f"💾 Saved averaged checkpoint to {output_path}")

