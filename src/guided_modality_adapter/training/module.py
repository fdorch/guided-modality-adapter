from typing import Any, Dict, Optional

import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from guided_modality_adapter.model.model import SAASRModel


class SAASRLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: SAASRModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 1000,
        max_steps: int = 100_000,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self._lr = lr
        self._weight_decay = weight_decay
        self._warmup_steps = warmup_steps
        self._max_steps = max_steps

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        # Expect SAASRModel.forward to accept:
        #   waveform: (B, T)
        #   targets_text_ids: (B, L)
        #   target_speaker_ids: optional (B, L) or similar
        waveforms = batch["waveforms"]
        transcript_ids = batch["transcript_ids"]
        speaker_segments = batch.get("speaker_segments", None)

        out = self.model(
            waveform=waveforms,
            targets_text_ids=transcript_ids,
            target_speaker_ids=speaker_segments,
        )
        return out

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        out = self.forward(batch)
        loss = out["loss"] if isinstance(out, dict) else out
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if isinstance(out, dict) and "losses" in out:
            for name, val in out["losses"].items():
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        out = self.forward(batch)
        loss = out["loss"] if isinstance(out, dict) else out
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if isinstance(out, dict) and "losses" in out:
            for name, val in out["losses"].items():
                self.log(f"val/{name}", val, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        # Only train adapters: filter parameters with requires_grad=True
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            params,
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._warmup_steps,
            num_training_steps=self._max_steps,
        )
        scheduler_cfg = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]