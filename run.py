import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.guided_modality_adapter.model.llm import LLMModel  # to get tokenizer
from src.guided_modality_adapter.training.data import SAASRDataModule
from src.guided_modality_adapter.training.module import SAASRLightningModule


def main():
    cwd = Path(os.getcwd())
    cfg_dir = cwd / "configs"

    model_cfg_path = cfg_dir / "model.yaml"
    data_cfg_path = cfg_dir / "data.yaml"

    model_cfg = OmegaConf.load(model_cfg_path)
    data_cfg = OmegaConf.load(data_cfg_path)
    # merge into a single config tree for LightningModule / SAASRModel
    config = OmegaConf.merge(model_cfg, data_cfg)

    # Build tokenizer via your LLM wrapper so data module can tokenize transcripts
    # Assumes config.model.llama_name_or_path exists
    llama_name = config.model.get("llama_name_or_path", None)
    if llama_name is None:
        raise ValueError("config.model.llama_name_or_path must be set in model.yaml")

    llm_wrapper = LLMModel(
        model_name_or_path=llama_name,
        freeze_llm=True,  # training adapters only; can be overridden in config
        load_model=False,  # if your wrapper supports lazy loading of full weights
    )
    tokenizer = llm_wrapper.tokenizer

    datamodule = SAASRDataModule(data_cfg=data_cfg, tokenizer=tokenizer)

    lightning_module = SAASRLightningModule(config=config)

    t_cfg = config.training
    trainer = pl.Trainer(
        max_epochs=int(t_cfg.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=int(t_cfg.get("precision", 16)),
        gradient_clip_val=float(t_cfg.get("grad_clip", 1.0)),
        log_every_n_steps=int(t_cfg.get("log_every_n_steps", 50)),
    )

    trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()