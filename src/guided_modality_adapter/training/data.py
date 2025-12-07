import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class UtteranceExample:
    audio_path: str
    transcript_ids: List[int]
    speaker_segments: Optional[List[Tuple[float, float, int]]] = None


class SAASRDataset(Dataset):
    """
    Generic dataset for SA-ASR.

    Assumes an external manifest (e.g., JSON or TSV) has been parsed into
    a list of UtteranceExample objects. Here we keep it abstract and expect
    a list of dicts/structures to be passed in.
    """

    def __init__(
        self,
        examples: List[UtteranceExample],
        tokenizer,
        sample_rate: int = 16000,
        max_duration_sec: Optional[float] = None,
        audio_loader=None,
    ):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        # audio_loader: callable(path, sample_rate) -> Tensor (samples,)
        self.audio_loader = audio_loader or self._default_audio_loader

    def __len__(self) -> int:
        return len(self.examples)

    def _default_audio_loader(self, path: str, sample_rate: int) -> Tensor:
        raise NotImplementedError(
            "Provide an audio_loader that returns a 1D float tensor at the target sample rate."
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]

        waveform = self.audio_loader(ex.audio_path, self.sample_rate)  # (samples,)
        transcript_ids = torch.tensor(ex.transcript_ids, dtype=torch.long)

        item: Dict[str, Any] = {
            "waveform": waveform,
            "transcript_ids": transcript_ids,
            "audio_path": ex.audio_path,
        }
        if ex.speaker_segments is not None:
            item["speaker_segments"] = ex.speaker_segments
        return item


def saasr_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, Any]:
    """
    Collate function for variable-length audio and text.

    Returns:
        {
          "waveforms": (B, T_max),
          "waveform_lengths": (B,),
          "transcript_ids": (B, L_max),
          "transcript_lengths": (B,),
          "speaker_segments": list[list[(start, end, spk_id)]]
        }
    """
    # Waveforms
    waveforms = [b["waveform"] for b in batch]
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())
    B = len(batch)

    padded_waveforms = waveforms[0].new_zeros(B, max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, : w.shape[-1]] = w

    # Transcripts
    transcripts = [b["transcript_ids"] for b in batch]
    t_lens = torch.tensor([t.shape[-1] for t in transcripts], dtype=torch.long)
    max_t_len = int(t_lens.max().item())

    padded_transcripts = transcripts[0].new_full((B, max_t_len), pad_token_id)
    for i, t in enumerate(transcripts):
        padded_transcripts[i, : t.shape[-1]] = t

    # Speaker segments (kept as list of lists; model utilities can align later)
    speaker_segments = [b.get("speaker_segments", []) for b in batch]

    return {
        "waveforms": padded_waveforms,
        "waveform_lengths": lengths,
        "transcript_ids": padded_transcripts,
        "transcript_lengths": t_lens,
        "speaker_segments": speaker_segments,
        "audio_paths": [b["audio_path"] for b in batch],
    }


class SAASRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_examples: List[UtteranceExample],
        val_examples: List[UtteranceExample],
        tokenizer,
        sample_rate: int = 16000,
        batch_size: int = 4,
        num_workers: int = 4,
        audio_loader=None,
    ):
        super().__init__()
        self.train_examples = train_examples
        self.val_examples = val_examples
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.audio_loader = audio_loader

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SAASRDataset(
            self.train_examples,
            tokenizer=self.tokenizer,
            sample_rate=self.sample_rate,
            audio_loader=self.audio_loader,
        )
        self.val_dataset = SAASRDataset(
            self.val_examples,
            tokenizer=self.tokenizer,
            sample_rate=self.sample_rate,
            audio_loader=self.audio_loader,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda b: saasr_collate_fn(
                b, pad_token_id=self.tokenizer.pad_token_id
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: saasr_collate_fn(
                b, pad_token_id=self.tokenizer.pad_token_id
            ),
        )
