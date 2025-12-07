from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
 
from guided_modality_adapter.model.utils.audio import w2t


@dataclass
class UtteranceExample:
    audio_path: str
    transcript: str
    speaker_segments: Optional[List[Tuple[float, float, int]]] = None


def load_manifest_jsonl(path: Path) -> List[UtteranceExample]:
    examples: List[UtteranceExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            segs = obj.get("speaker_segments")
            if segs is not None:
                segs = [(float(s), float(e), int(spk)) for s, e, spk in segs]
            examples.append(
                UtteranceExample(
                    audio_path=obj["audio_path"],
                    transcript=obj["transcript"],
                    speaker_segments=segs,
                )
            )
    return examples


class SAASRDataset(Dataset):
    """
    Generic dataset: waveform + transcript (+ optional speaker segments).
    """

    def __init__(
        self,
        examples: List[UtteranceExample],
        tokenizer,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        # use your audio util to load and resample
        waveform: Tensor = w2t(ex.audio_path, self.sample_rate)  # (samples,)

        tok = self.tokenizer(
            ex.transcript,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # assume tokenizer returns input_ids with shape (1, L)
        transcript_ids = tok["input_ids"].squeeze(0).to(torch.long)

        item: Dict[str, Any] = {
            "waveform": waveform,
            "transcript_ids": transcript_ids,
            "audio_path": ex.audio_path,
        }
        if ex.speaker_segments is not None:
            item["speaker_segments"] = ex.speaker_segments
        return item


def _collate_saasr(
    batch: List[Dict[str, Any]], pad_token_id: int
) -> Dict[str, Any]:
    # waveforms: pad to max length
    waveforms = [b["waveform"] for b in batch]
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())
    B = len(batch)

    padded_waveforms = waveforms[0].new_zeros(B, max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, : w.shape[-1]] = w

    # transcripts: pad with pad_token_id
    transcripts = [b["transcript_ids"] for b in batch]
    t_lens = torch.tensor([t.shape[-1] for t in transcripts], dtype=torch.long)
    max_t_len = int(t_lens.max().item())

    padded_transcripts = transcripts[0].new_full((B, max_t_len), pad_token_id)
    for i, t in enumerate(transcripts):
        padded_transcripts[i, : t.shape[-1]] = t

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
    """
    LightningDataModule configurable from configs/data.yaml
    """

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        tokenizer,
    ):
        super().__init__()
        self.data_cfg = data_cfg["data"]
        self.tokenizer = tokenizer

        self.root = Path(self.data_cfg["root_dir"])
        self.sample_rate = int(self.data_cfg.get("sample_rate", 16000))
        self.batch_size = int(self.data_cfg.get("batch_size", 4))
        self.num_workers = int(self.data_cfg.get("num_workers", 4))

        self.train_dataset: Optional[SAASRDataset] = None
        self.val_dataset: Optional[SAASRDataset] = None

    def setup(self, stage: Optional[str] = None):
        train_manifest = self.root / self.data_cfg["train_manifest"]
        val_manifest = self.root / self.data_cfg["val_manifest"]

        train_examples = load_manifest_jsonl(train_manifest)
        val_examples = load_manifest_jsonl(val_manifest)

        self.train_dataset = SAASRDataset(
            train_examples, tokenizer=self.tokenizer, sample_rate=self.sample_rate
        )
        self.val_dataset = SAASRDataset(
            val_examples, tokenizer=self.tokenizer, sample_rate=self.sample_rate
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda b: _collate_saasr(
                b, pad_token_id=self.tokenizer.pad_token_id
            ),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: _collate_saasr(
                b, pad_token_id=self.tokenizer.pad_token_id
            ),
        )