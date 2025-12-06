from typing import Any, Dict, Optional

import torch.nn as nn
from components.attn import CrossAttentionBlock
from components.qformer import QueryFormerBase
from torch import Tensor


class SpeakerQueryFormer(QueryFormerBase):
    def __init__(
        self,
        spk_in_dim: int,
        num_queries: int,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_dim=spk_in_dim,
            num_queries=num_queries,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x: Tensor, input_mask: Optional[Tensor] = None) -> Tensor:
        return super().forward(x, input_mask)


class SpeechQueryFormer(QueryFormerBase):
    def __init__(
        self,
        speech_in_dim: int,
        num_queries: int,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_dim=speech_in_dim,
            num_queries=num_queries,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x: Tensor, input_mask: Optional[Tensor] = None) -> Tensor:
        return super().forward(x, input_mask)


class UnifiedProjection(nn.Module):
    """
    Fusion module that:
      - takes Q_spk := SpeakerQueryFormer output (B, Q_spk, D)
      - takes Q_speech := SpeechQueryFormer output (B, Q_speech, D)
      - runs alternating cross-attention blocks so speaker->speech and speech->speaker information is exchanged
      - returns updated speech queries (B, Q_speech, D) as fused sequence
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_blocks: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        # blocks: reused two CrossAttentionBlocks per round:
        # 1) speaker queries attend to speech queries
        # 2) speech queries attend to speaker queries
        self.spk2speech_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_model * ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.speech2spk_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_model * ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # final projection to match LLaMA embedding dim if needed; identity by default
        self.out_proj = nn.Identity()

    def forward(
        self,
        Q_spk: Tensor,
        Q_speech: Tensor,
        spk_mask: Optional[Tensor] = None,
        speech_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
          Q_spk: (B, Q_spk, D)
          Q_speech: (B, Q_speech, D)
          spk_mask: (B, Q_spk) boolean mask where True indicates padding (optional)
          speech_mask: (B, Q_speech) boolean mask where True indicates padding (optional)

        Returns:
          fused_speech_queries: (B, Q_speech, D)
        """
        # alternate updates
        spk = Q_spk
        speech = Q_speech

        for i in range(self.n_blocks):
            # speaker queries query speech queries -> speaker queries receive speech context
            spk = self.spk2speech_blocks[i](spk, speech, kv_mask=speech_mask)

            # speech queries query (updated) speaker queries -> speech queries receive speaker context
            speech = self.speech2spk_blocks[i](speech, spk, kv_mask=spk_mask)

        # final projection
        fused = self.out_proj(speech)  # (B, Q_speech, D)
        return fused


class SAASRAdapter(nn.Module):
    """
    Projectors wrapper:
      - builds both QueryFormers and UnifiedProjection
      - forward: (H_speech_frames, H_spk_windows, masks) -> fused_speech_queries
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.speech_qformer = SpeechQueryFormer(
            cfg["speech"].get("speech_in_dim", 1024),
            cfg["speech"].get("q_speech", 256),
            cfg["speech"].get("d_model", 4096),
            n_heads=cfg["speech"].get("n_heads", 4),
            dropout=cfg["speech"].get("dropout", 0.1),
        )
        self.spk_qformer = SpeakerQueryFormer(
            cfg["speaker"].get("spk_in_dim", 512),
            cfg["speaker"].get("q_spk", 128),
            cfg["speaker"].get("d_model", 4096),
            n_heads=cfg["speaker"].get("n_heads", 4),
            dropout=cfg["speaker"].get("dropout", 0.1),
        )
        self.unified = UnifiedProjection(
            d_model=cfg["unified"].get("d_model", 4096),
            n_heads=cfg["unified"].get("n_heads", 4),
            n_blocks=cfg["unified"].get("n_blocks", 2),
            dropout=cfg["unified"].get("dropout", 0.1),
        )

    def forward(
        self,
        H_speech: Tensor,
        H_spk: Tensor,
        speech_mask: Optional[Tensor] = None,
        spk_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
          H_speech: (B, T_speech, speech_in_dim)
          H_spk:    (B, N_spk, spk_in_dim)

        Returns:
          fused_speech_queries: (B, Q_speech, d_model)
        """
        Q_speech = self.speech_qformer(
            H_speech, input_mask=speech_mask
        )  # (B, Q_speech, D)
        Q_spk = self.spk_qformer(H_spk, input_mask=spk_mask)  # (B, Q_spk, D)

        fused = self.unified(
            Q_spk, Q_speech, spk_mask=spk_mask, speech_mask=speech_mask
        )
        return fused  # (B, Q_speech, D)
