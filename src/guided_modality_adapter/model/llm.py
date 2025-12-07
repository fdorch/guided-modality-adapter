from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch import LongTensor, Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.extend_tokenizer import add_timestamp_tokens, extend_tokenizer_with_speakers


class LLMModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super(LLMModel, self).__init__()
        self.cfg = cfg
        self.model_name = cfg.get("model_name", "meta-llama/Llama-3.2-1B-Instruct")

        # Load tokenizer and ensure pad token exists (previous fix)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Add special tokens
        self.spk_tokens = extend_tokenizer_with_speakers(
            self.tokenizer, cfg.get("num_speakers", 3)
        )
        self.timestamp_tokens = add_timestamp_tokens(
            self.tokenizer, cfg.get("max_time_s", 30.0), cfg.get("step_ms", 20)
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load LLM and resize embeddings after adding tokens
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # PEFT / LoRA
        target_modules = cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        lora_config = LoraConfig(
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            target_modules=target_modules,
            lora_dropout=cfg.get("lora_dropout", 0.05),
            task_type="CAUSAL_LM",
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # ---------------------------
        # Projector: adapter_dim -> llm token embedding dim
        # ---------------------------
        llm_embed_dim = self.model.get_input_embeddings().weight.shape[1]
        adapter_dim = cfg.get("adapter_dim", cfg.get("projector_adapter_dim", 4096))
        self.fused_projector = nn.Linear(adapter_dim, llm_embed_dim).to(self.device)
        # optional layernorm
        self.fused_ln = nn.LayerNorm(llm_embed_dim).to(self.device)

    def build_inputs_embeds(
        self,
        fused_emb: Tensor,
        prompt_texts: Optional[List[str]] = None,
        prompt_token_ids: Optional[torch.LongTensor] = None,
        prompt_as_text_first: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        fused_emb: (B, L_fused, adapter_dim)  -- adapter_dim must match cfg adapter_dim
        prompt_texts: list[str] length B, or a single string (broadcast)
        prompt_token_ids: NOT used in this implementation; keep for compatibility
        Returns:
            inputs_embeds: (B, L_total, llm_embed_dim)
            attention_mask: (B, L_total) torch.long
        """

        # Basic checks + broadcasting of prompt_texts
        B = fused_emb.size(0)
        device = fused_emb.device

        if prompt_texts is None and prompt_token_ids is None:
            # use empty prompt
            prompt_texts = [""] * B
        elif isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts] * B
        elif isinstance(prompt_texts, list) and len(prompt_texts) != B:
            # If provided single prompt as list of words (bug seen), broadcast the whole string
            if len(prompt_texts) > 0 and all(isinstance(x, str) for x in prompt_texts):
                # assume user accidentally passed token list; fallback to single prompt broadcast
                prompt_text = " ".join(prompt_texts)
                prompt_texts = [prompt_text] * B
            else:
                # fallback: if mismatch, raise clearer error
                raise ValueError(
                    f"prompt_texts length ({len(prompt_texts)}) != fused batch ({B})"
                )

        # Tokenize prompts (safe: padding token exists)
        enc = self.tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True
        )
        prompt_ids = enc["input_ids"].to(device)  # (B, Lp)
        prompt_mask = enc["attention_mask"].to(device)  # (B, Lp)

        # Prompt embeddings (from LLM embedding layer)
        with torch.no_grad():
            prompt_embeds = self.model.get_input_embeddings()(prompt_ids)  # (B, Lp, E)

        # Project fused embeddings to LLM embedding dim
        fused_proj = self.fused_projector(fused_emb)  # (B, Lf, E)
        fused_proj = self.fused_ln(fused_proj)

        # If dims mismatch (defensive), raise helpful error
        if prompt_embeds.size(-1) != fused_proj.size(-1):
            raise RuntimeError(
                f"Embedding dim mismatch: prompt_embeds {prompt_embeds.size(-1)} != fused_proj {fused_proj.size(-1)}"
            )

        # Concatenate in the chosen order
        if prompt_as_text_first:
            inputs_embeds = torch.cat(
                [prompt_embeds, fused_proj], dim=1
            )  # (B, Lp+Lf, E)
            attention_mask = torch.cat(
                [prompt_mask, torch.ones(B, fused_proj.size(1), device=device)], dim=1
            )
        else:
            inputs_embeds = torch.cat([fused_proj, prompt_embeds], dim=1)
            attention_mask = torch.cat(
                [torch.ones(B, fused_proj.size(1), device=device), prompt_mask], dim=1
            )

        return inputs_embeds, attention_mask.long()

    def forward(
        self,
        fused_emb: Tensor,
        prompt_texts: Optional[List[str]] = None,
        prompt_token_ids: Optional[LongTensor] = None,
        prompt_as_text_first: bool = True,
        labels: Optional[LongTensor] = None,
    ) -> Dict[str, Any]:
        inputs_embeds, attention_mask = self.build_inputs_embeds(
            fused_emb,
            prompt_texts=prompt_texts,
            prompt_token_ids=prompt_token_ids,
            prompt_as_text_first=prompt_as_text_first,
        )

        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels.to(self.device) if labels is not None else None,
        )
        return outputs
