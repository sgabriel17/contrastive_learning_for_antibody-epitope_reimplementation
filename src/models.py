"""Contrastive encoder wrappers for AntiBERTy and ESM-2."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, BertConfig, BertModel, BitsAndBytesConfig


ANTIBERTY_HIDDEN_SIZE = 512
ESM2_HIDDEN_SIZE = 1280
ESM2_MODEL_ID = "facebook/esm2_t33_650M_UR50D"


class Mixer(nn.Module):
    """Six linear layers with ReLU between layers, matching Holt et al."""

    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _bitsandbytes_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def _apply_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int,
    alpha: int,
    dropout: float,
    load_in_4bit: bool,
) -> nn.Module:
    matches = [name for name, _ in model.named_modules() if any(name.endswith(target) for target in target_modules)]
    if not matches:
        raise ValueError(f"No LoRA target modules found for {target_modules}. Verify module names with named_modules().")

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        for param in model.parameters():
            param.requires_grad = False

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, config)


def _mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    special_token_ids: set[int],
) -> torch.Tensor:
    special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for token_id in special_token_ids:
        special_mask |= input_ids == token_id
    mask = attention_mask.bool() & ~special_mask
    mask = mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-9)
    return summed / counts


def _antiberty_model_dir(model_dir: str | Path | None = None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    return Path(files("antiberty").joinpath("trained_models", "AntiBERTy_md_smooth"))


def _from_pretrained_kwargs(load_in_4bit: bool) -> dict:
    quantization_config = _bitsandbytes_config(load_in_4bit)
    if quantization_config is None:
        return {}
    return {"quantization_config": quantization_config, "device_map": "auto"}


def _config_token_id(config, attr: str, default: int) -> int:
    token_id = getattr(config, attr, None)
    return default if token_id is None else int(token_id)


def _load_antiberty_encoder(model_dir: str | Path | None, load_in_4bit: bool) -> BertModel:
    model_path = _antiberty_model_dir(model_dir)
    try:
        return BertModel.from_pretrained(str(model_path), **_from_pretrained_kwargs(load_in_4bit))
    except OSError:
        config = BertConfig.from_pretrained(str(model_path))
        model = BertModel(config)
        state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        if load_in_4bit:
            raise RuntimeError("AntiBERTy 4-bit loading requires a Hugging Face-style checkpoint directory.") from None
        return model


class AntiBERTyContrastive(nn.Module):
    """AntiBERTy heavy-chain-only encoder with a 512D contrastive head."""

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.3,
        load_in_4bit: bool = False,
        model_dir: str | Path | None = None,
    ):
        super().__init__()
        self.load_in_4bit = load_in_4bit
        self.encoder = _load_antiberty_encoder(model_dir, load_in_4bit=load_in_4bit)
        self.encoder = _apply_lora(
            self.encoder,
            target_modules=["query", "value"],
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            load_in_4bit=load_in_4bit,
        )
        self.mixer = Mixer(ANTIBERTY_HIDDEN_SIZE)
        self.special_token_ids = {
            _config_token_id(self.encoder.config, "pad_token_id", 0),
            _config_token_id(self.encoder.config, "cls_token_id", 2),
            _config_token_id(self.encoder.config, "sep_token_id", 3),
        }

    def forward(self, h_input_ids: torch.Tensor, h_attention_mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=h_input_ids.long(), attention_mask=h_attention_mask.long())
        pooled = _mean_pool(outputs.last_hidden_state, h_attention_mask, h_input_ids, self.special_token_ids)
        return F.normalize(self.mixer(pooled), p=2, dim=1)


class ESM2Contrastive(nn.Module):
    """ESM-2 H+L concatenated encoder with a 1280D contrastive head."""

    def __init__(
        self,
        model_id: str = ESM2_MODEL_ID,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.3,
        load_in_4bit: bool = True,
    ):
        super().__init__()
        self.load_in_4bit = load_in_4bit
        self.encoder = AutoModel.from_pretrained(model_id, **_from_pretrained_kwargs(load_in_4bit))
        self.encoder = _apply_lora(
            self.encoder,
            target_modules=["q_proj", "v_proj"],
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            load_in_4bit=load_in_4bit,
        )
        self.mixer = Mixer(ESM2_HIDDEN_SIZE)
        self.special_token_ids = {
            _config_token_id(self.encoder.config, "pad_token_id", 1),
            _config_token_id(self.encoder.config, "cls_token_id", 0),
            _config_token_id(self.encoder.config, "eos_token_id", 2),
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        pooled = _mean_pool(outputs.last_hidden_state, attention_mask, input_ids, self.special_token_ids)
        return F.normalize(self.mixer(pooled), p=2, dim=1)


def build_model(
    encoder: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.3,
    load_in_4bit: bool | None = None,
) -> nn.Module:
    encoder = encoder.lower()
    if encoder == "antiberty":
        return AntiBERTyContrastive(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            load_in_4bit=torch.cuda.is_available() if load_in_4bit is None else load_in_4bit,
        )
    if encoder == "esm2":
        return ESM2Contrastive(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            load_in_4bit=torch.cuda.is_available() if load_in_4bit is None else load_in_4bit,
        )
    raise ValueError(f"Unsupported encoder '{encoder}'. Expected 'antiberty' or 'esm2'.")


def embedding_dim(encoder: str) -> int:
    if encoder.lower() == "antiberty":
        return ANTIBERTY_HIDDEN_SIZE
    if encoder.lower() == "esm2":
        return ESM2_HIDDEN_SIZE
    raise ValueError(f"Unsupported encoder '{encoder}'. Expected 'antiberty' or 'esm2'.")


def move_model_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Move non-quantized models normally; keep 4-bit encoders on their device map."""

    if getattr(model, "load_in_4bit", False):
        model.mixer.to(device)
        return model
    return model.to(device)
