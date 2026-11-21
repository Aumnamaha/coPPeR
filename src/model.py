"""CopperForCausalLM - from-scratch ~1B parameter causal Transformer.

Llama/Mistral-style architecture with:
- RMSNorm (pre-normalisation)
- RoPE rotary embeddings
- Grouped-query attention (GQA)
- SwiGLU feed-forward network
- torch.nn.functional.scaled_dot_product_attention for attn computation
- KV-cache autoregressive generation
- Gradient checkpointing support
- safetensors save / load helpers
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CopperConfig:
    """Hyper-parameters for the ~1B parameter coPPeR model."""

    vocab_size: int = 32768
    hidden_size: int = 1536
    intermediate_size: int = 5632
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4096
    rope_theta: float = 100_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    use_gradient_checkpointing: bool = False

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        n = self.num_hidden_layers
        d = self.hidden_size
        head_dim = d // self.num_attention_heads
        kv_groups = self.num_attention_heads // self.num_key_value_heads
        kv_dim = head_dim * kv_groups

        # Embedding + unembedding (untied)
        params = 2 * self.vocab_size * d

        # Per-layer parameters
        per_layer = 0
        per_layer += 2 * d  # rms_norm (gamma for attn, gamma for mlp)
        # Attention projections (no bias)
        per_layer += d * d + d * kv_dim + d * kv_dim + d * d  # q,k,v,o
        # MLP SwiGLU (no bias)
        per_layer += (
            d * self.intermediate_size * 2 + self.intermediate_size * d
        )  # gate, up, down
        params += n * per_layer

        return params


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (pre-norm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Embeddings
# ---------------------------------------------------------------------------


def _precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10_000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin frequency cache for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Apply rotary embeddings to x (batch, heads, seq_len, head_dim)."""
    if position_ids is not None:
        cos = cos.index_select(0, position_ids)
        sin = sin.index_select(0, position_ids)
    else:
        seq_len = x.shape[2]
        cos = cos[:seq_len]
        sin = sin[:seq_len]

    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()

    cos = cos.squeeze(1).unsqueeze(0).unsqueeze(0)
    sin = sin.squeeze(1).unsqueeze(0).unsqueeze(0)

    out = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return out.reshape(*x.shape).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary embedding cache with support for KV-cache offset."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.register_buffer(
            "cos_sin",
            torch.zeros(max_position_embeddings, 1, head_dim // 2, 2),
            persistent=False,
        )
        cos, sin = _precompute_freqs_cis(head_dim, max_position_embeddings, theta)
        self.cos_sin[:, :, :, 0] = cos
        self.cos_sin[:, :, :, 1] = sin

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors for rotary embedding."""
        if position_ids is not None:
            start = position_ids.max().item() + 1
        else:
            seq_len = x.shape[2]
            start = self.cos_sin.shape[0] - seq_len
            start = max(start, 0)
        end = min(start + x.shape[2], self.cos_sin.shape[0])
        chunk = self.cos_sin[start:end]
        cos = chunk[:, :, :, 0].to(x.dtype)
        sin = chunk[:, :, :, 1].to(x.dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Attention - GQA + KV-cache support
# ---------------------------------------------------------------------------


class CopperAttention(nn.Module):
    """Multi-head attention with grouped-query and optional KV cache."""

    def __init__(self, config: CopperConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert config.hidden_size % self.num_heads == 0

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.rotary = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query-head count for GQA."""
        if self.kv_groups == 1:
            return x
        b, _h, slen, d = x.shape
        return (
            x.view(b, self.num_kv_heads, 1, slen, d)
            .expand(-1, -1, self.kv_groups, -1, -1)
            .reshape(b, self.num_heads, slen, d)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor] | None]:
        """Forward pass through the attention layer."""
        bsz, q_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Rotary embeddings — apply BEFORE concatenation so each tensor gets
        # the correct position offset.  Cached K/V already have their RoPE from
        # when they were generated.
        if past_key_value is not None:
            past_len = past_key_value[0].shape[2]
            q_pos = torch.arange(
                past_len, past_len + q_len, device=query_states.device
            ).unsqueeze(0)
            cos_q, sin_q = self.rotary(query_states, position_ids=q_pos)
            query_states = _apply_rotary_emb(query_states, cos_q, sin_q)

            k_pos = torch.arange(
                past_len, past_len + q_len, device=key_states.device
            ).unsqueeze(0)
            cos_k, sin_k = self.rotary(key_states, position_ids=k_pos)
            key_states = _apply_rotary_emb(key_states, cos_k, sin_k)
        else:
            # Training path: all tokens at [0..seq_len-1]
            cos, sin = self.rotary(query_states)
            query_states = _apply_rotary_emb(query_states, cos, sin)
            key_states = _apply_rotary_emb(key_states, cos, sin)

        # KV-cache concatenation (after RoPE so cached K/V keep original embeddings)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        next_kv_cache = (key_states, value_states) if use_cache else None

        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=(past_key_value is None and q_len > 1),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output), next_kv_cache


# ---------------------------------------------------------------------------
# MLP - SwiGLU feed-forward network
# ---------------------------------------------------------------------------


class CopperMLP(nn.Module):
    """SwiGLU feed-forward network (no bias)."""

    def __init__(self, config: CopperConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder layer - Pre-Norm residual connections
# ---------------------------------------------------------------------------


class CopperDecoderLayer(nn.Module):
    """A single Transformer decoder layer."""

    def __init__(self, config: CopperConfig) -> None:
        super().__init__()
        self.self_attn = CopperAttention(config)
        self.mlp = CopperMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor] | None]:
        """Forward pass with pre-normalisation residual connections."""
        attn_output, next_kv = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output

        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + mlp_output

        return hidden_states, next_kv


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class CopperForCausalLM(nn.Module):
    """~1B parameter causal language model built from scratch."""

    config_class = CopperConfig

    def __init__(self, config: CopperConfig | None = None) -> None:
        super().__init__()
        self.config = config or CopperConfig()
        cfg = self.config

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [CopperDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self._use_gradient_checkpointing = False
        for layer in self.layers:
            _ = layer.self_attn.rotary

    @property
    def use_gradient_checkpointing(self) -> bool:
        return self._use_gradient_checkpointing

    def enable_gradient_checkpointing(self) -> None:
        self._use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._use_gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        past_key_values: list[tuple[torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """Forward pass through the model."""
        hidden_states = self.embed_tokens(input_ids)

        # Pass through decoder layers - causality handled entirely by
        # CopperAttention.is_causal (no explicit attention_mask needed).
        presents = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            kwargs: dict[str, Any] = {
                "hidden_states": hidden_states,
                "use_cache": use_cache,
            }
            if past_key_values is not None:
                kwargs["past_key_value"] = past_key_values[idx]

            if self._use_gradient_checkpointing and self.training:
                _layer = layer

                def _checkpoint_fn(hs, layer=_layer):
                    return layer.forward(
                        hs, attention_mask=None, past_key_value=None, use_cache=False
                    )

                out_tuple = torch.utils.checkpoint.checkpoint(
                    _checkpoint_fn,
                    kwargs["hidden_states"],
                    use_reentrant=True,
                )
                hidden_states = (
                    out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                )
            else:
                hidden_states, next_kv = layer(**kwargs)
                if presents is not None:
                    presents.append(next_kv)

        hidden_states = self.norm(hidden_states)
        logits_fn = (
            self.lm_head if not self.config.tie_word_embeddings else self.embed_tokens
        )
        logits = logits_fn(hidden_states)

        loss = None
        if labels is not None:
            # Standard causal LM loss: shift logits and labels
            shift_logits = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)
            valid_mask = shift_labels != -100
            if not valid_mask.any():
                # All labels masked — return zero loss instead of nan
                loss = torch.tensor(
                    0.0, device=shift_logits.device, dtype=shift_logits.dtype
                )
            else:
                loss = F.cross_entropy(
                    shift_logits[valid_mask],
                    shift_labels[valid_mask],
                    ignore_index=-100,
                )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": presents if use_cache else None,
        }

    # ------------------------------------------------------------------
    # Generation (autoregressive with KV cache)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        eos_token_id: int = 3,
    ) -> torch.LongTensor:
        """Autoregressive generation with KV caching."""
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                out = self(input_ids=generated, use_cache=True)
                logits = out["logits"][:, -1, :]
                past_key_values = out["past_key_values"]
            else:
                last_token = generated[:, -1:]
                out = self(
                    input_ids=last_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = out["logits"][:, -1, :]
                past_key_values = out["past_key_values"]

            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated

    # ------------------------------------------------------------------
    # Save / Load helpers
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str | Path) -> None:
        """Save model weights (safetensors) and config to save_dir."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        from safetensors.torch import save_file

        state_dict = self.state_dict()
        save_file(state_dict, str(save_dir / "model.safetensors"))

        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_dir: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> CopperForCausalLM:
        """Load model weights (safetensors) and config from load_dir."""
        load_dir = Path(load_dir)
        with open(load_dir / "config.json", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        config = CopperConfig(**cfg_dict)
        model = cls(config).to(dtype).to(device)

        from safetensors.torch import load_file

        state_dict = load_file(str(load_dir / "model.safetensors"), device=device)
        model.load_state_dict(state_dict, strict=True)
        return model

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return self.config.count_parameters()

    def parameter_count_bytes(self, dtype: torch.dtype = torch.bfloat16) -> int:
        """Estimate bytes needed to store all weights in dtype."""
        byte_size = 2 if dtype == torch.bfloat16 else 4
        return self.config.count_parameters() * byte_size


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_model_config() -> CopperConfig:
    """Return the default ~1B parameter configuration for coPPeR."""
    return CopperConfig()
