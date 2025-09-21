import math
from typing import Final, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code taken from Pytorch's DiT official implementation and timm layers.

def use_fused_attn() -> bool:
    """Return True when torch's fused scaled dot product attention is available."""
    return hasattr(F, "scaled_dot_product_attention")


def maybe_add_mask(attn: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attn_mask is None:
        return attn
    return attn + attn_mask


def _expand_attn_mask(
    attn_mask: torch.Tensor,
    query_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if attn_mask.dim() == 2:
        if attn_mask.shape[0] != query_len:
            raise ValueError("attn_mask with 2 dims must match query length")
        attn_mask = attn_mask.unsqueeze(0)
    if attn_mask.dim() == 3:
        attn_mask = attn_mask.unsqueeze(1)
    elif attn_mask.dim() == 4:
        pass
    else:
        raise ValueError("attn_mask must have 2, 3, or 4 dims")

    if attn_mask.dtype == torch.bool:
        bool_mask = attn_mask
        attn_mask = torch.zeros_like(bool_mask, dtype=dtype)
        attn_mask = attn_mask.masked_fill(~bool_mask, torch.finfo(dtype).min)
    else:
        attn_mask = attn_mask.to(dtype)
    return attn_mask


def _expand_key_padding_mask(
    key_padding_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if key_padding_mask.dim() != 2:
        raise ValueError("key_padding_mask must be 2D (B, S)")
    expanded = torch.zeros(
        key_padding_mask.shape[0],
        1,
        1,
        key_padding_mask.shape[1],
        dtype=dtype,
        device=key_padding_mask.device,
    )
    expanded = expanded.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(dtype).min)
    return expanded


def build_attention_mask(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    query_len: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    combined: Optional[torch.Tensor] = None
    if attn_mask is not None:
        combined = _expand_attn_mask(attn_mask, query_len, dtype)
    if key_padding_mask is not None:
        kp_mask = _expand_key_padding_mask(key_padding_mask, dtype)
        combined = kp_mask if combined is None else combined + kp_mask
    return combined


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def build_rotary_cache(
    positions: torch.Tensor,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = sin.repeat_interleave(2, dim=-1).to(dtype)
    cos = cos.repeat_interleave(2, dim=-1).to(dtype)
    return sin, cos


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings as described in DiT / diffusion literature."""
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float().view(-1)
    device = timesteps.device
    half_dim = dim // 2
    freq_seq = torch.linspace(0, 1, half_dim, device=device)
    frequency = torch.exp(-math.log(max_period) * freq_seq)
    args = timesteps[:, None] * frequency[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepMLP(nn.Module):
    """MLP that projects sinusoidal timestep embeddings into AdaLN conditioning space."""

    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, cond_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(cond_dim, cond_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(timesteps, self.linear1.in_features)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm with AdaLN-Zero modulation and optional gating."""

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        use_gate: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.use_gate = use_gate
        modulation_dim = hidden_size * 3 if use_gate else hidden_size * 2

        self.modulation = nn.Linear(cond_dim, modulation_dim, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_gate:
            shift, scale, gate = self.modulation(cond).chunk(3, dim=-1)
        else:
            shift, scale = self.modulation(cond).chunk(2, dim=-1)

        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        if self.use_gate:
            gate = torch.tanh(gate.unsqueeze(1))
        else:
            gate = torch.ones_like(shift, device=x.device, dtype=x.dtype).unsqueeze(1)

        return x, gate


class FeedForward(nn.Module):
    """Gated GELU feed-forward network."""

    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float):
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if qk_norm or scale_norm:
            if norm_layer is None:
                raise ValueError("norm_layer must be provided if qk_norm or scale_norm is True")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        combined_mask = build_attention_mask(attn_mask, key_padding_mask, x.shape[1], x.dtype)
        attn_out = self.attn(x, attn_mask=combined_mask)
        return attn_out


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        dropout: float,
        *,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(context_dim, hidden_size, bias=True)
        self.v_proj = nn.Linear(context_dim, hidden_size, bias=True)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, query_len, _ = x.shape
        _, key_len, _ = context.shape

        q = self.q_proj(x).reshape(B, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        combined_mask = build_attention_mask(attn_mask, key_padding_mask, query_len, q.dtype)

        if self.use_rope:
            if rope_positions is None:
                raise ValueError("rope_positions must be provided when use_rope is True")
            if rope_positions.dim() == 2:
                rope_positions = rope_positions[0]
            elif rope_positions.dim() != 1:
                raise ValueError("rope_positions must be 1D or 2D")
            if rope_positions.shape[0] != key_len:
                raise ValueError("rope_positions length must match context sequence length")
            sin, cos = build_rotary_cache(
                positions=rope_positions.to(k.device),
                head_dim=self.head_dim,
                base=self.rope_base,
                device=k.device,
                dtype=k.dtype,
            )
            sin = sin.unsqueeze(0).unsqueeze(0)
            cos = cos.unsqueeze(0).unsqueeze(0)
            k = apply_rotary_pos_emb(k, sin, cos)

        if self.fused_attn:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=combined_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, combined_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn_out = attn @ v

        attn_out = attn_out.transpose(1, 2).reshape(B, query_len, self.num_heads * self.head_dim)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        cond_dim: int,
        self_attn_dropout: float,
        cross_attn_dropout: float,
        mlp_dropout: float,
        history_use_rope: bool = False,
        history_rope_base: float = 10000.0,
        condition_use_gate: bool = True,
    ):
        super().__init__()

        self.self_norm = AdaLayerNorm(hidden_size, cond_dim, use_gate=False)
        self.self_attn = SelfAttention(hidden_size, num_heads, self_attn_dropout)

        self.text_norm = AdaLayerNorm(hidden_size, cond_dim, use_gate=condition_use_gate)
        self.text_attn = CrossAttention(
            hidden_size,
            hidden_size,
            num_heads,
            cross_attn_dropout,
        )

        self.history_norm = AdaLayerNorm(hidden_size, cond_dim, use_gate=condition_use_gate)
        self.history_attn = CrossAttention(
            hidden_size,
            hidden_size,
            num_heads,
            cross_attn_dropout,
            use_rope=history_use_rope,
            rope_base=history_rope_base,
        )

        self.mlp_norm = AdaLayerNorm(hidden_size, cond_dim, use_gate=False)
        self.mlp = FeedForward(hidden_size, mlp_ratio, mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        *,
        text_tokens: Optional[torch.Tensor] = None,
        history_context: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        history_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h, gate = self.self_norm(x, cond)
        x = x + gate * self.self_attn(h)

        if text_tokens is not None:
            h, gate = self.text_norm(x, cond)
            text_out = self.text_attn(
                h,
                text_tokens,
                key_padding_mask=text_mask,
            )
            x = x + gate * text_out

        if history_context is not None:
            h, gate = self.history_norm(x, cond)
            history_out = self.history_attn(
                h,
                history_context,
                key_padding_mask=history_mask,
                rope_positions=history_positions,
            )
            x = x + gate * history_out

        h, gate = self.mlp_norm(x, cond)
        x = x + gate * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, cond_dim: int, *, use_gate: bool = False):
        super().__init__()
        self.norm = AdaLayerNorm(hidden_size, cond_dim, use_gate=use_gate)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x, gate = self.norm(x, cond)
        x = self.proj(x)
        return gate * x
