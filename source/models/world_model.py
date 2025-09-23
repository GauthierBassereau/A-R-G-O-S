from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.configs import WorldModelFMConfig
from source.models.components import TimestepMLP, FinalLayer, TransformerBlock


class RectifiedFlow(nn.Module):
    """Wrapper that handles flow-matching loss and sampling around a world model."""

    def __init__(
        self,
        net: "WorldModelFM",
        logit_normal_sampling_t: bool = True,
        predict_velocity: bool = True,
        default_sample_steps: int = 50,
    ) -> None:
        super().__init__()
        self.net = net
        self.logit_normal_sampling_t = logit_normal_sampling_t
        self.predict_velocity = predict_velocity
        self.default_sample_steps = default_sample_steps

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.logit_normal_sampling_t:
            base = torch.randn(batch_size, device=device, generator=generator)
            return base.sigmoid()
        return torch.rand(batch_size, device=device, generator=generator)

    def forward(
        self,
        clean_embeddings: torch.Tensor,
        context_instructions: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        context_observations: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Alias for compute_loss so the wrapper behaves like a module."""

        return self.compute_loss(
            clean_embeddings,
            context_instructions=context_instructions,
            text_mask=text_mask,
            context_observations=context_observations,
            history_mask=history_mask,
            generator=generator,
        )

    def compute_loss(
        self,
        clean_embeddings: torch.Tensor,
        *,
        context_instructions: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        context_observations: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = clean_embeddings.shape[0]
        device = clean_embeddings.device
        timesteps = self._sample_timesteps(
            batch_size,
            device=device,
            generator=generator,
        ).to(dtype=clean_embeddings.dtype)
        timestep_factors = timesteps.view(batch_size, 1, 1)

        noise = torch.randn_like(clean_embeddings) # TODO Generator cannot be an argument, use randn
        noisy_embeddings = (1.0 - timestep_factors) * clean_embeddings + timestep_factors * noise

        prediction = self.net(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            context_observations=context_observations,
            context_instructions=context_instructions,
            history_mask=history_mask,
            text_mask=text_mask,
        )

        if self.predict_velocity:
            target = noise - clean_embeddings
        else:
            target = clean_embeddings

        loss = F.mse_loss(prediction, target)

        return loss, {
            "timesteps": timesteps.detach(),
            "noisy_embeddings": noisy_embeddings.detach(),
            "target": target.detach(),
            "prediction": prediction.detach(),
        }

    def _predict_step_cfg(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        context_observations: Optional[torch.Tensor],
        context_instructions: Optional[torch.Tensor],
        history_mask: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        cfg_scale: float,
    ) -> torch.Tensor:
        conditional = self.net(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            context_observations=None,
            context_instructions=None,
            history_mask=None,
            text_mask=None,
        )
        
        if cfg_scale == 0:
            return conditional
        
        unconditional = self.net(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            context_observations=context_observations,
            context_instructions=context_instructions,
            history_mask=history_mask,
            text_mask=text_mask,
        )

        return unconditional + cfg_scale * (conditional - unconditional)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        token_shape: Tuple[int, int],
        context_observations: Optional[torch.Tensor] = None,
        context_instructions: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        sample_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        try:
            num_tokens, feature_dim = token_shape
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError("token_shape must be a tuple of (num_tokens, feature_dim)") from exc

        steps = sample_steps if sample_steps is not None else self.default_sample_steps
        if steps < 1:
            raise ValueError("sample_steps must be >= 1")

        param = next(self.net.parameters())
        device = param.device
        dtype = param.dtype

        if initial_noise is not None:
            if initial_noise.shape != (batch_size, num_tokens, feature_dim):
                raise ValueError(
                    "initial_noise must have shape (batch_size, num_tokens, feature_dim)"
                )
            current = initial_noise.to(device=device, dtype=dtype)
        else:
            current = torch.randn(
                batch_size,
                num_tokens,
                feature_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )

        def _move_to_device(
            tensor: Optional[torch.Tensor],
            name: str,
            expect_bool: bool = False,
        ) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch dimension ({tensor.shape[0]}) does not match batch_size {batch_size}"
                )
            if expect_bool:
                return tensor.to(device=device, dtype=torch.bool)
            return tensor.to(device=device, dtype=dtype)

        context_observations = _move_to_device(
            context_observations,
            name="context_observations",
        )
        context_instructions = _move_to_device(
            context_instructions,
            name="context_instructions",
        )
        history_mask = _move_to_device(history_mask, name="history_mask", expect_bool=True)
        text_mask = _move_to_device(text_mask, name="text_mask", expect_bool=True)

        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)

        for idx in range(steps):
            t_curr = timesteps[idx]
            t_next = timesteps[idx + 1]
            t_batch = torch.full((batch_size,), float(t_curr), device=device, dtype=dtype)

            if cfg_scale == 0.0:
                prediction = self.net(
                    noisy_embeddings=current,
                    timesteps=t_batch,
                    context_observations=context_observations,
                    context_instructions=context_instructions,
                    history_mask=history_mask,
                    text_mask=text_mask,
                )
            else:
                prediction = self._predict_step_cfg(
                    noisy_embeddings=current,
                    timesteps=t_batch,
                    context_observations=context_observations,
                    context_instructions=context_instructions,
                    history_mask=history_mask,
                    text_mask=text_mask,
                    cfg_scale=cfg_scale,
                )

            if self.predict_velocity:
                velocity = prediction
            else:
                denom = torch.clamp(t_batch.view(batch_size, 1, 1), min=1e-5)
                velocity = (current - prediction) / denom

            current = current + (t_next - t_curr) * velocity

        return current

class WorldModelFM(nn.Module):
    """DiT-style denoiser that operates in the DINOv3 latent space with optional text and history context."""

    def __init__(
        self,
        input_dim: int = 1024,
        latent_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        time_embed_dim: int = 256,
        time_cond_dim: int = 1024,
        attention_dropout: float = 0.0,
        cross_attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        text_context_dim: int = 1024,
        history_context_dim: int = 1024,
        use_history_rope: bool = True,
        history_rope_base: float = 10000.0,
        condition_use_gate: bool = True,
    ):
        super().__init__()

        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be non-negative")

        self.num_register_tokens = num_register_tokens
        self.register_tokens = (
            nn.Parameter(torch.zeros(num_register_tokens, latent_dim))
            if num_register_tokens > 0
            else None
        )
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)

        self.input_proj = (
            nn.Linear(input_dim, latent_dim)
            if input_dim != latent_dim
            else nn.Identity()
        )

        self.timestep_mlp = TimestepMLP(time_embed_dim, time_cond_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=latent_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cond_dim=time_cond_dim,
                    self_attn_dropout=attention_dropout,
                    cross_attn_dropout=cross_attention_dropout,
                    mlp_dropout=mlp_dropout,
                    history_use_rope=use_history_rope,
                    history_rope_base=history_rope_base,
                    condition_use_gate=condition_use_gate,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(latent_dim, time_cond_dim, use_gate=False)
        self.output_proj = (
            nn.Linear(latent_dim, input_dim)
            if input_dim != latent_dim
            else nn.Identity()
        )

        self.text_proj = (
            nn.Linear(text_context_dim, latent_dim)
            if text_context_dim != latent_dim
            else nn.Identity()
        )

        self.obs_proj = (
            nn.Linear(history_context_dim, latent_dim)
            if history_context_dim != latent_dim
            else nn.Identity()
        )

    def forward(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        context_observations: Optional[torch.Tensor] = None,
        context_instructions: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_embeddings: Tensor of shape (B, N, input_dim) representing noisy world tokens.
            timesteps: Tensor of shape (B,) or (B, 1) with diffusion/flow times in [0, 1].
            context_observations: Optional tensor (B, N_frames, N_tokens, C_obs) for past observation
                tokens ordered from most recent to oldest (index 0 is the closest frame).
            context_instructions: Optional tensor (B, T, C_txt) for text conditioning.
            history_mask: Optional key padding mask for observation tokens with True entries marking
                padding. Accepts shapes (B, N_frames) or (B, N_frames, N_tokens).
            text_mask: Optional key padding mask for text tokens (B, T).

        Returns:
            Tensor of shape (B, N, input_dim) representing predicted clean embeddings / flow targets.
        """

        x = self.input_proj(noisy_embeddings)
        cond = self.timestep_mlp(timesteps).to(x.dtype)

        expanded_register_tokens = None
        if self.register_tokens is not None:
            batch_size = x.shape[0]
            expanded_register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            expanded_register_tokens = expanded_register_tokens.to(dtype=x.dtype)
            x = torch.cat([expanded_register_tokens, x], dim=1)

        text_tokens, text_padding_mask = self._prepare_text_context(
            context_instructions=context_instructions,
            text_mask=text_mask,
        )
        
        history_tokens, history_padding_mask, history_positions = self._prepare_history_context(
            context_observations=context_observations,
            history_mask=history_mask,
        )

        for block in self.blocks:
            x = block(
                x,
                cond,
                text_tokens=text_tokens,
                history_context=history_tokens,
                text_mask=text_padding_mask,
                history_mask=history_padding_mask,
                history_positions=history_positions,
            )

        x = self.final_layer(x, cond)
        x = self.output_proj(x)

        if expanded_register_tokens is not None:
            x = x[:, self.num_register_tokens :]

        return x

    @classmethod
    def from_config(cls, config: WorldModelFMConfig) -> "WorldModelFM":
        return cls(**config.__dict__)

    def _prepare_text_context(
        self,
        context_instructions: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if context_instructions is None:
            if text_mask is not None:
                raise ValueError("text_mask provided without context_instructions")
            return None, None

        if context_instructions.ndim != 3:
            raise ValueError("context_instructions must have shape (B, T, C_txt)")

        text_tokens = self.text_proj(context_instructions)

        if text_mask is None:
            return text_tokens, None

        if text_mask.shape != context_instructions.shape[:2]:
            raise ValueError("text_mask must have shape (B, T) matching instructions")

        return text_tokens, text_mask.to(torch.bool)

    def _prepare_history_context(
        self,
        context_observations: Optional[torch.Tensor],
        history_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if context_observations is None:
            if history_mask is not None:
                raise ValueError("history_mask provided without context_observations")
            return None, None, None

        if context_observations.ndim != 4:
            raise ValueError(
                "context_observations must have shape (B, N_frames, N_tokens, C_obs)"
            )

        batch_size, num_frames, tokens_per_frame, _ = context_observations.shape
        flattened = context_observations.reshape(batch_size, num_frames * tokens_per_frame, -1)
        history_tokens = self.obs_proj(flattened)
        history_positions = self._build_history_positions(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            device=history_tokens.device,
        )
        history_padding_mask = self._prepare_history_mask(
            history_mask=history_mask,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            batch_size=batch_size,
        )

        return history_tokens, history_padding_mask, history_positions

    @staticmethod
    def _prepare_history_mask(
        history_mask: Optional[torch.Tensor],
        num_frames: int,
        tokens_per_frame: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """Normalize history masks to shape (B, num_frames * tokens_per_frame)."""

        if history_mask is None:
            return None

        if history_mask.dim() == 2:
            if history_mask.shape != (batch_size, num_frames):
                raise ValueError("history_mask must have shape (B, num_frames)")
            expanded = history_mask.unsqueeze(-1).expand(-1, -1, tokens_per_frame)
        elif history_mask.dim() == 3:
            if history_mask.shape[0] != batch_size or history_mask.shape[1] != num_frames:
                raise ValueError("history_mask must have shape (B, num_frames, tokens_per_frame)")
            if history_mask.shape[2] == 1:
                expanded = history_mask.expand(-1, -1, tokens_per_frame)
            elif history_mask.shape[2] == tokens_per_frame:
                expanded = history_mask
            else:
                raise ValueError("history_mask last dimension must match tokens_per_frame")
        else:
            raise ValueError("history_mask must have 2 or 3 dimensions")

        return expanded.reshape(batch_size, num_frames * tokens_per_frame).to(torch.bool)

    @staticmethod
    def _build_history_positions(
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(num_frames, device=device)
        positions = positions.repeat_interleave(tokens_per_frame)
        return positions
