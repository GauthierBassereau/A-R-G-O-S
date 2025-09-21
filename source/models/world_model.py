from typing import Optional, Tuple

import torch
import torch.nn as nn

from source.configs import WorldModelFMConfig
from source.models.components import TimestepMLP, FinalLayer, TransformerBlock

class WorldModelFM(nn.Module):
    """
    Slightly modified DiT, entirely denoising in the Dinov3 latent space, prediciting futur states auto-regressively given an instruction.
    Conditions:
        - Current and past states. For the moment, it is only conditioned using cross-attention layers with added temporal embeddings (RoPE). Need to explore using causal masking on self attention layer. Also could explore conditioned on AdaLN-zero alone/with.
        - Instructions. Encoded by text encoder trained with Dinov3. Need to explore other encoders too.
        
    For the moment, states are only past encoded images, but could be extended to past proprioception too.
    
    Cool drawing:
    
                             Less Noisy                                                  
                          World Embeddings                                               
                                  ▲                                                      
                         ┌────────┴─────────────────────────────┐                        
┌────────────────────┐   │                                      │                        
│                    │   │                                      │                        
│ Linear+SiLU+Linear │──▶│                 DiT                  │                        
│                    │   │                                      │                        
└──────────▲─────────┘   │                                      │                        
           │             └────────▲──────────────────▲──────────┘                        
       Timestep                   │               **CFG** on cross-attention layer
                             Noisy World             └──┬─────────────────────┐          
                             Embeddings                 │                     │          
                                  ┌─────────────────────┤                     │          
                                  │                     │                     │          
                         ┌─────────────────┐  ┌───────────────────┐       
                         │                 │  │   Text Encoder    │     Motor Sensors ?
                         │World Embeddings │  │(DinoTxt, CLIP ...)│
                         │     Encoder     │  └─────────▲─────────┘
                         │                 │            │
                         │ (DINO, JEPA...) │      Instructions
                         │                 │
                         └────────▲────────┘                                             
                                  │                                                      
                                Past                                                     
                            Observations                                                 
    """

    def __init__(
        self,
        input_dim: int = 1024,
        latent_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
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
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

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
        return x

    def forward_with_cfg(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        context_observations: Optional[torch.Tensor] = None,
        context_instructions: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        history_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        uncond_context_observations: Optional[torch.Tensor] = None,
        uncond_context_instructions: Optional[torch.Tensor] = None,
        uncond_history_mask: Optional[torch.Tensor] = None,
        uncond_text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run conditional and unconditional passes then blend with classifier-free guidance."""

        conditional = self.forward(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            context_observations=context_observations,
            context_instructions=context_instructions,
            history_mask=history_mask,
            text_mask=text_mask,
        )

        unconditional = self.forward(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            context_observations=uncond_context_observations,
            context_instructions=uncond_context_instructions,
            history_mask=uncond_history_mask,
            text_mask=uncond_text_mask,
        )

        return unconditional + cfg_scale * (conditional - unconditional)

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


if __name__ == "__main__":
    device = 'mps'
    cfg = WorldModelFMConfig()
    model = WorldModelFM.from_config(cfg)
    model.eval().to(device)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    

    B, N = 2, 128
    noisy = torch.randn(B, N, cfg.input_dim).to(device)
    timesteps = torch.rand(B).to(device)
    num_frames, tokens_per_frame = 4, 128
    context_obs = torch.randn(B, num_frames, tokens_per_frame, cfg.history_context_dim).to(device)
    context_txt = torch.randn(B, 8, cfg.text_context_dim).to(device)
    history_mask = torch.zeros(B, num_frames, dtype=torch.bool).to(device)
    history_mask[0, -1] = True

    with torch.no_grad():
        out = model(
            noisy_embeddings=noisy,
            timesteps=timesteps,
            context_observations=context_obs,
            context_instructions=context_txt,
            history_mask=history_mask,
        )
        out_cfg = model.forward_with_cfg(
            noisy_embeddings=noisy,
            timesteps=timesteps,
            context_observations=context_obs,
            context_instructions=context_txt,
            history_mask=history_mask,
            cfg_scale=2.0,
        )
    print("Output shape:", tuple(out.shape))
    print("CFG output shape:", tuple(out_cfg.shape))
