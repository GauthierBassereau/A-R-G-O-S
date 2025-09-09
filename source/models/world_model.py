import torch
import torch.nn as nn

from source.settings import WorldModelFMConfig

class WorldModelFM(nn.Module):
    """
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
       Timestep                   │               **CFG**                                
                             Noisy World             └──┬─────────────────────┐          
                             Embeddings                 │                     │          
                                  ┌─────────────────────┤                     │          
                                  │                     │                     │          
                         ┌─────────────────┐  ┌───────────────────┐ ┌───────────────────┐
                         │                 │  │   Text Encoder    │ │  Action Encoder   │
                         │World Embeddings │  │ (umT5, CLIP ...)  │ │       (MLP)       │
                         │     Encoder     │  └─────────▲─────────┘ └─────────▲─────────┘
                         │                 │            │                     │          
                         │ (DINO, JEPA...) │      Instructions            Actions        
                         │                 │                                             
                         └────────▲────────┘                                             
                                  │                                                      
                                Past                                                     
                            Observations                                                 
    """
    def __init__(self, config: WorldModelFMConfig):
        super().__init__()
        
    def forward(self, context_world, context_instructions=None, context_actions=None, cfg_scale=3.0):
        """
        context_world: (B, C, H, W)  past world observations (only images for now)
        context_instructions: (B, S_text, D_text)  text embeddings (e.g., T5 or/and CLIP)
        context_actions: (B, S_action, D_action)  action embeddings (e.g., MLP on joint angles)
        """

        world_embeddings = self.world_encoder(context_world)

        context_tokens = [world_embeddings]
        if context_instructions is not None:
            context_tokens.append(self.text_encoder(context_instructions))
        if context_actions is not None:
            context_tokens.append(self.action_encoder(context_actions))
        context_tokens = torch.cat(context_tokens, dim=1)  # (B, S, D)
        
        # Prepare for diffusion model
        B, S, D = context_tokens.shape
        context_tokens = context_tokens.permute(0, 2, 1).view(B, D, 1, S)
        context_mask = torch.ones(B, 1, 1, S).to(context_tokens.device)
        context = {'tokens': context_tokens, 'mask': context_mask}
        x = torch.randn(B, D, 16, 16).to(context_tokens.device)
        timesteps = torch.randint(0, self.num_timesteps, (B,), device=context_tokens.device).long()
        noise = torch.randn_like(x)
        noisy_x = self.q_sample(x, timesteps, noise)
        if cfg_scale > 0:
            # Duplicate for classifier free guidance
            noisy_x = torch.cat([noisy_x] * 2, dim=0)
            timesteps = torch.cat([timesteps] * 2, dim=0)
            context['tokens'] = torch.cat([context['tokens'], torch.zeros_like(context['tokens'])], dim=0)
            context['mask'] = torch.cat([context['mask'], torch.zeros_like(context['mask'])], dim=0)
        # Predict the noise
        pred_noise = self.fm_backbone(noisy_x, timesteps, context)
        if cfg_scale > 0:
            pred_noise, uncond_pred_noise = pred_noise.chunk(2, dim=0)
            pred_noise = uncond_pred_noise + cfg_scale * (pred_noise - uncond_pred_noise)
        # Get the denoised sample
        x_recon = self.predict_start_from_noise(pred_noise, timesteps, noisy_x)
        return x_recon
    