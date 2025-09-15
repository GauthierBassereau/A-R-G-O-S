import logging
import torch
import wandb
from source.configs import HFStreamConfig, TrainerConfig, config_logging, ImageDecoderTransposeConfig
from source.datasets.dataset_hf import HFAsyncImageDataLoader
from source.models.image_decoder_transpose import ImageDecoderTranspose
from source.models.image_encoder_dinov3 import ImageEncoderDinov3
from source.utils.image_transforms import make_transform, IMAGENET_MEAN, IMAGENET_STD

config_logging("INFO")
log = logging.getLogger(__name__)

trainer_cfg = TrainerConfig()
dataset_cfg = HFStreamConfig(batch_size=trainer_cfg.batch_size)
image_decoder_cfg = ImageDecoderTransposeConfig()

dataset_loader = HFAsyncImageDataLoader.from_config(dataset_cfg, transform=make_transform(512))

encoder = ImageEncoderDinov3().to(trainer_cfg.device).eval()
decoder = ImageDecoderTranspose.from_config(image_decoder_cfg).to(trainer_cfg.device).train()

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(decoder.parameters(), lr=trainer_cfg.learning_rate)

wandb.init(
    project="argos",      # Name of your project
    name="decoder",  # Custom run name (optional)
    config={              # Hyperparameters
        "batch_size": trainer_cfg.batch_size,
        "lr": trainer_cfg.learning_rate,
        "device": trainer_cfg.device
    })

mean = torch.tensor(IMAGENET_MEAN, device=trainer_cfg.device).view(1,3,1,1)
std = torch.tensor(IMAGENET_STD, device=trainer_cfg.device).view(1,3,1,1)

step = 0
for batch in dataset_loader:
    images = batch["images"].to(trainer_cfg.device)
    
    with torch.inference_mode():
        enc_out = encoder(images, text_head=False, normalize=True)    
        
    feats = enc_out["patch_backbone"].clone()
    recons = decoder(feats)

    loss = criterion(recons, images)
    wandb.log({"train/loss": loss.item()}, step=step)
    log.info("  loss: %.6f", float(loss))
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 100 == 0 and step != 0:
        # Denormalize for visualization so images aren't black in logs
        with torch.no_grad():
            recon_vis = (recons[0].detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).squeeze(0)
            target_vis = (batch[0].detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).squeeze(0)
        wandb.log({
            "reconstruction": wandb.Image(recon_vis.permute(1, 2, 0).numpy(), caption=f"recon step {step}"),
            "target": wandb.Image(target_vis.permute(1, 2, 0).numpy(), caption=f"target step {step}")
        }, step=step)

    step += 1