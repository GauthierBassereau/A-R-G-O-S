import logging
import torch
import wandb

from source.configs import (
    HFStreamConfig,
    TrainerConfig,
    config_logging,
    ImageDecoderTransposeConfig,
)
from source.datasets.dataset_hf import HFAsyncImageDataLoader
from source.models.image_decoder_transpose import ImageDecoderTranspose
from source.models.image_encoder_dinov3 import ImageEncoderDinov3
from source.utils.image_transforms import make_transform, IMAGENET_MEAN, IMAGENET_STD
from source.utils.utils import collect_configs

config_logging("INFO")
log = logging.getLogger(__name__)

trainer_cfg = TrainerConfig()
dataset_cfg = HFStreamConfig(batch_size=trainer_cfg.batch_size)
image_decoder_cfg = ImageDecoderTransposeConfig()

dataset_loader = HFAsyncImageDataLoader.from_config(
    dataset_cfg,
    transform=make_transform(dataset_cfg.image_size),
)

encoder = ImageEncoderDinov3().to(trainer_cfg.device).eval()
decoder = ImageDecoderTranspose.from_config(image_decoder_cfg).to(trainer_cfg.device).train()

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(decoder.parameters(), lr=trainer_cfg.learning_rate)

run_config = collect_configs(
    trainer=trainer_cfg,
    dataset=dataset_cfg,
    image_decoder=image_decoder_cfg,
)

log.info("Initializing wandb run with config: %s", run_config)

wandb.init(
    project="argos",
    name="decoder",
    config=run_config,
)

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

    if (
        trainer_cfg.image_log_frequency > 0
        and step != 0
        and step % trainer_cfg.image_log_frequency == 0
    ):
        with torch.no_grad():
            recon_vis = (recons[0].detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).squeeze(0)
            target_vis = (images[0].detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).squeeze(0)
        wandb.log({
            "reconstruction": wandb.Image(recon_vis.permute(1, 2, 0).numpy(), caption=f"recon step {step}"),
            "target": wandb.Image(target_vis.permute(1, 2, 0).numpy(), caption=f"target step {step}")
        }, step=step)

    step += 1
