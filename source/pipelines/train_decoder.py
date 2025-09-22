import logging
from pathlib import Path

import torch
import wandb

from source.configs import (
    HFStreamConfig,
    TrainDecoderConfig,
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

trainer_cfg = TrainDecoderConfig()
dataset_cfg = HFStreamConfig(batch_size=trainer_cfg.batch_size)
image_decoder_cfg = ImageDecoderTransposeConfig()

dataset_loader = HFAsyncImageDataLoader.from_config(
    dataset_cfg,
    transform=make_transform(dataset_cfg.image_size),
)
encoder = ImageEncoderDinov3().to(trainer_cfg.device).eval()
decoder = ImageDecoderTranspose.from_config(image_decoder_cfg).to(trainer_cfg.device).train()

criterion = torch.nn.MSELoss()
optim = torch.optim.AdamW(decoder.parameters(), lr=trainer_cfg.learning_rate)

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

checkpoint_dir = Path(trainer_cfg.checkpoint_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

accum_steps = trainer_cfg.gradient_accumulation_steps
micro_step = 0
step = 0
accum_loss = 0.0

optim.zero_grad()

for batch in dataset_loader:
    images = batch["images"].to(trainer_cfg.device)
    
    with torch.inference_mode():
        enc_out = encoder(images, text_head=False, normalize=True)    
        
    feats = enc_out["patch_backbone"].clone()
    recons = decoder(feats)

    loss = criterion(recons, images)
    loss_value = float(loss.item())
    accum_loss += loss_value
    loss = loss / accum_steps
    loss.backward()
    micro_step += 1

    if micro_step % accum_steps != 0:
        continue

    optim.step()
    optim.zero_grad()

    step += 1
    mean_loss = accum_loss / accum_steps
    wandb.log({"train/loss": mean_loss}, step=step)
    log.info("  loss: %.6f", mean_loss)
    accum_loss = 0.0
    micro_step = 0

    if (trainer_cfg.checkpoint_frequency > 0
        and step % trainer_cfg.checkpoint_frequency == 0
    ):
        checkpoint_path = checkpoint_dir / f"dummy.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
            },
            checkpoint_path,
        )
        
        # Log checkpoint as W&B artifact
        artifact = wandb.Artifact("decoder-checkpoints", type="model")
        artifact.add_file(str(checkpoint_path), name=f"step_{step}")
        wandb.log_artifact(artifact)

    if (trainer_cfg.image_log_frequency > 0
        and step % trainer_cfg.image_log_frequency == 0
    ):
        with torch.no_grad():
            recon_vis = (recons[0].detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).squeeze(0)
            target_vis = (images[0].detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).squeeze(0)
        wandb.log({
            "reconstruction": wandb.Image(recon_vis.permute(1, 2, 0).numpy(), caption=f"recon step {step}"),
            "target": wandb.Image(target_vis.permute(1, 2, 0).numpy(), caption=f"target step {step}")
        }, step=step)
