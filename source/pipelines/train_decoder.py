import logging
import torch
import wandb
from source.configs import HTTPConfig, StreamConfig, config_logging, ImageDecoderTransposeConfig
from source.datasets.dataset_stream import DatasetStream, batch_iterator
from source.models.image_decoder_transpose import ImageDecoderTranspose
from source.models.image_encoder_dinov3 import ImageEncoderDinov3
from source.utils.utils import debug_show_img
from source.utils.image_transforms import IMAGENET_MEAN, IMAGENET_STD

config_logging("INFO")
log = logging.getLogger(__name__)


http_cfg = HTTPConfig()
stream_cfg = StreamConfig()
image_decoder_cfg = ImageDecoderTransposeConfig()
# Dataset Laion400M
BASE = "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta"
UUID = "5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36"
N_SHARDS = 32
remote_parquets = [f"{BASE}/part-{i:05d}-{UUID}-c000.snappy.parquet" for i in range(N_SHARDS)]
# Training
DEVICE = "mps"
LR = 1e-4
mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1,3,1,1)
std = torch.tensor(IMAGENET_STD, device=DEVICE).view(1,3,1,1)


wandb.init(
    project="argos",      # Name of your project
    name="decoder",  # Custom run name (optional)
    config={              # Hyperparameters
        "batch_size": stream_cfg.batch_size,
        "lr": LR,
        "device": DEVICE
    }
)

dataset = DatasetStream(remote_parquets, stream_cfg, http_cfg)

encoder = ImageEncoderDinov3().to(DEVICE).eval()
decoder = ImageDecoderTranspose.from_config(image_decoder_cfg).to(DEVICE).train()

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(decoder.parameters(), lr=LR)

step = 0
for batch in batch_iterator(iter(dataset), batch_size=stream_cfg.batch_size):
    # Move batch to the same device as the models
    batch = batch.to(DEVICE, non_blocking=True)
    
    with torch.inference_mode():
        enc_out = encoder(batch, text_head=False, normalize=True)    
        
    feats = enc_out["patch_backbone"].clone()
    recons = decoder(feats)

    loss = criterion(recons, batch)
    wandb.log({"train/loss": loss.item()}, step=step)
    log.info("  loss: %.6f", float(loss))
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 10 == 0 and step != 0:
        # Denormalize for visualization so images aren't black in logs
        with torch.no_grad():
            recon_vis = (recons[0].detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).squeeze(0)
            target_vis = (batch[0].detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).squeeze(0)
        wandb.log({
            "reconstruction": wandb.Image(recon_vis.permute(1, 2, 0).numpy(), caption=f"recon step {step}"),
            "target": wandb.Image(target_vis.permute(1, 2, 0).numpy(), caption=f"target step {step}")
        }, step=step)
    
    step += 1
    if step >= 1000:
        break
