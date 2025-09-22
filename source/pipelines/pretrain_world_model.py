import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple
import torch
import wandb

from source.configs import (
    HFStreamConfig,
    ImageDecoderTransposeConfig,
    PretrainWorldModelConfig,
    WorldModelFMConfig,
    config_logging,
)
from source.datasets.dataset_hf import HFAsyncImageDataLoader
from source.models.image_decoder_transpose import ImageDecoderTranspose
from source.models.image_encoder_dinov3 import ImageEncoderDinov3
from source.models.text_encoder_dinov3 import TextEncoderDinov3
from source.models.world_model import RectifiedFlow, WorldModelFM
from source.utils.image_transforms import IMAGENET_MEAN, IMAGENET_STD, make_transform
from source.utils.utils import collect_configs


config_logging("INFO")
log = logging.getLogger(__name__)


def _prepare_text_condition(
    text_encoder: TextEncoderDinov3,
    texts: Sequence[str],
    device: torch.device,
    dropout_prob: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Encode texts and apply classifier-free guidance dropout."""

    batch_size = len(texts)
    if batch_size == 0:
        keep_mask = torch.zeros(0, dtype=torch.bool, device=device)
        return None, None, keep_mask

    cleaned_texts = [text if isinstance(text, str) else "" for text in texts]
    with torch.inference_mode():
        text_features = text_encoder(cleaned_texts, normalize=True)

    text_features = text_features.to(device=device)
    text_features = text_features.unsqueeze(1)  # (B, 1, D)

    has_text = torch.tensor(
        [bool(txt.strip()) for txt in cleaned_texts],
        dtype=torch.bool,
        device=device,
    )
    dropout_mask = torch.rand(batch_size, device=device) < dropout_prob
    keep_mask = has_text & ~dropout_mask

    if not keep_mask.any():
        return None, None, keep_mask

    text_tokens = text_features.clone()
    text_tokens[~keep_mask] = 0.0
    text_mask = (~keep_mask).unsqueeze(-1)

    return text_tokens, text_mask, keep_mask


def _tensor_to_image(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    vis = (tensor * std + mean).clamp(0, 1)
    return vis[0].detach().cpu().permute(1, 2, 0).numpy()


trainer_cfg = PretrainWorldModelConfig()
dataset_cfg = HFStreamConfig(batch_size=trainer_cfg.batch_size)
world_model_cfg = WorldModelFMConfig(device=trainer_cfg.device)
image_decoder_cfg = ImageDecoderTransposeConfig()

device = torch.device(trainer_cfg.device)


dataset_loader = HFAsyncImageDataLoader.from_config(
    dataset_cfg,
    transform=make_transform(dataset_cfg.image_size),
)

image_encoder = ImageEncoderDinov3().to(device).eval()
text_encoder = TextEncoderDinov3().to(device).eval()
world_model = WorldModelFM.from_config(world_model_cfg).to(device).train()
image_decoder = ImageDecoderTranspose.from_config(image_decoder_cfg).to(device).eval()
rectified_flow = RectifiedFlow(
    net=world_model,
    device=device,
    logit_normal_sampling_t=trainer_cfg.rectified_flow_logit_normal_sampling_t,
    predict_velocity=trainer_cfg.rectified_flow_predict_velocity,
    default_sample_steps=trainer_cfg.rectified_flow_sample_steps,
)

for param in image_encoder.parameters():
    param.requires_grad_(False)
for param in text_encoder.parameters():
    param.requires_grad_(False)
for param in image_decoder.parameters():
    param.requires_grad_(False)


decoder_checkpoint_loaded = False
if trainer_cfg.decoder_checkpoint_path is not None:
    ckpt_path = Path(trainer_cfg.decoder_checkpoint_path)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        image_decoder.load_state_dict(state_dict)
        decoder_checkpoint_loaded = True
        log.info("Loaded image decoder weights from %s", ckpt_path)
    else:
        log.warning("Decoder checkpoint not found at %s; image logging disabled", ckpt_path)
else:
    log.warning("No decoder checkpoint provided; image logging disabled")


optimizer = torch.optim.AdamW(
    world_model.parameters(),
    lr=trainer_cfg.learning_rate,
    betas=trainer_cfg.adam_betas,
    weight_decay=trainer_cfg.weight_decay,
)


run_config = collect_configs(
    trainer=trainer_cfg,
    dataset=dataset_cfg,
    world_model=world_model_cfg,
    image_decoder=image_decoder_cfg,
)

log.info("Initializing wandb run with config: %s", run_config)

wandb.init(
    project="argos",
    name="world-model-pretrain",
    config=run_config,
)


mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

checkpoint_dir = Path(trainer_cfg.checkpoint_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

accum_steps = trainer_cfg.gradient_accumulation_steps
micro_step = 0
step = 0
accum_loss = 0.0
text_fraction_accum = 0.0
timestep_accum = 0.0

optimizer.zero_grad(set_to_none=True)


for batch in dataset_loader:
    if step >= trainer_cfg.max_steps:
        break

    images = batch["images"].to(device, non_blocking=True)
    texts = batch["texts"]

    with torch.inference_mode():
        encoded = image_encoder(images, text_head=False, normalize=False)
        clean_embeddings = encoded["patch_backbone"].detach()

    clean_embeddings = clean_embeddings.to(device)

    text_tokens, text_mask, keep_mask = _prepare_text_condition(
        text_encoder=text_encoder,
        texts=texts,
        device=device,
        dropout_prob=trainer_cfg.text_dropout_prob,
    )

    loss, flow_state = rectified_flow.compute_loss(
        clean_embeddings=clean_embeddings,
        context_instructions=text_tokens,
        text_mask=text_mask,
    )

    accum_loss += float(loss.item())
    (loss / accum_steps).backward()
    micro_step += 1

    text_condition_fraction = (
        keep_mask.float().mean().item() if keep_mask.numel() > 0 else 0.0
    )
    text_fraction_accum += text_condition_fraction
    timestep_accum += flow_state["timesteps"].mean().item()

    snapshot_payload: Optional[dict] = None
    next_step = step + 1
    if (
        decoder_checkpoint_loaded
        and trainer_cfg.image_log_frequency > 0
        and micro_step % accum_steps == 0
        and next_step % trainer_cfg.image_log_frequency == 0
    ):
        sample_text = texts[0] if texts else ""
        with torch.no_grad():
            sample_tokens, sample_mask, _ = _prepare_text_condition(
                text_encoder=text_encoder,
                texts=[sample_text],
                device=device,
                dropout_prob=0.0,
            )
            sampled_latents = rectified_flow.sample(
                batch_size=1,
                token_shape=(clean_embeddings.shape[1], clean_embeddings.shape[2]),
                context_observations=None,
                context_instructions=sample_tokens,
                text_mask=sample_mask,
                cfg_scale=trainer_cfg.cfg_log_scale,
                sample_steps=trainer_cfg.rectified_flow_sample_steps,
            )
            decoded = image_decoder(sampled_latents)
            sample_image = _tensor_to_image(decoded, mean, std)
            reference_image = _tensor_to_image(images[:1], mean, std)

        snapshot_payload = {
            "train/sample_from_noise": wandb.Image(
                sample_image,
                caption=f"text: {sample_text or '<empty>'}",
            ),
            "train/sample_reference": wandb.Image(
                reference_image,
                caption="dataset reference",
            ),
            "train/sample_text": sample_text,
        }

    if micro_step % accum_steps != 0:
        continue

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    step += 1
    mean_loss = accum_loss / accum_steps
    mean_text_fraction = text_fraction_accum / accum_steps
    mean_timestep = timestep_accum / accum_steps

    log_payload = {
        "train/loss": mean_loss,
        "train/text_condition_fraction": mean_text_fraction,
        "train/timestep_mean": mean_timestep,
    }
    if snapshot_payload is not None:
        log_payload.update(snapshot_payload)

    wandb.log(log_payload, step=step)
    log.info("step %d loss: %.6f", step, mean_loss)

    accum_loss = 0.0
    micro_step = 0
    text_fraction_accum = 0.0
    timestep_accum = 0.0

    if (
        trainer_cfg.checkpoint_frequency > 0
        and step % trainer_cfg.checkpoint_frequency == 0
    ):
        checkpoint_path = checkpoint_dir / f"world_model_step_{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": world_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        artifact = wandb.Artifact("world-model-checkpoints", type="model")
        artifact.add_file(str(checkpoint_path), name=f"step_{step}")
        wandb.log_artifact(artifact)

log.info("Training finished at step %d", step)
