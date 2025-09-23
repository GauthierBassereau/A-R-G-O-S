"""Minimal training loop to pretrain the world model with flow matching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

from source.configs import (
    HFStreamConfig,
    PretrainWorldModelConfig,
    WorldModelFMConfig,
    config_logging,
)
from source.datasets.dataset_hf import HFAsyncImageDataLoader
from source.models.world_model import RectifiedFlow, WorldModelFM
from source.utils.utils import collect_configs, make_transform


def _select_device(spec: str) -> torch.device:
    """Resolve a device string while being resilient to unavailable backends."""
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps":
        has_mps = getattr(torch.backends, "mps", None)
        if not (has_mps and torch.backends.mps.is_available()):
            logging.warning("MPS requested but not available; falling back to CPU.")
            return torch.device("cpu")
    return device


def _prepare_text_context(
    text_embeddings: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    dropout_prob: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a batch of pooled text embeddings into contextual tokens."""
    if text_embeddings is None:
        return None, None

    if text_embeddings.dim() != 2:
        raise ValueError("text_embeddings must have shape (B, D)")

    instructions = text_embeddings.to(device=device, dtype=dtype).unsqueeze(1)

    if dropout_prob <= 0:
        return instructions, None

    drop_mask = torch.rand(instructions.shape[0], device=device) < dropout_prob
    if not drop_mask.any():
        return instructions, None

    instructions = instructions.clone()
    instructions[drop_mask] = 0.0
    text_mask = torch.zeros(
        (instructions.shape[0], instructions.shape[1]),
        dtype=torch.bool,
        device=device,
    )
    text_mask[drop_mask] = True
    return instructions, text_mask


def _save_checkpoint(
    directory: Path,
    step: int,
    model: RectifiedFlow,
    optimizer: torch.optim.Optimizer,
    run_config: dict,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": run_config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    config_logging("INFO")
    log = logging.getLogger(__name__)

    trainer_cfg = PretrainWorldModelConfig()
    device = _select_device(trainer_cfg.device)
    trainer_cfg.device = str(device)

    dataset_cfg = HFStreamConfig(
        batch_size=trainer_cfg.batch_size,
        encode_images=True,
        encode_texts=True,
        image_encoder_text_head=False,
        image_encoder_device=trainer_cfg.device,
        text_encoder_device=trainer_cfg.device,
    )
    dataset_cfg.transform = make_transform(dataset_cfg.image_size)

    world_model_cfg = WorldModelFMConfig()

    run_config = collect_configs(
        trainer=trainer_cfg,
        dataset=dataset_cfg,
        model=world_model_cfg,
    )
    log.info("Starting world model pretraining with config: %s", run_config)

    loader = HFAsyncImageDataLoader.from_config(dataset_cfg)
    world_model = WorldModelFM.from_config(world_model_cfg)
    flow_model = RectifiedFlow(
        world_model,
        logit_normal_sampling_t=trainer_cfg.rectified_flow_logit_normal_sampling_t,
        predict_velocity=trainer_cfg.rectified_flow_predict_velocity,
        default_sample_steps=trainer_cfg.rectified_flow_sample_steps,
    ).to(device)
    flow_model.train()

    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=trainer_cfg.learning_rate,
        betas=trainer_cfg.adam_betas,
        weight_decay=trainer_cfg.weight_decay,
    )

    checkpoint_dir = Path(trainer_cfg.checkpoint_dir)
    accum_steps = max(1, trainer_cfg.gradient_accumulation_steps)
    accum_loss = 0.0
    micro_step = 0
    global_step = 0
    model_dtype = next(flow_model.parameters()).dtype

    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        image_embeddings = batch.get("image_embeddings")
        if not image_embeddings or "patch_backbone" not in image_embeddings:
            raise RuntimeError("Dataset batch missing DINO patch embeddings.")

        clean_embeddings = image_embeddings["patch_backbone"].to(
            device=device,
            dtype=model_dtype,
        )

        text_embeddings = batch.get("text_embeddings")
        context_instructions, text_mask = _prepare_text_context(
            text_embeddings,
            device=device,
            dtype=model_dtype,
            dropout_prob=trainer_cfg.text_dropout_prob,
        )

        loss, _ = flow_model(
            clean_embeddings,
            context_instructions=context_instructions,
            text_mask=text_mask,
        )

        loss_value = float(loss.detach())
        (loss / accum_steps).backward()
        accum_loss += loss_value
        micro_step += 1

        if micro_step % accum_steps != 0:
            continue

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        mean_loss = accum_loss / accum_steps
        accum_loss = 0.0
        micro_step = 0

        if trainer_cfg.log_frequency > 0 and global_step % trainer_cfg.log_frequency == 0:
            log.info("step=%d loss=%.6f", global_step, mean_loss)

        if (
            trainer_cfg.checkpoint_frequency > 0
            and global_step % trainer_cfg.checkpoint_frequency == 0
        ):
            ckpt_path = _save_checkpoint(
                checkpoint_dir,
                global_step,
                flow_model,
                optimizer,
                run_config,
            )
            log.info("Saved checkpoint to %s", ckpt_path)
            
    if global_step > 0:
        ckpt_path = _save_checkpoint(
            checkpoint_dir,
            global_step,
            flow_model,
            optimizer,
            run_config,
        )
        log.info("Saved final checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()
