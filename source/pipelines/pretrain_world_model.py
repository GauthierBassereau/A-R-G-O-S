"""Minimal training loop to pretrain the world model with flow matching + wandb loss logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import wandb  # NEW: wandb

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

    instructions = text_embeddings.to(device=device, dtype=dtype).unsqueeze(1).clone()

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
    dataset_cfg.batch_size = trainer_cfg.batch_size

    world_model_cfg = WorldModelFMConfig()

    run_config = collect_configs(
        trainer=trainer_cfg,
        dataset=dataset_cfg,
        model=world_model_cfg,
    )
    log.info("Starting world model pretraining with config: %s", run_config)

    log.info("Initializing wandb run with config.")
    wandb.init(
        project="argos",
        name="world-model-fm",
        config=run_config,
    )

    try:
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
            ).clone()

            text_embeddings = batch.get("text_embeddings")
            context_instructions, text_mask = _prepare_text_context(
                text_embeddings,
                device=device,
                dtype=model_dtype,
                dropout_prob=trainer_cfg.text_dropout_prob,
            )

            loss, out = flow_model(
                clean_embeddings,
                context_instructions=context_instructions,
                text_mask=text_mask,
            )
            with torch.no_grad():
                import matplotlib
                matplotlib.use("Agg")  # non-interactive backend (safe on server)
                import matplotlib.pyplot as plt
                import numpy as np

                # Take sample 0, token 0
                clean_vec = clean_embeddings[0, 0].detach().cpu().float()
                mix_vec   = out["noisy_embeddings"][0, 0].detach().cpu().float()
                t_used    = float(out["timesteps"][0].item())

                # Reconstruct the PURE noise vector used this step
                # If predict_velocity=True, target = noise - clean  -> noise = target + clean
                # Else, target = clean (x0 prediction), so noise isn't in 'target'; fall back to mix + ...
                if trainer_cfg.rectified_flow_predict_velocity:
                    noise_vec = (out["target"][0, 0].detach().cpu().float() + clean_vec).detach().cpu().float()
                else:
                    # When predicting x0, we don't have ε directly. Recover it from the linear mix:
                    # x_t = (1 - t) x0 + t ε  =>  ε = (x_t - (1 - t) x0) / t
                    eps = 1e-6
                    noise_vec = ((mix_vec - (1.0 - t_used) * clean_vec) / max(t_used, eps)).detach().cpu().float()

                # --- Overlaid histograms (per-dimension distributions) ---
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                bins = 60
                axes[0].hist(clean_vec.numpy(), bins=bins, density=True, alpha=0.45, label="clean (x₀)")
                axes[0].hist(noise_vec.numpy(), bins=bins, density=True, alpha=0.45, label="pure noise (ε)")
                axes[0].hist(mix_vec.numpy(),   bins=bins, density=True, alpha=0.45, label=f"mixture (xₜ, t={t_used:.3f})")
                axes[0].set_title("Per-dim distributions (token 0)")
                axes[0].set_xlabel("value"); axes[0].set_ylabel("density")
                axes[0].legend(loc="upper right")

                # --- QQ plot: clean vs pure noise ---
                q = np.linspace(0.01, 0.99, 199)  # avoid extreme tails noise
                clean_q = np.quantile(clean_vec.numpy(), q)
                noise_q = np.quantile(noise_vec.numpy(), q)
                mn = float(min(clean_q.min(), noise_q.min()))
                mx = float(max(clean_q.max(), noise_q.max()))

                axes[1].plot(noise_q, clean_q, ".", markersize=3)
                axes[1].plot([mn, mx], [mn, mx], linewidth=1)  # y=x reference
                axes[1].set_title("QQ plot: clean vs pure noise")
                axes[1].set_xlabel("noise quantiles (ε)")
                axes[1].set_ylabel("clean quantiles (x₀)")

                # Annotate means/stds for quick sanity
                def _ms(x): 
                    return float(x.mean().item()), float(x.std(unbiased=False).item())
                m0, s0 = _ms(clean_vec); me, se = _ms(noise_vec); mt, st = _ms(mix_vec)
                stats_txt = f"x₀: μ={m0:.3f}, σ={s0:.3f}\nε: μ={me:.3f}, σ={se:.3f}\nxₜ: μ={mt:.3f}, σ={st:.3f}"
                axes[0].text(0.02, 0.98, stats_txt, transform=axes[0].transAxes,
                            va="top", ha="left", bbox=dict(boxstyle="round", alpha=0.15, pad=0.3))

                fig.tight_layout()
                fig.savefig("first_token_dists.png", dpi=150)
                plt.close(fig)

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

            wandb.log({"train/loss": mean_loss}, step=global_step)

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
    finally:
        # --- NEW: ensure wandb run is properly closed ---
        wandb.finish()


if __name__ == "__main__":
    main()
