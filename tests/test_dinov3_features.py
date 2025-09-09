#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from source.models.world_encoder_dinov3 import WorldEncoderDinov3
from source.utils.image_transforms import make_transform

def round_to_multiple(x, m=16):
    return int(math.ceil(x / m) * m)

def load_image(path):
    img = Image.open(path).convert("RGB")
    tfm = make_transform(1024)   # always resize to 256x256 for DINOv3
    tensor = tfm(img).unsqueeze(0)  # (1,3,H,W) in [0,1] normalized
    return img, tensor

def compute_patch_similarity(dinov3, img_tensor, yx=None, patch_size=16):
    """
    img_tensor: (1,3,H,W) after normalization and resized to multiples of 16
    yx: (y,x) pixel coordinates in the (resized) tensor space. If None, pick center.
    """
    out = dinov3.encode_image(img_tensor, text_head=True, normalize=True)
    patch_bb = out["patch_backbone"]  # (1, P, C) features after final LN if normalize=True in your call
    assert patch_bb is not None, "Backbone patch tokens missing"

    # L2-normalize per token for cosine similarity
    feats = F.normalize(patch_bb[0], dim=-1)  # (P, C)

    patch_size = 16
    H = W = 1024
    hp, wp = H // patch_size, W // patch_size  # → 14x14 patches
    P = hp * wp
    assert feats.shape[0] == P, f"Unexpected token count: {feats.shape[0]} vs {P}"

    if yx is None:
        ri, rj = hp // 2, wp // 2
    else:
        y, x = yx
        # clamp to [0, H-1]/[0, W-1], then to patch indices
        y = max(0, min(H - 1, int(y)))
        x = max(0, min(W - 1, int(x)))
        ri, rj = y // patch_size, x // patch_size

    ref_idx = int(ri * wp + rj)
    ref = feats[ref_idx]  # (C,)

    sim = feats @ ref  # (P,)
    sim_map = sim.view(hp, wp).unsqueeze(0).unsqueeze(0)  # (1,1,hp,wp)
    heat = F.interpolate(sim_map, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    # Normalize for visualization to [0,1]
    # Use scalar quantiles to avoid device mismatch (CPU/MPS/CUDA)
    heat_min, heat_max = torch.quantile(heat, 0.01), torch.quantile(heat, 0.99)
    heat = (heat - heat_min) / (heat_max - heat_min + 1e-6)
    heat = heat.clamp(0, 1)
    return heat.cpu().numpy(), (ri, rj), (H, W)

def show_overlay(pil_img, heatmap, pick_xy, resized_shape, patch_size=16, title="Patch similarity"):
    """
    pil_img: original-size image (PIL)
    heatmap: (H_resized, W_resized) np array in [0,1]
    pick_xy: (x, y) in original image coordinate space (if provided), or None
    """
    # If we resized for the backbone, make a resized version for display to match heatmap resolution
    H_resized, W_resized = resized_shape
    disp_img = pil_img.resize((W_resized, H_resized), Image.BICUBIC)

    plt.imshow(disp_img)
    plt.imshow(heatmap, alpha=0.55, cmap="jet")  # default colormap; no fixed colors specified by us
    # Mark the selected patch center
    if pick_xy is not None:
        x, y = pick_xy
        # Map original coordinate to resized coordinate
        x = x * (W_resized / pil_img.width)
        y = y * (H_resized / pil_img.height)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def interactive_pick_and_show(dinov3, pil_img, img_tensor):
    """
    If user did not pass coordinates, let them click on the image (resized view),
    then recompute and show the heatmap.
    """
    H, W = img_tensor.shape[-2:]
    disp_img = pil_img.resize((W, H), Image.BICUBIC)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(disp_img)
    ax.set_title("Click to choose a pixel; close window to exit.")
    ax.axis("off")

    picked = {"xy": None}

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        picked["xy"] = (x, y)
        plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if picked["xy"] is None:
        return  # user closed without clicking

    # Compute similarity and show overlay
    heat, (ri, rj), (Hr, Wr) = compute_patch_similarity(dinov3, img_tensor, yx=(picked["xy"][1], picked["xy"][0]))
    show_overlay(pil_img, heat, pick_xy=picked["xy"], resized_shape=(Hr, Wr),
                 title=f"Patch similarity (picked ~ patch [{ri},{rj}])")

def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 backbone patch similarity by pixel.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image")
    parser.add_argument("--x", type=int, default=None, help="Pixel x (in original image)")
    parser.add_argument("--y", type=int, default=None, help="Pixel y (in original image)")
    parser.add_argument("--device", type=str, default=None, help='e.g., "cuda", "mps", or "cpu"')
    parser.add_argument("--dtype", type=str, default=None, help='e.g., "float16" or "bfloat16"')
    parser.add_argument("--head_ckpt", type=str, default=None)
    parser.add_argument("--backbone_ckpt", type=str, default=None)
    args = parser.parse_args()

    # dtype parsing
    torch_dtype = None
    if args.dtype is not None:
        s = args.dtype.lower()
        if s in ("float16", "fp16", "half"):
            torch_dtype = torch.float16
        elif s in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        elif s in ("float32", "fp32"):
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype: {args.dtype}")

    # 1) Load image (PIL for original view, tensor for model)
    pil_img, img_tensor = load_image(Path(args.image))

    # 2) Init DINOv3
    dinov3 = WorldEncoderDinov3(
        head_ckpt=args.head_ckpt,
        backbone_ckpt=args.backbone_ckpt,
        device=args.device,
        dtype=torch_dtype,
    )

    # 3) If coordinates are provided, compute & show directly
    if args.x is not None and args.y is not None:
        # Map original (x,y) to resized tensor space
        x_resized = args.x * (img_tensor.shape[-1] / pil_img.width)
        y_resized = args.y * (img_tensor.shape[-2] / pil_img.height)
        heat, (ri, rj), (H, W) = compute_patch_similarity(dinov3, img_tensor, yx=(y_resized, x_resized))
        show_overlay(
            pil_img,
            heat,
            pick_xy=(args.x, args.y),
            resized_shape=(H, W),
            title=f"Patch similarity (picked ~ patch [{ri},{rj}])"
        )
    else:
        # 4) Otherwise, let the user click to pick a pixel
        interactive_pick_and_show(dinov3, pil_img, img_tensor)

if __name__ == "__main__":
    main()
