import torch
from typing import Dict, Iterable, Optional, Literal

class WorldEncoderDinov3:
    def __init__(self,
                 head_ckpt: Optional[str] = None,
                 backbone_ckpt: Optional[str] = None,
                 repo: str = "facebookresearch/dinov3",
                 hub_entry: str = "dinov3_vitl16_dinotxt_tet1280d20h24l",
                 source: Literal["github", "local"] = "github",
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available()
                                               else "mps" if torch.backends.mps.is_available()
                                               else "cpu")
        )
        self.dtype = dtype
        kwargs = {}
        if head_ckpt is not None:
            kwargs["weights"] = head_ckpt
        if backbone_ckpt is not None:
            kwargs["backbone_weights"] = backbone_ckpt

        self.model, self.tokenizer = torch.hub.load(repo, hub_entry, source=source, **kwargs)
        self.model.eval().to(self.device)
        if self.dtype is not None:
            self.model.to(self.dtype)

    @torch.inference_mode()
    def encode_image(self, images: torch.Tensor, text_head: bool = True, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Returns:
          - head cls/patch (post-head), pooled head features,
          - backbone cls/patch/register (pre-head).
        """
        images = images.to(self.device, dtype=self.dtype if self.dtype is not None else images.dtype)

        V = self.model.visual_model                      # VisionTower (DINOv3)
        B = V.backbone                                   # ViT backbone
        H = V.head                                       # VisionHead aligned to text
        num_reg = V.num_register_tokens

        # Mirror VisionTower.get_backbone_features from DinoV3 repo
        tokens = B.get_intermediate_layers(
            images,
            n=V.patch_token_layer,
            return_class_token=True,
            return_extra_tokens=True,
            norm=normalize
        )
        # tokens[-1][1]: class_token (backbone)
        # tokens[0][0]: patch_tokens (backbone)
        # tokens[0][2]: register_tokens (backbone)
        cls_bb = tokens[-1][1]               # [B, C]
        patch_bb = tokens[0][0]              # [B, P, C]
        reg_bb = tokens[0][2]                # [B, R, C]

        if text_head:
            # Build the input to the head exactly like VisionTower.get_class_and_patch_tokens
            image_tokens = torch.cat([cls_bb.unsqueeze(1), reg_bb, patch_bb], dim=1)  # [B, 1+R+P, C]
            image_tokens = H(image_tokens)  # head blocks + LN + (optional) linear projection

            # Head outputs
            cls_head = image_tokens[:, 0]                               # [B, D]
            patch_head = image_tokens[:, num_reg + 1 :]                 # [B, P, D]

            # Pool to the final head feature (what VisionTower.forward returns as `features`)
            feats = []
            if V.use_class_token:
                feats.append(cls_head)
            if V.use_patch_tokens:
                if V.patch_tokens_pooler_type == "mean":
                    feats.append(patch_head.mean(dim=1))
                elif V.patch_tokens_pooler_type == "max":
                    feats.append(patch_head.max(dim=1).values)
                else:
                    raise ValueError(f"Unknown pooler: {V.patch_tokens_pooler_type}")
            head_feat = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

            if normalize:
                head_feat = torch.nn.functional.normalize(head_feat, dim=-1)
        else:
            head_feat = None
            patch_head = None

        return {
            # backbone
            "cls_backbone": cls_bb,
            "patch_backbone": patch_bb,
            "register_backbone": reg_bb,
            # head
            "cls_text": head_feat,
            "patch_text": patch_head,
        }

    @torch.inference_mode()
    def encode_text(self, texts: Iterable[str], normalize: bool = True) -> torch.Tensor:
        # Keep tokens as int64; only move device
        tokens = self.tokenizer.tokenize(list(texts)).to(self.device)
        # DO NOT cast tokens to self.dtype
        feats = self.model.encode_text(tokens, normalize=normalize)
        return feats



def _assert_same_tokens(dino_tok, texts):
    import clip
    d = dino_tok.tokenize(texts)
    c = clip.tokenize(texts)
    if d.shape != c.shape:
        raise AssertionError(f"token shapes differ: {tuple(d.shape)} vs {tuple(c.shape)}")
    if not torch.equal(d, c):
        raise AssertionError("token ids differ")

if __name__ == "__main__":
    dinov3 = WorldEncoderDinov3(device="cpu")
    print(f"device={dinov3.device}")

    texts = ["robot arm resting", "robot arm grasping a dinosaur"]
    
    _assert_same_tokens(dinov3.tokenizer, texts)

    B, H, W = 2, 224, 224
    img = torch.randn(B, 3, H, W)
    out = dinov3.encode_image(img, text_head=False, normalize=True)
    for k, v in out.items():
        if v is not None:
            print(f"{k}: {tuple(v.shape)}")