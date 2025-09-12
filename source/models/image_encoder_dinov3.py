import torch
from typing import Dict

class ImageEncoderDinov3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model, _ = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vitl16_dinotxt_tet1280d20h24l",
            source="github")
        self.visual_model = model.visual_model
        
        # Free up memory
        del model
        import gc
        gc.collect()

    def forward(self, images: torch.Tensor, text_head: bool = True, normalize: bool = True) -> Dict[str, torch.Tensor]:
        V = self.visual_model
        B = V.backbone
        H = V.head
        num_reg = V.num_register_tokens

        tokens = B.get_intermediate_layers(
            images,
            n=V.patch_token_layer,
            return_class_token=True,
            return_extra_tokens=True,
            norm=False if text_head else normalize, # if no text head, return normalized features, else raw features for head processing
        )
        cls_bb = tokens[-1][1]         # [B, C]
        patch_bb = tokens[0][0]        # [B, P, C]
        reg_bb = tokens[0][2]          # [B, R, C]

        head_feat = None
        patch_head = None

        if text_head:
            image_tokens = torch.cat([cls_bb.unsqueeze(1), reg_bb, patch_bb], dim=1)  # [B, 1+R+P, C]
            image_tokens = H(image_tokens)

            cls_head = image_tokens[:, 0]                # [B, D]
            patch_head = image_tokens[:, num_reg + 1 :]  # [B, P, D]

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

        return {
            "cls_backbone": cls_bb,
            "patch_backbone": patch_bb,
            "register_backbone": reg_bb,
            "cls_text": head_feat,
            "patch_text": patch_head,
        }


if __name__ == "__main__":
    model = ImageEncoderDinov3()
    model.to("mps")
    B, H, W = 2, 224, 224
    x = torch.randn(B, 3, H, W).to("mps")
    out = model(x, text_head=True, normalize=True)
    for k, v in out.items():
        if v is not None:
            print(k, tuple(v.shape))
